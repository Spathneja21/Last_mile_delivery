#!/usr/bin/env python3
"""
Road-path waypoint navigator (ROS1 / husky_catkin_ws) — drives through an
ordered list of road-snapped waypoints using a PID heading controller, fed
directly by GPS position + raw IMU orientation.

Takes the dense, road-snapped waypoint list published on /gps/path by
map_app.py (road_network.route_latlon() — Dijkstra shortest path over the
saved OSM graph, see road_network.py) and drives through the nodes in order:
rotate to face the next node, drive to it, advance to the next once within
tolerance, repeat until the path is exhausted.

Pose feedback deliberately does NOT go through /odometry/filtered (the EKF).
Position comes straight from /navsat/fix (GPS) — the exact same topic that
drives the live map marker in map_app.py/map.html, so "where the controller
thinks the robot is" and "where the map shows it" can never diverge. Heading
comes straight from /imu/data (raw orientation, bypassing the EKF's fusion
entirely). This sidesteps a real bug found this session: with the EKF's IMU
fused differentially (no absolute heading anchor), /odometry/filtered's yaw
could drift to an arbitrary offset from true heading, causing the robot to
confidently "reach" waypoints in its own (wrong) frame while actually
driving somewhere else on the real map. GPS+raw-IMU has no such fusion step
to drift.

Unlike waypoint_path_node.py, this does NOT use move_base / actionlib and
has no obstacle-avoidance of its own — it assumes an open environment
(husky_empty_world.launch) where the road graph itself is the only routing
constraint. Use waypoint_path_node.py instead if you need real-time
obstacle avoidance via move_base's costmap.

Do not run this alongside gps_navigator_node.py or waypoint_path_node.py —
all three publish to the same /cmd_vel and will fight each other. Run only
one navigator node at a time.

Tolerances: intermediate waypoints use a loose tolerance (just pass through
without stopping to precisely re-align at every node — the road graph's
nodes can be as little as ~15m apart, so tight tolerance here would look
like constant stop-and-go). Only the final waypoint uses a tight tolerance,
matching gps_navigator_node.py's default "arrived" precision.

PID note: only the heading loop is a full PID (P handles the base response,
I corrects any steady-state bias, D damps oscillation). Linear velocity
keeps the simpler "rotate first, then drive proportional to distance"
gating from the original controller — that part was never the problem.
Gains below (Kp=1.5, Ki=0.02, Kd=0.3) are starting values, not exhaustively
tuned — expect to adjust after watching real behavior in Gazebo. Ki starts
deliberately small: an earlier Ki=0.1 was found (via live testing) to wind
up enough to make the robot circle its target indefinitely rather than
converge — see the sign-change integral reset and tightened anti-windup
below, both added specifically to prevent that failure mode from recurring
even if Ki needs to go back up later.
"""
import json
import math

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist

EARTH_RADIUS_M = 6371000.0


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Z-axis (yaw) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def latlon_to_local_xy(lat_deg, lon_deg, anchor_lat_deg, anchor_lon_deg):
    """Equirectangular projection around the anchor — same math as
    gps_navigator_node.py / waypoint_path_node.py, kept duplicated since
    these are independently runnable scripts, not a shared package."""
    lat0 = math.radians(anchor_lat_deg)
    x_east = EARTH_RADIUS_M * math.radians(lon_deg - anchor_lon_deg) * math.cos(lat0)
    y_north = EARTH_RADIUS_M * math.radians(lat_deg - anchor_lat_deg)
    return x_east, y_north


def compute_command_pid(x, y, yaw, gx, gy, tolerance, p, integral, prev_error, dt):
    """PID go-to-goal control law. State (integral, prev_error) is threaded
    in/out explicitly rather than stored on an object, so this stays a pure,
    unit-testable function like the rest of this session's controllers.

    Returns (linear, angular, reached, new_integral, new_error, dist) — dist
    and new_error (the raw heading error, radians) are exposed so the caller
    can log the live position/heading error being minimized, not just the
    discrete reached/not-reached outcome."""
    dist = math.hypot(gx - x, gy - y)
    if dist <= tolerance:
        return 0.0, 0.0, True, integral, prev_error, dist

    heading_error = normalize_angle(math.atan2(gy - y, gx - x) - yaw)

    # If the error just crossed zero, any accumulated integral is from the
    # *other* side of the target and is no longer meaningful — carrying it
    # forward is exactly what causes a stale bias to keep steering the robot
    # after it's already aligned, turning a converging approach into a
    # sustained circle (the error introduced by circling then keeps feeding
    # the same integral, making it self-sustaining). Reset on sign change.
    if prev_error != 0.0 and (heading_error > 0) != (prev_error > 0):
        integral = 0.0

    integral += heading_error * dt
    if p['ki_angular'] > 0:
        # Anti-windup: clamp so the I-term's contribution alone is capped at
        # a small fraction of the actuator limit, not the full range — the
        # I-term should only ever nudge/correct, never dominate the output.
        max_i_contribution = 0.2 * p['max_angular_speed']
        max_integral = max_i_contribution / p['ki_angular']
        integral = max(-max_integral, min(max_integral, integral))

    derivative = (heading_error - prev_error) / dt if dt > 0 else 0.0

    angular = (p['kp_angular'] * heading_error
              + p['ki_angular'] * integral
              + p['kd_angular'] * derivative)
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular))

    if abs(heading_error) > p['rotate_in_place_angle']:
        linear = 0.0                              # turn to face the goal first
    else:
        linear = min(p['max_linear_speed'], p['kp_linear'] * dist)

    return linear, angular, False, integral, heading_error, dist


class RoadPathNavigatorNode:
    def __init__(self):
        gp = rospy.get_param
        self.anchor_lat = gp('~anchor_lat', 31.781306)
        self.anchor_lon = gp('~anchor_lon', 76.997611)
        self.intermediate_tolerance = gp('~intermediate_tolerance', 1.5)
        self.final_tolerance = gp('~final_tolerance', 0.5)
        self.p = {
            'max_linear_speed': gp('~max_linear_speed', 0.6),
            'max_angular_speed': gp('~max_angular_speed', 1.0),
            'kp_linear': gp('~kp_linear', 0.5),
            'kp_angular': gp('~kp_angular', 1.5),
            'ki_angular': gp('~ki_angular', 0.02),
            'kd_angular': gp('~kd_angular', 0.3),
            'rotate_in_place_angle': gp('~rotate_in_place_angle', 0.6),
        }
        self.pose_timeout = gp('~pose_timeout', 1.0)
        gps_topic = gp('~gps_topic', '/navsat/fix')
        imu_topic = gp('~imu_topic', '/imu/data')
        path_topic = gp('~path_topic', '/gps/path')
        cmd_vel_topic = gp('~cmd_vel_topic', '/cmd_vel')
        control_rate = gp('~control_rate', 10.0)
        self.dt = 1.0 / float(control_rate)

        # --- Heading offset auto-calibration (GPS course vs raw IMU yaw).
        # The raw IMU yaw sits at a fixed, unknown offset from the ENU frame
        # our GPS-derived (x, y) uses — a hector_gazebo frame-convention
        # mismatch. Rather than guess the transform, we measure it: while
        # driving mostly-straight-forward, the direction between consecutive
        # GPS fixes (course-over-ground) IS the true heading, so
        # offset = course - imu_yaw. corrected_yaw = imu_yaw + offset then
        # feeds the controller. Works for any constant offset (90 deg, 78 deg,
        # anything) and self-corrects any residual slow drift.
        self.course_min_dist = gp('~course_min_dist', 0.4)    # m of travel before a course fix is trusted
        self.offset_filter_alpha = gp('~offset_filter_alpha', 0.3)  # low-pass on the measured offset
        self.calib_min_speed = gp('~calib_min_speed', 0.15)   # only calibrate while driving forward...
        self.calib_max_turn = gp('~calib_max_turn', 0.25)     # ...and not turning hard
        self.bootstrap_speed = gp('~bootstrap_speed', 0.3)    # straight-drive speed until offset is known

        self.gps_position = None        # (x, y), local frame, from /navsat/fix
        self.gps_latlon = None           # raw (lat, lon), for logging only
        self.imu_yaw = None              # yaw, from /imu/data (raw, no EKF)
        self.last_gps_time = None
        self.last_imu_time = None

        self.heading_offset = 0.0        # corrected_yaw = imu_yaw + heading_offset
        self.offset_initialized = False
        self.course_baseline = None      # (x, y) position at the last course measurement
        self.last_cmd_v = 0.0            # last commanded v/w, so gps_cb knows if we were
        self.last_cmd_w = 0.0            # driving straight enough to trust a course fix

        self.waypoints = []             # list of (x, y, lat, lon) still to visit, in order
        self.covered = []               # list of (lat, lon) already reached, for this path
        self.total_waypoints = 0

        self.integral_heading = 0.0     # PID state, reset per-leg and per-new-path
        self.prev_heading_error = 0.0

        rospy.Subscriber(path_topic, String, self.path_cb, queue_size=1)
        rospy.Subscriber(gps_topic, NavSatFix, self.gps_cb, queue_size=10)
        rospy.Subscriber(imu_topic, Imu, self.imu_cb, queue_size=10)
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        rospy.Timer(rospy.Duration(self.dt), self.control_cb)
        rospy.loginfo('Road-path navigator ready (GPS+IMU, PID heading), '
                      'waiting for a path on %s.', path_topic)

    def path_cb(self, msg: String):
        try:
            waypoints = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            rospy.logwarn('road_path_navigator: could not parse path JSON: %r', msg.data)
            return

        local_path = [
            latlon_to_local_xy(wp['latitude'], wp['longitude'],
                               self.anchor_lat, self.anchor_lon) + (wp['latitude'], wp['longitude'])
            for wp in waypoints
        ]
        self.waypoints = local_path     # a new path always replaces the current one
        self.covered = []               # fresh path -> reset covered-points log
        self.total_waypoints = len(local_path)
        self._reset_pid()               # fresh path -> no carried-over I/D state
        rospy.loginfo('New path received: %d waypoint(s). Starting navigation.', len(local_path))
        # First waypoint is already the nearest road node to the robot's
        # current position (map_app.py's route_latlon() snaps the start to
        # it, and prepends the robot's exact GPS position ahead of that) —
        # so driving to self.waypoints[0] first IS "go to nearest node, then
        # follow the line."
        self._log_next_target()

    def _reset_pid(self):
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0

    def _log_next_target(self):
        if not self.waypoints:
            rospy.loginfo('[ROAD PATH] Path complete — %d/%d point(s) covered.',
                         len(self.covered), self.total_waypoints)
            return
        _, _, lat, lon = self.waypoints[0]
        rospy.loginfo('[ROAD PATH] Next target (%d/%d): (%.6f, %.6f)',
                     len(self.covered) + 1, self.total_waypoints, lat, lon)

    def gps_cb(self, msg: NavSatFix):
        self.last_gps_time = rospy.Time.now()
        x, y = latlon_to_local_xy(msg.latitude, msg.longitude,
                                  self.anchor_lat, self.anchor_lon)
        self.gps_position = (x, y)
        self.gps_latlon = (msg.latitude, msg.longitude)   # raw, for logging only
        self._update_heading_offset(x, y)

    def _update_heading_offset(self, x, y):
        """Measure the IMU-yaw-to-ENU offset from GPS course-over-ground while
        driving mostly straight forward. See the calibration note in __init__."""
        if self.course_baseline is None:
            self.course_baseline = (x, y)
            return
        bx, by = self.course_baseline
        dx, dy = x - bx, y - by
        if math.hypot(dx, dy) < self.course_min_dist:
            return                                  # not enough travel yet — keep accumulating
        # Moved far enough for a course fix. Advance the baseline so the next
        # window starts fresh (never spans a turn).
        self.course_baseline = (x, y)

        # Only trust this fix if we were driving forward and not turning hard —
        # course-over-ground is meaningless during in-place rotation.
        if (self.imu_yaw is None
                or self.last_cmd_v < self.calib_min_speed
                or abs(self.last_cmd_w) > self.calib_max_turn):
            return

        gps_course = math.atan2(dy, dx)
        measured_offset = normalize_angle(gps_course - self.imu_yaw)
        if not self.offset_initialized:
            self.heading_offset = measured_offset
            self.offset_initialized = True
            rospy.loginfo('[ROAD PATH] Heading offset calibrated: %.1f deg '
                         '(IMU yaw frame vs GPS/ENU).', math.degrees(self.heading_offset))
        else:
            # Circular low-pass filter toward the newly measured offset.
            delta = normalize_angle(measured_offset - self.heading_offset)
            self.heading_offset = normalize_angle(
                self.heading_offset + self.offset_filter_alpha * delta)

    def corrected_yaw(self):
        return normalize_angle(self.imu_yaw + self.heading_offset)

    def imu_cb(self, msg: Imu):
        self.last_imu_time = rospy.Time.now()
        q = msg.orientation
        self.imu_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def control_cb(self, _event):
        v, w = 0.0, 0.0
        now = rospy.Time.now()
        gps_stale = (self.last_gps_time is None
                    or (now - self.last_gps_time).to_sec() > self.pose_timeout)
        imu_stale = (self.last_imu_time is None
                    or (now - self.last_imu_time).to_sec() > self.pose_timeout)

        if (self.gps_position is not None and self.imu_yaw is not None
                and not gps_stale and not imu_stale and self.waypoints
                and not self.offset_initialized):
            # Bootstrap: we don't yet know the IMU->ENU heading offset, so we
            # can't steer meaningfully. Drive straight forward (in whatever
            # direction the robot currently faces) so gps_cb can measure the
            # offset from course-over-ground. This is deadlock-free: it never
            # gates on heading, guaranteeing the forward motion needed to
            # calibrate. Once offset_initialized flips true (~course_min_dist
            # of travel), the normal controller below takes over next tick.
            v, w = self.bootstrap_speed, 0.0
            rospy.loginfo_throttle(1.0, '[ROAD PATH] Calibrating heading offset — '
                                        'driving straight to read GPS course...')

        elif (self.gps_position is not None and self.imu_yaw is not None
                and not gps_stale and not imu_stale):
            x, y = self.gps_position
            yaw = self.corrected_yaw()      # IMU yaw + measured GPS-course offset
            # Pop through any waypoints already within tolerance in this same
            # tick (handles closely-spaced nodes / control-loop gaps cleanly,
            # rather than a one-tick stutter at each one).
            while self.waypoints:
                gx, gy, glat, glon = self.waypoints[0]
                is_final = len(self.waypoints) == 1
                tolerance = self.final_tolerance if is_final else self.intermediate_tolerance

                v, w, reached, self.integral_heading, self.prev_heading_error, dist = compute_command_pid(
                    x, y, yaw, gx, gy, tolerance, self.p,
                    self.integral_heading, self.prev_heading_error, self.dt)
                if not reached:
                    # Throttled (1 Hz) so this doesn't flood the terminal at
                    # the 10 Hz control rate — shows the live error the PID
                    # is actually driving toward zero. Robot and target
                    # coordinates are logged side by side specifically so a
                    # stuck/non-progressing robot (dist not shrinking despite
                    # v>0) is immediately visible by comparing them directly,
                    # rather than having to cross-reference two topics by hand.
                    rlat, rlon = self.gps_latlon
                    rospy.loginfo_throttle(
                        1.0,
                        '[ROAD PATH] Robot: (%.6f, %.6f)  Target (%d/%d): (%.6f, %.6f)  '
                        'dist=%.2fm heading=%.1fdeg  cmd: v=%.2f w=%.2f',
                        rlat, rlon, len(self.covered) + 1, self.total_waypoints, glat, glon,
                        dist, math.degrees(self.prev_heading_error), v, w)
                    break

                self.waypoints.pop(0)
                self.covered.append((glat, glon))
                self._reset_pid()   # fresh leg -> no carried-over I/D state
                covered_str = ' -> '.join(f'({lat:.5f},{lon:.5f})' for lat, lon in self.covered)
                rospy.loginfo('[ROAD PATH] Reached (%d/%d): (%.6f, %.6f)',
                             len(self.covered), self.total_waypoints, glat, glon)
                rospy.loginfo('[ROAD PATH] Covered so far: %s', covered_str)
                self._log_next_target()
                v, w = 0.0, 0.0

        # Record what we just commanded so gps_cb's calibration can tell
        # whether the robot was driving straight enough to trust a course fix.
        self.last_cmd_v = v
        self.last_cmd_w = w

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('road_path_navigator_node')
    node = RoadPathNavigatorNode()
    rospy.on_shutdown(lambda: node.cmd_pub.publish(Twist()))
    rospy.spin()
