#!/usr/bin/env python3
# Main /cmd_vel producer for the delivery robot: follows the road path from
# map_app.py using GPS+IMU heading PID, blends in vision-based obstacle
# avoidance, and triggers an automatic replan when pushed off the route.
"""
Fusion navigator (ROS1 / husky_catkin_ws) — road-path following + vision
obstacle avoidance. The single owner of /cmd_vel in the merged pipeline
(see MERGED_NAVIGATION_PLAN.md).

Forked from road_path_navigator_node.py — keeps its GPS(/navsat/fix)+IMU
(/imu/data) heading PID and its self-calibrating IMU-yaw->ENU heading offset
(course-over-ground while driving straight) verbatim; those are the hard-won,
drift-free parts. What's added:

  1. Vision arbitration. Subscribes /vision/avoidance (published by
     vision_avoidance_node.py). When the path straight ahead is CLEAR it steers
     toward the next densified road viewpoint (normal line-following). When
     vision reports the way BLOCKED it overrides steering toward the clearest
     camera bin nearest the road direction — i.e. it deviates around the
     obstacle while still biasing toward the goal.

  2. Cross-track + replan. It measures how far the robot has been pushed off
     the planned line (perpendicular distance to the current segment). Once the
     detour has taken it past a threshold AND vision reports clear again
     (obstacle passed), it publishes /replan_request; map_app.py reroutes from
     the robot's current GPS to the stored destination and republishes a fresh
     /gps/path, which replaces the active path here. Debounced so one detour =
     one replan.

Do not run this alongside road_path_navigator_node.py / gps_navigator_node.py /
waypoint_path_node.py / webcam_navigator_node.py — all publish /cmd_vel and
would fight. Run the vision ADVISOR (vision_avoidance_node.py, which does not
touch /cmd_vel) plus this node.

Angle conventions: both the GPS-derived heading error toward a viewpoint and
the vision bin bearings are body-frame "turn this much from current heading"
angles with +left/+yaw (REP-103), so they arbitrate directly with no sign flip
(scene_decision.py sets bin_bearing = +left; verified).
"""
import json
import math

import rospy
from std_msgs.msg import String, Empty
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist

EARTH_RADIUS_M = 6371000.0  # mean Earth radius, used for the local flat-earth lat/lon projection


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
    road_path_navigator_node.py / gps_navigator_node.py."""
    lat0 = math.radians(anchor_lat_deg)
    x_east = EARTH_RADIUS_M * math.radians(lon_deg - anchor_lon_deg) * math.cos(lat0)
    y_north = EARTH_RADIUS_M * math.radians(lat_deg - anchor_lat_deg)
    return x_east, y_north


def point_segment_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from point (px,py) to the segment (ax,ay)-(bx,by),
    clamped at the endpoints. Used for cross-track error off the planned line."""
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def heading_pid(heading_error, p, integral, prev_error, dt):
    """Full PID heading loop (P base response, I steady-state bias, D damping),
    with the sign-change integral reset + tight anti-windup from
    road_path_navigator_node.py. Pure/threaded state so both the follow and
    avoid branches can drive it with whatever heading_error they choose.

    Returns (angular, new_integral, new_error)."""
    # Error crossed zero -> accumulated integral is from the other side of the
    # target and would keep steering after alignment (a converging approach
    # turns into a sustained circle). Reset on sign change.
    if prev_error != 0.0 and (heading_error > 0) != (prev_error > 0):
        integral = 0.0

    integral += heading_error * dt
    if p['ki_angular'] > 0:
        # Anti-windup: cap the I-term's contribution alone at a small fraction
        # of the actuator limit — it should nudge, never dominate.
        max_i_contribution = 0.2 * p['max_angular_speed']
        max_integral = max_i_contribution / p['ki_angular']
        integral = max(-max_integral, min(max_integral, integral))

    derivative = (heading_error - prev_error) / dt if dt > 0 else 0.0
    angular = (p['kp_angular'] * heading_error
               + p['ki_angular'] * integral
               + p['kd_angular'] * derivative)
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular))
    return angular, integral, heading_error


class FusionNavigatorNode:
    def __init__(self):
        # Values below come from config/fusion_navigator.yaml, which
        # run_fusion_navigator.sh loads onto the param server before this
        # node starts — edit that file to tune behavior instead of these defaults.
        gp = rospy.get_param
        self.anchor_lat = gp('~anchor_lat', 31.781306)  # local-xy projection origin; matches map_app.py's anchor
        self.anchor_lon = gp('~anchor_lon', 76.997611)
        self.intermediate_tolerance = gp('~intermediate_tolerance', 1.0)  # m, radius to count a mid-path waypoint reached
        self.final_tolerance = gp('~final_tolerance', 0.5)  # m, tighter radius for the last waypoint
        self.p = {
            'max_linear_speed': gp('~max_linear_speed', 0.6),
            'max_angular_speed': gp('~max_angular_speed', 1.0),
            'kp_linear': gp('~kp_linear', 0.5),
            'kp_angular': gp('~kp_angular', 1.5),
            'ki_angular': gp('~ki_angular', 0.02),
            'kd_angular': gp('~kd_angular', 0.3),
            'rotate_in_place_angle': gp('~rotate_in_place_angle', 0.6),  # rad heading error above which the robot stops and turns instead of driving forward
        }
        self.pose_timeout = gp('~pose_timeout', 1.0)  # s, GPS/IMU older than this is treated as stale -> robot halts

        # --- Vision arbitration params.
        self.vision_timeout = gp('~vision_timeout', 0.5)   # advisory older than this -> treat as clear
        self.clear_score_min = gp('~clear_score_min', 0.0)  # per-bin clear score to count a bin as drivable

        # --- Cross-track / replan params.
        self.replan_dist = gp('~replan_dist', 3.0)          # m off the line before a replan is warranted
        self.replan_min_interval = gp('~replan_min_interval', 5.0)  # s debounce between replans

        gps_topic = gp('~gps_topic', '/navsat/fix')
        imu_topic = gp('~imu_topic', '/imu/data')
        path_topic = gp('~path_topic', '/gps/path')
        advisory_topic = gp('~advisory_topic', '/vision/avoidance')
        replan_topic = gp('~replan_topic', '/replan_request')
        cmd_vel_topic = gp('~cmd_vel_topic', '/cmd_vel')
        control_rate = gp('~control_rate', 10.0)
        self.dt = 1.0 / float(control_rate)

        # --- Heading offset auto-calibration (see road_path_navigator_node.py).
        self.course_min_dist = gp('~course_min_dist', 0.4)  # m the robot must move before a new GPS-course sample is taken
        self.offset_filter_alpha = gp('~offset_filter_alpha', 0.3)  # low-pass weight for updating heading_offset from new samples
        self.calib_min_speed = gp('~calib_min_speed', 0.15)  # m/s below which GPS course is too noisy to trust for calibration
        self.calib_max_turn = gp('~calib_max_turn', 0.25)  # rad/s above which the robot isn't driving straight enough to calibrate
        self.bootstrap_speed = gp('~bootstrap_speed', 0.3)  # m/s used to drive straight while heading_offset is still uncalibrated

        self.gps_position = None
        self.gps_latlon = None
        self.imu_yaw = None
        self.last_gps_time = None
        self.last_imu_time = None

        self.heading_offset = 0.0     # measured IMU-yaw -> GPS/ENU heading correction
        self.offset_initialized = False  # True once heading_offset has a real (non-zero-guess) value
        self.course_baseline = None   # (x, y) GPS position the current calibration sample is measured from
        self.last_cmd_v = 0.0         # last published linear speed, used to gate calibration on "driving straight"
        self.last_cmd_w = 0.0         # last published angular speed, same purpose

        self.waypoints = []             # (x, y, lat, lon) still to visit, in order
        self.covered = []               # (lat, lon) already reached
        self.total_waypoints = 0
        self.segment_start = None       # (x, y) of the last passed viewpoint, for cross-track

        self.integral_heading = 0.0
        self.prev_heading_error = 0.0

        # --- Latest vision advisory (parsed dict) + its receive time.
        self.vision = None
        self.last_vision_time = None
        self.last_replan_time = rospy.Time(0)

        rospy.Subscriber(path_topic, String, self.path_cb, queue_size=1)
        rospy.Subscriber(gps_topic, NavSatFix, self.gps_cb, queue_size=10)
        rospy.Subscriber(imu_topic, Imu, self.imu_cb, queue_size=10)
        rospy.Subscriber(advisory_topic, String, self.vision_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.replan_pub = rospy.Publisher(replan_topic, Empty, queue_size=1)

        rospy.Timer(rospy.Duration(self.dt), self.control_cb)
        rospy.loginfo('Fusion navigator ready (GPS+IMU PID + vision avoidance), '
                      'waiting for a path on %s.', path_topic)

    # ------------------------------------------------------------------ inputs

    def path_cb(self, msg: String):
        try:
            waypoints = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            rospy.logwarn('fusion_navigator: could not parse path JSON: %r', msg.data)
            return

        local_path = [
            latlon_to_local_xy(wp['latitude'], wp['longitude'],
                               self.anchor_lat, self.anchor_lon) + (wp['latitude'], wp['longitude'])
            for wp in waypoints
        ]
        self.waypoints = local_path     # a new path always replaces the current one
        self.covered = []
        self.total_waypoints = len(local_path)
        # Cross-track baseline starts at the first point (the robot's own start
        # position — map_app prepends it), so deviation is measured from the
        # very first leg.
        self.segment_start = (local_path[0][0], local_path[0][1]) if local_path else None
        self._reset_pid()
        rospy.loginfo('New path received: %d waypoint(s). Starting navigation.', len(local_path))
        self._log_next_target()

    def vision_cb(self, msg: String):
        try:
            self.vision = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            rospy.logwarn('fusion_navigator: could not parse vision advisory: %r', msg.data)
            return
        self.last_vision_time = rospy.Time.now()

    def gps_cb(self, msg: NavSatFix):
        self.last_gps_time = rospy.Time.now()
        x, y = latlon_to_local_xy(msg.latitude, msg.longitude,
                                  self.anchor_lat, self.anchor_lon)
        self.gps_position = (x, y)
        self.gps_latlon = (msg.latitude, msg.longitude)
        self._update_heading_offset(x, y)

    def imu_cb(self, msg: Imu):
        self.last_imu_time = rospy.Time.now()
        q = msg.orientation
        self.imu_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    # --------------------------------------------------- heading calibration

    def _update_heading_offset(self, x, y):
        """Measure the IMU-yaw-to-ENU offset from GPS course-over-ground while
        driving mostly straight forward (see road_path_navigator_node.py)."""
        if self.course_baseline is None:
            self.course_baseline = (x, y)
            return
        bx, by = self.course_baseline
        dx, dy = x - bx, y - by
        if math.hypot(dx, dy) < self.course_min_dist:
            return
        self.course_baseline = (x, y)

        if (self.imu_yaw is None
                or self.last_cmd_v < self.calib_min_speed
                or abs(self.last_cmd_w) > self.calib_max_turn):
            return

        gps_course = math.atan2(dy, dx)
        measured_offset = normalize_angle(gps_course - self.imu_yaw)
        if not self.offset_initialized:
            self.heading_offset = measured_offset
            self.offset_initialized = True
            rospy.loginfo('[FUSION] Heading offset calibrated: %.1f deg '
                          '(IMU yaw frame vs GPS/ENU).', math.degrees(self.heading_offset))
        else:
            delta = normalize_angle(measured_offset - self.heading_offset)
            self.heading_offset = normalize_angle(
                self.heading_offset + self.offset_filter_alpha * delta)

    def corrected_yaw(self):
        return normalize_angle(self.imu_yaw + self.heading_offset)

    # ---------------------------------------------------------------- helpers

    def _reset_pid(self):
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0

    def _log_next_target(self):
        if not self.waypoints:
            rospy.loginfo('[FUSION] Path complete — %d/%d point(s) covered.',
                          len(self.covered), self.total_waypoints)
            return
        _, _, lat, lon = self.waypoints[0]
        rospy.loginfo('[FUSION] Next target (%d/%d): (%.6f, %.6f)',
                      len(self.covered) + 1, self.total_waypoints, lat, lon)

    def _vision_fresh(self):
        return (self.vision is not None and self.last_vision_time is not None
                and (rospy.Time.now() - self.last_vision_time).to_sec() <= self.vision_timeout)

    def _arbitrate(self, road_err, dist):
        """Blend road heading with vision avoidance.
        Returns (heading_error, linear, mode)."""
        vision = self.vision if self._vision_fresh() else None

        if vision is None or not vision.get('blocked', False):
            # Path ahead clear (or no fresh vision) -> follow the line.
            linear = (0.0 if abs(road_err) > self.p['rotate_in_place_angle']
                      else min(self.p['max_linear_speed'], self.p['kp_linear'] * dist))
            return road_err, linear, 'FOLLOW'

        # Blocked -> steer toward the clearest bin nearest the road direction.
        bearings = vision.get('bin_bearings', [])
        clears = vision.get('bin_clear', [])
        block_amount = float(vision.get('block_amount', 1.0))
        clear_idx = [i for i, s in enumerate(clears) if s >= self.clear_score_min]

        if clear_idx:
            best_i = min(clear_idx,  # index of the drivable camera bin closest to the road heading
                         key=lambda i: abs(normalize_angle(bearings[i] - road_err)))
            heading_error = bearings[best_i]
            linear = (0.0 if abs(heading_error) > self.p['rotate_in_place_angle']
                      else self.p['max_linear_speed'] * (1.0 - block_amount))  # slow down proportionally to how blocked the view is
            return heading_error, linear, 'AVOID'

        # Nothing drivable -> rotate in place toward the least-bad direction.
        heading_error = float(vision.get('desired_bearing', 0.0)) 
        return heading_error, 0.0, 'BLOCKED'

    def _maybe_replan(self, x, y):
        """Fire a debounced /replan_request once the robot has been pushed off
        the line and vision is clear again (the obstacle is behind us)."""
        if not self.offset_initialized or not self.waypoints or self.segment_start is None:
            return
        # Only replan when the way is clear again — otherwise we're mid-dodge and
        # a reroute-through-the-obstacle would just get abandoned next tick.
        if self._vision_fresh() and self.vision.get('blocked', False):
            return
        gx, gy, _, _ = self.waypoints[0]
        sx, sy = self.segment_start
        cross_track = point_segment_distance(x, y, sx, sy, gx, gy)
        if cross_track < self.replan_dist:
            return
        now = rospy.Time.now()
        if (now - self.last_replan_time).to_sec() < self.replan_min_interval:
            return
        self.last_replan_time = now
        rospy.logwarn('[FUSION] Off the line by %.1fm and clear — requesting replan.',
                      cross_track)
        self.replan_pub.publish(Empty())

    # ------------------------------------------------------------- control loop

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
            # Bootstrap: don't yet know the IMU->ENU offset, so steering is
            # meaningless. Drive straight so gps_cb can measure it from course.
            v, w = self.bootstrap_speed, 0.0
            rospy.loginfo_throttle(1.0, '[FUSION] Calibrating heading offset — '
                                        'driving straight to read GPS course...')

        elif (self.gps_position is not None and self.imu_yaw is not None
                and not gps_stale and not imu_stale and self.waypoints):
            x, y = self.gps_position
            yaw = self.corrected_yaw()

            # Pop any waypoints already within tolerance this tick.
            while self.waypoints:
                gx, gy, glat, glon = self.waypoints[0]
                is_final = len(self.waypoints) == 1
                tolerance = self.final_tolerance if is_final else self.intermediate_tolerance
                if math.hypot(gx - x, gy - y) > tolerance:
                    break
                self.segment_start = (gx, gy)
                self.waypoints.pop(0)
                self.covered.append((glat, glon))
                self._reset_pid()
                rospy.loginfo('[FUSION] Reached (%d/%d): (%.6f, %.6f)',
                              len(self.covered), self.total_waypoints, glat, glon)
                self._log_next_target()

            if self.waypoints:
                gx, gy, glat, glon = self.waypoints[0]
                dist = math.hypot(gx - x, gy - y)
                road_err = normalize_angle(math.atan2(gy - y, gx - x) - yaw)

                heading_error, v, mode = self._arbitrate(road_err, dist)
                w, self.integral_heading, self.prev_heading_error = heading_pid(
                    heading_error, self.p, self.integral_heading,
                    self.prev_heading_error, self.dt)

                self._maybe_replan(x, y)

                rlat, rlon = self.gps_latlon
                rospy.loginfo_throttle(
                    1.0,
                    '[FUSION] %-7s Robot: (%.6f, %.6f)  Target (%d/%d): (%.6f, %.6f)  '
                    'dist=%.2fm herr=%.1fdeg  cmd: v=%.2f w=%.2f',
                    mode, rlat, rlon, len(self.covered) + 1, self.total_waypoints,
                    glat, glon, dist, math.degrees(heading_error), v, w)

        self.last_cmd_v = v
        self.last_cmd_w = w

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('fusion_navigator_node')
    node = FusionNavigatorNode()
    rospy.on_shutdown(lambda: node.cmd_pub.publish(Twist()))
    rospy.spin()
