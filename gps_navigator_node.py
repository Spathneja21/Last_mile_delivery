#!/usr/bin/env python3
"""
GPS go-to-goal navigator (ROS1 / husky_catkin_ws).

Takes a target lat/lon (published on /gps/target by map_app.py whenever the
user clicks the map) and drives the Husky to it with a proportional
go-to-goal controller, publishing /cmd_vel. Stops within goal_tolerance.

Frame note: the hector_gazebo GPS plugin in husky.urdf.xacro has
referenceHeading=0 and referenceLatitude/Longitude set to the same anchor
used here, so its world (x=East, y=North) axes line up with ENU. The Husky
spawns at that same world origin with yaw=0, so /odometry/filtered's (x, y)
axes are already aligned with ENU. That means the target's lat/lon only
needs to be converted to local (x, y) meters *once*, on arrival; the live
control loop then runs entirely off /odometry/filtered, which is far
higher-rate and less noisy than re-deriving position from /navsat/fix every
tick.
"""
import math

import rospy
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
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
    """Equirectangular projection around the anchor (fine at robot-navigation scales)."""
    lat0 = math.radians(anchor_lat_deg)
    x_east = EARTH_RADIUS_M * math.radians(lon_deg - anchor_lon_deg) * math.cos(lat0)
    y_north = EARTH_RADIUS_M * math.radians(lat_deg - anchor_lat_deg)
    return x_east, y_north


def compute_command(x, y, yaw, gx, gy, p):
    """Pure go-to-goal control law (no ROS) so it can be unit-tested."""
    dist = math.hypot(gx - x, gy - y)
    if dist <= p['goal_tolerance']:
        return 0.0, 0.0, True

    heading_error = normalize_angle(math.atan2(gy - y, gx - x) - yaw)
    angular = p['kp_angular'] * heading_error
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular))

    if abs(heading_error) > p['rotate_in_place_angle']:
        linear = 0.0                              # turn to face the goal first
    else:
        linear = min(p['max_linear_speed'], p['kp_linear'] * dist)

    return linear, angular, False


class GpsNavigatorNode:
    def __init__(self):
        gp = rospy.get_param
        self.anchor_lat = gp('~anchor_lat', 31.781306)
        self.anchor_lon = gp('~anchor_lon', 76.997611)
        self.p = {
            'goal_tolerance': gp('~goal_tolerance', 0.5),
            'max_linear_speed': gp('~max_linear_speed', 0.6),
            'max_angular_speed': gp('~max_angular_speed', 1.0),
            'kp_linear': gp('~kp_linear', 0.5),
            'kp_angular': gp('~kp_angular', 1.5),
            'rotate_in_place_angle': gp('~rotate_in_place_angle', 0.6),
        }
        self.odom_timeout = gp('~odom_timeout', 1.0)
        odom_topic = gp('~odom_topic', '/odometry/filtered')
        target_topic = gp('~target_topic', '/gps/target')
        cmd_vel_topic = gp('~cmd_vel_topic', '/cmd_vel')
        control_rate = gp('~control_rate', 10.0)

        self.pose = None            # (x, y, yaw) from odom
        self.goal = None            # (x, y) in the odom frame, or None until a target arrives
        self.reached = False
        self.last_odom_time = None

        rospy.Subscriber(target_topic, NavSatFix, self.target_cb)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / float(control_rate)), self.control_cb)
        rospy.loginfo('GPS navigator ready, waiting for a target on %s.', target_topic)

    def target_cb(self, msg: NavSatFix):
        gx, gy = latlon_to_local_xy(msg.latitude, msg.longitude, self.anchor_lat, self.anchor_lon)
        self.goal = (gx, gy)
        self.reached = False
        rospy.loginfo('New target: lat=%.6f lon=%.6f -> odom (%.2f, %.2f)',
                       msg.latitude, msg.longitude, gx, gy)

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = rospy.Time.now()
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def control_cb(self, _event):
        v, w = 0.0, 0.0
        stale = (self.last_odom_time is None or
                 (rospy.Time.now() - self.last_odom_time).to_sec() > self.odom_timeout)

        if self.pose is not None and self.goal is not None and not self.reached and not stale:
            x, y, yaw = self.pose
            gx, gy = self.goal
            v, w, just_reached = compute_command(x, y, yaw, gx, gy, self.p)
            if just_reached:
                self.reached = True
                rospy.loginfo('Target reached. Stopping.')

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('gps_navigator_node')
    node = GpsNavigatorNode()
    rospy.on_shutdown(lambda: node.cmd_pub.publish(Twist()))
    rospy.spin()
