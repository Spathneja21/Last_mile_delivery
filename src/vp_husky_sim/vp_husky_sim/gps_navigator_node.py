#!/usr/bin/env python3
"""
GPS go-to-goal navigator.

Takes a target lat/lon (published on /gps/target by map_app.py whenever the
user clicks the map) and drives the Husky to it with a proportional
go-to-goal controller, publishing /cmd_vel. Stops within goal_tolerance.

Frame note: the world's <spherical_coordinates> anchor (see vp_track.world /
vp_city.world) has heading_deg=0, so world ENU axes line up with (x=East,
y=North). The Husky spawns at that same world origin with yaw=0, so /odom's
(x, y) axes are already aligned with ENU. That means the target's lat/lon
only needs to be converted to local (x, y) meters *once*, on arrival; the
live control loop then runs entirely off /odom, which is far higher-rate and
less noisy than re-deriving position from /gps/fix every tick.
"""
import math

import rclpy
from rclpy.node import Node
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


class GpsNavigatorNode(Node):
    def __init__(self):
        super().__init__('gps_navigator_node')

        self.declare_parameter('target_topic', '/gps/target')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('anchor_lat', 37.7749)
        self.declare_parameter('anchor_lon', -122.4194)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('max_linear_speed', 0.6)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('rotate_in_place_angle', 0.6)
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('odom_timeout', 1.0)

        gp = self.get_parameter
        self.anchor_lat = gp('anchor_lat').value
        self.anchor_lon = gp('anchor_lon').value
        self.p = {
            'goal_tolerance': gp('goal_tolerance').value,
            'max_linear_speed': gp('max_linear_speed').value,
            'max_angular_speed': gp('max_angular_speed').value,
            'kp_linear': gp('kp_linear').value,
            'kp_angular': gp('kp_angular').value,
            'rotate_in_place_angle': gp('rotate_in_place_angle').value,
        }
        self.odom_timeout = gp('odom_timeout').value

        self.pose = None            # (x, y, yaw) from /odom
        self.goal = None            # (x, y) in the odom frame, or None until a target arrives
        self.reached = False
        self.last_odom_time = None

        self.create_subscription(NavSatFix, gp('target_topic').value, self.target_cb, 10)
        self.create_subscription(Odometry, gp('odom_topic').value, self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, gp('cmd_vel_topic').value, 10)

        self.create_timer(1.0 / float(gp('control_rate').value), self.control_cb)
        self.get_logger().info('GPS navigator ready, waiting for a target on /gps/target.')

    def target_cb(self, msg: NavSatFix):
        gx, gy = latlon_to_local_xy(msg.latitude, msg.longitude, self.anchor_lat, self.anchor_lon)
        self.goal = (gx, gy)
        self.reached = False
        self.get_logger().info(
            f'New target: lat={msg.latitude:.6f} lon={msg.longitude:.6f} -> odom ({gx:.2f}, {gy:.2f})')

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = self.get_clock().now()
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def control_cb(self):
        v, w = 0.0, 0.0
        stale = (self.last_odom_time is None or
                 (self.get_clock().now() - self.last_odom_time).nanoseconds * 1e-9 > self.odom_timeout)

        if self.pose is not None and self.goal is not None and not self.reached and not stale:
            x, y, yaw = self.pose
            gx, gy = self.goal
            v, w, just_reached = compute_command(x, y, yaw, gx, gy, self.p)
            if just_reached:
                self.reached = True
                self.get_logger().info('Target reached. Stopping.')

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = GpsNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
