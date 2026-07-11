#!/usr/bin/env python3
"""
AutoDrive node — end-to-end camera driving.

Subscribes to a camera, buffers the previous frame, runs the trained AutoDrive
model on (prev, curr), and converts its (distance, curvature, flag) outputs into
a geometry_msgs/Twist on /cmd_vel.

NOTE: AutoDrive is an end-to-end "drive along the road" model — it has NO
destination input, so this node does not navigate to a goal. For goal-directed
obstacle avoidance use scene_seg_navigator_node instead.

Control law (diff-drive, path curvature kappa = omega / v):
    v = max_linear_speed * clamp((distance - stop_distance)/(distance_ref - stop_distance), 0, 1) * flag_gate
    w = curvature * v * curvature_gain        (clamped to +/- max_angular_speed)
"""
import math

from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from vp_car_sim.vp_models.inference.autodrive_infer import AutoDriveInfer


def speed_from_distance(distance_m, p):
    span = max(1e-3, p['distance_ref'] - p['stop_distance'])
    frac = (distance_m - p['stop_distance']) / span
    return p['max_linear_speed'] * max(0.0, min(1.0, frac))


def flag_gate(flag_prob, p):
    mode = p['flag_mode']
    if mode == 'proceed_if_high':
        return 1.0 if flag_prob >= p['flag_threshold'] else 0.0
    if mode == 'stop_if_high':
        return 0.0 if flag_prob >= p['flag_threshold'] else 1.0
    return 1.0  # 'ignore'


def compute_command(distance_m, curvature, flag_prob, p):
    """Pure control-law mapping (no ROS) so it can be unit-tested."""
    v = speed_from_distance(distance_m, p) * flag_gate(flag_prob, p)
    w = curvature * v * p['curvature_gain']
    w = max(-p['max_angular_speed'], min(p['max_angular_speed'], w))
    return v, w


class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('image_topic', '/husky/front_camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_speed', 0.6)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('distance_ref', 30.0)
        self.declare_parameter('stop_distance', 3.0)
        self.declare_parameter('curvature_gain', 1.0)
        self.declare_parameter('flag_mode', 'ignore')          # ignore | proceed_if_high | stop_if_high
        self.declare_parameter('flag_threshold', 0.5)
        self.declare_parameter('command_smoothing', 0.3)        # low-pass alpha
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('image_timeout', 2.0)

        gp = self.get_parameter
        checkpoint_path = gp('checkpoint_path').value
        if not checkpoint_path:
            raise ValueError('No checkpoint_path provided (AutoDrive .pth).')

        self.p = {
            'max_linear_speed': gp('max_linear_speed').value,
            'max_angular_speed': gp('max_angular_speed').value,
            'distance_ref': gp('distance_ref').value,
            'stop_distance': gp('stop_distance').value,
            'curvature_gain': gp('curvature_gain').value,
            'flag_mode': gp('flag_mode').value,
            'flag_threshold': gp('flag_threshold').value,
        }
        self.alpha = gp('command_smoothing').value
        self.image_timeout = gp('image_timeout').value

        self.bridge = CvBridge()
        self.model = AutoDriveInfer(checkpoint_path=checkpoint_path)
        self.get_logger().info('AutoDrive model loaded.')

        self.prev_frame = None          # previous PIL image
        self.cmd = (0.0, 0.0)           # latest smoothed (v, w)
        self.last_image_time = None

        self.create_subscription(Image, gp('image_topic').value, self.image_cb, 1)
        self.cmd_pub = self.create_publisher(Twist, gp('cmd_vel_topic').value, 10)

        self.create_timer(1.0 / float(gp('control_rate').value), self.control_cb)
        self.get_logger().info(
            f"AutoDrive driving from {gp('image_topic').value} "
            f"(flag_mode={self.p['flag_mode']})")

    def image_cb(self, msg: Image):
        self.last_image_time = self.get_clock().now()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        curr = PILImage.fromarray(cv_image)

        if self.prev_frame is None:
            self.prev_frame = curr               # need two frames; skip first
            return

        distance_m, curvature, flag_prob = self.model.inference(self.prev_frame, curr)
        self.prev_frame = curr

        v_raw, w_raw = compute_command(distance_m, curvature, flag_prob, self.p)
        # Low-pass smooth to avoid jerky commands.
        v = self.alpha * self.cmd[0] + (1.0 - self.alpha) * v_raw
        w = self.alpha * self.cmd[1] + (1.0 - self.alpha) * w_raw
        self.cmd = (v, w)

        self.get_logger().info(
            f'dist={distance_m:6.1f}m curv={curvature:+.3f} flag={flag_prob:.2f} '
            f'-> v={v:.2f} w={w:+.2f}', throttle_duration_sec=1.0)

    def control_cb(self):
        v, w = self.cmd
        if self.last_image_time is not None:
            age = (self.get_clock().now() - self.last_image_time).nanoseconds * 1e-9
            if age > self.image_timeout:
                v, w = 0.0, 0.0                   # watchdog: perception stalled
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AutoDriveNode()
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
