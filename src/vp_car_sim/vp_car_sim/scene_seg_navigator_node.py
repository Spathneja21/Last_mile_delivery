#!/usr/bin/env python3
"""
SceneSeg-based reactive navigator.

Uses the (trained) SceneSeg model as the only perception source:
  class 0 = background, 1 = foreground object (obstacle), 2 = drivable road.

Each frame it splits the lower image into vertical bins, scores every bin by
  (goal alignment) + (road / free space) - (obstacle density),
picks the best bin, and steers toward it while heading to a goal (goal_x, goal_y
in the odom frame). It avoids foreground obstacles and stops at the goal.

This is a reactive controller (no global planner / map), suited to open-ish
scenes like vp_city.world. Publishes /cmd_vel and a debug overlay.
"""
import math

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from vp_car_sim.vp_models.pth_pipeline.inference.scene_seg_infer import SceneSegNetworkInfer

# SceneSeg input size + class ids
MODEL_WIDTH = 640
MODEL_HEIGHT = 320
CLS_FOREGROUND = 1  # SceneSeg class id: obstacle
CLS_ROAD = 2  # SceneSeg class id: drivable road (class 0 = background, implicit)


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Z-axis (yaw) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_command(prediction, heading_error, dist_to_goal, p):
    """
    Pure decision function (no ROS) so it can be unit-tested.

    prediction    : (H, W) int array of SceneSeg class ids
    heading_error : signed angle to goal in robot frame [rad] (+ = goal to the left)
    dist_to_goal  : metres
    p             : dict of parameters (see node defaults)

    Returns (linear_x, angular_z, info) where info has per-bin debug fields.
    """
    h, w = prediction.shape
    roi = prediction[int(p['roi_top_frac'] * h):, :]            # lower band = near road
    n = p['num_bins']  # number of vertical scan bins across the frame
    cols = np.array_split(np.arange(w), n)

    half_fov = math.radians(p['camera_hfov_deg']) / 2.0

    bin_bearing = np.zeros(n)
    fg_frac = np.zeros(n)
    road_frac = np.zeros(n)
    for i, c in enumerate(cols):
        seg = roi[:, c[0]:c[-1] + 1]
        fg_frac[i] = float(np.mean(seg == CLS_FOREGROUND))
        road_frac[i] = float(np.mean(seg == CLS_ROAD))
        u = (((c[0] + c[-1]) / 2.0) - w / 2.0) / (w / 2.0)      # [-1 (left) .. +1 (right)]
        bin_bearing[i] = -u * half_fov                          # +bearing = left = +yaw

    # Goal alignment in [0,1] (1 = bin points exactly at the goal bearing)
    goal_align = 1.0 - np.abs(bin_bearing - heading_error) / math.pi
    score = (p['goal_weight'] * goal_align
             + p['road_weight'] * road_frac
             - p['obstacle_weight'] * fg_frac)

    best = int(np.argmax(score))
    desired_bearing = float(bin_bearing[best])

    # How blocked is the path straight ahead? (central third of the bins)
    lo = n // 3
    hi = n - n // 3
    center_fg = float(np.max(fg_frac[lo:hi])) if hi > lo else float(fg_frac[best])

    # Steering: proportional to the chosen bearing
    angular = p['kp_steer'] * desired_bearing
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular))

    # Speed: slow down as the path ahead gets blocked; stop & rotate if blocked or
    # the goal needs a big turn.
    block = min(1.0, center_fg / max(1e-3, p['blocked_stop_fraction']))
    linear = p['max_linear_speed'] * (1.0 - block)
    if center_fg >= p['blocked_stop_fraction'] or abs(desired_bearing) > p['rotate_in_place_angle']:
        linear = 0.0                                            # rotate in place to find a way

    info = {'fg_frac': fg_frac, 'road_frac': road_frac, 'bin_bearing': bin_bearing,
            'best': best, 'center_fg': center_fg, 'cols': cols}
    return linear, angular, info


def overlay_image(prediction, info, p):
    """Colour the seg mask (road=green, obstacle=red) and mark the chosen bin."""
    h, w = prediction.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[prediction == CLS_ROAD] = (40, 180, 40)
    img[prediction == CLS_FOREGROUND] = (220, 40, 40)
    img[prediction == 0] = (60, 60, 60)
    cols = info['cols'][info['best']]
    c0, c1 = cols[0], cols[-1]
    img[:, c0:c1 + 1, :] = (img[:, c0:c1 + 1, :] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    return img


class SceneSegNavigator(Node):
    def __init__(self):
        super().__init__('scene_seg_navigator')

        # --- parameters ---
        self.declare_parameter('checkpoint_path', '')  # path to SceneSeg .pth weights
        self.declare_parameter('image_topic', '/husky/front_camera/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('overlay_topic', '/vision_pilot/navigator/overlay')
        self.declare_parameter('goal_x', 0.0)  # goal position in odom frame
        self.declare_parameter('goal_y', 0.0)  # goal position in odom frame
        self.declare_parameter('goal_tolerance', 1.5)  # distance (m) within which the goal counts as reached
        self.declare_parameter('max_linear_speed', 0.6)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_steer', 1.5)  # proportional steering gain
        self.declare_parameter('num_bins', 9)  # number of vertical scan bins across the frame
        self.declare_parameter('roi_top_frac', 0.4)  # fraction of image height cropped off the top before scoring
        self.declare_parameter('goal_weight', 1.0)  # bin score weight for goal-bearing alignment
        self.declare_parameter('road_weight', 0.6)  # bin score weight for drivable-road coverage
        self.declare_parameter('obstacle_weight', 2.0)  # bin score penalty weight for obstacle coverage
        self.declare_parameter('blocked_stop_fraction', 0.4)  # central obstacle fraction above which the path counts as blocked
        self.declare_parameter('rotate_in_place_angle', 0.6)  # bearing magnitude (rad) beyond which the robot stops and rotates
        self.declare_parameter('camera_hfov_deg', 50.0)  # camera horizontal field of view, used to map bins to bearings
        self.declare_parameter('control_rate', 10.0)  # Hz, rate of the cmd_vel control timer
        self.declare_parameter('image_timeout', 2.0)  # seconds without a new image before the watchdog stops the robot

        gp = self.get_parameter
        checkpoint_path = gp('checkpoint_path').value
        if not checkpoint_path:
            raise ValueError('No checkpoint_path provided (SceneSeg .pth).')

        self.image_topic = gp('image_topic').value
        self.goal = (gp('goal_x').value, gp('goal_y').value)
        self.goal_tolerance = gp('goal_tolerance').value
        self.image_timeout = gp('image_timeout').value
        self.p = {
            'max_linear_speed': gp('max_linear_speed').value,
            'max_angular_speed': gp('max_angular_speed').value,
            'kp_steer': gp('kp_steer').value,
            'num_bins': gp('num_bins').value,
            'roi_top_frac': gp('roi_top_frac').value,
            'goal_weight': gp('goal_weight').value,
            'road_weight': gp('road_weight').value,
            'obstacle_weight': gp('obstacle_weight').value,
            'blocked_stop_fraction': gp('blocked_stop_fraction').value,
            'rotate_in_place_angle': gp('rotate_in_place_angle').value,
            'camera_hfov_deg': gp('camera_hfov_deg').value,
        }

        self.bridge = CvBridge()
        self.model = SceneSegNetworkInfer(checkpoint_path=checkpoint_path)
        self.get_logger().info('SceneSeg model loaded for navigation.')

        self.pose = None            # (x, y, yaw)
        self.cmd = (0.0, 0.0)       # latest (v, w)
        self.last_image_time = None
        self.reached = False        # latched True once goal_tolerance is reached, to stop and stay stopped

        self.create_subscription(Image, self.image_topic, self.image_cb, 1)
        self.create_subscription(Odometry, gp('odom_topic').value, self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, gp('cmd_vel_topic').value, 10)
        self.overlay_pub = self.create_publisher(Image, gp('overlay_topic').value, 1)

        period = 1.0 / float(gp('control_rate').value)
        self.create_timer(period, self.control_cb)
        self.get_logger().info(
            f'Navigating to goal {self.goal} (tol {self.goal_tolerance} m) '
            f'using {self.image_topic}')

    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def image_cb(self, msg: Image):
        self.last_image_time = self.get_clock().now()
        if self.pose is None or self.reached:
            return

        x, y, yaw = self.pose
        gx, gy = self.goal
        dist = math.hypot(gx - x, gy - y)
        if dist <= self.goal_tolerance:
            self.cmd = (0.0, 0.0)
            if not self.reached:
                self.reached = True
                self.get_logger().info(f'Goal reached (dist {dist:.2f} m). Stopping.')
            return
        heading_error = normalize_angle(math.atan2(gy - y, gx - x) - yaw)

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil = PILImage.fromarray(cv_image).resize((MODEL_WIDTH, MODEL_HEIGHT))
        prediction = self.model.inference(pil)

        v, w, info = compute_command(prediction, heading_error, dist, self.p)
        self.cmd = (v, w)

        if self.overlay_pub.get_subscription_count() > 0:
            ov = overlay_image(prediction, info, self.p)
            out = self.bridge.cv2_to_imgmsg(ov, encoding='rgb8')
            out.header = msg.header
            self.overlay_pub.publish(out)

    def control_cb(self):
        v, w = self.cmd
        # Watchdog: stop if perception has stalled.
        if self.last_image_time is not None:
            age = (self.get_clock().now() - self.last_image_time).nanoseconds * 1e-9
            if age > self.image_timeout:
                v, w = 0.0, 0.0
        if self.pose is None:
            v, w = 0.0, 0.0
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = SceneSegNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort stop on shutdown.
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
