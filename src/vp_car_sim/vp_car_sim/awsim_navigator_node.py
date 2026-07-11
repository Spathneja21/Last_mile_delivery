#!/usr/bin/env python3
"""
SceneSeg + Scene3D reactive obstacle monitor for AWSIM (depth-gated braking, Option B
from AWSIM_INTEGRATION_PLAN.md S4.4a/S4.4b).

Drives AWSIM's ego vehicle via the Autoware vehicle interface (S4.2-4.4): a one-time
/input/control_mode_request -> AUTONOMOUS plus a latched GearCommand.DRIVE, then every
frame publishes autoware_control_msgs/msg/Control (steering_tire_angle + velocity) on
/control/command/control_cmd. A debug Twist + annotated overlay are also published for
visibility. No goal-seeking yet (would need AWSIM's GNSS pose wired into a goal frame);
today this just steers toward the most open/road-like direction while avoiding obstacles.

Brake logic (Option B, no metric depth calibration needed):
  area_blocked  : centre-bin foreground fraction >= blocked_stop_fraction (old SceneSeg-
                  only check - a *wide* obstacle blocks regardless of distance)
  depth_blocked : nearest centre-bin foreground pixel's per-frame-normalized Scene3D
                  value >= brake_depth_threshold (NEW - catches a *small but close*
                  obstacle that area_blocked alone would miss)
  blocked = area_blocked or depth_blocked

Steering refinement (optional, AutoSteer 2.0): when enable_autosteer is true, a third
model predicts ego-path waypoints directly from the camera frame. lane_center_offset()
turns those into an image-space lane-centering bearing correction (no homography/world
calibration - same "Option B" philosophy as the brake logic above) which is blended with
the SceneSeg/Scene3D bin-scored bearing, weighted by AutoSteer's own confidence. Braking
is untouched by this - it remains purely SceneSeg/Scene3D-driven.
"""
import math

import numpy as np
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import GearCommand, VelocityReport
from autoware_vehicle_msgs.srv import ControlModeCommand

from vp_car_sim.vp_models.inference.scene_seg_infer import SceneSegNetworkInfer
from vp_car_sim.vp_models.inference.scene_3d_infer import Scene3DNetworkInfer
from vp_car_sim.vp_models.inference.auto_steer_infer import AutoSteerNetworkInfer

MODEL_WIDTH = 640
MODEL_HEIGHT = 320
CLS_FOREGROUND = 1
CLS_ROAD = 2

# AutoSteer 2.0 has its own fixed input contract (1024x512, BGR, no ImageNet normalize)
# - separate from the 640x320 SceneSeg/Scene3D contract above.
AUTOSTEER_WIDTH = 1024
AUTOSTEER_HEIGHT = 512

# AWSIM's subscriptions on these two topics require RELIABLE + TRANSIENT_LOCAL -
# a default (VOLATILE) publisher silently fails to match. See AWSIM_INTEGRATION_PLAN.md S4.2.
CONTROL_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)


def compute_command(seg_pred, depth_pred, p):
    """
    Pure decision function (no ROS) so it can be unit-tested.

    seg_pred   : (H, W) int array of SceneSeg class ids
    depth_pred : (H, W) float array of Scene3D raw relative depth (higher = nearer,
                 confirmed empirically - see AWSIM_INTEGRATION_PLAN.md S4.4a)
    p          : dict of parameters (see node defaults)

    Returns (linear_x, angular_z, info) where info has per-bin debug fields.
    """
    h, w = seg_pred.shape
    top = int(p['roi_top_frac'] * h)
    roi_seg = seg_pred[top:, :]
    roi_depth = depth_pred[top:, :]

    # Relative depth only - normalize within the ROI, per frame (no fixed scale exists).
    d_min, d_max = float(roi_depth.min()), float(roi_depth.max())
    norm_depth = (roi_depth - d_min) / max(d_max - d_min, 1e-6)

    n = p['num_bins']
    cols = np.array_split(np.arange(w), n)
    half_fov = math.radians(p['camera_hfov_deg']) / 2.0

    bin_bearing = np.zeros(n)
    fg_frac = np.zeros(n)
    road_frac = np.zeros(n)
    fg_proximity = np.zeros(n)     # nearest obstacle pixel in this bin, 0 if none
    for i, c in enumerate(cols):
        seg = roi_seg[:, c[0]:c[-1] + 1]
        depth = norm_depth[:, c[0]:c[-1] + 1]
        fg_mask = (seg == CLS_FOREGROUND)
        fg_frac[i] = float(np.mean(fg_mask))
        road_frac[i] = float(np.mean(seg == CLS_ROAD))
        fg_proximity[i] = float(depth[fg_mask].max()) if np.any(fg_mask) else 0.0
        u = (((c[0] + c[-1]) / 2.0) - w / 2.0) / (w / 2.0)      # [-1 (left) .. +1 (right)]
        bin_bearing[i] = -u * half_fov                          # +bearing = left = +yaw

    # Depth-weighted obstacle penalty: a near obstacle (fg_proximity -> 1) penalizes a
    # bin up to 2x harder than a far one (fg_proximity -> 0) of the same pixel area.
    score = (p['road_weight'] * road_frac
             - p['obstacle_weight'] * fg_frac * (0.5 + 0.5 * fg_proximity))

    best = int(np.argmax(score))
    desired_bearing = float(bin_bearing[best])

    # Central third of the bins = path straight ahead.
    lo, hi = n // 3, n - n // 3
    center_fg = float(np.max(fg_frac[lo:hi])) if hi > lo else float(fg_frac[best])
    center_proximity = float(np.max(fg_proximity[lo:hi])) if hi > lo else float(fg_proximity[best])

    angular = p['kp_steer'] * desired_bearing
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular))

    area_blocked = center_fg >= p['blocked_stop_fraction']
    depth_blocked = center_proximity >= p['brake_depth_threshold']
    blocked = area_blocked or depth_blocked

    block_amount = max(center_fg / max(1e-3, p['blocked_stop_fraction']),
                        center_proximity / max(1e-3, p['brake_depth_threshold']))
    block_amount = min(1.0, block_amount)
    linear = p['max_linear_speed'] * (1.0 - block_amount)
    if blocked or abs(desired_bearing) > p['rotate_in_place_angle']:
        linear = 0.0

    info = {
        'fg_frac': fg_frac, 'road_frac': road_frac, 'fg_proximity': fg_proximity,
        'bin_bearing': bin_bearing, 'best': best, 'cols': cols,
        'desired_bearing': desired_bearing,
        'center_fg': center_fg, 'center_proximity': center_proximity,
        'area_blocked': area_blocked, 'depth_blocked': depth_blocked, 'blocked': blocked,
    }
    return linear, angular, info


def lane_center_offset(xp, h_vector, p):
    """
    Pure decision function (no ROS) so it can be unit-tested.

    Turns AutoSteer 2.0's raw waypoint output into an image-space lane-centering
    correction. No homography/world-frame projection (Option B, same no-metric-
    calibration philosophy as the depth-gated braking above) - this only knows about
    normalized image coordinates.

    xp, h_vector : each (64,) arrays from AutoSteerNetworkInfer.inference(). xp[i] is
                   the normalized [0,1] column of the ego-path at row i; h_vector[i] is
                   AutoSteer's confidence for that waypoint (row i itself is implicit -
                   evenly spaced top-to-bottom, see auto_steer_infer.py).
    p            : dict of parameters (see node defaults)

    Returns (lateral_offset, confidence):
      lateral_offset in [-1, 1]: offset of the ego-path from image center, averaged
                                  over the near-field (bottom-third, i.e. closest-to-
                                  the-car) waypoints. +1 = path is at the right edge.
      confidence in [0, 1]: fraction of near-field waypoints AutoSteer considers valid
                             (h_vector >= lane_confidence_threshold). 0.0 means "not
                             enough signal" - the caller should fall back entirely to
                             its other steering source.
    """
    n = len(xp)
    near = slice(2 * n // 3, n)  # bottom third of the 64 rows = nearest to the car
    near_xp = np.asarray(xp[near])
    near_valid = np.asarray(h_vector[near]) >= p['lane_confidence_threshold']

    if int(near_valid.sum()) < p['min_valid_lane_points']:
        return 0.0, 0.0

    lateral_offset = float(2.0 * (near_xp[near_valid].mean() - 0.5))
    confidence = float(near_valid.mean())
    return lateral_offset, confidence


def overlay_image(seg_pred, depth_pred, info, p):
    """Road=green, obstacle red-intensity scaled by proximity, chosen bin highlighted,
    BRAKE/CLEAR banner."""
    h, w = seg_pred.shape
    top = int(p['roi_top_frac'] * h)
    norm_depth = np.zeros((h, w), dtype=np.float32)
    roi_depth = depth_pred[top:, :]
    d_min, d_max = float(roi_depth.min()), float(roi_depth.max())
    norm_depth[top:, :] = (roi_depth - d_min) / max(d_max - d_min, 1e-6)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[seg_pred == CLS_ROAD] = (40, 180, 40)
    img[seg_pred == 0] = (60, 60, 60)
    fg_mask = seg_pred == CLS_FOREGROUND
    # Closer obstacle pixels -> brighter red; farther ones -> dimmer red.
    red = (80 + 175 * norm_depth).astype(np.uint8)
    img[fg_mask] = np.stack([red[fg_mask], np.zeros_like(red[fg_mask]), np.zeros_like(red[fg_mask])], axis=-1)

    cols = info['cols'][info['best']]
    c0, c1 = cols[0], cols[-1]
    img[:, c0:c1 + 1, :] = (img[:, c0:c1 + 1, :] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)

    banner = (0, 0, 200) if info['blocked'] else (0, 160, 0)
    img[0:24, :, :] = banner
    label = f"BRAKE area={info['center_fg']:.2f} prox={info['center_proximity']:.2f}" \
        if info['blocked'] else f"CLEAR area={info['center_fg']:.2f} prox={info['center_proximity']:.2f}"
    cv2.putText(img, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class AwsimNavigatorNode(Node):
    def __init__(self):
        super().__init__('awsim_navigator_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('depth_checkpoint_path', '')
        self.declare_parameter('autosteer_checkpoint_path', '')
        self.declare_parameter('enable_autosteer', True)
        self.declare_parameter('image_topic', '/sensing/camera/traffic_light/image_raw')
        self.declare_parameter('overlay_topic', '/vision_pilot/awsim_navigator/overlay')
        self.declare_parameter('debug_cmd_topic', '/vision_pilot/awsim_navigator/debug_cmd')
        self.declare_parameter('max_linear_speed', 0.6)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_steer', 1.5)
        self.declare_parameter('num_bins', 9)
        self.declare_parameter('roi_top_frac', 0.4)
        self.declare_parameter('road_weight', 0.6)
        self.declare_parameter('obstacle_weight', 2.0)
        self.declare_parameter('blocked_stop_fraction', 0.4)
        self.declare_parameter('brake_depth_threshold', 0.75)
        self.declare_parameter('rotate_in_place_angle', 0.6)
        self.declare_parameter('camera_hfov_deg', 50.0)
        # AutoSteer lane-centering blend - will need empirical tuning, not a solved number.
        self.declare_parameter('kp_lane_center', 1.0)
        self.declare_parameter('lane_confidence_threshold', 0.5)  # matches the production
                                                                    # C++ convention (lateral_fusion.hpp)
        self.declare_parameter('min_valid_lane_points', 5)
        self.declare_parameter('control_rate', 5.0)
        # Combined SceneSeg+Scene3D inference on this CPU-only host takes ~4-6s/frame
        # (observed live) - a short watchdog zeroes the command between almost every
        # frame. Set comfortably above the observed worst case.
        self.declare_parameter('image_timeout', 10.0)
        # AWSIM-Demo/sample-config.json EgoVehicleSettings._max{Steer,Acceleration,Deceleration}Input
        self.declare_parameter('max_steer_tire_angle_deg', 35.0)
        self.declare_parameter('max_acceleration', 2.0)
        self.declare_parameter('max_deceleration', 2.0)
        self.declare_parameter('accel_gain', 2.0)

        gp = self.get_parameter
        checkpoint_path = gp('checkpoint_path').value
        depth_checkpoint_path = gp('depth_checkpoint_path').value
        if not checkpoint_path:
            raise ValueError('No checkpoint_path provided (SceneSeg .pth).')
        if not depth_checkpoint_path:
            raise ValueError('No depth_checkpoint_path provided (Scene3D .pth).')

        self.enable_autosteer = gp('enable_autosteer').value
        autosteer_checkpoint_path = gp('autosteer_checkpoint_path').value
        if self.enable_autosteer and not autosteer_checkpoint_path:
            raise ValueError(
                'No autosteer_checkpoint_path provided (AutoSteer .pth) while '
                'enable_autosteer=True. Set enable_autosteer:=false to skip lane-centering.')

        self.image_topic = gp('image_topic').value
        self.image_timeout = gp('image_timeout').value
        self.max_steer_tire_angle_rad = math.radians(gp('max_steer_tire_angle_deg').value)
        self.max_acceleration = gp('max_acceleration').value
        self.max_deceleration = gp('max_deceleration').value
        self.accel_gain = gp('accel_gain').value
        self.p = {
            'max_linear_speed': gp('max_linear_speed').value,
            'max_angular_speed': gp('max_angular_speed').value,
            'kp_steer': gp('kp_steer').value,
            'num_bins': gp('num_bins').value,
            'roi_top_frac': gp('roi_top_frac').value,
            'road_weight': gp('road_weight').value,
            'obstacle_weight': gp('obstacle_weight').value,
            'blocked_stop_fraction': gp('blocked_stop_fraction').value,
            'brake_depth_threshold': gp('brake_depth_threshold').value,
            'rotate_in_place_angle': gp('rotate_in_place_angle').value,
            'camera_hfov_deg': gp('camera_hfov_deg').value,
            'kp_lane_center': gp('kp_lane_center').value,
            'lane_confidence_threshold': gp('lane_confidence_threshold').value,
            'min_valid_lane_points': gp('min_valid_lane_points').value,
        }

        self.bridge = CvBridge()
        self.seg_model = SceneSegNetworkInfer(checkpoint_path=checkpoint_path)
        self.depth_model = Scene3DNetworkInfer(checkpoint_path=depth_checkpoint_path)
        self.get_logger().info('SceneSeg + Scene3D models loaded.')

        self.autosteer_model = None
        if self.enable_autosteer:
            self.autosteer_model = AutoSteerNetworkInfer(checkpoint_path=autosteer_checkpoint_path)
            self.get_logger().info('AutoSteer model loaded.')

        self.cmd = (0.0, 0.0)
        self.steer = 0.0
        self.current_velocity = 0.0
        self.last_image_time = None

        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(VelocityReport, '/vehicle/status/velocity_status', self.velocity_cb, 10)
        self.overlay_pub = self.create_publisher(Image, gp('overlay_topic').value, 1)
        self.debug_cmd_pub = self.create_publisher(Twist, gp('debug_cmd_topic').value, 10)
        self.control_pub = self.create_publisher(Control, '/control/command/control_cmd', CONTROL_QOS)
        self.gear_pub = self.create_publisher(GearCommand, '/control/command/gear_cmd', CONTROL_QOS)
        self.control_mode_client = self.create_client(ControlModeCommand, '/input/control_mode_request')

        self.engage()

        period = 1.0 / float(gp('control_rate').value)
        self.create_timer(period, self.control_cb)
        self.get_logger().info(
            f'Monitoring {self.image_topic} for obstacles '
            f'(blocked_stop_fraction={self.p["blocked_stop_fraction"]}, '
            f'brake_depth_threshold={self.p["brake_depth_threshold"]})')

    def engage(self):
        """One-time: put AWSIM in DRIVE gear and request AUTONOMOUS control mode."""
        now = self.get_clock().now().to_msg()
        gear_msg = GearCommand()
        gear_msg.stamp = now
        gear_msg.command = GearCommand.DRIVE
        self.gear_pub.publish(gear_msg)

        if not self.control_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                '/input/control_mode_request not available - vehicle may stay in MANUAL mode.')
            return
        req = ControlModeCommand.Request()
        req.stamp = self.get_clock().now().to_msg()
        req.mode = ControlModeCommand.Request.AUTONOMOUS
        future = self.control_mode_client.call_async(req)
        future.add_done_callback(self._on_engage_response)

    def _on_engage_response(self, future):
        try:
            res = future.result()
            self.get_logger().info(f'control_mode_request -> success={res.success}')
        except Exception as exc:
            self.get_logger().error(f'control_mode_request failed: {exc}')

    def velocity_cb(self, msg: VelocityReport):
        self.current_velocity = msg.longitudinal_velocity

    def image_cb(self, msg: Image):
        self.last_image_time = self.get_clock().now()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil = PILImage.fromarray(cv_image).resize((MODEL_WIDTH, MODEL_HEIGHT))

        seg_pred = np.asarray(self.seg_model.inference(pil))
        depth_pred = np.asarray(self.depth_model.inference(pil)).squeeze(-1)

        v, w, info = compute_command(seg_pred, depth_pred, self.p)
        self.cmd = (v, w)

        # Lane-centering refinement (steering only - braking above is untouched).
        # AutoSteer's own confidence is the blend weight: confidence=0 (no usable lane
        # markings) degrades exactly to today's SceneSeg-bin-scored bearing.
        desired_bearing = info['desired_bearing']
        lane_offset, lane_confidence = 0.0, 0.0
        if self.enable_autosteer:
            bgr_image = cv_image[:, :, ::-1]  # rgb8 -> bgr: AutoSteer's native contract
            autosteer_input = cv2.resize(bgr_image, (AUTOSTEER_WIDTH, AUTOSTEER_HEIGHT))
            xp, h_vector = self.autosteer_model.inference(autosteer_input)
            lane_offset, lane_confidence = lane_center_offset(xp, h_vector, self.p)
            lane_bearing = -lane_offset * self.p['kp_lane_center']  # same sign convention
                                                                      # as bin_bearing: "+bearing=left"
            desired_bearing = (lane_confidence * lane_bearing
                                + (1.0 - lane_confidence) * desired_bearing)

        # Ackermann steering: the blended bearing *is* the steering angle (no
        # curvature/angular-velocity conversion - see AWSIM_INTEGRATION_PLAN.md S4.4),
        # clamped to AWSIM's real steering limit.
        self.steer = max(-self.max_steer_tire_angle_rad,
                          min(self.max_steer_tire_angle_rad, desired_bearing))

        state = 'BRAKE' if info['blocked'] else 'clear'
        self.get_logger().info(
            f'{state}  area={info["center_fg"]:.2f}  prox={info["center_proximity"]:.2f}  '
            f'lane_conf={lane_confidence:.2f}  '
            f'v={v:.2f}  w={w:.2f}  steer={math.degrees(self.steer):.1f}deg')

        if self.overlay_pub.get_subscription_count() > 0:
            ov = overlay_image(seg_pred, depth_pred, info, self.p)
            out = self.bridge.cv2_to_imgmsg(ov, encoding='rgb8')
            out.header = msg.header
            self.overlay_pub.publish(out)

    def control_cb(self):
        v, w = self.cmd
        steer = self.steer
        if self.last_image_time is not None:
            age = (self.get_clock().now() - self.last_image_time).nanoseconds * 1e-9
            if age > self.image_timeout:
                v, w, steer = 0.0, 0.0, 0.0

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.debug_cmd_pub.publish(twist)

        # AWSIM needs an explicit acceleration, not just a target velocity - confirmed
        # empirically (a velocity-only Control message was silently not driving the
        # vehicle; adding acceleration + is_defined_acceleration=True fixed it). Closed
        # -loop P-control against real feedback (/vehicle/status/velocity_status),
        # clamped to AWSIM's real accel/decel limits (sample-config.json).
        accel = self.accel_gain * (v - self.current_velocity)
        accel = max(-self.max_deceleration, min(self.max_acceleration, accel))

        now = self.get_clock().now().to_msg()
        control_msg = Control()
        control_msg.stamp = now
        control_msg.lateral.stamp = now
        control_msg.lateral.steering_tire_angle = float(steer)
        control_msg.longitudinal.stamp = now
        control_msg.longitudinal.velocity = float(v)
        control_msg.longitudinal.acceleration = float(accel)
        control_msg.longitudinal.is_defined_acceleration = True
        self.control_pub.publish(control_msg)

    def stop(self):
        """Publish an explicit stop (velocity=0, full braking) - call before shutdown.
        AWSIM keeps honoring the last received command rather than stopping on its own
        when commands stop arriving, so skipping this leaves the vehicle coasting at
        whatever it was last doing."""
        now = self.get_clock().now().to_msg()
        control_msg = Control()
        control_msg.stamp = now
        control_msg.lateral.stamp = now
        control_msg.lateral.steering_tire_angle = 0.0
        control_msg.longitudinal.stamp = now
        control_msg.longitudinal.velocity = 0.0
        control_msg.longitudinal.acceleration = -self.max_deceleration
        control_msg.longitudinal.is_defined_acceleration = True
        self.control_pub.publish(control_msg)
        self.debug_cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = AwsimNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
