#!/data/archit0030/miniforge3/envs/vp_gpu/bin/python3
"""
SceneSeg + Scene3D obstacle-avoidance ADVISOR (perception only).

ROS 1 (Noetic) port. This is the perception half of the merged
road-follow + vision-avoid pipeline (see MERGED_NAVIGATION_PLAN.md). It runs
the exact same SceneSeg/Scene3D inference + compute_command() decision logic as
webcam_navigator_node.py, but it does NOT publish /cmd_vel. Instead it publishes
what it *sees* as a JSON advisory on /vision/avoidance, which fusion_navigator_node.py
consumes and arbitrates against the GPS road-following heading.

Splitting perception from actuation is the whole point: exactly one node
(the fusion navigator) may own /cmd_vel. webcam_navigator_node.py is left
untouched and still runnable standalone for the vision-only pipeline.

Advisory JSON on /vision/avoidance (std_msgs/String):
  {
    "stamp":          <float, image capture time, secs>,
    "blocked":        <bool,  central path straight ahead is blocked>,
    "block_amount":   <float 0..1, how blocked (0 clear, 1 fully)>,
    "desired_bearing":<float rad, clearest bin overall (+left/+yaw, REP-103)>,
    "bin_bearings":   [<float rad>, ...]  per-bin bearing, +left,
    "bin_clear":      [<float>, ...]      per-bin clear score (higher = clearer),
    "center_proximity":<float 0..1, nearest central obstacle depth>
  }
bin_bearing sign matches scene_decision.py (+bearing = left = +yaw), which is
the same convention as the fusion navigator's GPS heading_error — so the two
can be arbitrated directly without a sign flip.
"""
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image as PILImage

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

from vp_car_sim.vp_models.trt_pipeline.decision.scene_decision import compute_command, overlay_image
from vp_car_sim.vp_models.trt_pipeline.inference.scene_seg_infer_trt import SceneSegNetworkInferTRT
from vp_car_sim.vp_models.trt_pipeline.inference.scene_3d_infer_trt import Scene3DNetworkInferTRT

MODEL_WIDTH = 640
MODEL_HEIGHT = 320


def imgmsg_to_rgb8(msg):
    """Convert a sensor_msgs/Image (rgb8 or bgr8) to an HxWx3 uint8 ndarray."""
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1].copy()
    return arr


def rgb8_to_imgmsg(arr, header=None):
    """Convert an HxWx3 uint8 RGB ndarray to a sensor_msgs/Image (rgb8)."""
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * 3
    msg.data = arr.tobytes()
    return msg


class VisionAvoidanceNode:
    def __init__(self):
        rospy.init_node('vision_avoidance_node', anonymous=False)

        seg_engine_path   = rospy.get_param('~seg_engine_path', '')
        depth_engine_path = rospy.get_param('~depth_engine_path', '')
        image_topic       = rospy.get_param('~image_topic', '/webcam/image_raw')
        advisory_topic    = rospy.get_param('~advisory_topic', '/vision/avoidance')
        overlay_topic     = rospy.get_param('~overlay_topic',
                                            '/vision_pilot/vision_avoidance/overlay')

        if not seg_engine_path:
            rospy.logfatal('No seg_engine_path provided (SceneSeg TensorRT .engine).')
            sys.exit(1)
        if not depth_engine_path:
            rospy.logfatal('No depth_engine_path provided (Scene3D TensorRT .engine).')
            sys.exit(1)

        # Same decision params as webcam_navigator_node.py — compute_command()
        # needs them to bin the frame and score obstacles. The speed/steer gains
        # (max_linear_speed, kp_steer, ...) are only used for the v/w this node
        # discards; the fusion navigator applies its own gains to the advisory.
        self.p = {
            'max_linear_speed':     rospy.get_param('~max_linear_speed',     0.6),
            'max_angular_speed':    rospy.get_param('~max_angular_speed',    1.0),
            'kp_steer':             rospy.get_param('~kp_steer',             1.5),
            'num_bins':             rospy.get_param('~num_bins',             9),
            'roi_top_frac':         rospy.get_param('~roi_top_frac',         0.4),
            'road_weight':          rospy.get_param('~road_weight',          0.6),
            'obstacle_weight':      rospy.get_param('~obstacle_weight',      2.0),
            'blocked_stop_fraction':rospy.get_param('~blocked_stop_fraction',0.4),
            'brake_depth_threshold':rospy.get_param('~brake_depth_threshold',0.75),
            'rotate_in_place_angle':rospy.get_param('~rotate_in_place_angle',0.6),
            'camera_hfov_deg':      rospy.get_param('~camera_hfov_deg',     50.0),
        }

        self.seg_model   = SceneSegNetworkInferTRT(engine_path=seg_engine_path)
        self.depth_model = Scene3DNetworkInferTRT(engine_path=depth_engine_path)
        rospy.loginfo('SceneSeg + Scene3D TensorRT engines loaded.')

        self.advisory_pub = rospy.Publisher(advisory_topic, String, queue_size=1)
        self.overlay_pub  = rospy.Publisher(overlay_topic, Image, queue_size=1)

        rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=1,
                         buff_size=2**24)
        rospy.loginfo('Vision advisor: monitoring %s, publishing advisory on %s',
                      image_topic, advisory_topic)

    def image_cb(self, msg: Image):
        t0 = time.perf_counter()

        cv_image = imgmsg_to_rgb8(msg)
        pil = PILImage.fromarray(cv2.resize(cv_image, (MODEL_WIDTH, MODEL_HEIGHT)))

        # launch both kernels on their own CUDA streams — GPU runs them concurrently
        self.seg_model.launch_async(pil)
        self.depth_model.launch_async(pil)
        seg_pred = self.seg_model.fetch_result()
        depth_pred = self.depth_model.fetch_result().squeeze(-1)

        _, _, info = compute_command(seg_pred, depth_pred, self.p)

        # block_amount is computed inside compute_command() but not returned in
        # info — recompute it here from the central-bin features it does expose
        # (identical formula to scene_decision.py).
        block_amount = max(
            info['center_fg'] / max(1e-3, self.p['blocked_stop_fraction']),
            info['center_proximity'] / max(1e-3, self.p['brake_depth_threshold']))
        block_amount = min(1.0, block_amount)

        advisory = {
            'stamp': msg.header.stamp.to_sec(),
            'blocked': bool(info['blocked']),
            'block_amount': float(block_amount),
            'desired_bearing': float(info['desired_bearing']),
            'bin_bearings': [float(b) for b in info['bin_bearing']],
            'bin_clear': [float(s) for s in info['score']],
            'center_proximity': float(info['center_proximity']),
        }
        self.advisory_pub.publish(String(data=json.dumps(advisory)))

        dt_ms = (time.perf_counter() - t0) * 1e3
        state = 'BLOCKED' if info['blocked'] else 'clear'
        rospy.loginfo_throttle(
            1.0, '[VISION] %s  block=%.2f  best_bearing=%.1fdeg  (%.1f ms)',
            state, block_amount, np.degrees(info['desired_bearing']), dt_ms)

        if self.overlay_pub.get_num_connections() > 0:
            ov = overlay_image(seg_pred, depth_pred, info, self.p)
            self.overlay_pub.publish(rgb8_to_imgmsg(ov, header=msg.header))


def main():
    VisionAvoidanceNode()
    rospy.spin()


if __name__ == '__main__':
    main()
