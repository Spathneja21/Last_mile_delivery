#!/data/archit0030/miniforge3/envs/lerobot/bin/python3
"""
SceneSeg + Scene3D reactive steering/velocity monitor for a live webcam feed.

ROS 1 (Noetic) port.

Runs depth-gated obstacle-avoidance decision logic against a real camera.
Publishes geometry_msgs/Twist on /cmd_vel
  linear.x  = velocity
  angular.z = steering / turn rate
"""
import sys
import struct
import csv
import json
import time
import cv2
import numpy as np
from PIL import Image as PILImage

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from vp_car_sim.vp_models.trt_pipeline.decision.scene_decision import compute_command, overlay_image
from vp_car_sim.vp_models.trt_pipeline.decision.csv_logger import CommandLogger
from vp_car_sim.vp_models.trt_pipeline.inference.scene_seg_infer_trt import SceneSegNetworkInferTRT
from vp_car_sim.vp_models.trt_pipeline.inference.scene_3d_infer_trt import Scene3DNetworkInferTRT

MODEL_WIDTH = 640
MODEL_HEIGHT = 320


# ---------------------------------------------------------------------------
# Pure-Python cv_bridge replacement (avoids the NumPy 2.x ABI breakage in the
# system cv_bridge C++ extension while keeping zero new installs).
# ---------------------------------------------------------------------------

def imgmsg_to_rgb8(msg):
    """Convert a sensor_msgs/Image (rgb8 or bgr8) to an HxWx3 uint8 ndarray."""
    dtype = np.uint8
    n_channels = 3
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_channels)
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


# ---------------------------------------------------------------------------

class WebcamNavigatorNode:
    def __init__(self):
        rospy.init_node('webcam_navigator_node', anonymous=False)

        seg_engine_path   = rospy.get_param('~seg_engine_path', '')
        depth_engine_path = rospy.get_param('~depth_engine_path', '')
        image_topic       = rospy.get_param('~image_topic', '/webcam/image_raw')
        cmd_vel_topic     = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        overlay_topic     = rospy.get_param('~overlay_topic',
                                            '/vision_pilot/webcam_navigator/overlay')

        if not seg_engine_path:
            rospy.logfatal('No seg_engine_path provided (SceneSeg TensorRT .engine).')
            sys.exit(1)
        if not depth_engine_path:
            rospy.logfatal('No depth_engine_path provided (Scene3D TensorRT .engine).')
            sys.exit(1)

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

        csv_log_path = rospy.get_param('~csv_log_path', 'pipeline_log.csv')
        self.cmd_logger = CommandLogger(csv_log_path)
        rospy.loginfo(f'Logging per-frame commands to {csv_log_path}')

        timing_log_path = rospy.get_param('~timing_log_path', 'timing_log.csv')
        self._timing_file = open(timing_log_path, 'w', newline='')
        self._timing_writer = csv.writer(self._timing_file)
        self._timing_writer.writerow(
            ['frame', 'pre_ms', 'seg_ms', 'depth_ms', 'post_ms', 'pub_ms', 'total_ms', 'e2e_ms'])
        self._frame = 0
        rospy.loginfo(f'Logging per-frame timings to {timing_log_path}')

        self.cmd_pub     = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.overlay_pub = rospy.Publisher(overlay_topic, Image, queue_size=1)

        # --- NL-command override state (set by nl_command_node via /vision_pilot/nl_intent).
        # These are applied as bounded post-processing on top of compute_command()'s
        # output below — they never bypass the obstacle-avoidance/braking logic itself.
        self.manual_stop    = False
        self.speed_scale    = 1.0
        self.turn_bias      = 0.0
        self.turn_bias_until = rospy.Time(0)

        nl_intent_topic = rospy.get_param('~nl_intent_topic', '/vision_pilot/nl_intent')
        rospy.Subscriber(nl_intent_topic, String, self.nl_intent_cb, queue_size=10)
        rospy.loginfo(f'Listening for NL-derived intents on {nl_intent_topic}')

        rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=1,
                         buff_size=2**24)
        rospy.loginfo(f'Monitoring {image_topic}, publishing Twist on {cmd_vel_topic}')

    def nl_intent_cb(self, msg: String):
        try:
            intent = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            rospy.logwarn(f'nl_intent: could not parse {msg.data!r}')
            return

        action = intent.get('action')

        if action == 'stop':
            self.manual_stop = True
            rospy.loginfo('nl_intent: STOP (manual override engaged)')
        elif action == 'go':
            self.manual_stop = False
            rospy.loginfo('nl_intent: GO (manual override cleared)')
        elif action == 'set_speed_scale':
            try:
                scale = float(intent['scale'])
            except (KeyError, TypeError, ValueError):
                rospy.logwarn(f'nl_intent: invalid scale in {intent!r}')
                return
            # Defense in depth: re-clamp even though nl_command_node already did.
            self.speed_scale = max(0.2, min(1.5, scale))
            rospy.loginfo(f'nl_intent: speed_scale={self.speed_scale:.2f}')
        elif action == 'turn':
            direction = intent.get('direction')
            if direction not in ('left', 'right'):
                rospy.logwarn(f'nl_intent: invalid turn direction in {intent!r}')
                return
            duration = max(0.1, min(5.0, float(intent.get('duration_s', 1.5))))
            sign = 1.0 if direction == 'left' else -1.0
            self.turn_bias = sign * 0.5 * self.p['max_angular_speed']
            self.turn_bias_until = rospy.Time.now() + rospy.Duration(duration)
            rospy.loginfo(f'nl_intent: turn {direction} for {duration:.1f}s '
                         f'(bias={self.turn_bias:.2f})')
        else:
            rospy.loginfo(f'nl_intent: ignored (action={action!r})')

    def image_cb(self, msg: Image):
        t0 = time.perf_counter()

        cv_image = imgmsg_to_rgb8(msg)
        pil = PILImage.fromarray(cv2.resize(cv_image, (MODEL_WIDTH, MODEL_HEIGHT)))
        t1 = time.perf_counter()

        # launch both kernels on their own CUDA streams — GPU runs them concurrently
        self.seg_model.launch_async(pil)
        self.depth_model.launch_async(pil)

        seg_pred = self.seg_model.fetch_result()          # syncs seg stream
        t2 = time.perf_counter()
        depth_pred = self.depth_model.fetch_result().squeeze(-1)   # syncs depth stream
        t3 = time.perf_counter()

        v, w, info = compute_command(seg_pred, depth_pred, self.p)
        self.cmd_logger.log(info['best'], v, w, info['blocked'])
        t4 = time.perf_counter()

        # --- Apply NL-command overrides (bounded post-processing only — the
        # obstacle-avoidance/braking logic above always runs unmodified first).
        if self.turn_bias != 0.0 and rospy.Time.now() < self.turn_bias_until and not info['blocked']:
            w = max(-self.p['max_angular_speed'],
                    min(self.p['max_angular_speed'], w + self.turn_bias))

        v = max(0.0, min(self.p['max_linear_speed'], v * self.speed_scale))

        if self.manual_stop:
            v, w = 0.0, 0.0

        twist = Twist()
        twist.linear.x  = float(v)
        twist.angular.z = float(w)

        e2e_ms = (rospy.Time.now() - msg.header.stamp).to_sec() * 1e3
        self.cmd_pub.publish(twist)
        t5 = time.perf_counter()

        ms = lambda a, b: (b - a) * 1e3
        pre_ms   = ms(t0, t1)
        seg_ms   = ms(t1, t2)
        depth_ms = ms(t2, t3)
        post_ms  = ms(t3, t4)
        pub_ms   = ms(t4, t5)
        total_ms = ms(t0, t5)

        self._timing_writer.writerow([
            self._frame, f'{pre_ms:.2f}', f'{seg_ms:.2f}', f'{depth_ms:.2f}',
            f'{post_ms:.2f}', f'{pub_ms:.2f}', f'{total_ms:.2f}', f'{e2e_ms:.2f}'])
        self._frame += 1

        state = 'BRAKE' if info['blocked'] else 'clear'
        rospy.loginfo(
            f'{state}  v={v:.2f}  w={w:.2f}  |  '
            f'pre={pre_ms:.1f}  seg={seg_ms:.1f}  depth={depth_ms:.1f}  '
            f'post={post_ms:.1f}  pub={pub_ms:.1f}  '
            f'total={total_ms:.1f}ms  e2e={e2e_ms:.1f}ms')

        if self.overlay_pub.get_num_connections() > 0:
            ov  = overlay_image(seg_pred, depth_pred, info, self.p)
            out = rgb8_to_imgmsg(ov, header=msg.header)
            self.overlay_pub.publish(out)

    def spin(self):
        rospy.spin()
        self._timing_file.flush()
        self._timing_file.close()


def main():
    node = WebcamNavigatorNode()
    node.spin()


if __name__ == '__main__':
    main()
