#!/usr/bin/env python3
"""
AutoSteer 2.0 node - ego-path waypoint visualization.

Standalone, read-only: subscribes a camera topic, runs AutoSteerNetworkInfer, and
publishes the predicted ego-path waypoints drawn over the live frame. Mirrors
scene_3d_node.py's structure/role - confirm the model's output looks sane against a
real camera feed (e.g. AWSIM's) before relying on it for steering in
awsim_navigator_node.

Note the BGR input contract: unlike every other node in this workspace (which uses
rgb8), AutoSteer's preprocessing expects BGR (confirmed against the production C++
header) and skips ImageNet normalization - so this node requests bgr8 from cv_bridge
and never converts to RGB.
"""
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vp_car_sim.vp_models.inference.auto_steer_infer import AutoSteerNetworkInfer

# Model input size required by AutoSteer 2.0
MODEL_WIDTH = 1024
MODEL_HEIGHT = 512
CONFIDENCE_THRESHOLD = 0.5  # matches the production C++ convention (lateral_fusion.hpp)

WAYPOINT_COLOR = (0, 255, 255)  # yellow, drawn directly on the bgr8 frame


class AutoSteerNode(Node):
    def __init__(self):
        super().__init__('auto_steer_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('input_topic', '/sensing/camera/traffic_light/image_raw')
        self.declare_parameter('output_topic', '/vision_pilot/auto_steer/overlay')

        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        if not checkpoint_path:
            raise ValueError(
                'No checkpoint_path provided. Set the "checkpoint_path" parameter to an '
                'AutoSteer 2.0 .pth file (1024x512 variant).'
            )

        self.bridge = CvBridge()
        self.model = AutoSteerNetworkInfer(checkpoint_path=checkpoint_path)
        self.get_logger().info('AutoSteer model loaded.')

        self.sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'Subscribed to {input_topic}, publishing overlay on {output_topic}')

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        orig_h, orig_w = cv_image.shape[:2]
        resized = cv2.resize(cv_image, (MODEL_WIDTH, MODEL_HEIGHT))

        xp, h_vector = self.model.inference(resized)

        scale_x = orig_w / MODEL_WIDTH
        scale_y = orig_h / MODEL_HEIGHT
        rows = np.linspace(0, MODEL_HEIGHT - 1, len(xp))

        overlay = cv_image.copy()
        points = []
        for x_norm, conf, row in zip(xp, h_vector, rows):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            u = int(x_norm * MODEL_WIDTH * scale_x)
            v = int(row * scale_y)
            points.append((u, v))
            cv2.circle(overlay, (u, v), 4, WAYPOINT_COLOR, -1)

        for p0, p1 in zip(points, points[1:]):
            cv2.line(overlay, p0, p1, WAYPOINT_COLOR, 2)

        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AutoSteerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
