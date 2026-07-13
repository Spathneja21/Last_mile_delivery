#!/usr/bin/env python3
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vp_car_sim.vp_models.pth_pipeline.inference.scene_seg_infer import SceneSegNetworkInfer

# Model input size required by SceneSeg
MODEL_WIDTH = 640
MODEL_HEIGHT = 320

# Visualization colours — 3-class palette (matches sceneseg_test/eval_sceneseg.py):
#   0 = background → grey, 1 = foreground object → red, 2 = drivable road → green
CLASS_COLOURS = {
    0: (90, 90, 90),
    1: (220, 40, 40),
    2: (40, 180, 40),
}
ALPHA = 0.5


class SceneSegNode(Node):
    def __init__(self):
        super().__init__('scene_seg_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('input_topic', '/vp_car/front_camera/image_raw')
        self.declare_parameter('output_topic', '/vision_pilot/scene_seg/overlay')

        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        if not checkpoint_path:
            raise ValueError(
                'No checkpoint_path provided. Set the "checkpoint_path" parameter to a '
                'SceneSeg .pth file (see Models/model_library/SceneSeg/README.md).'
            )

        self.bridge = CvBridge()
        self.model = SceneSegNetworkInfer(checkpoint_path=checkpoint_path)
        self.get_logger().info('SceneSeg model loaded.')

        self.sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'Subscribed to {input_topic}, publishing overlay on {output_topic}')

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        orig_h, orig_w = cv_image.shape[:2]

        pil_image = PILImage.fromarray(cv_image).resize((MODEL_WIDTH, MODEL_HEIGHT))

        prediction = self.model.inference(pil_image)

        # Colour every class (background grey, foreground red, road green).
        overlay = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
        for class_id, colour in CLASS_COLOURS.items():
            overlay[prediction == class_id] = colour

        overlay_resized = np.array(PILImage.fromarray(overlay).resize((orig_w, orig_h)))

        blended = (overlay_resized.astype(np.float32) * ALPHA +
                   cv_image.astype(np.float32) * (1 - ALPHA)).astype(np.uint8)

        out_msg = self.bridge.cv2_to_imgmsg(blended, encoding='rgb8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SceneSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
