#!/usr/bin/env python3
# ROS2 node that runs the Scene3D depth model on the front camera feed.
# Subscribes to an image topic (default /vp_car/front_camera/image_raw) and
# publishes a colorized depth overlay blended over the original frame
# (default /vision_pilot/scene_3d/overlay).
import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vp_car_sim.vp_models.pth_pipeline.inference.scene_3d_infer import Scene3DNetworkInfer

# Model input size required by Scene3D
MODEL_WIDTH = 640
MODEL_HEIGHT = 320

ALPHA = 0.5  # overlay blend weight (0=only camera image, 1=only depth overlay)


class Scene3DNode(Node):
    def __init__(self):
        super().__init__('scene_3d_node')

        self.declare_parameter('checkpoint_path', '')  # path to Scene3D .pth weights
        self.declare_parameter('input_topic', '/vp_car/front_camera/image_raw')
        self.declare_parameter('output_topic', '/vision_pilot/scene_3d/overlay')

        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        if not checkpoint_path:
            raise ValueError(
                'No checkpoint_path provided. Set the "checkpoint_path" parameter to a '
                'Scene3D .pth file (see Models/model_library/Scene3D/README.md).'
            )

        self.bridge = CvBridge()
        self.model = Scene3DNetworkInfer(checkpoint_path=checkpoint_path)
        self.get_logger().info('Scene3D model loaded.')

        self.sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'Subscribed to {input_topic}, publishing depth overlay on {output_topic}')

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        orig_h, orig_w = cv_image.shape[:2]

        pil_image = PILImage.fromarray(cv_image).resize((MODEL_WIDTH, MODEL_HEIGHT))

        # Raw relative depth, (320,640,1) -> higher value = nearer (confirmed empirically,
        # see AWSIM_INTEGRATION_PLAN.md S4.4a). No fixed scale, so stretch per-frame.
        raw_depth = np.asarray(self.model.inference(pil_image)).squeeze(-1)

        stretched = 255.0 * (raw_depth - raw_depth.min()) / max(raw_depth.max() - raw_depth.min(), 1e-6)
        depth_bgr = cv2.applyColorMap(stretched.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        overlay_resized = np.array(PILImage.fromarray(depth_rgb).resize((orig_w, orig_h)))

        blended = (overlay_resized.astype(np.float32) * ALPHA +
                   cv_image.astype(np.float32) * (1 - ALPHA)).astype(np.uint8)

        out_msg = self.bridge.cv2_to_imgmsg(blended, encoding='rgb8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Scene3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()