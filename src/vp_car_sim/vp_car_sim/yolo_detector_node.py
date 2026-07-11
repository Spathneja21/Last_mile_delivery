#!/data/archit0030/miniforge3/envs/lerobot/bin/python3
"""
YOLOv8 object detection node for a live webcam feed.

ROS 1 (Noetic).

Runs independently of webcam_navigator_node.py / /cmd_vel: subscribes to the
same raw image topic, draws detection boxes + class labels, and publishes an
annotated overlay image plus a JSON string of per-frame detections. Does not
touch the driving control loop.
"""
import sys
import json

import numpy as np
import cv2

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Pure-Python cv_bridge replacement (matches webcam_navigator_node.py so both
# nodes stay independent of the system cv_bridge C++ extension).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------

class YoloDetectorNode:
    def __init__(self):
        rospy.init_node('yolo_detector_node', anonymous=False)

        weights_path  = rospy.get_param('~weights_path', '')
        image_topic   = rospy.get_param('~image_topic', '/webcam/image_raw')
        overlay_topic = rospy.get_param('~overlay_topic',
                                        '/vision_pilot/yolo_detector/overlay')
        detections_topic = rospy.get_param('~detections_topic',
                                           '/vision_pilot/yolo_detector/detections')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.4)

        if not weights_path:
            rospy.logfatal('No weights_path provided (YOLOv8 .pt checkpoint).')
            sys.exit(1)

        self.model = YOLO(weights_path)
        rospy.loginfo(f'YOLO model loaded from {weights_path} '
                      f'({len(self.model.names)} classes).')

        self.overlay_pub    = rospy.Publisher(overlay_topic, Image, queue_size=1)
        self.detections_pub = rospy.Publisher(detections_topic, String, queue_size=10)

        rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=1,
                         buff_size=2**24)
        rospy.loginfo(f'Monitoring {image_topic}, publishing overlay on {overlay_topic} '
                      f'and detections on {detections_topic}')

    def image_cb(self, msg: Image):
        rgb = imgmsg_to_rgb8(msg)

        results = self.model.predict(rgb, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        annotated = rgb.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]
            conf   = float(box.conf[0])

            detections.append({
                'label': label, 'confidence': round(conf, 3),
                'box_xyxy': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]),
                         (0, 255, 0), -1)
            cv2.putText(annotated, text, (p1[0] + 2, p1[1] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        self.detections_pub.publish(String(data=json.dumps(detections)))

        if self.overlay_pub.get_num_connections() > 0:
            out = rgb8_to_imgmsg(annotated, header=msg.header)
            self.overlay_pub.publish(out)

        if detections:
            summary = ', '.join(f"{d['label']}({d['confidence']:.2f})" for d in detections)
            rospy.loginfo(f'{len(detections)} detection(s): {summary}')

    def spin(self):
        rospy.spin()


def main():
    node = YoloDetectorNode()
    node.spin()


if __name__ == '__main__':
    main()
