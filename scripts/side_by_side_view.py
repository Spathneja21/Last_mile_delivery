#!/usr/bin/env python3
"""
side_by_side_view.py
Combines the raw webcam feed, the SceneSeg/Scene3D overlay, and (if running)
the YOLO detection overlay into a single image, published on one topic.

Panels appear left-to-right as: raw | SceneSeg/Scene3D overlay | YOLO overlay.
The YOLO panel is only included once at least one YOLO overlay frame has
arrived, so this still works unchanged if yolo_detector_node isn't running.
"""
import numpy as np
from PIL import Image as PILImage

import rospy
from sensor_msgs.msg import Image

latest_raw = None
latest_seg_overlay = None
latest_yolo_overlay = None


def imgmsg_to_rgb8(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1].copy()
    return arr


def rgb8_to_imgmsg(arr, header=None):
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


def _resize_to(arr, h, w):
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return np.asarray(PILImage.fromarray(arr).resize((w, h)))


def _publish_combined(pub, header):
    if latest_raw is None:
        return
    h, w = latest_raw.shape[:2]

    panels = [latest_raw]
    if latest_seg_overlay is not None:
        panels.append(_resize_to(latest_seg_overlay, h, w))
    if latest_yolo_overlay is not None:
        panels.append(_resize_to(latest_yolo_overlay, h, w))

    combined = np.concatenate(panels, axis=1)
    pub.publish(rgb8_to_imgmsg(combined, header=header))


def raw_cb(msg, pub):
    global latest_raw
    latest_raw = imgmsg_to_rgb8(msg)
    _publish_combined(pub, msg.header)


def seg_overlay_cb(msg, pub):
    global latest_seg_overlay
    latest_seg_overlay = imgmsg_to_rgb8(msg)
    _publish_combined(pub, msg.header)


def yolo_overlay_cb(msg, pub):
    global latest_yolo_overlay
    latest_yolo_overlay = imgmsg_to_rgb8(msg)
    _publish_combined(pub, msg.header)


def main():
    rospy.init_node('side_by_side_view', anonymous=True)
    raw_topic = rospy.get_param('~raw_topic', '/webcam/image_raw')
    overlay_topic = rospy.get_param('~overlay_topic', '/vision_pilot/webcam_navigator/overlay')
    yolo_overlay_topic = rospy.get_param('~yolo_overlay_topic',
                                        '/vision_pilot/yolo_detector/overlay')
    out_topic = rospy.get_param('~out_topic', '/vision_pilot/side_by_side')

    pub = rospy.Publisher(out_topic, Image, queue_size=1)
    rospy.Subscriber(raw_topic, Image, raw_cb, callback_args=pub, queue_size=1, buff_size=2**24)
    rospy.Subscriber(overlay_topic, Image, seg_overlay_cb, callback_args=pub, queue_size=1, buff_size=2**24)
    rospy.Subscriber(yolo_overlay_topic, Image, yolo_overlay_cb, callback_args=pub, queue_size=1, buff_size=2**24)

    rospy.loginfo(f'Publishing side-by-side (raw | seg overlay | yolo overlay) on {out_topic}')
    rospy.spin()


if __name__ == '__main__':
    main()
