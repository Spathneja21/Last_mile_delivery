#!/usr/bin/env bash
# run_webcam_publisher.sh
# Reads from /dev/video0 and publishes to /webcam/image_raw

VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
export PYTHONPATH="$VP_GPU_SITE:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
/data/archit0030/miniforge3/envs/vp_gpu/bin/python3 -c "
import cv2
import rospy
from sensor_msgs.msg import Image
import numpy as np

rospy.init_node('webcam_publisher', anonymous=True)
pub = rospy.Publisher('/webcam/image_raw', Image, queue_size=1)
cap = cv2.VideoCapture(0)  # /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('Webcam publisher started on /webcam/image_raw...')

rate = rospy.Rate(30)  # 30 Hz
while not rospy.is_shutdown():
    ret, frame = cap.read()
    if ret:
        # Convert BGR (cv2) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create Image msg (rgb8)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0  # byte order flag (little-endian, standard on x86).
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        
        pub.publish(msg)
    rate.sleep()

cap.release()
"
