#!/usr/bin/env bash
# run_yolo_detector.sh
# Runs yolo_detector_node.py under ROS 1 Noetic + vp_gpu venv (CUDA torch).
# Independent of webcam_navigator_node.py — does not touch /cmd_vel.
#
# Usage:
#   ./run_yolo_detector.sh

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$WS_DIR/models"
SRC_DIR="$WS_DIR/src/vp_car_sim"
NODE="$SRC_DIR/vp_car_sim/yolo_detector_node.py"
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

# ---------- model checkpoint ----------
WEIGHTS_PATH="$MODELS_DIR/yolov8n.pt"

# ---------- ROS params (override as needed) ----------
IMAGE_TOPIC="/webcam/image_raw"
OVERLAY_TOPIC="/vision_pilot/yolo_detector/overlay"
DETECTIONS_TOPIC="/vision_pilot/yolo_detector/detections"
CONF_THRESHOLD="0.4"

# ---------- source ROS 1 ----------
source /opt/ros/noetic/setup.bash

# ---------- PYTHONPATH priority (order matters!, mirrors run_webcam_navigator.sh) ----------
export PYTHONPATH="\
$VP_GPU_SITE:\
$SRC_DIR:\
/usr/lib/python3.8/dist-packages:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

# ---------- ROS env (must have roscore running first) ----------
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export PYTHONPATH

echo "=================================================="
echo " yolo_detector_node  (ROS 1 / vp_gpu venv, CUDA)"
echo "=================================================="
echo "  weights        : $WEIGHTS_PATH"
echo "  image_topic    : $IMAGE_TOPIC"
echo "  overlay        : $OVERLAY_TOPIC"
echo "  detections     : $DETECTIONS_TOPIC"
echo "  conf_threshold : $CONF_THRESHOLD"
echo ""
echo "  Make sure 'roscore' and a camera node are running first!"
echo "=================================================="
echo ""

exec "$VP_GPU_PY" "$NODE" \
    _weights_path:="$WEIGHTS_PATH" \
    _image_topic:="$IMAGE_TOPIC" \
    _overlay_topic:="$OVERLAY_TOPIC" \
    _detections_topic:="$DETECTIONS_TOPIC" \
    _conf_threshold:="$CONF_THRESHOLD"
