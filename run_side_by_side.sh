#!/usr/bin/env bash
# run_side_by_side.sh
# Combines /webcam/image_raw + SceneSeg/Scene3D overlay + YOLO overlay (if
# yolo_detector_node is running) into one topic for viewing.

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE="$WS_DIR/scripts/side_by_side_view.py"
LEROBOT_SITE="/data/archit0030/miniforge3/envs/lerobot/lib/python3.10/site-packages"
LEROBOT_PY="/data/archit0030/miniforge3/envs/lerobot/bin/python3"

# ---------- ROS params (override as needed) ----------
RAW_TOPIC="/webcam/image_raw"
OVERLAY_TOPIC="/vision_pilot/webcam_navigator/overlay"
YOLO_OVERLAY_TOPIC="/vision_pilot/yolo_detector/overlay"
OUT_TOPIC="/vision_pilot/side_by_side"

source /opt/ros/noetic/setup.bash

export PYTHONPATH="\
$LEROBOT_SITE:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

echo "Publishing side-by-side (raw | seg overlay | yolo overlay) on $OUT_TOPIC"
exec "$LEROBOT_PY" "$NODE" \
    _raw_topic:="$RAW_TOPIC" \
    _overlay_topic:="$OVERLAY_TOPIC" \
    _yolo_overlay_topic:="$YOLO_OVERLAY_TOPIC" \
    _out_topic:="$OUT_TOPIC"
