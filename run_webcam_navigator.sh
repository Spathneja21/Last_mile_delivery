#!/usr/bin/env bash
# run_webcam_navigator.sh
# Runs webcam_navigator_node.py under ROS 1 Noetic + vp_gpu venv (CUDA torch).
# Usage:
#   ./run_webcam_navigator.sh [video_device]
#
# Examples:
#   ./run_webcam_navigator.sh
#   ./run_webcam_navigator.sh /dev/video2

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$WS_DIR/models"
SRC_DIR="$WS_DIR/src/vp_car_sim"
NODE="$SRC_DIR/vp_car_sim/webcam_navigator_node.py"
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

# ---------- model checkpoints (TensorRT FP16 engines) ----------
SEG_ENGINE_PATH="$MODELS_DIR/SceneSeg_FP16.engine"
DEPTH_ENGINE_PATH="$MODELS_DIR/Scene3D_FP16.engine"
CSV_LOG_PATH="$WS_DIR/logs/pipeline_log.csv"

# ---------- ROS params (override as needed) ----------
IMAGE_TOPIC="/webcam/image_raw"
CMD_VEL_TOPIC="/cmd_vel"
OVERLAY_TOPIC="/vision_pilot/webcam_navigator/overlay"

# ---------- source ROS 1 ----------
source /opt/ros/noetic/setup.bash

# ---------- PYTHONPATH priority (order matters!):
#   1. vp_gpu site-pkgs         — numpy, CUDA torch 2.1.0, torchvision 0.16.1, PIL, cv2, etc.
#   2. vp_car_sim src           — vp_car_sim package imports
#   3. /usr/lib/python3.8/dist-packages — TensorRT 8.5.2 Python bindings (system-installed,
#      ABI-compatible with vp_gpu's python3.8; not present anywhere inside vp_gpu itself)
#   4. system dist-pkgs         — rospkg, catkin_pkg
#   5. /opt/ros/noetic/.../dist — rospy, sensor_msgs, geometry_msgs, genpy, etc.
# ----------
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
echo " webcam_navigator_node  (ROS 1 / vp_gpu venv, CUDA)"
echo "=================================================="
echo "  SceneSeg  : $SEG_ENGINE_PATH"
echo "  Scene3D   : $DEPTH_ENGINE_PATH"
echo "  image_topic : $IMAGE_TOPIC"
echo "  cmd_vel   : $CMD_VEL_TOPIC"
echo "  overlay   : $OVERLAY_TOPIC"
echo "  csv log   : $CSV_LOG_PATH"
echo ""
echo "  Make sure 'roscore' and a camera node are running first!"
echo "=================================================="
echo ""

exec "$VP_GPU_PY" "$NODE" \
    _seg_engine_path:="$SEG_ENGINE_PATH" \
    _depth_engine_path:="$DEPTH_ENGINE_PATH" \
    _image_topic:="$IMAGE_TOPIC" \
    _cmd_vel_topic:="$CMD_VEL_TOPIC" \
    _overlay_topic:="$OVERLAY_TOPIC" \
    _csv_log_path:="$CSV_LOG_PATH"
