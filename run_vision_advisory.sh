#!/usr/bin/env bash
# run_vision_advisory.sh
# Runs vision_avoidance_node.py (perception-only advisor) under ROS 1 Noetic +
# vp_gpu venv (CUDA torch). Publishes /vision/avoidance for fusion_navigator_node.py.
# Same env-layering as run_webcam_navigator.sh — only the node + topics differ.
#
# Precondition: roscore + a camera node publishing IMAGE_TOPIC must be running.

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$WS_DIR/models"
SRC_DIR="$WS_DIR/src/vp_car_sim"
NODE="$SRC_DIR/vp_car_sim/vision_avoidance_node.py"
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

# ---------- model checkpoints (TensorRT FP16 engines) ----------
SEG_ENGINE_PATH="$MODELS_DIR/SceneSeg_FP16.engine"
DEPTH_ENGINE_PATH="$MODELS_DIR/Scene3D_FP16.engine"

# ---------- ROS params (override as needed) ----------
# Webcam for initial bring-up; retarget to husky/front_camera/image_raw for sim vision.
IMAGE_TOPIC="/webcam/image_raw"
ADVISORY_TOPIC="/vision/avoidance"
OVERLAY_TOPIC="/vision_pilot/vision_avoidance/overlay"

# ---------- source ROS 1 ----------
source /opt/ros/noetic/setup.bash

# ---------- PYTHONPATH priority (order matters — see run_webcam_navigator.sh) ----------
export PYTHONPATH="\
$VP_GPU_SITE:\
$SRC_DIR:\
/usr/lib/python3.8/dist-packages:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export PYTHONPATH

# ---------- tunable params (steering gain, ROI, weights, thresholds) ----------
# Loaded onto the param server first; the _name:=value CLI args below still
# take precedence over these since they're applied when the node starts.
rosparam load "$WS_DIR/config/vision_avoidance.yaml" /vision_avoidance_node

echo "=================================================="
echo " vision_avoidance_node  (advisor, ROS 1 / vp_gpu, CUDA)"
echo "=================================================="
echo "  SceneSeg    : $SEG_ENGINE_PATH"
echo "  Scene3D     : $DEPTH_ENGINE_PATH"
echo "  image_topic : $IMAGE_TOPIC"
echo "  advisory    : $ADVISORY_TOPIC"
echo "  overlay     : $OVERLAY_TOPIC"
echo ""
echo "  Make sure 'roscore' and a camera node are running first!"
echo "=================================================="
echo ""

exec "$VP_GPU_PY" "$NODE" \
    _seg_engine_path:="$SEG_ENGINE_PATH" \
    _depth_engine_path:="$DEPTH_ENGINE_PATH" \
    _image_topic:="$IMAGE_TOPIC" \
    _advisory_topic:="$ADVISORY_TOPIC" \
    _overlay_topic:="$OVERLAY_TOPIC"
