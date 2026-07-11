#!/usr/bin/env bash
# run_nl_command.sh
# Runs nl_command_node.py under ROS 1 Noetic + vp_gpu venv.
# Requires GEMINI_API_KEY to be exported in your shell before running:
#   export GEMINI_API_KEY="your-key-here"
#   ./run_nl_command.sh
#
# The key is never hardcoded here or in the node — it's read from the
# environment only, so it never ends up committed to the repo.

set -e

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set." >&2
    echo "Run:  export GEMINI_API_KEY=\"your-key-here\"" >&2
    echo "...then re-run this script." >&2
    exit 1
fi

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$WS_DIR/src/vp_car_sim"
NODE="$SRC_DIR/vp_car_sim/nl_command_node.py"
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

# ---------- ROS params (override as needed) ----------
COMMAND_TOPIC="/vision_pilot/nl_command"
INTENT_TOPIC="/vision_pilot/nl_intent"
DETECTIONS_TOPIC="/vision_pilot/yolo_detector/detections"

source /opt/ros/noetic/setup.bash

export PYTHONPATH="\
$VP_GPU_SITE:\
$SRC_DIR:\
/usr/lib/python3.8/dist-packages:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

echo "=================================================="
echo " nl_command_node  (ROS 1 / vp_gpu venv)"
echo "=================================================="
echo "  command_topic    : $COMMAND_TOPIC"
echo "  intent_topic     : $INTENT_TOPIC"
echo "  detections_topic : $DETECTIONS_TOPIC"
echo "  GEMINI_API_KEY   : (set, hidden)"
echo ""
echo "  Make sure 'roscore' is running first!"
echo "  For 'find and stop' commands, yolo_detector_node must also be running."
echo "=================================================="
echo ""

exec "$VP_GPU_PY" "$NODE" \
    _command_topic:="$COMMAND_TOPIC" \
    _intent_topic:="$INTENT_TOPIC" \
    _detections_topic:="$DETECTIONS_TOPIC"
