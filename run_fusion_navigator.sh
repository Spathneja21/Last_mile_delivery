#!/usr/bin/env bash
# run_fusion_navigator.sh
# Runs fusion_navigator_node.py — the single /cmd_vel owner in the merged
# road-follow + vision-avoid pipeline. Subscribes /gps/path, /navsat/fix,
# /imu/data, /vision/avoidance; publishes /cmd_vel and /replan_request.
#
# Preconditions (see cmds.md "merged pipeline"):
#   - roscore + husky_empty_world.launch (sim: /navsat/fix, /imu/data)
#   - a camera node + run_vision_advisory.sh (publishes /vision/avoidance)
#   - map_app.py (publishes /gps/path, handles /replan_request)
# Do NOT also run road_path_navigator / gps_navigator / waypoint_path /
# webcam_navigator — they publish /cmd_vel too and would fight this node.

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE="$WS_DIR/fusion_navigator_node.py"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

source /opt/ros/noetic/setup.bash

# Same PYTHONPATH the cmds.md road_path_navigator invocation uses (rospy etc.).
export PYTHONPATH="\
/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages:\
/usr/lib/python3.8/dist-packages:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# ---------- tunable params (gains, tolerances, topics, calibration) ----------
rosparam load "$WS_DIR/config/fusion_navigator.yaml" /fusion_navigator_node

echo "=================================================="
echo " fusion_navigator_node  (road PID + vision avoid)"
echo "   sole /cmd_vel owner — do not run other navigators"
echo "=================================================="

exec "$VP_GPU_PY" "$NODE"
