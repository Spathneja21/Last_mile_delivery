#!/usr/bin/env bash
# run_road_path_navigator.sh
# Runs road_path_navigator_node.py — drives through the road-snapped
# waypoint list from map_app.py using direct control (no move_base).
#
# Do NOT run this alongside gps_navigator_node.py or waypoint_path_node.py —
# all three publish to /cmd_vel and will fight each other.

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE="$WS_DIR/road_path_navigator_node.py"
VP_GPU_SITE="/data/archit0030/miniforge3/envs/vp_gpu/lib/python3.8/site-packages"
VP_GPU_PY="/data/archit0030/miniforge3/envs/vp_gpu/bin/python3"

source /opt/ros/noetic/setup.bash

export PYTHONPATH="\
$VP_GPU_SITE:\
/usr/lib/python3.8/dist-packages:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

echo "=================================================="
echo " road_path_navigator_node  (ROS 1 / vp_gpu venv)"
echo "=================================================="
echo "  Drives through /gps/path waypoints via direct control (no move_base)."
echo "  Make sure 'roscore', Gazebo/Husky, and map_app.py are running first!"
echo "=================================================="
echo ""

exec "$VP_GPU_PY" "$NODE"
