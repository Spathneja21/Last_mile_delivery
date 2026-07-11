#!/usr/bin/env bash
# run_send_nl_command.sh
# Interactive CLI: type a plain-English command, it's published to
# /vision_pilot/nl_command for nl_command_node.py to parse.

set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE="$WS_DIR/scripts/send_nl_command.py"
LEROBOT_SITE="/data/archit0030/miniforge3/envs/lerobot/lib/python3.10/site-packages"
LEROBOT_PY="/data/archit0030/miniforge3/envs/lerobot/bin/python3"

COMMAND_TOPIC="/vision_pilot/nl_command"

source /opt/ros/noetic/setup.bash

export PYTHONPATH="\
$LEROBOT_SITE:\
/usr/lib/python3/dist-packages:\
/opt/ros/noetic/lib/python3/dist-packages:\
$PYTHONPATH"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

exec "$LEROBOT_PY" "$NODE" _command_topic:="$COMMAND_TOPIC"
