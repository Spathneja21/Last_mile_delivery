"""
One-command Husky + AutoDrive bringup.

Includes husky_sim.launch.py (gz + Husky + bridges [+ RViz]) and starts the
AutoDrive node from vp_car_sim, wired to the Husky's camera/cmd_vel. AutoDrive is
an end-to-end "drive along the road" model — it does not target a destination
(use husky_navigate.launch.py for goal-directed obstacle avoidance).

No code is duplicated: the model/inference/node all live in vp_car_sim; this just
launches them against the Husky topics.

Example:
  ros2 launch vp_husky_sim husky_autodrive.launch.py world:=vp_city.world rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    husky_share = get_package_share_directory('vp_husky_sim')

    args = [
        DeclareLaunchArgument('world', default_value='vp_city.world'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'checkpoint_path',
            default_value='/home/iit/ros2_humble/vision_pilot_sim_ws/models/autodrive.pth'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.6'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.0'),
        DeclareLaunchArgument('curvature_gain', default_value='1.0'),
        DeclareLaunchArgument('flag_mode', default_value='ignore'),
    ]

    # 1) Bring up the Husky simulation + bridges (+ RViz).
    husky_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(husky_share, 'launch', 'husky_sim.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'rviz': LaunchConfiguration('rviz'),
        }.items(),
    )

    # 2) AutoDrive node (from vp_car_sim), pointed at the Husky camera. Delayed so
    #    gz + the camera bridge are up before the model starts driving.
    auto_drive = Node(
        package='vp_car_sim',
        executable='auto_drive_node',
        name='auto_drive_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'image_topic': '/husky/front_camera/image_raw',
            'cmd_vel_topic': '/cmd_vel',
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'curvature_gain': LaunchConfiguration('curvature_gain'),
            'flag_mode': LaunchConfiguration('flag_mode'),
            'use_sim_time': True,
        }],
    )

    return LaunchDescription(args + [husky_sim, TimerAction(period=8.0, actions=[auto_drive])])
