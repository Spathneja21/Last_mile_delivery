"""
One-command Husky + GPS-navigator bringup.

Includes husky_sim.launch.py (gz + Husky + bridges [+ RViz]) and starts
gps_navigator_node, which drives the Husky to whatever target is published on
/gps/target — set by clicking the map in map.html (run map_app.py separately).

Example:
  ros2 launch vp_husky_sim husky_gps_navigate.launch.py world:=vp_city.world rviz:=true
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
        DeclareLaunchArgument('world', default_value='vp_track.world'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('anchor_lat', default_value='37.7749'),
        DeclareLaunchArgument('anchor_lon', default_value='-122.4194'),
        DeclareLaunchArgument('goal_tolerance', default_value='0.5'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.6'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.0'),
    ]

    husky_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(husky_share, 'launch', 'husky_sim.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'rviz': LaunchConfiguration('rviz'),
        }.items(),
    )

    navigator = Node(
        package='vp_husky_sim',
        executable='gps_navigator_node',
        name='gps_navigator_node',
        output='screen',
        parameters=[{
            'anchor_lat': LaunchConfiguration('anchor_lat'),
            'anchor_lon': LaunchConfiguration('anchor_lon'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'use_sim_time': True,
        }],
    )

    return LaunchDescription(args + [husky_sim, TimerAction(period=8.0, actions=[navigator])])
