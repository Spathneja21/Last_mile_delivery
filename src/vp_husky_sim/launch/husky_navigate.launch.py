"""
One-command Husky + SceneSeg-navigator bringup.

Includes husky_sim.launch.py (gz + Husky + bridges [+ RViz]) and starts the
SceneSeg-based reactive navigator from vp_car_sim, wired to the Husky's
camera/odom/cmd_vel. This is the goal-directed obstacle-avoidance autonomy:
drives toward (goal_x, goal_y) in the odom frame while avoiding obstacles.

No code is duplicated: the model/inference/node all live in vp_car_sim.

Example:
  ros2 launch vp_husky_sim husky_navigate.launch.py \
      world:=vp_city.world rviz:=true goal_x:=40.0 goal_y:=0.0
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
            default_value='/home/iit/ros2_humble/vision_pilot_sim_ws/models/SceneSeg.pth'),
        DeclareLaunchArgument('goal_x', default_value='40.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('goal_tolerance', default_value='1.5'),
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
        package='vp_car_sim',
        executable='scene_seg_navigator',
        name='scene_seg_navigator',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'image_topic': '/husky/front_camera/image_raw',
            'odom_topic': '/odom',
            'cmd_vel_topic': '/cmd_vel',
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'use_sim_time': True,
        }],
    )

    return LaunchDescription(args + [husky_sim, TimerAction(period=8.0, actions=[navigator])])
