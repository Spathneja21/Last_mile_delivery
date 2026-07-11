"""
Launch the SceneSeg-based reactive navigator.

Drives a robot toward (goal_x, goal_y) in the odom frame while avoiding
foreground obstacles seen by SceneSeg. Defaults target the Husky; override
image_topic/odom_topic for vp_car.

Example:
  ros2 launch vp_car_sim scene_seg_navigate.launch.py \
      checkpoint_path:=/ws/models/SceneSeg.pth goal_x:=40.0 goal_y:=0.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('checkpoint_path',
                              description='Absolute path to the SceneSeg .pth checkpoint'),
        DeclareLaunchArgument('image_topic', default_value='/husky/front_camera/image_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('goal_x', default_value='40.0'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('goal_tolerance', default_value='1.5'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.6'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.0'),
    ]

    navigator = Node(
        package='vp_car_sim',
        executable='scene_seg_navigator',
        name='scene_seg_navigator',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'use_sim_time': True,
        }],
    )

    return LaunchDescription(args + [navigator])
