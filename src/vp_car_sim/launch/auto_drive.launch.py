"""
Launch the AutoDrive end-to-end driving node.

Drives a robot (default the Husky) by feeding its camera into the trained
AutoDrive model and publishing /cmd_vel. AutoDrive follows the road; it does NOT
target a destination (use scene_seg_navigate.launch.py for goal-directed driving).

Example:
  ros2 launch vp_car_sim auto_drive.launch.py \
      checkpoint_path:=/ws/models/autodrive.pth \
      image_topic:=/husky/front_camera/image_raw
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('checkpoint_path',
                              description='Absolute path to the AutoDrive .pth checkpoint'),
        DeclareLaunchArgument('image_topic', default_value='/husky/front_camera/image_raw'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.6'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.0'),
        DeclareLaunchArgument('curvature_gain', default_value='1.0'),
        DeclareLaunchArgument('flag_mode', default_value='ignore'),
    ]

    auto_drive = Node(
        package='vp_car_sim',
        executable='auto_drive_node',
        name='auto_drive_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'curvature_gain': LaunchConfiguration('curvature_gain'),
            'flag_mode': LaunchConfiguration('flag_mode'),
            'use_sim_time': True,
        }],
    )

    return LaunchDescription(args + [auto_drive])
