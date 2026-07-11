from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        description='Absolute path to the SceneSeg .pth checkpoint '
                     '(see autoware/autoware_vision_pilot/Models/model_library/SceneSeg/README.md)'
    )
    depth_checkpoint_arg = DeclareLaunchArgument(
        'depth_checkpoint_path',
        description='Absolute path to the Scene3D .pth checkpoint (see '
                     'autoware_vision_pilot/Models/model_library/Scene3D/README.md).'
    )
    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='V4L2 device path for the external USB camera.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/webcam/image_raw',
        description='Topic the camera publishes raw frames on (also where this node subscribes).',
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Topic to publish the computed Twist (linear.x=velocity, angular.z=steering) on.',
    )
    overlay_topic_arg = DeclareLaunchArgument(
        'overlay_topic',
        default_value='/vision_pilot/webcam_navigator/overlay',
        description='Topic to publish the debug overlay on.',
    )

    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='webcam',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('video_device'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('image_topic')),
        ],
    )

    webcam_navigator_node = Node(
        package='vp_car_sim',
        executable='webcam_navigator_node',
        name='webcam_navigator_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'depth_checkpoint_path': LaunchConfiguration('depth_checkpoint_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'overlay_topic': LaunchConfiguration('overlay_topic'),
        }]
    )

    return LaunchDescription([
        checkpoint_arg,
        depth_checkpoint_arg,
        video_device_arg,
        image_topic_arg,
        cmd_vel_topic_arg,
        overlay_topic_arg,
        camera_node,
        webcam_navigator_node,
    ])
