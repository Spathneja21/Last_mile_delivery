# ROS2 launch file for a live-webcam demo: starts a v4l2 USB camera node feeding
# SceneSeg (segmentation), and optionally Scene3D (depth) if a checkpoint is given.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        description='Absolute path to the SceneSeg .pth checkpoint '
                     '(see autoware/autoware_vision_pilot/Models/model_library/SceneSeg/README.md)'
    )
    depth_checkpoint_arg = DeclareLaunchArgument(
        'depth_checkpoint_path',
        default_value='',
        description='Absolute path to the Scene3D .pth checkpoint (see '
                     'autoware_vision_pilot/Models/model_library/Scene3D/README.md). '
                     'Leave empty to skip Scene3D and only run SceneSeg.',
    )
    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='V4L2 device path for the external USB camera.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/webcam/image_raw',
        description='Topic the camera publishes raw frames on (also where SceneSeg/Scene3D subscribe).',
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/vision_pilot/scene_seg/overlay',
        description='Topic to publish the SceneSeg overlay on.',
    )
    depth_output_topic_arg = DeclareLaunchArgument(
        'depth_output_topic',
        default_value='/vision_pilot/scene_3d/overlay',
        description='Topic to publish the Scene3D depth overlay on.',
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

    scene_seg_node = Node(
        package='vp_car_sim',
        executable='scene_seg_node',
        name='scene_seg_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'input_topic': LaunchConfiguration('image_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
        }]
    )

    scene_3d_node = Node(
        package='vp_car_sim',
        executable='scene_3d_node',
        name='scene_3d_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('depth_checkpoint_path'), "' != ''"])
        ),
        parameters=[{
            'checkpoint_path': LaunchConfiguration('depth_checkpoint_path'),
            'input_topic': LaunchConfiguration('image_topic'),
            'output_topic': LaunchConfiguration('depth_output_topic'),
        }]
    )

    return LaunchDescription([
        checkpoint_arg,
        depth_checkpoint_arg,
        video_device_arg,
        image_topic_arg,
        output_topic_arg,
        depth_output_topic_arg,
        camera_node,
        scene_seg_node,
        scene_3d_node,
    ])