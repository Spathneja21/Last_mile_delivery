# ROS2 launch file that starts the scene_seg_node (SceneSeg segmentation-inference
# node) standalone, wired to a camera image topic and publishing a class overlay.
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

    # Input camera topic — defaults to the vp_car camera, but can be pointed at the
    # Husky (or any robot) camera, e.g. image_topic:=/husky/front_camera/image_raw
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/vp_car/front_camera/image_raw',
        description='Camera image topic to run SceneSeg on.',
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/vision_pilot/scene_seg/overlay',
        description='Topic to publish the SceneSeg overlay on.',
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

    return LaunchDescription([
        checkpoint_arg,
        image_topic_arg,
        output_topic_arg,
        scene_seg_node,
    ])
