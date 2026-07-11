from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        description='Absolute path to the AutoSteer 2.0 .pth checkpoint (1024x512 variant).'
    )

    # Input camera topic — defaults to AWSIM's traffic-light camera, but can be pointed
    # at any robot/sim camera, e.g. image_topic:=/vp_car/front_camera/image_raw
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/sensing/camera/traffic_light/image_raw',
        description='Camera image topic to run AutoSteer on.',
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/vision_pilot/auto_steer/overlay',
        description='Topic to publish the AutoSteer waypoint overlay on.',
    )

    auto_steer_node = Node(
        package='vp_car_sim',
        executable='auto_steer_node',
        name='auto_steer_node',
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
        auto_steer_node,
    ])
