from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        description='Absolute path to the Scene3D .pth checkpoint '
                     '(see autoware_vision_pilot/Models/model_library/Scene3D/README.md)'
    )

    # Input camera topic — defaults to the vp_car camera, but can be pointed at any
    # robot/sim camera, e.g. image_topic:=/sensing/camera/traffic_light/image_raw (AWSIM)
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/vp_car/front_camera/image_raw',
        description='Camera image topic to run Scene3D on.',
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/vision_pilot/scene_3d/overlay',
        description='Topic to publish the Scene3D depth overlay on.',
    )

    scene_3d_node = Node(
        package='vp_car_sim',
        executable='scene_3d_node',
        name='scene_3d_node',
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
        scene_3d_node,
    ])
