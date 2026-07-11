from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        description='Absolute path to the SceneSeg .pth checkpoint.'
    )
    depth_checkpoint_arg = DeclareLaunchArgument(
        'depth_checkpoint_path',
        description='Absolute path to the Scene3D .pth checkpoint.'
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/sensing/camera/traffic_light/image_raw',
        description='AWSIM camera topic to monitor for obstacles.',
    )
    enable_autosteer_arg = DeclareLaunchArgument(
        'enable_autosteer',
        default_value='true',
        description='Blend AutoSteer 2.0 lane-centering into steering. Set false to '
                     'fall back to SceneSeg/Scene3D bin-scored steering only (also '
                     'drops autosteer_checkpoint_path as a requirement, and avoids '
                     'the extra per-frame inference cost on CPU-only hosts).',
    )
    autosteer_checkpoint_arg = DeclareLaunchArgument(
        'autosteer_checkpoint_path',
        default_value='',
        description='Absolute path to the AutoSteer 2.0 .pth checkpoint (1024x512 '
                     'variant). Required iff enable_autosteer:=true.',
    )

    awsim_navigator_node = Node(
        package='vp_car_sim',
        executable='awsim_navigator_node',
        name='awsim_navigator_node',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'depth_checkpoint_path': LaunchConfiguration('depth_checkpoint_path'),
            'autosteer_checkpoint_path': LaunchConfiguration('autosteer_checkpoint_path'),
            'enable_autosteer': ParameterValue(LaunchConfiguration('enable_autosteer'), value_type=bool),
            'image_topic': LaunchConfiguration('image_topic'),
            # AWSIM publishes /clock (sim time) and stamps its own status messages
            # with it; our outgoing Control message stamps must match or AWSIM
            # appears to disregard them as implausible (confirmed empirically -
            # see AWSIM_INTEGRATION_PLAN.md S4.4c).
            'use_sim_time': True,
        }]
    )

    return LaunchDescription([
        checkpoint_arg,
        depth_checkpoint_arg,
        image_topic_arg,
        enable_autosteer_arg,
        autosteer_checkpoint_arg,
        awsim_navigator_node,
    ])
