import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('vp_car_sim')

    xacro_file = os.path.join(pkg_share, 'urdf', 'vp_car.urdf.xacro')

    # World is selectable: defaults to the light track, pass world:=vp_city.world
    # for the CitySim-style urban scene.
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='vp_track.world',
        description='World file (in the package worlds/ dir) to load in gz sim.',
    )
    world_file = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    # Optionally start RViz2 with the pre-built car config.
    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Start RViz2 with the vp_car visualization config.',
    )
    rviz_config = os.path.join(pkg_share, 'rviz', 'vp_car.rviz')

    # Wrap as a string parameter, otherwise launch tries to parse the URDF XML as
    # YAML and robot_state_publisher fails to start (which silently breaks spawning).
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # Start Gazebo with our world. This host's Gazebo Fortress install only ships
    # the pre-rebrand `ign` CLI (no `gz` binary — confirmed: `ignition-tools` 1.5.0
    # provides /usr/bin/ign, not /usr/bin/gz; `gz-tools` would add the `gz` shim but
    # isn't pulled in as a dependency). `ign gazebo -r` is the equivalent invocation.
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        output='screen'
    )

    # Publish robot_description / TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # Spawn the car into the running world
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'vp_car',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.15'
        ],
        output='screen'
    )

    # Bridge camera, cmd_vel and odom between gz and ROS2
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Sim clock → ROS, so use_sim_time nodes (incl. their control timers) run.
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/vp_car/front_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/vp_car/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # Wheel joint angles → robot_state_publisher, so RViz can place the wheels.
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        declare_world,
        declare_rviz,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        rviz,
    ])
