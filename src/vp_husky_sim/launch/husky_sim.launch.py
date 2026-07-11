import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('vp_husky_sim')

    xacro_file = os.path.join(pkg_share, 'urdf', 'husky.urdf.xacro')

    # World is selectable: defaults to the light track, pass world:=vp_city.world
    # for the CitySim-style urban scene.
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='vp_track.world',
        description='World file (in the package worlds/ dir) to load in gz sim.',
    )
    world_file = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Start RViz2 with the husky visualization config.',
    )
    rviz_config = os.path.join(pkg_share, 'rviz', 'husky.rviz')

    # Wrap as a string parameter so robot_state_publisher gets the URDF as XML.
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # gz sim resolves package://vp_husky_sim/meshes/... by searching its resource
    # path for a directory named vp_husky_sim. pkg_share is .../share/vp_husky_sim,
    # so add its parent (.../share) to the resource path.
    set_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.dirname(pkg_share),
    )

    # This host's Gazebo Fortress install only ships the pre-rebrand `ign` CLI
    # (no `gz` binary — see vp_car_sim.launch.py for details). `ign gazebo -r` is
    # the equivalent invocation.
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'husky',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.2'
        ],
        output='screen'
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Sim clock → ROS, so use_sim_time nodes (incl. their control timers) run.
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/husky/front_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/husky/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
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
        set_resource_path,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        rviz,
    ])
