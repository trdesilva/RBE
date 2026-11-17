from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ros_gz_sim_pkg_path = get_package_share_directory('ros_gz_sim')
    pkg_path = FindPackageShare('lab12_pkg')
    gz_launch_path = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])

    return LaunchDescription([
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            PathJoinSubstitution([pkg_path, 'models'])
        ),
        SetEnvironmentVariable(
            'GZ_SIM_PLUGIN_PATH',
            PathJoinSubstitution([pkg_path, 'plugins'])
        ),
        DeclareLaunchArgument(name="model", default_value=PathJoinSubstitution([pkg_path, 'models', 'cam.urdf.xacro']),
                              description="Absolute path to robot urdf file"
                              ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': PathJoinSubstitution([pkg_path, 'worlds', 'world.sdf']),
                'on_exit_shutdown': 'True'
            }.items(),
        ),
        # start edge_detect first so ros_gz_bridge has subscribers ready on startup
        Node(
            package='lab12_pkg',
            executable='edge_detect'
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": ParameterValue(Command([
                    "xacro ",
                    LaunchConfiguration("model"),
                    " is_ignition:=",
                    "True"
                ]),
                value_type=str
            ),
            "use_sim_time": True}]
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=["-topic", "robot_description",
                       "-name", "model_cam"],
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image', # camera built into world
                       '/model_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image', # camera from cam.urdf.xacro model
                   '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'],
            output='screen'
        )
    ])