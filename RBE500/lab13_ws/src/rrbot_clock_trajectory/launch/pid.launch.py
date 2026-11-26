# Copyright 2021 Stogl Robotics Consulting UG (haftungsbeschränkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,  ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile

from visualization_msgs.msg import Marker, MarkerArray
from rosidl_runtime_py import message_to_yaml
import math

def generate_launch_description():
    #  make the clock
    markers = MarkerArray()
    for i in range(1, 13):
        angle = (i/12.)*2*math.pi
        marker = Marker()
        marker.header.frame_id = "/world"

        marker.ns = "clock"
        marker.id = i

        marker.type = Marker.TEXT_VIEW_FACING

        marker.action = Marker.ADD

        # radius of 1m centered 1m above top of link1
        marker.pose.position.x = 0.95*math.sin(angle)
        marker.pose.position.y = -0.1 # in front of the arm
        marker.pose.position.z = 2.95 + 0.95*math.cos(angle)

        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.text = f'{i}'
        markers.markers.append(marker)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start RViz2 automatically with this launch file.",
            ),
            DeclareLaunchArgument(
                "joint_prefix",
                default_value="joint",
                description="Prefix for joint names (used with allow_substs in spawner).",
            ),
            # Control node
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[{"update_rate": 10}],
                output="log",
            ),
            # robot_state_publisher with robot_description from xacro
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="log",
                parameters=[
                    {
                        "robot_description": Command(
                            [
                                "xacro",
                                " ",
                                PathSubstitution(FindPackageShare("rrbot_clock_trajectory"))
                                / "urdf"
                                / "rrbot.urdf.xacro",
                            ]
                        )
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="log",
                arguments=[
                    "-d",
                    PathSubstitution(FindPackageShare("ros2_control_demo_description"))
                    / "rrbot/rviz/rrbot.rviz",
                ],
                condition=IfCondition(LaunchConfiguration("gui")),
            ),
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once', '-w', '1',
                    '/clock_numbers',
                    'visualization_msgs/msg/MarkerArray',
                    message_to_yaml(markers),
                ],
                output='log'
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "joint_state_broadcaster",
                    "--param-file",
                    PathSubstitution(FindPackageShare("rrbot_clock_trajectory"))
                    / "config"
                    / "rrbot_pid.yaml",
                ],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                parameters=[
                    {"joint_prefix": LaunchConfiguration("joint_prefix")},
                    ParameterFile(
                        PathSubstitution(FindPackageShare("rrbot_clock_trajectory"))
                        / "config"
                        / "rrbot_pid.yaml",
                        allow_substs=True,
                    ),
                ],
                arguments=[
                    "rrbot_pid_controller",
                ],
            ),
            # Node(
            #     package='rrbot_clock_trajectory',
            #     executable='setpoints',
            #     output='both',
            # ),
        ]
    )
