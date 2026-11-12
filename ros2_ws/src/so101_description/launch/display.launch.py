"""Show the SO-101 in RViz: robot_state_publisher + joint sliders (+ RViz).

  ros2 launch so101_description display.launch.py              # sliders + RViz
  ros2 launch so101_description display.launch.py gui:=false   # all joints at 0, no slider window
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("so101_description")
    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")

    # xacro expands the .xacro file into plain URDF at launch time
    robot_description = ParameterValue(
        Command(["xacro ", PathJoinSubstitution([pkg, "urdf", "so101.urdf.xacro"])]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="joint slider window"),
        DeclareLaunchArgument("rviz", default_value="true", description="start RViz"),

        # Publishes /robot_description and turns /joint_states into TF frames (= forward kinematics)
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),

        # Publishes /joint_states: from sliders (gui) or all zeros (no gui)
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             condition=IfCondition(gui)),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             condition=UnlessCondition(gui)),

        Node(package="rviz2", executable="rviz2",
             arguments=["-d", PathJoinSubstitution([pkg, "rviz", "display.rviz"])],
             condition=IfCondition(rviz)),
    ])
