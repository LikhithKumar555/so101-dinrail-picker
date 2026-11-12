"""SO-101 with ros2_control: robot_state_publisher + controller manager + controllers (+ RViz).

  ros2 launch so101_bringup bringup.launch.py                  # mock hardware + RViz
  ros2 launch so101_bringup bringup.launch.py rviz:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc = FindPackageShare("so101_description")
    bringup = FindPackageShare("so101_bringup")
    use_mock = LaunchConfiguration("use_mock_hardware")

    robot_description = ParameterValue(
        Command(["xacro ", PathJoinSubstitution([desc, "urdf", "so101.urdf.xacro"]),
                 " use_mock_hardware:=", use_mock]),
        value_type=str)
    controllers = PathJoinSubstitution([bringup, "config", "controllers.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("use_mock_hardware", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),

        # The controller manager loads the hardware plugin named in <ros2_control>.
        # It reads the robot description from the /robot_description topic.
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[controllers],
             remappings=[("~/robot_description", "/robot_description")],
             output="screen"),

        # Load + activate the controllers
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster", "arm_controller", "gripper_controller",
                        "--controller-manager", "/controller_manager"]),

        Node(package="rviz2", executable="rviz2",
             arguments=["-d", PathJoinSubstitution([desc, "rviz", "display.rviz"])],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
