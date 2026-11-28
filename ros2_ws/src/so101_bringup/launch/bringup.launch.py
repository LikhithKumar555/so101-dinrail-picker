"""SO-101 with ros2_control: robot_state_publisher + controller manager + controllers (+ RViz).

  ros2 launch so101_bringup bringup.launch.py                       # mock hardware + RViz
  ros2 launch so101_bringup bringup.launch.py rviz:=false
  ros2 launch so101_bringup bringup.launch.py hardware:=feetech     # Feetech bus at /tmp/so101_bus
  ros2 launch so101_bringup bringup.launch.py hardware:=feetech port:=/dev/ttyACM0   # real arm

hardware:=feetech needs something on the serial port first, e.g. the virtual bus:
  python sim/scripts/run_virtual_bus.py --viewer
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

    robot_description = ParameterValue(
        Command(["xacro ", PathJoinSubstitution([desc, "urdf", "so101.urdf.xacro"]),
                 " hardware:=", LaunchConfiguration("hardware"),
                 " port:=", LaunchConfiguration("port"),
                 " timeout_ms:=", LaunchConfiguration("timeout_ms")]),
        value_type=str)
    controllers = PathJoinSubstitution([bringup, "config", "controllers.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock",
                              description="mock (no physics) or feetech (serial servo bus)"),
        DeclareLaunchArgument("port", default_value="/tmp/so101_bus",
                              description="serial device for hardware:=feetech"),
        DeclareLaunchArgument("timeout_ms", default_value="10",
                              description="bus reply timeout per transaction (hardware:=feetech)"),
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
