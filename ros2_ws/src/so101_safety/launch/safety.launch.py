"""Safety monitor (+ simulated telemetry until the real arm is connected).

Start the robot first (ros2 launch so101_bringup bringup.launch.py), then:
  ros2 launch so101_safety safety.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("so101_safety"), "config", "safety.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("fake_telemetry", default_value="true"),
        Node(package="so101_safety", executable="safety_monitor", parameters=[config],
             output="screen"),
        Node(package="so101_safety", executable="fake_telemetry.py", name="fake_telemetry",
             output="screen", condition=IfCondition(LaunchConfiguration("fake_telemetry"))),
    ])
