"""Everything for MoveIt in one go: robot + ros2_control (mock hardware) + move_group + RViz
with the MotionPlanning panel.   ros2 launch so101_moveit_config demo.launch.py
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from so101_moveit_config_utils import moveit_config  # noqa: E402


def generate_launch_description():
    cfg = moveit_config()
    bringup = os.path.join(get_package_share_directory("so101_bringup"), "launch", "bringup.launch.py")
    rviz_cfg = os.path.join(get_package_share_directory("so101_moveit_config"), "config", "moveit.rviz")
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        # GroupAction scopes the include: its rviz:=false stays inside the group.
        # Without it, launch arguments of an include leak into this file and
        # would switch off the MoveIt RViz below as well.
        GroupAction([
            IncludeLaunchDescription(PythonLaunchDescriptionSource(bringup),
                                     launch_arguments={"rviz": "false"}.items()),
        ]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[cfg.to_dict()]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg],
             parameters=[cfg.robot_description, cfg.robot_description_semantic,
                         cfg.robot_description_kinematics, cfg.planning_pipelines, cfg.joint_limits],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
