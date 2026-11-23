"""MoveIt 2 move_group for the SO-101 (start so101_bringup first, or use demo.launch.py)."""
from launch import LaunchDescription
from launch_ros.actions import Node

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from so101_moveit_config_utils import moveit_config  # noqa: E402


def generate_launch_description():
    cfg = moveit_config()
    return LaunchDescription([
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[cfg.to_dict()]),
    ])
