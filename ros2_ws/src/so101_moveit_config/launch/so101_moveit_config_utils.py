"""Shared MoveIt configuration (imported by the launch files in this folder)."""
import os

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def moveit_config():
    xacro = os.path.join(get_package_share_directory("so101_description"), "urdf", "so101.urdf.xacro")
    return (MoveItConfigsBuilder("so101", package_name="so101_moveit_config")
            .robot_description(file_path=xacro)
            .robot_description_semantic(file_path="config/so101.srdf")
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .joint_limits(file_path="config/joint_limits.yaml")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
            .to_moveit_configs())
