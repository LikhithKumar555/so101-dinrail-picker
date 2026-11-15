"""Simulated hand-eye calibration: a checkerboard on a bracket on the gripper, seen by the
fixed overhead camera. The board is added to the robot model in code (MjSpec), so the
Menagerie files stay untouched. In simulation we know the TRUE camera pose, so we can measure
exactly how good the calibration is - impossible on the real robot.
"""
from pathlib import Path

import mujoco
import numpy as np

from so101_perception.calibration import Board
from so101_sim.kinematics import ARM_JOINTS, make_transform
from so101_sim.scene import SCENE_XML, scene_kinematics

from so101_perception.handeye import rot_to_quat

HANDEYE_BOARD = Board(cols=7, rows=4, square=0.015)   # 8 x 5 squares, 120 x 75 mm (+ margin)
# Bracket: board centre 9 cm to the side of the TCP (along the jaw-closing axis z_tcp) and
# 3 cm behind it (towards the wrist), facing the camera when the gripper points down.
BRACKET_SIDE, BRACKET_BACK = 0.09, 0.03


def board_mount_in_tcp(board: Board) -> np.ndarray:
    """T_tcp_board: board x = z_tcp, board z = x_tcp (board normal points along the approach)."""
    R = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    centre_b = np.array([(board.cols - 1) * board.square / 2, (board.rows - 1) * board.square / 2, 0])
    centre_tcp = np.array([-BRACKET_BACK, 0.0, BRACKET_SIDE])
    return make_transform(R, centre_tcp - R @ centre_b)


def build_handeye_model(board: Board = HANDEYE_BOARD):
    """Scene + board on the gripper. Returns (model, true T_gripper_board)."""
    kin = scene_kinematics()
    T_gripper_board = kin.T_last_tcp @ board_mount_in_tcp(board)
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    body = spec.body("gripper").add_body(name="calib_board", pos=T_gripper_board[:3, 3],
                                         quat=rot_to_quat(T_gripper_board[:3, :3]))
    s, h = board.square, 0.0002
    # board z points AWAY from the camera: black squares end at z = 0 (the corner plane),
    # white plate behind them
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(board.cols + 3) * s / 2, (board.rows + 3) * s / 2, 0.0015],
                  pos=[(board.cols - 1) * s / 2, (board.rows - 1) * s / 2, 2 * h + 0.0015],
                  rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    for a in range(board.cols + 1):
        for b in range(board.rows + 1):
            if (a + b) % 2 == 0:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2, s / 2, h],
                              pos=[(a - 0.5) * s, (b - 0.5) * s, h], rgba=[0.05, 0.05, 0.05, 1],
                              contype=0, conaffinity=0)
    return spec.compile(), T_gripper_board


def set_arm(model, data, q):
    for j, qj in zip(ARM_JOINTS, q):
        data.qpos[model.joint(j).qposadr[0]] = qj
    mujoco.mj_forward(model, data)
