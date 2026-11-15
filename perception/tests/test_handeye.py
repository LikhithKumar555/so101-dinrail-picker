"""Hand-eye solver on exact synthetic data: it must recover the known X and Y."""
import cv2
import numpy as np
import pytest

from so101_perception.calibration import Board
from so101_perception.handeye import (calibrate_eye_to_hand, inv, pose_error, rot_exp,
                                      solve_ax_xb)
from so101_sim.kinematics import make_transform

K = np.array([[1100.0, 0, 480], [0, 1100.0, 360], [0, 0, 1]])
DIST = np.zeros(5)
BOARD = Board(7, 4, 0.015)
rng = np.random.default_rng(0)

X_TRUE = make_transform(rot_exp(np.array([np.pi, 0, 0]) + rng.normal(0, 0.05, 3)), [0.22, 0.0, 0.50])
Y_TRUE = make_transform(rot_exp(rng.normal(0, 0.5, 3)), [0.01, 0.02, 0.09])


def random_gripper_poses(n, rotate=True):
    poses = []
    while len(poses) < n:
        R = rot_exp(rng.normal(0, 0.35, 3)) if rotate else rot_exp([0, 0, rng.uniform(-1, 1)])
        A = make_transform(R, [rng.uniform(0.15, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(0.0, 0.1)])
        T_cam_board = inv(X_TRUE) @ A @ Y_TRUE
        if T_cam_board[2, 3] > 0.1:
            poses.append(A)
    return poses


def project(A, noise_px=0.0):
    T = inv(X_TRUE) @ A @ Y_TRUE
    p, _ = cv2.projectPoints(BOARD.object_points().astype(float), cv2.Rodrigues(T[:3, :3])[0],
                             T[:3, 3], K, DIST)
    return (p.reshape(-1, 1, 2) + rng.normal(0, noise_px, (len(p), 1, 2))).astype(np.float32)


def test_closed_form_ax_xb_exact():
    A = random_gripper_poses(8)
    B = [inv(X_TRUE) @ a @ Y_TRUE for a in A]
    C = [A[j] @ inv(A[i]) for i in range(8) for j in range(i + 1, 8)]
    D = [B[j] @ inv(B[i]) for i in range(8) for j in range(i + 1, 8)]
    dt, dr = pose_error(solve_ax_xb(C, D), X_TRUE)
    assert dt < 1e-6 and dr < 1e-6


def test_full_calibration_with_pixel_noise():
    A = random_gripper_poses(20)
    res = calibrate_eye_to_hand(A, [project(a, 0.3) for a in A], BOARD, K, DIST)
    dt, dr = pose_error(res.T_base_cam, X_TRUE)
    assert res.rms_px < 0.5
    assert dt < 2.0 and dr < 0.2          # mm, deg


def test_rotation_about_one_axis_only_is_degenerate():
    """Classic pitfall: only yawing the gripper cannot determine the camera pose."""
    A = random_gripper_poses(10, rotate=False)
    B = [inv(X_TRUE) @ a @ Y_TRUE for a in A]
    C = [A[j] @ inv(A[i]) for i in range(10) for j in range(i + 1, 10)]
    D = [B[j] @ inv(B[i]) for i in range(10) for j in range(i + 1, 10)]
    with pytest.raises(ValueError):
        solve_ax_xb(C, D)


def test_one_bad_view_is_rejected():
    """A view with a flipped corner order must not ruin the calibration."""
    A = random_gripper_poses(15)
    pts = [project(a, 0.3) for a in A]
    pts[4] = pts[4][::-1].copy()                      # 180-degree flipped corner order
    res = calibrate_eye_to_hand(A, pts, BOARD, K, DIST)
    assert 4 in res.rejected
    dt, dr = pose_error(res.T_base_cam, X_TRUE)
    assert dt < 2.0 and dr < 0.2
