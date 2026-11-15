"""Hand-eye calibration for a FIXED camera looking at the robot ("eye-to-hand").

Unknowns:  X = T_base_cam     where the camera is in the robot base frame (what we need)
           Y = T_gripper_board where the checkerboard sits on the gripper (bracket, unknown)
Per robot pose i we know:
           A_i = T_base_gripper  from forward kinematics
           B_i = T_cam_board     from the camera (solvePnP on the detected corners)
The chain base -> camera -> board must equal base -> gripper -> board:

           X B_i = A_i Y                              ("AX = YB" form)

Taking two poses i, j and eliminating Y gives the classic   C X = X D  with
           C = A_j A_i^-1  (gripper motion, base frame),  D = B_j B_i^-1  (board motion, camera frame)
which we solve in closed form (Park & Martin 1994), then refine X and Y together by minimising
the pixel reprojection error of all board corners in all images (the gold standard).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from so101_perception.calibration import Board
from so101_sim.ik import rot_to_quat
from so101_sim.kinematics import axis_angle_to_rot, make_transform


# ------------------------------------------------------------------ SO(3) helpers
def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    q = rot_to_quat(R)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    return np.zeros(3) if s < 1e-12 else q[1:] / s * 2 * np.arctan2(s, q[0])


def rot_exp(v: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-12 else axis_angle_to_rot(v / a, a)


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def pose_to_vec(T):
    return np.concatenate([rot_log(T[:3, :3]), T[:3, 3]])


def vec_to_pose(v):
    return make_transform(rot_exp(v[:3]), v[3:])


# ------------------------------------------------------------------ solvers
def solve_ax_xb(C: list, D: list) -> np.ndarray:
    """Closed-form C_k X = X D_k (Park & Martin). Needs rotations about >= 2 different axes."""
    M = sum(np.outer(rot_log(d[:3, :3]), rot_log(c[:3, :3])) for c, d in zip(C, D))
    w, V = np.linalg.eigh(M.T @ M)
    if w.min() < 1e-10:
        raise ValueError("degenerate motions: the robot must rotate about at least two different axes")
    R = V @ np.diag(w ** -0.5) @ V.T @ M.T
    A = np.vstack([c[:3, :3] - np.eye(3) for c in C])
    b = np.concatenate([R @ d[:3, 3] - c[:3, 3] for c, d in zip(C, D)])
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return make_transform(R, t)


@dataclass
class HandEyeResult:
    T_base_cam: np.ndarray
    T_gripper_board: np.ndarray
    rms_px: float               # reprojection error after refinement (inlier views)
    n_views: int                # views used
    rejected: list              # indices of views rejected as outliers


def _view_rms(X, Y, A_i, ip, obj, K, dist) -> float:
    T = inv(X) @ A_i @ Y
    proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - np.asarray(ip).reshape(-1, 2)) ** 2, 1))))


def calibrate_eye_to_hand(A: list, image_points: list, board: Board, K, dist,
                          inlier_px: float = 3.0, ransac_iters: int = 100, seed: int = 0) -> HandEyeResult:
    """A: T_base_gripper per view (from FK). image_points: detected corners per view.

    One bad view (a flipped corner order, the planar-PnP ambiguity, a blurred image) can ruin
    the closed-form solution for all views, so it is wrapped in RANSAC: solve from random
    4-view subsets, keep the solution that explains the most views within `inlier_px`, then
    refine X and Y on those inliers by minimising the reprojection error.
    """
    obj = board.object_points().astype(np.float64)
    B = []
    for ip in image_points:
        ok, rvec, tvec = cv2.solvePnP(obj, ip, K, dist, flags=cv2.SOLVEPNP_IPPE)
        ok, rvec, tvec = cv2.solvePnP(obj, ip, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        B.append(make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()))

    def closed_form(idx):
        pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
        X = solve_ax_xb([A[j] @ inv(A[i]) for i, j in pairs], [B[j] @ inv(B[i]) for i, j in pairs])
        return X, inv(A[idx[0]]) @ X @ B[idx[0]]

    n = len(A)
    rng = np.random.default_rng(seed)
    best = (-1, None)
    for _ in range(ransac_iters):
        idx = list(rng.choice(n, size=min(4, n), replace=False))
        try:
            X, Y = closed_form(idx)
        except ValueError:
            continue
        inliers = [i for i in range(n) if _view_rms(X, Y, A[i], image_points[i], obj, K, dist) < inlier_px]
        if len(inliers) > best[0]:
            best = (len(inliers), inliers)
    inliers = best[1] if best[1] and len(best[1]) >= 3 else list(range(n))
    X0, Y0 = closed_form(inliers)

    def residuals(p):
        X, Y = vec_to_pose(p[:6]), vec_to_pose(p[6:])
        res = []
        for i in inliers:
            T_cam_board = inv(X) @ A[i] @ Y
            rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
            proj, _ = cv2.projectPoints(obj, rvec, T_cam_board[:3, 3], K, dist)
            res.append((proj.reshape(-1, 2) - np.asarray(image_points[i]).reshape(-1, 2)).ravel())
        return np.concatenate(res)

    sol = least_squares(residuals, np.concatenate([pose_to_vec(X0), pose_to_vec(Y0)]), method="lm")
    rms = float(np.sqrt(np.mean(sol.fun.reshape(-1, 2) ** 2) * 2))
    rejected = [i for i in range(n) if i not in inliers]
    return HandEyeResult(vec_to_pose(sol.x[:6]), vec_to_pose(sol.x[6:]), rms, len(inliers), rejected)


def pose_error(T_est: np.ndarray, T_true: np.ndarray):
    """(translation error [mm], rotation error [deg])."""
    dt = np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000
    dr = np.degrees(np.linalg.norm(rot_log(T_est[:3, :3].T @ T_true[:3, :3])))
    return dt, dr
