"""Printed test sheet (step 4.2b): layout, board pose from a rendered photo, geometry chain."""
import cv2
import numpy as np
import pytest

from so101_perception.pose_estimation import Camera, estimate_block_pose, is_plausible_block
from so101_perception.print_sheet import (A4_H_MM, A4_W_MM, LAYOUT, PAGE_MARGIN_MM, SHEET_BOARD,
                                          background_rects_mm, block_corners_mm, look_at_sheet, page_to_board,
                                          pose_from_corners, quiet_zone_rect_mm, rasterize_sheet,
                                          render_sheet_view, sheet_pose, truth)

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
DIST = np.array([0.06, -0.12, 0.0005, -0.0003, 0.02])
SIZE = (1280, 720)


@pytest.fixture(scope="module")
def sheet():
    return rasterize_sheet(8.0)


def _inside(p, rect, tol=0.0):
    x0, y0, x1, y1 = rect
    return x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol


def test_layout_is_valid():
    assert sorted(b.cls for b in LAYOUT) == [0] * 4 + [1] * 4 + [2] * 4
    r = np.hypot(22.5, 6.1)                                   # circumradius of the widest block
    for i, a in enumerate(LAYOUT):
        for b in LAYOUT[i + 1:]:
            assert np.hypot(*np.subtract(a.centre_mm, b.centre_mm)) >= 2 * r + 5
        for p in block_corners_mm(a):
            assert PAGE_MARGIN_MM <= p[0] <= A4_W_MM - PAGE_MARGIN_MM
            assert PAGE_MARGIN_MM <= p[1] <= A4_H_MM - PAGE_MARGIN_MM
            assert not _inside(p, quiet_zone_rect_mm(), tol=2.0), "block too close to the board"
            assert any(_inside(p, rect) for rect in background_rects_mm()), "block off the background"


def test_sheet_pose_from_rendered_photo(sheet):
    rng = np.random.default_rng(1)
    for tilt_x, tilt_y, roll in [(0, 0, 0), (15, -10, 25), (-20, 12, -40)]:
        T_true = look_at_sheet(0.36, tilt_x, tilt_y, roll)
        img = render_sheet_view(*sheet, K, DIST, T_true, SIZE, rng=rng)
        res = sheet_pose(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K, DIST)
        assert res is not None, "board not found"
        T, _, rms = res
        assert rms < 0.5
        assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 1.5e-3            # < 1.5 mm
        dR = T[:3, :3].T @ T_true[:3, :3]
        assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 0.3


def test_mirrored_corner_order_is_corrected(sheet):
    T_true = look_at_sheet(0.36, 10, 5, 15)
    img = render_sheet_view(*sheet, K, DIST, T_true, SIZE)
    T, corners, _ = sheet_pose(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K, DIST)
    mirrored = corners.reshape(SHEET_BOARD.rows, SHEET_BOARD.cols, 1, 2)[:, ::-1].reshape(-1, 1, 2)
    T_m, _, _ = pose_from_corners(mirrored, K, DIST)
    np.testing.assert_allclose(T_m, T, atol=1e-6)


def test_true_corners_through_estimated_camera_give_true_poses(sheet):
    """Detector replaced by the exact corners: what is left is calibration + PnP + back-projection."""
    T_true = look_at_sheet(0.36, 12, -8, 20)
    img = render_sheet_view(*sheet, K, DIST, T_true, SIZE)
    T_cam_board, _, _ = sheet_pose(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K, DIST)
    cam_true = Camera(K, DIST, np.linalg.inv(T_true))
    cam_est = Camera(K, DIST, np.linalg.inv(T_cam_board))
    for b, (cls, xy, yaw, width) in zip(LAYOUT, truth()):
        corners3 = np.column_stack([page_to_board(block_corners_mm(b)), np.zeros(4)])
        px = cam_true.project(corners3)
        est = estimate_block_pose(cam_est, px, cls, top_z=0.0)
        assert np.linalg.norm(est.position[:2] - xy) < 0.5e-3                 # < 0.5 mm
        d_yaw = (est.yaw - yaw + np.pi / 2) % np.pi - np.pi / 2
        assert abs(np.degrees(d_yaw)) < 0.3
        assert abs(est.width_mm - width) < 0.2
        assert abs(est.length_mm - 45.0) < 0.3
        assert is_plausible_block(est)
    # a checkerboard column (75 x 17 mm) seen as a "block" must fail the size check
    column = np.array([[0.0, -0.015, 0], [0.0, 0.06, 0], [0.017, 0.06, 0], [0.017, -0.015, 0]])
    fake = estimate_block_pose(cam_est, cam_true.project(column), 2, top_z=0.0)
    assert not is_plausible_block(fake)
