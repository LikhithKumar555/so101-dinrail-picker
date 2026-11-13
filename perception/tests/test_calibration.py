"""Calibrate on synthetic images of a known camera and check we recover it."""
import cv2
import numpy as np

from so101_perception.calibration import (Board, calibrate_robust, find_corners, render_board_image,
                                          undistort_normalized)

K_TRUE = np.array([[820.0, 0, 330.0], [0, 815.0, 235.0], [0, 0, 1]])
DIST_TRUE = np.array([-0.12, 0.05, 0.001, -0.0005, 0.0])  # mild barrel distortion, typical webcam
BOARD = Board(9, 6, 0.025)


def random_view(rng):
    """Board 25-45 cm away, tilted up to ~35 deg, spread over the whole image.

    Spreading matters: distortion is strongest at the image edges, and it can only be
    estimated where the calibration images actually put corners.
    """
    rvec = rng.uniform(-0.6, 0.6, 3) * [1, 1, 0.5]
    R, _ = cv2.Rodrigues(rvec)
    centre = np.array([BOARD.cols - 1, BOARD.rows - 1, 0]) * BOARD.square / 2
    t = np.array([rng.uniform(-0.09, 0.09), rng.uniform(-0.06, 0.06), rng.uniform(0.28, 0.45)])
    return R, t - R @ centre


def test_recovers_known_camera():
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 20:
        R, t = random_view(rng)
        img = render_board_image(BOARD, K_TRUE, DIST_TRUE, R, t, rng=rng)
        c = find_corners(img, BOARD)
        if c is not None:
            pts.append(c)
    res, keep, rejected = calibrate_robust(pts, BOARD, (640, 480))
    assert len(keep) >= 15
    assert res.rms < 0.5                                       # good calibrations: < 0.5 px
    np.testing.assert_allclose(res.K[[0, 1], [0, 1]], K_TRUE[[0, 1], [0, 1]], rtol=0.01)
    np.testing.assert_allclose(res.K[:2, 2], K_TRUE[:2, 2], atol=3.0)
    # Distortion: compare how far each model shifts a pixel near the image corner
    # (raw k1/k2/k3 are correlated - different combinations fit equally well).
    for res_K, res_d in [(res.K, res.dist)]:
        np.testing.assert_allclose(distortion_shift([580.0, 420.0], res_K, res_d),
                                   distortion_shift([580.0, 420.0], K_TRUE, DIST_TRUE), atol=1.0)
    assert rejected == [] or len(rejected) <= 3


def distortion_shift(pixel, K, dist):
    """Pixel displacement caused by lens distortion at `pixel` (distorted -> ideal)."""
    xy = undistort_normalized(np.array([[pixel]], float), K, dist).reshape(2)
    ideal = K[:2, :2] @ xy + K[:2, 2]
    return np.asarray(pixel) - ideal


def test_board_not_found_returns_none():
    assert find_corners(np.full((480, 640), 200, np.uint8), BOARD) is None


def test_wrong_detection_is_rejected_before_calibration():
    """A view with mis-ordered corners must be filtered out.

    (A mirrored corner order would NOT be wrong: it looks like the board seen from behind.
    Swapping two rows, however, is impossible for any flat board.)"""
    rng = np.random.default_rng(1)
    pts = []
    while len(pts) < 12:
        R, t = random_view(rng)
        rvec, _ = cv2.Rodrigues(R)
        p, _ = cv2.projectPoints(BOARD.object_points(), rvec, t, K_TRUE, DIST_TRUE)
        p = p.reshape(-1, 1, 2).astype(np.float32)
        if p.min() > 10 and p[..., 0].max() < 630 and p[..., 1].max() < 470:
            pts.append(p + rng.normal(0, 0.2, p.shape).astype(np.float32))
    grid = pts[5].reshape(BOARD.rows, BOARD.cols, 1, 2).copy()
    grid[[0, 2]] = grid[[2, 0]]                       # rows 0 and 2 swapped: not a flat board
    pts[5] = grid.reshape(-1, 1, 2)
    res, keep, rejected = calibrate_robust(pts, BOARD, (640, 480))
    assert 5 in rejected
    assert res.rms < 0.5
