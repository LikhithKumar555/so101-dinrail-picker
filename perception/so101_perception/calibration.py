"""Camera intrinsic calibration with a checkerboard (Zhang's method, via OpenCV).

Camera model (pinhole + Brown-Conrady distortion), for a 3D point X_c = (X, Y, Z) in the
camera frame:
    x = X/Z, y = Y/Z                                   normalised image coordinates
    r^2 = x^2 + y^2
    x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)   (radial + tangential)
    y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
    u = fx x_d + cx,  v = fy y_d + cy                  pixels   ->  K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
Calibration finds K and (k1, k2, p1, p2, k3) - plus one board pose per image - that minimise
the reprojection error: the pixel distance between detected corners and projected 3D corners.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Board:
    cols: int = 9          # inner corners along the long side
    rows: int = 6          # inner corners along the short side
    square: float = 0.025  # edge length of one square in metres (MEASURE your print!)

    @property
    def pattern_size(self):
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """(N, 3) corner coordinates in the board frame (z = 0 plane), metres."""
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        pts = np.zeros((self.cols * self.rows, 3), np.float32)
        pts[:, :2] = grid * self.square
        return pts


def undistort_normalized(pixels: np.ndarray, K, dist, iterations: int = 50) -> np.ndarray:
    """Pixels (N,1,2) -> undistorted normalised coordinates (N,1,2), iterating to convergence.

    (OpenCV 4 has a separate undistortPointsIter; OpenCV 5 merged it into undistortPoints.)
    """
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, 1e-10)
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(pixels, K, dist, None, None, crit)
    return cv2.undistortPoints(pixels, K, dist, R=None, P=None, criteria=crit)


def canonical_corner_order(gray: np.ndarray, corners: np.ndarray, board: Board) -> np.ndarray:
    """Make the corner order unique: OpenCV may return it rotated by 180 deg between images.

    For intrinsic calibration this does not matter, but hand-eye calibration needs the SAME
    board frame in every image. In our boards the square between corners (i, j) and
    (i+1, j+1) is black when i + j is even. With cols + rows odd, a 180-degree rotation turns
    this colouring into its negative - so we sample every inner square and reverse the order
    if the image looks like the negative.
    """
    if (board.cols + board.rows) % 2 == 0:
        return corners                          # symmetric board: order cannot be fixed
    c = corners.reshape(-1, 2).astype(np.float64)
    grid = np.mgrid[0:board.cols, 0:board.rows].T.reshape(-1, 2).astype(np.float64)
    H, _ = cv2.findHomography(grid, c, 0)
    ij = np.mgrid[0:board.cols - 1, 0:board.rows - 1].T.reshape(-1, 2)
    centres = cv2.perspectiveTransform((ij + 0.5).reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    h, w = gray.shape[:2]
    xs = np.clip(np.round(centres[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(centres[:, 1]).astype(int), 0, h - 1)
    values = gray[ys, xs].astype(np.float64)
    expected_black = (ij.sum(axis=1) % 2) == 0
    score = values[~expected_black].mean() - values[expected_black].mean()   # > 0: as expected
    return corners if score >= 0 else corners[::-1].copy()


def find_corners(gray: np.ndarray, board: Board, border_px: int = 10):
    """Sub-pixel checkerboard corners as (N, 1, 2) float32, or None if the board is not found.

    Detections with corners closer than `border_px` to the image edge are rejected: a board
    that is cut off by the image border is a common source of wrong detections.
    """
    h, w = gray.shape[:2]

    def near_border(c):
        c = c.reshape(-1, 2)
        return (c.min() < border_px or (c[:, 0] > w - 1 - border_px).any()
                or (c[:, 1] > h - 1 - border_px).any())

    ok, corners = cv2.findChessboardCornersSB(
        gray, board.pattern_size,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok:
        if near_border(corners):
            return None
        return canonical_corner_order(gray, corners.astype(np.float32), board)
    # Fallback: classic detector + sub-pixel refinement
    ok, corners = cv2.findChessboardCorners(
        gray, board.pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return None if near_border(corners) else canonical_corner_order(gray, corners, board)


@dataclass
class CalibrationResult:
    K: np.ndarray
    dist: np.ndarray                 # (k1, k2, p1, p2, k3)
    image_size: tuple                # (width, height)
    rms: float                       # overall RMS reprojection error [px]
    per_view_error: np.ndarray       # RMS per image [px]
    rvecs: list = field(repr=False, default_factory=list)
    tvecs: list = field(repr=False, default_factory=list)


# Lens distortion models: how many radial terms are estimated (k1 r^2 + k2 r^4 + k3 r^6).
# More terms fit better INSIDE the area covered by the board, but a high-order polynomial can
# explode OUTSIDE it (extrapolation) - so a small board or one far from the image corners needs
# a simpler model. Phones also correct most lens distortion in software already.
DIST_MODELS = {"full": 0, "k1k2": cv2.CALIB_FIX_K3, "k1": cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3}


def calibrate(image_points: list, board: Board, image_size: tuple, flags: int = 0) -> CalibrationResult:
    obj = [board.object_points()] * len(image_points)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, image_points, image_size, None, None,
                                                     flags=flags)
    per_view = []
    for o, ip, r, t in zip(obj, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = proj.reshape(-1, 2) - np.asarray(ip).reshape(-1, 2)   # shapes differ between versions
        per_view.append(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    return CalibrationResult(K, dist.ravel(), tuple(image_size), float(rms),
                             np.array(per_view), list(rvecs), list(tvecs))


def distortion_check(K, dist, image_size, image_points=()):
    """Is the distortion model physically sensible over the WHOLE image?

    Returns a dict:
      corner_shift_px  - how far the model moves the image corners (distorted vs pinhole)
      monotonic        - the distorted radius must grow with the true radius up to the image
                         corners; if not, the polynomial folds back (two directions map to the
                         same pixel) - a sure sign of extrapolation outside the calibrated area
      coverage         - how far the calibration corners reach towards the image corners
                         (1.0 = all the way)
    """
    K, d = np.asarray(K, float), np.asarray(dist, float).ravel()
    w, h = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    # normalised pinhole coordinates of the image corners (approximately the true directions)
    xy = np.column_stack([(corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy])
    obj = np.column_stack([xy, np.ones(4)])
    proj, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, d)
    shift = float(np.max(np.linalg.norm(proj.reshape(-1, 2) - corners, axis=1)))
    r_max = float(np.max(np.linalg.norm(xy, axis=1)))
    k1, k2, k3 = d[0], d[1], d[4] if len(d) > 4 else 0.0
    r = np.linspace(0, r_max, 400)
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    coverage = float("nan")
    if len(image_points):
        pts = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in image_points])
        r_pts = np.linalg.norm(np.column_stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy]), axis=1)
        coverage = float(r_pts.max() / r_max)
    return {"corner_shift_px": shift, "monotonic": bool(np.all(np.diff(rd) > 0)), "coverage": coverage}


def homography_rms(corners: np.ndarray, board: Board) -> float:
    """How well the detected corners fit a plane-to-image homography [px].

    A flat board seen by an undistorted pinhole camera maps EXACTLY by a homography, so for a
    correct detection the residual is small (only lens distortion + noise, a few px at most).
    Mis-ordered or wrong detections give tens of pixels. Cheap, needs no calibration.
    """
    obj = board.object_points()[:, :2].astype(np.float64)
    img = np.asarray(corners, np.float64).reshape(-1, 2)
    H, _ = cv2.findHomography(obj, img, 0)
    proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))


def calibrate_robust(image_points: list, board: Board, image_size: tuple,
                     outlier_factor: float = 3.0, outlier_min_px: float = 1.0,
                     catastrophic_rms: float = 2.0, min_views: int = 8, log=None, names=None,
                     flags: int = 0):
    """Calibrate and remove bad views (blurred photos, wrong detections).

    0. Pre-filter: views whose corners do not fit a plane-to-image homography are wrong
       detections - removed before calibrating (fast, needs no calibration).
    1. Catastrophic views: one wrong detection can ruin the joint optimisation for ALL views.
       While the overall RMS is huge, drop the worst view: first try the view with the largest
       own error (fast); if that does not help, test removing each view in turn (slower).
    2. Outliers: drop views whose own error is > outlier_factor x median (and > outlier_min_px).
    Returns (result, kept_indices, rejected_indices).
    """
    log = log or (lambda *a: None)
    names = names or [f"view {i}" for i in range(len(image_points))]
    keep = list(range(len(image_points)))
    rejected = []

    h_rms = np.array([homography_rms(c, board) for c in image_points])
    h_limit = max(4.0, 4.0 * float(np.median(h_rms)))
    for i in [k for k in keep if h_rms[k] > h_limit]:
        if len(keep) > min_views:
            keep.remove(i)
            rejected.append(i)
            log(f"  pre-filter: {names[i]} does not fit a flat board "
                f"(homography RMS {h_rms[i]:.1f} px > {h_limit:.1f} px)")

    def run(idx):
        return calibrate([image_points[i] for i in idx], board, image_size, flags)

    res = run(keep)
    log(f"  initial calibration with {len(keep)} images: RMS {res.rms:.3f} px")
    while res.rms > catastrophic_rms and len(keep) > min_views:
        cand = keep[int(np.argmax(res.per_view_error))]
        if run([k for k in keep if k != cand]).rms > 0.7 * res.rms:   # did not help: search all
            log(f"  RMS still high - testing the removal of every image ...")
            cand = min((run([k for k in keep if k != i]).rms, i) for i in keep)[1]
        keep.remove(cand)
        rejected.append(cand)
        res = run(keep)
        log(f"  removed {names[cand]} -> RMS {res.rms:.3f} px")
    while len(keep) > min_views:
        j = int(np.argmax(res.per_view_error))
        limit = max(outlier_factor * float(np.median(res.per_view_error)), outlier_min_px)
        if res.per_view_error[j] <= limit:
            break
        name = names[keep[j]]
        rejected.append(keep.pop(j))
        res = run(keep)
        log(f"  removed outlier {name} -> RMS {res.rms:.3f} px")
    return res, keep, rejected


def save_camera_info_yaml(path: Path, res: CalibrationResult, camera_name: str, board: Board,
                          n_images: int):
    """Write in the ROS camera_info YAML format, so ROS camera drivers can load it directly."""
    w, h = res.image_size
    P = np.hstack([res.K, np.zeros((3, 1))])
    data = {
        "image_width": int(w), "image_height": int(h), "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in res.K.ravel()]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [float(v) for v in res.dist]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in np.eye(3).ravel()]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(v) for v in P.ravel()]},
        # extra fields (ignored by ROS) documenting how this was obtained
        "calibration": {"rms_reprojection_error_px": round(res.rms, 4), "n_images": n_images,
                        "board_inner_corners": [board.cols, board.rows],
                        "square_size_m": board.square},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def load_camera_info_yaml(path: Path):
    d = yaml.safe_load(Path(path).read_text())
    K = np.array(d["camera_matrix"]["data"]).reshape(3, 3)
    dist = np.array(d["distortion_coefficients"]["data"])
    return K, dist, (d["image_width"], d["image_height"])


@lru_cache(maxsize=4)
def _pixel_rays(K: tuple, dist: tuple, size: tuple) -> np.ndarray:
    """Normalised undistorted coordinates of every pixel (cached: the same for every view)."""
    w, h = size
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float64)
    return undistort_normalized(uv, np.array(K).reshape(3, 3), np.array(dist)).reshape(-1, 2)


def render_board_image(board: Board, K, dist, R, t, size=(640, 480), margin_squares=1.0,
                       noise=2.0, rng=None) -> np.ndarray:
    """Synthetic grey image of the board seen by a camera (for testing the calibration).

    Every pixel is back-projected through the (distorted) camera onto the board plane
    and coloured by the checker pattern - an exact inverse of the camera model.
    """
    w, h = size
    xy = _pixel_rays(tuple(np.ravel(K)), tuple(np.ravel(dist)), tuple(size))
    H = np.column_stack([R[:, 0], R[:, 1], t])                     # plane z=0 -> camera
    pts = np.linalg.solve(H, np.column_stack([xy, np.ones(len(xy))]).T).T
    bx, by = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]          # board-plane coordinates [m]
    # corners at (i*s, j*s); squares extend one square beyond the outer corners
    i = np.floor(bx / board.square + 1.0)
    j = np.floor(by / board.square + 1.0)
    inside = (i >= 0) & (i <= board.cols) & (j >= 0) & (j <= board.rows)
    img = np.full(len(bx), 235.0)                                  # white paper
    img[inside & ((i + j) % 2 == 0)] = 25.0                        # black squares
    behind = pts[:, 2] * np.sign(np.linalg.det(H)) < 0
    img[behind] = 128.0
    img = img.reshape(h, w)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    if noise:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
