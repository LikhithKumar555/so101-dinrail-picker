"""Printed 1:1 test sheet for the real-camera perception test (step 4.2b).

The sheet carries a small checkerboard and 12 terminal-block top faces (4 per width) at known
positions. One photo of the sheet gives everything needed for a real-camera test:

    checkerboard  --(find corners, solvePnP with the REAL calibrated intrinsics)-->  T_cam_board
    detector      --(oriented boxes)-->  pixels  --(rays, plane z = 0 of the sheet)-->  block poses
    ground truth  =  the printed layout, in the same board frame

Frames: the board frame has its origin at inner corner (0, 0), x along the long side of the
board (= page right), y = page down, z into the paper. Page coordinates are millimetres from
the top-left corner of the A4 sheet (landscape), y measured downwards, so board and page axes
are parallel and the block yaw is the same number in both.

A uniform print scale error (e.g. "fit to page" at 97 %) scales the board AND the blocks, so the
measured positions in "board millimetres" stay correct - only the absolute size changes.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from so101_perception.calibration import Board, _pixel_rays, find_corners

A4_W_MM, A4_H_MM = 297.0, 210.0
SHEET_BOARD = Board(cols=7, rows=4, square=0.015)    # 8 x 5 squares = 120 x 75 mm, asymmetric
BOARD_ORIGIN_MM = np.array([35.0, 130.0])            # page position of inner corner (0, 0)
BLOCK_LENGTH_MM = 45.0
WIDTHS_MM = {0: 5.2, 1: 8.2, 2: 12.2}                # class id -> nominal width
PAGE_MARGIN_MM = 8.0                                 # printers cannot print to the paper edge

# Block colours like the synthetic training data: light grey, blue (N), darker grey
COLOURS = {"grey": (0.80, 0.80, 0.76), "blue": (0.15, 0.35, 0.80), "dark": (0.55, 0.55, 0.55)}
COLOURS_GRAYSCALE = {"grey": (0.80, 0.80, 0.80), "blue": (0.62, 0.62, 0.62), "dark": (0.55, 0.55, 0.55)}
BACKGROUNDS = {"dark": 0.30, "mid": 0.45}             # table tone behind the blocks (grey level)


@dataclass(frozen=True)
class SheetBlock:
    cls: int                  # 0 / 1 / 2 = 5.2 / 8.2 / 12.2 mm
    centre_mm: tuple          # page position of the top-face centre (x right, y down)
    yaw_deg: float            # long axis angle from page x towards page y (down)
    colour: str

    @property
    def width_mm(self) -> float:
        return WIDTHS_MM[self.cls]


# Hand-made layout: centres >= 56 mm apart (blocks cannot touch for any yaw), all on the dark
# background, none inside the white quiet zone around the checkerboard.
LAYOUT = (
    SheetBlock(0, (40.0, 35.0), 0.0, "grey"),
    SheetBlock(1, (96.0, 35.0), 30.0, "grey"),
    SheetBlock(2, (152.0, 35.0), -45.0, "grey"),
    SheetBlock(0, (208.0, 35.0), 90.0, "blue"),
    SheetBlock(1, (264.0, 35.0), 15.0, "grey"),
    SheetBlock(2, (40.0, 91.0), 10.0, "dark"),
    SheetBlock(0, (96.0, 91.0), -20.0, "grey"),
    SheetBlock(1, (152.0, 91.0), 15.0, "blue"),
    SheetBlock(2, (208.0, 91.0), -60.0, "grey"),
    SheetBlock(0, (264.0, 91.0), 45.0, "grey"),
    SheetBlock(1, (190.0, 160.0), -35.0, "dark"),
    SheetBlock(2, (255.0, 160.0), 5.0, "grey"),
)


def board_squares_rect_mm():
    """Page rectangle (x0, y0, x1, y1) covered by the checkerboard squares."""
    s = SHEET_BOARD.square * 1000
    x0, y0 = BOARD_ORIGIN_MM - s
    return x0, y0, x0 + (SHEET_BOARD.cols + 1) * s, y0 + (SHEET_BOARD.rows + 1) * s


def quiet_zone_rect_mm(margin_mm: float = 10.0):
    """White area around the board: the corner detector needs a light border around it."""
    x0, y0, x1, y1 = board_squares_rect_mm()
    return x0 - margin_mm, y0 - margin_mm, x1 + margin_mm, y1 + margin_mm


def background_rects_mm():
    """Dark 'table' regions: everything printable except the white area of the board."""
    qx0, qy0, qx1, qy1 = quiet_zone_rect_mm()
    m = PAGE_MARGIN_MM
    return [(m, m, qx1, qy0),                         # band above the board
            (qx1, m, A4_W_MM - m, A4_H_MM - m)]       # everything right of the board


def block_corners_mm(b: SheetBlock) -> np.ndarray:
    """4 x 2 page coordinates of the top-face corners (same order convention as the labels)."""
    c, s = np.cos(np.radians(b.yaw_deg)), np.sin(np.radians(b.yaw_deg))
    dx, dy = BLOCK_LENGTH_MM / 2, b.width_mm / 2
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    return np.asarray(b.centre_mm) + local @ np.array([[c, s], [-s, c]])


def page_to_board(xy_mm) -> np.ndarray:
    """Page millimetres -> board-frame metres (z = 0)."""
    return (np.asarray(xy_mm, float) - BOARD_ORIGIN_MM) / 1000.0


def truth():
    """Ground truth in the board frame: list of (cls, centre_xy [m], yaw [rad], width [mm])."""
    return [(b.cls, page_to_board(b.centre_mm), np.radians(b.yaw_deg), b.width_mm) for b in LAYOUT]


# --------------------------------------------------------------------------------- drawing
def draw_sheet(ax, background: str = "dark", grayscale: bool = False, labels: bool = True):
    """Draw the sheet into a matplotlib axes in page millimetres (y down)."""
    from matplotlib.patches import Polygon, Rectangle

    ax.set_xlim(0, A4_W_MM)
    ax.set_ylim(A4_H_MM, 0)                                  # y down, like the page
    ax.set_aspect("equal")
    ax.axis("off")
    g = BACKGROUNDS[background]
    for x0, y0, x1, y1 in background_rects_mm():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=(g, g, g), lw=0))
    s = SHEET_BOARD.square * 1000
    for i in range(-1, SHEET_BOARD.cols):
        for j in range(-1, SHEET_BOARD.rows):
            if (i + j) % 2 == 0:       # the colouring that canonical_corner_order expects
                ax.add_patch(Rectangle(tuple(BOARD_ORIGIN_MM + [i * s, j * s]), s, s,
                                       color="black", lw=0))
    colours = COLOURS_GRAYSCALE if grayscale else COLOURS
    for b in LAYOUT:
        rgb = np.array(colours[b.colour])
        ax.add_patch(Polygon(block_corners_mm(b), closed=True, facecolor=rgb,
                             edgecolor=rgb * 0.6, lw=0.6))
    if labels:
        # 100 mm scale bar in the white strip below the board: measure it after printing
        _, _, _, by1 = board_squares_rect_mm()
        x0, y = BOARD_ORIGIN_MM[0] - s, by1 + 7
        ax.plot([x0, x0 + 100], [y, y], color="black", lw=1.0)
        for x in (x0, x0 + 100):
            ax.plot([x, x], [y - 1.5, y + 1.5], color="black", lw=1.0)
        ax.text(x0 + 103, y, "100 mm", fontsize=5, va="center")
        ax.text(PAGE_MARGIN_MM + 2, PAGE_MARGIN_MM + 4,
                "SO-101 block test sheet - print at 100 % (actual size)",
                fontsize=5, color="white", va="center")


def save_sheet_pdf(path, background: str = "dark", grayscale: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(A4_W_MM / 25.4, A4_H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_sheet(ax, background, grayscale)
    fig.savefig(path)
    plt.close(fig)


def rasterize_sheet(px_per_mm: float = 8.0, background: str = "dark", grayscale: bool = False):
    """The sheet as an RGB image (for synthetic views). Returns (image, px_per_mm_x, px_per_mm_y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(A4_W_MM / 25.4, A4_H_MM / 25.4), dpi=px_per_mm * 25.4)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_sheet(ax, background, grayscale, labels=False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    h, w = img.shape[:2]
    return img, w / A4_W_MM, h / A4_H_MM


# --------------------------------------------------------------------------------- geometry
def pose_from_corners(corners: np.ndarray, K, dist):
    """Checkerboard corners (N,1,2) in canonical order -> (T_cam_board, corners, RMS px).

    A flat target has two PnP solutions that look alike (a tilted plane and its mirror tilt);
    IPPE returns the better one, Levenberg-Marquardt then refines it.
    canonical_corner_order fixes a 180 deg rotation of the corner order, but not a mirrored
    order; a mirrored order makes the board's z axis point AT the camera, which is impossible
    for a sheet seen from the front - so it is detected and corrected here.
    """
    obj = SHEET_BOARD.object_points()
    K, dist = np.asarray(K, float), np.asarray(dist, float)

    def solve(c):
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE)
        rvec, tvec = cv2.solvePnPRefineLM(obj, c, K, dist, rvec, tvec)
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), rvec

    c = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    R, t, rvec = solve(c)
    if R[2, 2] < 0:                       # board z axis (into the paper) points to the camera
        c = c.reshape(SHEET_BOARD.rows, SHEET_BOARD.cols, 1, 2)[:, ::-1].reshape(-1, 1, 2).copy()
        R, t, rvec = solve(c)
    proj, _ = cv2.projectPoints(obj, rvec, t, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c.reshape(-1, 2)) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, c, rms


def sheet_pose(gray: np.ndarray, K, dist):
    """Find the sheet's checkerboard in a grey image -> (T_cam_board, corners, RMS) or None."""
    corners = find_corners(gray, SHEET_BOARD)
    if corners is None:
        return None
    return pose_from_corners(corners, K, dist)


def board_px_per_mm(corners: np.ndarray) -> float:
    """Image scale at the board: mean distance between neighbouring corners / square size."""
    c = np.asarray(corners).reshape(SHEET_BOARD.rows, SHEET_BOARD.cols, 2)
    d = np.concatenate([np.linalg.norm(np.diff(c, axis=1), axis=2).ravel(),
                        np.linalg.norm(np.diff(c, axis=0), axis=2).ravel()])
    return float(d.mean() / (SHEET_BOARD.square * 1000))


# --------------------------------------------------------------------------------- synthetic views
def look_at_sheet(distance_m: float, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0,
                  roll_deg: float = 0.0, target_mm=(148.5, 105.0)) -> np.ndarray:
    """T_cam_board of a camera in front of the sheet, looking at `target_mm` (page position)."""
    def rot(axis, deg):
        return cv2.Rodrigues(np.radians(deg) * np.asarray(axis, float))[0]

    # camera axes in the board frame; identity = camera x right, y down, looking into the paper
    R_bc = rot([0, 0, 1], roll_deg) @ rot([1, 0, 0], tilt_x_deg) @ rot([0, 1, 0], tilt_y_deg)
    target = np.append(page_to_board(target_mm), 0.0)
    C = target - distance_m * R_bc[:, 2]
    T_bc = np.eye(4)
    T_bc[:3, :3], T_bc[:3, 3] = R_bc, C
    return np.linalg.inv(T_bc)


def render_sheet_view(sheet_rgb, ppmm_x, ppmm_y, K, dist, T_cam_board, size=(1280, 720),
                      desk=(0.42, 0.36, 0.30), noise=2.0, blur=0.6, rng=None) -> np.ndarray:
    """What a camera with (K, dist) at T_cam_board sees of the printed sheet (RGB uint8).

    Every pixel is back-projected through the distorted camera onto the paper plane (z = 0)
    and coloured from the rasterised sheet - the exact inverse of the camera model.
    """
    w, h = size
    xy = _pixel_rays(tuple(np.ravel(K)), tuple(np.ravel(dist)), tuple(size))
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    H = np.column_stack([R[:, 0], R[:, 1], t])                 # (x_b, y_b, 1) -> camera ray
    p = np.linalg.solve(H, np.column_stack([xy, np.ones(len(xy))]).T).T
    in_front = p[:, 2] * np.sign(np.linalg.det(H)) > 0
    bx, by = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]              # board-plane coordinates [m]
    page = BOARD_ORIGIN_MM + np.column_stack([bx, by]) * 1000
    map_x = (page[:, 0] * ppmm_x - 0.5).reshape(h, w).astype(np.float32)
    map_y = (page[:, 1] * ppmm_y - 0.5).reshape(h, w).astype(np.float32)
    map_x[~in_front.reshape(h, w)] = -1e4
    # low-pass the high-resolution sheet first, so resampling does not alias
    sigma = max(0.5 * ppmm_x / (K[0][0] / max(abs(t[2]) * 1000, 1.0)), 0.5)
    src = cv2.GaussianBlur(sheet_rgb.astype(np.float32), (0, 0), sigma)
    img = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=tuple(float(v) * 255 for v in desk))
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
