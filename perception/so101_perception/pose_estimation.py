"""From a detection in the image to a 3D block pose in the robot base frame.

A single camera cannot measure depth, but we know WHERE the blocks are: standing on the
table (top face at z = table + 45 mm) or seated on the rail (7.5 mm higher). So every pixel
is turned into a 3D ray and intersected with the plane of the block's top face:

    pixel --(undistort, K^-1)--> ray in camera frame --(T_base_cam)--> ray in base frame
          --(intersect with plane z = z_top)--> 3D point

All four corners of the oriented box are back-projected this way. Their mean is the top-face
centre, the long edge gives the yaw, and the edge lengths give a size check in millimetres.
"""
from dataclasses import dataclass

import numpy as np

from so101_perception.calibration import undistort_normalized
from so101_perception.synthetic import CLASS_NAMES, camera_intrinsics
from so101_sim.scene import BLOCK_DEPTH, BLOCK_HEIGHT, RAIL_TOP_Z, TABLE_Z

# Rail footprint in the base frame (from the scene): blocks detected here sit on the rail
RAIL_X_RANGE = (0.24 - 0.025, 0.24 + 0.025)
RAIL_Y_RANGE = (-0.13, 0.03)
BLOCK_NAMES = ["block_w5", "block_w8", "block_w12"]   # class id -> block type in the scene
# Physical prior: every terminal block's top face is 45 mm long and 5.2 - 12.2 mm wide.
# A detection whose measured size is far off cannot be a block (e.g. a checkerboard column).
LENGTH_TOL_MM = 7.0
WIDTH_RANGE_MM = (3.5, 14.5)


@dataclass
class Camera:
    K: np.ndarray            # 3x3 intrinsics
    dist: np.ndarray         # (k1, k2, p1, p2, k3)
    T_base_cam: np.ndarray   # 4x4 pose of the camera in the robot base frame (OpenCV axes:
                             # x right, y down, z forward) - from hand-eye calibration (step 4.4)

    def pixel_rays(self, pixels: np.ndarray):
        """Pixels (N,2) -> ray origin (3,) and unit directions (N,3) in the base frame."""
        xy = undistort_normalized(np.asarray(pixels, float), self.K, self.dist).reshape(-1, 2)
        d_cam = np.column_stack([xy, np.ones(len(xy))])
        d = d_cam @ self.T_base_cam[:3, :3].T
        return self.T_base_cam[:3, 3], d / np.linalg.norm(d, axis=1, keepdims=True)

    def backproject_to_plane(self, pixels: np.ndarray, z: float) -> np.ndarray:
        """Intersect the pixel rays with the horizontal plane at height z. Returns (N,3)."""
        o, d = self.pixel_rays(pixels)
        s = (z - o[2]) / d[:, 2]
        if np.any(s <= 0):
            raise ValueError("plane is behind the camera")
        return o + s[:, None] * d

    def project(self, points: np.ndarray) -> np.ndarray:
        """Base-frame points (N,3) -> pixels (N,2), with lens distortion (for tests/overlays)."""
        import cv2
        T_cam_base = np.linalg.inv(self.T_base_cam)
        rvec, _ = cv2.Rodrigues(T_cam_base[:3, :3])
        px, _ = cv2.projectPoints(np.asarray(points, float), rvec, T_cam_base[:3, 3], self.K, self.dist)
        return px.reshape(-1, 2)


def camera_from_mujoco(model, data, cam_name: str, width: int, height: int) -> Camera:
    """The simulated camera as a Camera (its 'true' calibration, known exactly in simulation)."""
    cid = model.camera(cam_name).id
    R_mj = data.cam_xmat[cid].reshape(3, 3)            # MuJoCo camera: x right, y up, looks along -z
    R = R_mj @ np.diag([1.0, -1.0, -1.0])               # -> OpenCV camera: y down, looks along +z
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, data.cam_xpos[cid]
    return Camera(camera_intrinsics(model, cid, width, height), np.zeros(5), T)


@dataclass
class BlockPose:
    cls: int
    name: str                # block type, e.g. "block_w8"
    confidence: float
    position: np.ndarray     # block centre (body origin) in the base frame [m]
    yaw: float               # rotation of the block's long axis about z [rad], in (-pi/2, pi/2]
    on_rail: bool
    length_mm: float         # measured top-face length (should be ~45 mm)
    width_mm: float          # measured top-face width (5.2 / 8.2 / 12.2 mm)

    def T_base_block(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = self.position
        return T


def _in_rail_region(p) -> bool:
    return RAIL_X_RANGE[0] <= p[0] <= RAIL_X_RANGE[1] and RAIL_Y_RANGE[0] <= p[1] <= RAIL_Y_RANGE[1]


def estimate_block_pose(camera: Camera, corners_px: np.ndarray, cls: int,
                        confidence: float = 1.0, top_z: float | None = None) -> BlockPose:
    """Oriented box of a block's top face (4 corner pixels) -> 3D block pose in the base frame.

    top_z: height of the top-face plane in the base frame. None = the robot cell (table or
    rail, decided from where the block lands). A number = a known flat surface, e.g. 0.0 for
    the printed test sheet (step 4.2b), whose frame is the base frame there.
    """
    if top_z is not None:
        pts = camera.backproject_to_plane(corners_px, top_z)
        on_rail = False
    else:
        # first assume the block stands on the table; if it lands on the rail footprint, redo it
        pts = camera.backproject_to_plane(corners_px, TABLE_Z + BLOCK_HEIGHT)
        on_rail = _in_rail_region(pts.mean(axis=0))
        if on_rail:
            pts = camera.backproject_to_plane(corners_px, RAIL_TOP_Z + BLOCK_HEIGHT)

    centre_top = pts.mean(axis=0)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    yaw = np.arctan2(long_e[1], long_e[0])
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2        # a box rotated by 180 deg is the same box
    position = centre_top - [0.0, 0.0, BLOCK_HEIGHT / 2]
    return BlockPose(cls, BLOCK_NAMES[cls], float(confidence), position, float(yaw), on_rail,
                     float(np.linalg.norm(long_e) * 1000), float(np.linalg.norm(short_e) * 1000))


class BlockLocalizer:
    """Detector + geometry: image in, list of 3D block poses out."""

    def __init__(self, weights, camera: Camera, imgsz: int = 960, conf: float = 0.5,
                 top_z: float | None = None, size_check: bool = True):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.camera, self.imgsz, self.conf, self.top_z = camera, imgsz, conf, top_z
        self.size_check = size_check
        self.rejected = []          # detections dropped by the size check in the last call

    def __call__(self, image_rgb: np.ndarray) -> list:
        # agnostic NMS: one box per object, even if two width classes both fire on it
        r = self.model.predict(image_rgb[..., ::-1], imgsz=self.imgsz, conf=self.conf,
                               agnostic_nms=True, verbose=False)[0]
        poses = []
        if r.obb is not None and len(r.obb):
            for cls, conf, pts in zip(r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy(),
                                      r.obb.xyxyxyxy.cpu().numpy()):
                poses.append(estimate_block_pose(self.camera, pts.reshape(4, 2), int(cls), float(conf),
                                                top_z=self.top_z))
        self.rejected = [p for p in poses if self.size_check and not is_plausible_block(p)]
        return [p for p in poses if not (self.size_check and not is_plausible_block(p))]


def is_plausible_block(p: BlockPose) -> bool:
    """Size check in millimetres - possible only because the 3D pose is metric."""
    return (abs(p.length_mm - BLOCK_DEPTH * 1000) <= LENGTH_TOL_MM
            and WIDTH_RANGE_MM[0] <= p.width_mm <= WIDTH_RANGE_MM[1])


def describe(p: BlockPose) -> str:
    return (f"{CLASS_NAMES[p.cls]:<11} conf {p.confidence:.2f}  "
            f"xyz = ({p.position[0]*1000:6.1f}, {p.position[1]*1000:6.1f}, {p.position[2]*1000:6.1f}) mm  "
            f"yaw {np.degrees(p.yaw):6.1f} deg  {'rail ' if p.on_rail else 'table'}  "
            f"size {p.length_mm:4.1f} x {p.width_mm:4.1f} mm")
