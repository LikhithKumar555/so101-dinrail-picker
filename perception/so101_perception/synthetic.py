"""Synthetic, automatically labelled camera images of the DIN-rail scene (domain randomisation).

Every sample randomises what a real camera would see differently from day to day:
block count / position / rotation / colour shade, blocks on the rail or on the table, the arm's
pose (occlusions), lighting, table colour, camera pose and field of view, sensor noise and blur.
Labels are exact (we know every pose), in Ultralytics YOLO-OBB format: the oriented box of
each block's TOP face, which is what the grasp needs (centre + rotation).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.kinematics import ARM_JOINTS
from so101_sim.scene import BLOCK_DEPTH, BLOCK_HEIGHT, PARK_Q, RAIL_TOP_Z, TABLE_Z, scene_kinematics

PERCEPTION_XML = Path(__file__).resolve().parents[2] / "sim" / "models" / "perception_scene.xml"
CLASSES = {"block_w5": 0, "block_w8": 1, "block_w12": 2}          # class id per width
CLASS_NAMES = ["block_5mm", "block_8mm", "block_12mm"]
WIDTHS = {0: 0.0052, 1: 0.0082, 2: 0.0122}
CAMERA = "overhead"
# Region of the table that the overhead camera sees well (robot base frame, metres)
TABLE_X, TABLE_Y = (0.08, 0.34), (-0.18, 0.18)
RAIL_X, RAIL_Y = 0.24, (-0.12, 0.02)


@dataclass
class Sample:
    image: np.ndarray        # RGB uint8
    labels: list             # (class_id, 4x2 corner pixels) per visible block


def block_class(body_name: str) -> int:
    return CLASSES["_".join(body_name.split("_")[:2])]


def camera_intrinsics(model, cam_id: int, width: int, height: int) -> np.ndarray:
    """Pinhole K of a MuJoCo camera: fovy is the vertical field of view."""
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


def project(points_w: np.ndarray, data, cam_id: int, K: np.ndarray) -> np.ndarray:
    """World points (N,3) -> pixels (N,2). MuJoCo cameras look along -z, with y up."""
    R = data.cam_xmat[cam_id].reshape(3, 3)           # columns: camera axes in world
    pc = (points_w - data.cam_xpos[cam_id]) @ R       # world -> camera frame
    x, y, z = pc[:, 0], -pc[:, 1], -pc[:, 2]          # -> OpenCV convention (x right, y down, z fwd)
    return np.column_stack([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])


def top_face_corners(data, body: str, width: float) -> np.ndarray:
    R = data.body(body).xmat.reshape(3, 3)
    c = data.body(body).xpos
    dx, dy, dz = BLOCK_DEPTH / 2, width / 2, BLOCK_HEIGHT / 2
    local = np.array([[dx, dy, dz], [-dx, dy, dz], [-dx, -dy, dz], [dx, -dy, dz]])
    return c + local @ R.T


class SyntheticRenderer:
    def __init__(self, width: int = 960, height: int = 720):
        self.model = mujoco.MjModel.from_xml_path(str(PERCEPTION_XML))
        self.data = mujoco.MjData(self.model)
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = self.model.camera(CAMERA).id
        m = self.model
        self.blocks = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("block_")]
        self.qadr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
        # nominal values to randomise around
        self.cam_pos0, self.cam_quat0 = m.cam_pos[self.cam].copy(), m.cam_quat[self.cam].copy()
        self.fovy0 = float(m.cam_fovy[self.cam])
        self.light_diffuse0 = m.light_diffuse.copy()
        self.light_pos0 = m.light_pos.copy()
        self.block_geoms = {b: m.geom(b).id for b in self.blocks}
        self.table_mat = m.material("table").id
        self.kin = scene_kinematics()
        self._arm_poses = None

    # ------------------------------------------------------------------ randomisation
    def _arm_pose_bank(self, rng, n=40):
        """Pre-compute some reachable arm poses (IK is slower than rendering)."""
        poses = [PARK_Q.copy()]
        while len(poses) < n:
            p = [rng.uniform(0.14, 0.30), rng.uniform(-0.15, 0.15), TABLE_Z + rng.uniform(0.06, 0.15)]
            res = solve_ik(self.kin, top_down_pose(p, rng.uniform(-np.pi, np.pi)), restarts=2)
            if res.success:
                poses.append(res.q)
        return poses

    def _place_blocks(self, rng):
        m, d = self.model, self.data
        n = rng.integers(1, len(self.blocks) + 1)
        chosen = list(rng.choice(self.blocks, size=n, replace=False))
        placed = []                                          # (x, y) of placed blocks
        for i, b in enumerate(self.blocks):
            adr = m.joint(b).qposadr[0]
            d.qpos[adr:adr + 7] = [3.0 + 0.1 * i, 3.0, 0.0, 1, 0, 0, 0]   # out of view
        for b in chosen:
            for _ in range(50):                              # rejection sampling: no overlaps
                on_rail = rng.random() < 0.3
                if on_rail:
                    x, y, z, yaw = RAIL_X, rng.uniform(*RAIL_Y), RAIL_TOP_Z + BLOCK_HEIGHT / 2, 0.0
                    yaw += np.radians(rng.normal(0, 1.5))
                else:
                    x, y = rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y)
                    if abs(x - RAIL_X) < 0.05 and RAIL_Y[0] - 0.04 < y < RAIL_Y[1] + 0.06:
                        continue                             # not on top of the rail
                    z, yaw = TABLE_Z + BLOCK_HEIGHT / 2, rng.uniform(-np.pi, np.pi)
                if all(np.hypot(x - px, y - py) > (0.022 if on_rail and pr else 0.05)
                       for px, py, pr in placed):
                    break
            else:
                continue
            placed.append((x, y, on_rail))
            adr = m.joint(b).qposadr[0]
            d.qpos[adr:adr + 7] = [x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            # colour: mostly light grey like real blocks, sometimes blue (N) or darker grey
            r = rng.random()
            base = (np.array([0.15, 0.35, 0.80]) if r < 0.12 else
                    np.array([0.45, 0.45, 0.45]) if r < 0.22 else np.array([0.80, 0.80, 0.76]))
            m.geom_rgba[self.block_geoms[b], :3] = np.clip(base + rng.normal(0, 0.04, 3), 0, 1)

    def _randomise_world(self, rng):
        m, d = self.model, self.data
        if self._arm_poses is None:
            self._arm_poses = self._arm_pose_bank(np.random.default_rng(1234))
        # half the images with the arm parked (how a snapshot for picking is taken),
        # half with the arm somewhere above the table (occlusions, shadows)
        d.qpos[self.qadr] = (PARK_Q if rng.random() < 0.5
                             else self._arm_poses[rng.integers(1, len(self._arm_poses))])
        # camera: position +-2 cm, height +-3 cm, small tilt, fov +-4 deg
        m.cam_pos[self.cam] = self.cam_pos0 + rng.uniform([-0.02, -0.02, -0.03], [0.02, 0.02, 0.03])
        tilt = rng.normal(0, np.radians(2), 3)
        qt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qt, tilt / (np.linalg.norm(tilt) + 1e-12), np.linalg.norm(tilt))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self.cam_quat0, qt)
        m.cam_quat[self.cam] = q
        m.cam_fovy[self.cam] = self.fovy0 + rng.uniform(-4, 4)
        # lighting and table colour
        m.light_diffuse[:] = self.light_diffuse0 * rng.uniform(0.4, 1.4)
        m.light_pos[:] = self.light_pos0 + rng.uniform(-0.3, 0.3, self.light_pos0.shape)
        m.vis.headlight.ambient[:] = rng.uniform(0.15, 0.45)
        m.vis.headlight.diffuse[:] = rng.uniform(0.3, 0.8)
        # table tones: wood, light grey, dark grey, greenish workbench mat, beige
        tone = np.array([[0.55, 0.45, 0.35], [0.80, 0.80, 0.80], [0.25, 0.25, 0.27],
                         [0.35, 0.45, 0.40], [0.75, 0.68, 0.55]])[rng.integers(5)]
        tone = np.clip(tone + rng.normal(0, 0.04, 3), 0.05, 0.95)
        # the table texture is (0.55, 0.45, 0.35); material rgba multiplies it
        m.mat_rgba[self.table_mat, :3] = tone / np.array([0.55, 0.45, 0.35])

    # ------------------------------------------------------------------ rendering + labels
    def sample(self, rng, min_visible: float = 0.35) -> Sample:
        self._place_blocks(rng)
        self._randomise_world(rng)
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.cam)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.cam)
        seg = self.renderer.render()[..., 0].copy()            # geom id per pixel
        self.renderer.disable_segmentation_rendering()

        K = camera_intrinsics(self.model, self.cam, self.w, self.h)
        labels = []
        for b in self.blocks:
            if self.data.body(b).xpos[0] > 2.0:
                continue
            cls = block_class(b)
            corners = project(top_face_corners(self.data, b, WIDTHS[cls]), self.data, self.cam, K)
            if (corners < 1).any() or (corners[:, 0] > self.w - 2).any() or (corners[:, 1] > self.h - 2).any():
                continue                                       # cut off by the image border
            area = cv2.contourArea(corners.astype(np.float32))
            visible = np.count_nonzero(seg == self.block_geoms[b])
            if area < 4 or visible < min_visible * area:
                continue                                       # mostly hidden (e.g. by the arm)
            labels.append((cls, corners))
        return Sample(augment_image(rgb, rng), labels)


def augment_image(rgb: np.ndarray, rng) -> np.ndarray:
    """Sensor effects: slight blur, noise, gain/offset (Ultralytics adds more during training)."""
    img = rgb.astype(np.float32)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    img += rng.normal(0, rng.uniform(1, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def yolo_obb_line(cls: int, corners: np.ndarray, w: int, h: int) -> str:
    n = corners / [w, h]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in n.ravel())


def draw_labels(rgb: np.ndarray, labels: list) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colours = [(0, 200, 255), (0, 255, 0), (255, 80, 0)]
    for cls, c in labels:
        cv2.polylines(out, [c.astype(np.int32)], True, colours[cls], 2)
        cv2.putText(out, CLASS_NAMES[cls], tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, colours[cls], 1)
    return out
