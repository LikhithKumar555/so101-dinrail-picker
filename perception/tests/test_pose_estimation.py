"""Geometry of pixel -> 3D (independent of the detector)."""
import numpy as np

from so101_perception.pose_estimation import Camera, camera_from_mujoco, estimate_block_pose
from so101_perception.synthetic import CLASSES, SyntheticRenderer, block_class
from so101_sim.scene import TABLE_Z


def look_down_camera(dist=np.zeros(5)):
    K = np.array([[1100.0, 0, 480], [0, 1100.0, 360], [0, 0, 1]])
    T = np.eye(4)
    # columns = camera axes in the base frame: image right = -y, image down = -x, view = -z
    T[:3, :3] = np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    T[:3, 3] = [0.22, 0.0, 0.50]
    return Camera(K, np.asarray(dist, float), T)


def test_backprojection_inverts_projection():
    rng = np.random.default_rng(0)
    for dist in (np.zeros(5), [0.08, -0.25, 0.001, -0.002, 0.1]):
        cam = look_down_camera(dist)
        z = TABLE_Z + 0.045
        pts = np.column_stack([rng.uniform(0.1, 0.34, 50), rng.uniform(-0.15, 0.15, 50), np.full(50, z)])
        back = cam.backproject_to_plane(cam.project(pts), z)
        np.testing.assert_allclose(back, pts, atol=1e-6)


def test_pose_from_ground_truth_labels_matches_simulation():
    """Use the exact labels (no detector): the recovered pose must match the true block pose."""
    r = SyntheticRenderer(960, 720, randomize_camera=False)
    rng = np.random.default_rng(3)
    n = 0
    for _ in range(4):
        s = r.sample(rng)
        cam = camera_from_mujoco(r.model, r.data, "overhead", 960, 720)
        for cls, corners in s.labels:
            est = estimate_block_pose(cam, corners, cls)
            # find the true block of that class closest to the estimate
            cands = [b for b in r.blocks if block_class(b) == cls and r.data.body(b).xpos[0] < 2]
            true = min(cands, key=lambda b: np.linalg.norm(r.data.body(b).xpos - est.position))
            assert np.linalg.norm(r.data.body(true).xpos - est.position) < 0.5e-3
            R = r.data.body(true).xmat.reshape(3, 3)
            true_yaw = np.arctan2(R[1, 0], R[0, 0])
            d = (est.yaw - true_yaw + np.pi / 2) % np.pi - np.pi / 2
            assert abs(np.degrees(d)) < 0.5
            assert abs(est.length_mm - 45.0) < 0.5
            n += 1
    assert n >= 10
