"""The automatic labels must line up with what is actually rendered."""
import cv2
import mujoco
import numpy as np

from so101_perception.synthetic import SyntheticRenderer, block_class


def test_labels_match_rendered_blocks():
    r = SyntheticRenderer(640, 480)
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(5):
        s = r.sample(rng)
        r.renderer.enable_segmentation_rendering()
        r.renderer.update_scene(r.data, camera=r.cam)
        seg = r.renderer.render()[..., 0]
        r.renderer.disable_segmentation_rendering()
        for cls, corners in s.labels:
            # the label's centre must lie on a rendered block of the same class
            cx, cy = corners.mean(axis=0).round().astype(int)
            geom = seg[cy, cx]
            name = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if not name.startswith("block_"):
                continue        # centre hidden by the arm (allowed: block is partly visible)
            assert block_class(name) == cls
            # top-face label area vs. rendered block pixels: same order of magnitude
            area = cv2.contourArea(corners.astype(np.float32))
            assert 0.3 * area < np.count_nonzero(seg == geom) < 4.0 * area
            n_checked += 1
    assert n_checked >= 10
