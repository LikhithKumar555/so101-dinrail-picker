"""End-to-end check of perception in 3D: image -> detector -> 3D block poses vs. ground truth.

  python perception/scripts/evaluate_pose_estimation.py --scenes 100

New random scenes are rendered with the camera at its calibrated pose (the hand-eye result).
For every true block it reports whether it was found with the right width, and the 3D position
and yaw errors in the robot base frame - the numbers that decide whether a grasp succeeds.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from so101_perception.pose_estimation import BlockLocalizer, camera_from_mujoco, describe
from so101_perception.synthetic import CLASS_NAMES, SyntheticRenderer, block_class

ROOT = Path(__file__).resolve().parents[2]
POS_TOL_MM, YAW_TOL_DEG = 3.0, 5.0       # what the gripper can tolerate (see step 2.5)


def wrap_half_pi(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def main(a):
    r = SyntheticRenderer(960, 720, randomize_camera=False)
    rng = np.random.default_rng(a.seed)
    r.sample(rng)                                            # initialises the camera state
    cam = camera_from_mujoco(r.model, r.data, "overhead", 960, 720)
    loc = BlockLocalizer(ROOT / a.weights, cam, imgsz=a.imgsz, conf=a.conf)

    n_true = n_found = n_cls_ok = n_graspable = n_fp = 0
    pos_err, z_err, yaw_err = [], [], []
    for k in range(a.scenes):
        s = r.sample(rng)
        poses = loc(s.image)
        truth = [b for b in r.blocks if r.data.body(b).xpos[0] < 2.0]
        # only blocks that are actually visible count (the labels list exactly those)
        visible = []
        for b in truth:
            px = cam.project(r.data.body(b).xpos[None])[0]
            if any(np.linalg.norm(c.mean(axis=0) - px) < 25 for _, c in s.labels):
                visible.append(b)
        used = set()
        for b in visible:
            n_true += 1
            p_true = r.data.body(b).xpos
            R = r.data.body(b).xmat.reshape(3, 3)
            yaw_true = np.arctan2(R[1, 0], R[0, 0])
            j = min(range(len(poses)), key=lambda i: np.linalg.norm(poses[i].position - p_true),
                    default=None)
            if j is None or j in used or np.linalg.norm(poses[j].position - p_true) > 0.015:
                continue
            used.add(j)
            p = poses[j]
            n_found += 1
            cls_ok = p.cls == block_class(b)
            n_cls_ok += cls_ok
            e_xy = np.linalg.norm(p.position[:2] - p_true[:2]) * 1000
            e_yaw = abs(np.degrees(wrap_half_pi(p.yaw - yaw_true)))
            pos_err.append(e_xy)
            z_err.append(abs(p.position[2] - p_true[2]) * 1000)
            yaw_err.append(e_yaw)
            n_graspable += cls_ok and e_xy <= POS_TOL_MM and e_yaw <= YAW_TOL_DEG
        n_fp += len(poses) - len(used)
        if k == 0:
            img = cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR)
            for i, p in enumerate(poses):
                c = cam.project(p.position[None] + [0, 0, 0.0225])[0].astype(int)
                cv2.circle(img, tuple(c), 4, (0, 0, 255), -1)
                cv2.putText(img, f"{CLASS_NAMES[p.cls][6:]} ({p.position[0]*1000:.0f},{p.position[1]*1000:.0f}) "
                            f"{np.degrees(p.yaw):.0f}deg", (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                print("  " + describe(p))
            cv2.imwrite(str(ROOT / "media" / "pose_estimation_example.png"), img)

    pe, ze, ye = map(np.array, (pos_err, z_err, yaw_err))
    print(f"\n{a.scenes} new scenes, {n_true} visible blocks")
    print(f"found                      : {n_found / n_true:6.1%}   ({n_fp} false detections)")
    print(f"width class correct        : {n_cls_ok / max(n_found, 1):6.1%} of found")
    print(f"xy position error med/p95/max: {np.median(pe):.2f} / {np.percentile(pe, 95):.2f} / {pe.max():.2f} mm")
    print(f"height error      median   : {np.median(ze):.2f} mm (wrong plane would be 7.5 mm)")
    print(f"yaw error     med/p95/max  : {np.median(ye):.2f} / {np.percentile(ye, 95):.2f} / {ye.max():.2f} deg")
    print(f"GRASPABLE (found, right width, <= {POS_TOL_MM:g} mm, <= {YAW_TOL_DEG:g} deg): "
          f"{n_graspable / n_true:6.1%}")
    print("wrote media/pose_estimation_example.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="outputs/yolo/blocks_obb/weights/best.pt")
    p.add_argument("--scenes", type=int, default=100)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.5)
    main(p.parse_args())
