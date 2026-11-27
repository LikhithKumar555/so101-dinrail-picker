"""Evaluate the block detector with the numbers that matter for GRASPING.

  python perception/scripts/evaluate_detector.py --weights outputs/yolo/blocks_obb/weights/best.pt

Besides the usual detection metrics, it reports for every ground-truth block:
  * detected? (matched by centre distance)    -> recall, false positives -> precision
  * width class correct?                       -> 3x3 confusion matrix
  * centre error in mm and rotation error in deg (what the gripper needs)
Millimetres per pixel are taken per block from its known 45 mm length, so no calibration is needed.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from so101_perception.synthetic import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
BLOCK_LENGTH_MM = 45.0


def obb_params(corners: np.ndarray):
    """Centre (px), long-axis angle in [0, 180) deg, long side length (px) of a 4-corner box."""
    c = corners.mean(axis=0)
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    angle = np.degrees(np.arctan2(long_edge[1], long_edge[0])) % 180.0
    return c, angle, np.linalg.norm(long_edge)


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def main(a):
    model = YOLO(str(ROOT / a.weights))
    img_dir = ROOT / a.data / "images" / a.split
    lbl_dir = ROOT / a.data / "labels" / a.split
    files = sorted(img_dir.glob("*.jpg"))
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), int)   # last column = missed
    n_fp, pos_err, ang_err = 0, [], []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gts = []
        for line in (lbl_dir / f"{f.stem}.txt").read_text().splitlines():
            v = line.split()
            gts.append((int(v[0]), np.array(v[1:], float).reshape(4, 2) * [w, h]))
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        preds = []
        if r.obb is not None and len(r.obb):
            for cls, pts in zip(r.obb.cls.cpu().numpy(), r.obb.xyxyxyxy.cpu().numpy()):
                preds.append((int(cls), pts.reshape(4, 2)))
        used = set()
        for g_cls, g_pts in gts:
            g_c, g_ang, g_len = obb_params(g_pts)
            mm_per_px = BLOCK_LENGTH_MM / g_len
            best, best_d = None, a.match_mm / mm_per_px
            for j, (p_cls, p_pts) in enumerate(preds):
                if j in used:
                    continue
                d = np.linalg.norm(obb_params(p_pts)[0] - g_c)
                if d < best_d:
                    best, best_d = j, d
            if best is None:
                conf[g_cls, -1] += 1
                continue
            used.add(best)
            p_cls, p_pts = preds[best]
            conf[g_cls, p_cls] += 1
            p_c, p_ang, _ = obb_params(p_pts)
            pos_err.append(np.linalg.norm(p_c - g_c) * mm_per_px)
            ang_err.append(angle_diff(p_ang, g_ang))
        n_fp += len(preds) - len(used)

    n_gt = conf.sum()
    n_det = conf[:, :-1].sum()
    correct = np.trace(conf[:, :-1])
    print(f"\n{len(files)} images ({a.split} split), {n_gt} blocks")
    print(f"recall (found)            : {n_det / n_gt:6.1%}")
    print(f"precision                 : {n_det / max(n_det + n_fp, 1):6.1%}  ({n_fp} false positives)")
    print(f"width class correct       : {correct / max(n_det, 1):6.1%} of found blocks")
    if pos_err:
        print(f"centre error   median/p95 : {np.median(pos_err):.2f} / "
              f"{np.percentile(pos_err, 95):.2f} mm")
        print(f"rotation error median/p95 : {np.median(ang_err):.2f} / "
              f"{np.percentile(ang_err, 95):.2f} deg")
    print("\nconfusion matrix (rows = true, columns = predicted, last = missed):")
    print(" " * 12 + "".join(f"{n[6:]:>8}" for n in CLASS_NAMES) + "  missed")
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:>12}" + "".join(f"{v:8d}" for v in conf[i]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="outputs/yolo/blocks_obb/weights/best.pt")
    p.add_argument("--data", default="data/synthetic_blocks")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--match-mm", type=float, default=10.0, help="max centre distance for a match")
    main(p.parse_args())
