"""Calibrate a camera from checkerboard photos and save the result (ROS camera_info YAML).

  python perception/scripts/calibrate_camera.py --images /mnt/d/so101_data/calib_webcam \\
      --square-mm 25.0 --name laptop_webcam

Writes perception/calibration/<name>.yaml and two report images in media/.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from so101_perception.calibration import (Board, calibrate_robust, find_corners, homography_rms,
                                          save_camera_info_yaml)

ROOT = Path(__file__).resolve().parents[2]


def main(a):
    board = Board(a.cols, a.rows, a.square_mm / 1000.0)
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"no images in {a.images}")
    pts, used, size = [], [], None
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"  skip {f.name}: different image size")
            continue
        c = find_corners(gray, board)
        if c is None:
            print(f"  {f.name}: board NOT found")
        else:
            print(f"  {f.name}: ok   (fit to a plane: {homography_rms(c, board):5.2f} px)")
        if c is not None:
            pts.append(c)
            used.append(f)
    print(f"\nboard found in {len(pts)}/{len(files)} images, image size {size[0]} x {size[1]}")
    if len(pts) < 8:
        raise SystemExit("need at least 8 good images")

    print("calibrating (normally a few seconds) ...", flush=True)
    t0 = time.time()
    res, keep, rejected = calibrate_robust(pts, board, size, log=lambda m: print(m, flush=True),
                                           names=[f.name for f in used])
    print(f"done in {time.time() - t0:.1f} s")
    for i in rejected:
        print(f"  rejected as outlier: {used[i].name}")
    print("\nper-image RMS reprojection error [px]:")
    for i, e in zip(keep, res.per_view_error):
        print(f"  {used[i].name}: {e:.3f}")
    K, d = res.K, res.dist
    print(f"\nRMS reprojection error: {res.rms:.3f} px   (good: < 0.5, acceptable: < 1.0)")
    print(f"fx = {K[0, 0]:.1f}  fy = {K[1, 1]:.1f}  cx = {K[0, 2]:.1f}  cy = {K[1, 2]:.1f}  [px]")
    print("k1 k2 p1 p2 k3 =", np.round(d, 4))
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    print(f"horizontal field of view = {hfov:.1f} deg")

    out = ROOT / "perception" / "calibration" / f"{a.name}.yaml"
    save_camera_info_yaml(out, res, a.name, board, len(keep))
    print("wrote", out)

    # Report 1: coverage - where in the image did we have corners?
    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    cov = np.full((size[1], size[0], 3), 255, np.uint8)
    for i in keep:
        for x, y in pts[i].reshape(-1, 2):
            cv2.circle(cov, (int(x), int(y)), 3, (200, 60, 0), -1)
    cv2.putText(cov, f"corner coverage, {len(keep)} images", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(media / f"calib_coverage_{a.name}.png"), cov)

    # Report 2: original vs undistorted (straight edges should be straight on the right)
    img = cv2.imread(str(used[keep[0]]))
    und = cv2.undistort(img, K, d)
    both = np.hstack([img, und])
    cv2.putText(both, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(both, "undistorted", (size[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(media / f"calib_undistort_{a.name}.png"), both)
    print("wrote media/calib_coverage_*.png and media/calib_undistort_*.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square-mm", type=float, required=True, help="MEASURED square size of your print")
    p.add_argument("--name", default="laptop_webcam")
    main(p.parse_args())
