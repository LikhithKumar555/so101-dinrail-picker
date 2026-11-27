"""Capture checkerboard photos from a webcam - run this on WINDOWS (WSL cannot see the webcam).

  Windows PowerShell:
    python capture_calib_images.py --out D:\\so101_data\\calib_webcam

Live view: corners are drawn when the board is detected. Keys:
  SPACE  save the current frame (only if the board is detected and the image is sharp)
  q      quit
Aim for 20-25 photos: board near every image corner and edge, tilted left/right/up/down by
up to ~45 deg, at different distances, and filling a good part of the image in some shots.
Hold the board still (or put it down) when saving - motion blur ruins corner accuracy.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()  # variance of the Laplacian: low = blurry


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # a changing focus changes the intrinsics
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {a.camera}: {w} x {h}")
    n = len(list(out.glob("img_*.png")))
    coverage = np.zeros((h, w), np.uint8)
    pattern = (a.cols, a.rows)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("cannot read from the camera")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        found, corners = cv2.findChessboardCorners(
            small, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        sharp = sharpness(gray)
        view = frame.copy()
        view[coverage > 0] = (0.6 * view[coverage > 0] + [0, 80, 0]).astype(np.uint8)
        if found:
            cv2.drawChessboardCorners(view, pattern, corners * 2, found)
        color = (0, 200, 0) if found and sharp > a.min_sharpness else (0, 0, 255)
        cv2.putText(view, f"saved {n}  sharpness {sharp:.0f}  {'BOARD' if found else 'no board'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("calibration capture (green tint = already covered)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not found:
                print("not saved: board not detected")
            elif sharp < a.min_sharpness:
                print(f"not saved: too blurry ({sharp:.0f} < {a.min_sharpness})")
            else:
                path = out / f"img_{n:02d}.png"
                cv2.imwrite(str(path), frame)
                hull = cv2.convexHull((corners * 2).astype(np.int32))
                cv2.fillConvexPoly(coverage, hull, 1)
                n += 1
                print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--min-sharpness", type=float, default=80.0)
    main(p.parse_args())
