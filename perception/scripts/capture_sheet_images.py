"""Photograph the printed block test sheet (step 4.2b) - run this on WINDOWS (WSL cannot see the webcam).

  Windows PowerShell:
    python capture_sheet_images.py --out D:\\so101_data\\sheet_test

Uses only OpenCV + NumPy (no project packages), like capture_calib_images.py.
Capture at the SAME resolution and focus setting as the calibration (default 1280 x 720,
autofocus off) - the intrinsics are only valid for that.

Live view: the checkerboard corners and the outline of the whole sheet are drawn. The status
line shows the block scale at the detector input (px/mm). The detector was trained at about
2.1 px/mm, so aim for 1.8 - 2.4 (green). For a 1280 x 720 laptop webcam that is roughly
30 - 40 cm between camera and sheet.
Keys:
  SPACE  save the frame (only if the board is found, the image is sharp and the sheet is fully visible)
  q      quit
Aim for 15 - 20 photos: different distances within the green range, the sheet tilted up to
~20 deg in different directions, different positions in the image, and different light
(ceiling light, desk lamp, daylight). Keep the sheet FLAT (tape it to a book or a box).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# must match perception/so101_perception/print_sheet.py
COLS, ROWS, SQUARE_MM = 7, 4, 15.0
BOARD_ORIGIN_MM = np.array([35.0, 130.0])            # page position of inner corner (0, 0)
A4_W_MM, A4_H_MM = 297.0, 210.0
SCALE_OK = (1.8, 2.4)                                # px/mm at the detector input


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()     # variance of the Laplacian: low = blurry


def orient(corners, gray):
    """Put the corners into the sheet's board order: x = page right, y = page down, corner
    (0, 0) next to a BLACK square. OpenCV may return them mirrored or rotated by 180 deg."""
    c = corners.reshape(ROWS, COLS, 2).astype(np.float64)
    x_axis, y_axis = c[0, -1] - c[0, 0], c[-1, 0] - c[0, 0]
    if x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0] < 0:     # mirrored order
        c = c[:, ::-1]

    def grey_at(p):
        x, y = np.clip(np.round(p).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
        return float(gray[y, x])

    sq00 = c[:2, :2].reshape(-1, 2).mean(axis=0)              # square between corners (0,0)-(1,1)
    sq10 = c[:2, 1:3].reshape(-1, 2).mean(axis=0)             # its right neighbour
    if grey_at(sq00) > grey_at(sq10):                         # (0,0) square must be the dark one
        c = c[::-1, ::-1]
    return c.reshape(-1, 2)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                        # a changing focus changes the intrinsics
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {a.camera}: {w} x {h}")
    if (w, h) != (a.width, a.height):
        print(f"WARNING: the camera gives {w} x {h}, not {a.width} x {a.height} - "
              "the calibration will not match")
    n = len(list(out.glob("sheet_*.png")))
    grid_mm = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2).astype(np.float64) * SQUARE_MM
    sheet_mm = np.array([[0, 0], [A4_W_MM, 0], [A4_W_MM, A4_H_MM], [0, A4_H_MM]]) - BOARD_ORIGIN_MM

    while True:
        ok, frame = cap.read()
        if not ok:
            print("cannot read from the camera")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        found, corners = cv2.findChessboardCorners(
            small, (COLS, ROWS), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        sharp = sharpness(gray)
        view = frame.copy()
        scale = None
        visible = False
        if found:
            c = orient(corners * 2, gray)
            cv2.drawChessboardCorners(view, (COLS, ROWS), c.reshape(-1, 1, 2).astype(np.float32), True)
            Hm, _ = cv2.findHomography(grid_mm, c)
            outline = cv2.perspectiveTransform(sheet_mm.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            margin = 15                                       # px; the outline ignores lens distortion
            visible = bool(np.all((outline >= margin) & (outline <= [w - 1 - margin, h - 1 - margin])))
            cv2.polylines(view, [outline.astype(np.int32)], True, (0, 255, 0) if visible else (0, 0, 255), 2)
            d = np.concatenate([np.linalg.norm(np.diff(c.reshape(ROWS, COLS, 2), axis=1), axis=2).ravel(),
                                np.linalg.norm(np.diff(c.reshape(ROWS, COLS, 2), axis=0), axis=2).ravel()])
            scale = d.mean() / SQUARE_MM * a.imgsz / max(w, h)
        scale_ok = scale is not None and SCALE_OK[0] <= scale <= SCALE_OK[1]
        sharp_ok = sharp > a.min_sharpness
        msg = (f"saved {n} | scale {scale:.2f} px/mm" if scale is not None else f"saved {n} | no board")
        msg += f" ({'ok' if scale_ok else 'move closer' if scale is not None and scale < SCALE_OK[0] else 'move away' if scale is not None else '-'})"
        msg += f" | sheet {'visible' if visible else 'CUT OFF'} | sharp {sharp:.0f}"
        good = found and visible and sharp_ok
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if good and scale_ok else (0, 165, 255) if good else (0, 0, 255), 2)
        cv2.imshow("sheet capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not found:
                print("not saved: board not detected")
            elif not visible:
                print("not saved: the sheet is cut off by the image border")
            elif not sharp_ok:
                print(f"not saved: too blurry ({sharp:.0f} < {a.min_sharpness})")
            else:
                path = out / f"sheet_{n:02d}.png"
                cv2.imwrite(str(path), frame)
                n += 1
                print(f"saved {path}  (scale {scale:.2f} px/mm{'' if scale_ok else ' - outside 1.8-2.4'})")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--imgsz", type=int, default=960, help="detector input size (for the scale readout)")
    p.add_argument("--min-sharpness", type=float, default=80.0)
    main(p.parse_args())
