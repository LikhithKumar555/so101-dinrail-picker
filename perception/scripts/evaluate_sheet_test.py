"""Real-camera perception test (step 4.2b): photos of the printed test sheet -> errors in mm.

  # real photos (taken with capture_sheet_images.py on Windows)
  python perception/scripts/evaluate_sheet_test.py --images /mnt/d/so101_data/sheet_test
  # the same test with synthetic photos of the same sheet (ideal camera = the baseline)
  python perception/scripts/evaluate_sheet_test.py --dry-run 30

Per photo:
  1. checkerboard -> solvePnP with the REAL intrinsics (laptop_webcam.yaml) -> T_cam_board
  2. detector (trained only on synthetic images) -> oriented boxes
  3. box corners -> rays -> plane z = 0 of the sheet -> centre, yaw and width in board millimetres
  4. match with the printed ground truth (same board frame) and score
The numbers answer: how much worse is the perception on a real lens, real noise and real light
than on the synthetic images it was trained on (the sim-to-real gap of perception)?
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from so101_perception.calibration import load_camera_info_yaml
from so101_perception.pose_estimation import BlockLocalizer, Camera
from so101_perception.print_sheet import (BOARD_ORIGIN_MM, LAYOUT, WIDTHS_MM, block_corners_mm, board_px_per_mm,
                                          look_at_sheet, page_to_board, quiet_zone_rect_mm,
                                          rasterize_sheet, render_sheet_view, sheet_pose, truth)
from so101_perception.synthetic import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
POS_TOL_MM, YAW_TOL_DEG = 3.0, 5.0          # what the gripper tolerates (same as in sim)
MATCH_RADIUS_MM = 15.0                      # a detection further away than this is a miss
TRAIN_PX_PER_MM = 2.07                      # block scale in the synthetic training images


def wrap_half_pi(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def width_class(width_mm: float, tol_mm: float):
    """Class from the measured width alone (None if no nominal width is within tol_mm)."""
    d = {c: abs(width_mm - w) for c, w in WIDTHS_MM.items()}
    c = min(d, key=d.get)
    return c if d[c] <= tol_mm else None


def on_board_area(xy_m) -> bool:
    """True if a board-frame point lies on the white checkerboard area of the sheet."""
    x0, y0, x1, y1 = quiet_zone_rect_mm(0.0)
    p = np.asarray(xy_m) * 1000 + BOARD_ORIGIN_MM
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def images_real(folder: Path):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise SystemExit(f"no images in {folder}")
    for p in paths:
        img = cv2.imread(str(p))
        yield p.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def images_synthetic(n: int, K, dist, size, imgsz: int, seed: int):
    """Random but realistic views of the sheet: scale near the training scale, tilted, rolled."""
    rng = np.random.default_rng(seed)
    sheet = rasterize_sheet(8.0)
    for k in range(n):
        eff = rng.uniform(1.8, 2.3)                                # px/mm at the detector input
        distance = K[0, 0] / (eff * max(size) / imgsz) / 1000      # metres
        # small roll only: at the training scale the whole A4 sheet just fits into the image
        T = look_at_sheet(distance, rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-8, 8),
                          target_mm=(148.5 + rng.uniform(-5, 5), 105 + rng.uniform(-5, 5)))
        img = render_sheet_view(*sheet, K, dist, T, size, noise=rng.uniform(1, 4),
                                blur=rng.uniform(0.4, 1.0), rng=rng)
        # lighting: gain, offset and a smooth brightness gradient across the image
        h, w = img.shape[:2]
        gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        light = rng.uniform(0.75, 1.15) * (1 + 0.15 * (rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy))
        img = np.clip(img * light[..., None] + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
        yield f"synthetic_{k:03d}", img


def draw_box(out, cam: Camera, p, colour, thickness):
    L, W = p.length_mm / 2000, p.width_mm / 2000
    cs, sn = np.cos(p.yaw), np.sin(p.yaw)
    local = np.array([[L, W], [-L, W], [-L, -W], [L, -W]])
    pts = p.position[:2] + local @ np.array([[cs, sn], [-sn, cs]])
    px = cam.project(np.column_stack([pts, np.zeros(4)]))
    cv2.polylines(out, [px.astype(np.int32)], True, colour, thickness)


def draw_overlay(img_rgb, cam: Camera, poses, matches, rejected=()) -> np.ndarray:
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for p in rejected:                                 # dropped by the size check: thin grey
        draw_box(out, cam, p, (160, 160, 160), 1)
    for b in LAYOUT:                                   # ground truth in green
        px = cam.project(np.column_stack([page_to_board(block_corners_mm(b)), np.zeros(4)]))
        cv2.polylines(out, [px.astype(np.int32)], True, (0, 255, 0), 1)
    colours = [(0, 200, 255), (255, 0, 255), (255, 80, 0)]
    for k, p in enumerate(poses):                      # detections, labelled with their error
        c = cam.project(np.array([[p.position[0], p.position[1], 0.0]]))[0]
        draw_box(out, cam, p, colours[p.cls], 2)
        m = matches.get(k)
        txt = f"{CLASS_NAMES[p.cls][6:]} {p.width_mm:.1f}mm" + (f" e={m:.1f}" if m is not None else " FP")
        cv2.putText(out, txt, (int(c[0]) + 8, int(c[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colours[p.cls], 1, cv2.LINE_AA)
    return out


def main(a):
    K, dist, size = load_camera_info_yaml(ROOT / a.calib)
    if a.dry_run:
        source = images_synthetic(a.dry_run, K, dist, size, a.imgsz, a.seed)
        csv_path = ROOT / (a.csv or "experiments/sheet_test_dryrun.csv")
        overlay_path = ROOT / (a.overlay or "outputs/sheet_test_dryrun_example.png")
    else:
        source = images_real(Path(a.images))
        csv_path = ROOT / (a.csv or "experiments/sheet_test_real.csv")
        overlay_path = ROOT / (a.overlay or "media/sheet_test_real_example.png")

    loc = BlockLocalizer(ROOT / a.weights, Camera(K, dist, np.eye(4)), imgsz=a.imgsz,
                         conf=a.conf, top_z=0.0, size_check=not a.no_size_check)
    gt = truth()
    rows, best = [], None
    n_img = n_used = n_fp = n_fp_board = n_rejected = 0
    print(f"{'image':<22} {'board':>8} {'px/mm*':>7} {'found':>6} {'FP':>3} {'xy med':>7}")
    for name, img in source:
        n_img += 1
        if (img.shape[1], img.shape[0]) != tuple(size):
            raise SystemExit(f"{name}: image is {img.shape[1]}x{img.shape[0]}, but the calibration "
                             f"is for {size[0]}x{size[1]} - capture at the calibrated resolution")
        res = sheet_pose(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K, dist)
        if res is None:
            print(f"{name:<22} board not found - skipped")
            continue
        T_cam_board, corners, rms = res
        n_used += 1
        scale = board_px_per_mm(corners) * a.imgsz / max(size)        # at the detector input
        cam = Camera(K, dist, np.linalg.inv(T_cam_board))
        loc.camera = cam
        poses = loc(img)
        n_rejected += len(loc.rejected)

        # greedy matching: closest detection per true block, each detection used once
        pairs = sorted((np.linalg.norm(p.position[:2] - g[1]) * 1000, i, k)
                       for i, g in enumerate(gt) for k, p in enumerate(poses))
        used_t, used_d, match_of = set(), set(), {}
        for d_mm, i, k in pairs:
            if d_mm > MATCH_RADIUS_MM or i in used_t or k in used_d:
                continue
            used_t.add(i)
            used_d.add(k)
            match_of[i] = k
        fps = [k for k in range(len(poses)) if k not in used_d]
        n_fp += len(fps)
        n_fp_board += sum(on_board_area(poses[k].position[:2]) for k in fps)

        errs, det_err = [], {}
        for i, (cls, xy, yaw, width) in enumerate(gt):
            row = {"image": name, "block": i, "true_class": cls, "found": i in match_of,
                   "board_rms_px": round(rms, 3), "px_per_mm_input": round(scale, 3),
                   "true_x_mm": round(float(xy[0]) * 1000, 3), "true_y_mm": round(float(xy[1]) * 1000, 3)}
            if i in match_of:
                p = poses[match_of[i]]
                e_xy = float(np.linalg.norm(p.position[:2] - xy) * 1000)
                e_yaw = float(abs(np.degrees(wrap_half_pi(p.yaw - yaw))))
                wc = width_class(p.width_mm, a.width_tol_mm)
                fused = wc if wc is not None else p.cls              # same rule as the pick-and-place
                row.update(est_x_mm=round(float(p.position[0]) * 1000, 3),
                           est_y_mm=round(float(p.position[1]) * 1000, 3),
                           det_class=p.cls, width_class=wc, fused_class=fused,
                           conf=round(p.confidence, 3), xy_err_mm=round(e_xy, 3),
                           yaw_err_deg=round(e_yaw, 3), width_mm=round(p.width_mm, 3),
                           width_err_mm=round(p.width_mm - width, 3), length_mm=round(p.length_mm, 3),
                           graspable=bool(fused == cls and e_xy <= POS_TOL_MM and e_yaw <= YAW_TOL_DEG))
                errs.append(e_xy)
                det_err[match_of[i]] = e_xy
            rows.append(row)
        med = f"{np.median(errs):6.2f}" if errs else "     -"
        print(f"{name:<22} {rms:6.2f}px {scale:7.2f} {len(match_of):3d}/12 {len(fps):3d} {med} mm")
        overlay = draw_overlay(img, cam, poses, det_err, loc.rejected)
        if a.save_all:                                  # every photo, to inspect misses and FPs
            Path(a.save_all).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.save_all) / f"{Path(name).stem}_overlay.png"), overlay)
        score = (len(match_of), -len(fps))              # the README image: the best photo
        if best is None or score > best[0]:
            best = (score, overlay)

    if not rows:
        raise SystemExit("no usable image (the checkerboard was never found)")
    found = [r for r in rows if r["found"]]
    n_true = len(rows)
    xy = np.array([r["xy_err_mm"] for r in found])
    yaw = np.array([r["yaw_err_deg"] for r in found])
    w_err = np.array([r["width_err_mm"] for r in found])
    scales = np.array(sorted({r["px_per_mm_input"] for r in rows}))
    print(f"\n{'SYNTHETIC (dry run)' if a.dry_run else 'REAL'} photos: {n_used}/{n_img} usable, "
          f"{n_true} true blocks, scale at detector input {np.median(scales):.2f} px/mm "
          f"(training: ~{TRAIN_PX_PER_MM})")
    print(f"found                         : {len(found) / n_true:6.1%}   "
          f"({n_fp} false detections, {n_fp_board} of them on the checkerboard)")
    print(f"rejected by the size check    : {n_rejected} detections"
          f"{' (size check OFF)' if a.no_size_check else ''}")
    if found:
        for key, label in (("det_class", "detector class"), ("width_class", "width-only class"),
                           ("fused_class", "fused class (pipeline)")):
            ok = np.mean([r[key] == r["true_class"] for r in found])
            print(f"{label + ' correct':<30}: {ok:6.1%} of found")
        print(f"xy error   med/p95/max        : {np.median(xy):.2f} / {np.percentile(xy, 95):.2f} / {xy.max():.2f} mm")
        print(f"yaw error  med/p95/max        : {np.median(yaw):.2f} / {np.percentile(yaw, 95):.2f} / {yaw.max():.2f} deg")
        print(f"width error mean +- std       : {w_err.mean():+.2f} +- {w_err.std():.2f} mm")
        for c, wn in WIDTHS_MM.items():
            m = [r["width_mm"] for r in found if r["true_class"] == c]
            if m:
                print(f"   {wn:4.1f} mm blocks measured    : {np.mean(m):5.2f} +- {np.std(m):.2f} mm  (n={len(m)})")
        g = np.mean([r.get("graspable", False) for r in rows])
        print(f"GRASPABLE (found, right class, <= {POS_TOL_MM:g} mm, <= {YAW_TOL_DEG:g} deg): {g:6.1%} of all blocks")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), best[1])
    print(f"wrote {csv_path.relative_to(ROOT)} and {overlay_path.relative_to(ROOT)} "
          f"(green = printed truth, coloured = detections with xy error in mm)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="/mnt/d/so101_data/sheet_test")
    p.add_argument("--dry-run", type=int, default=0, metavar="N",
                   help="use N synthetic photos of the sheet instead of real ones")
    p.add_argument("--calib", default="perception/calibration/laptop_webcam.yaml")
    p.add_argument("--weights", default="outputs/yolo/blocks_obb/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--width-tol-mm", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None)
    p.add_argument("--overlay", default=None)
    p.add_argument("--no-size-check", action="store_true",
                   help="keep detections whose size in mm cannot be a block")
    p.add_argument("--save-all", default=None, metavar="DIR",
                   help="also write an overlay image for every photo into DIR")
    main(p.parse_args())
