"""Camera-driven pick-and-place: the full classic pipeline, end to end, in simulation.

  python perception/scripts/run_camera_pick_place.py --episodes 20

For every block: arm to the 'look' (park) pose -> overhead image -> detector -> 3D pose with the
HAND-EYE-CALIBRATED camera (perception/calibration/handeye_sim.yaml) -> own IK -> pick-and-place.
Nothing about the blocks is read from the simulator, except to score the result at the end.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from so101_perception.pose_estimation import BlockLocalizer, Camera, camera_from_mujoco
from so101_perception.synthetic import CLASS_NAMES
from so101_sim.pick_place import GRIPPER_PARKED, SimRobot, pick_and_place
from so101_sim.scene import (BLOCKS, PARK_Q, SLOT_SITES, body_pose, load_scene, placement_error,
                             ready_q, reset)

ROOT = Path(__file__).resolve().parents[2]
W, H = 960, 720


def load_handeye_camera(model, data, path: Path) -> Camera:
    """Intrinsics of the camera (sim: exact; real robot: from calibrate_camera.py) +
    extrinsics from the hand-eye calibration result."""
    nominal = camera_from_mujoco(model, data, "overhead", W, H)
    T = np.array(yaml.safe_load(path.read_text())["T_base_cam"])
    return Camera(nominal.K, nominal.dist, T)


def main(a):
    rows, n_ok = [], 0
    for ep in range(a.episodes):
        model, data = load_scene()
        reset(model, data, rng=np.random.default_rng(a.seed + ep), arm_pose="park")
        renderer = mujoco.Renderer(model, height=H, width=W)
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 0                                # the camera does not see slot markers
        cam = load_handeye_camera(model, data, ROOT / a.handeye)
        if ep == 0:
            localizer = BlockLocalizer(ROOT / a.weights, cam, imgsz=a.imgsz, conf=0.5)
        localizer.camera = cam
        robot = SimRobot(model, data)
        robot.wait(0.5)                                     # let everything settle

        for block, slot in zip(BLOCKS, SLOT_SITES):
            # 1. look: arm parked out of view, take a picture, detect
            robot.move_joints(PARK_Q, 1.5)
            robot.wait(0.3)
            renderer.update_scene(data, camera="overhead", scene_option=opt)
            image = renderer.render().copy()
            t0 = time.time()
            poses = localizer(image)
            t_perc = (time.time() - t0) * 1000
            cls = [CLASS_NAMES.index(n) for n in CLASS_NAMES][BLOCKS.index(block)]
            # Two independent cues for the block type: the detector's class and the width
            # measured in 3D (5.2 / 8.2 / 12.2 mm, ~0.3 mm accurate). Trust the geometry when it
            # is unambiguous - picking the WRONG block is the worst failure, as it also blocks
            # the slot of the other one.
            def block_type(p):
                d = np.abs(np.array([b.width for b in BLOCKS]) * 1000 - p.width_mm)
                return int(np.argmin(d)) if d.min() < a.width_tol_mm else p.cls
            cands = [p for p in poses if block_type(p) == cls and not p.on_rail]
            T_true = body_pose(model, data, block.name)
            if not cands:
                rows.append({"episode": ep, "block": block.name, "success": 0, "message": "not detected"})
                print(f"episode {ep:3d}  {block.name:<10}: FAIL  not detected")
                continue
            est = max(cands, key=lambda p: p.confidence)
            T_est = est.T_base_block()
            e_xy = np.linalg.norm(T_est[:2, 3] - T_true[:2, 3]) * 1000
            yaw_true = np.arctan2(T_true[1, 0], T_true[0, 0])
            e_yaw = abs(np.degrees((est.yaw - yaw_true + np.pi / 2) % np.pi - np.pi / 2))
            # 2. act: ready pose, then the Phase-2 pick-and-place with the ESTIMATED pose
            robot.move_gripper(GRIPPER_PARKED, 0.2)
            robot.move_joints(ready_q(), 1.5)
            ok, msg = pick_and_place(robot, block, slot, T_block=T_est)
            robot.move_joints(ready_q(), 1.0)
            e = placement_error(model, data, block.name, slot)
            n_ok += ok
            rows.append({"episode": ep, "block": block.name, "success": int(ok), "message": msg,
                         "perception_err_xy_mm": round(e_xy, 2), "perception_err_yaw_deg": round(e_yaw, 2),
                         "place_err_along_mm": round(e["along"] * 1000, 2),
                         "place_err_across_mm": round(e["across"] * 1000, 2),
                         "perception_ms": round(t_perc, 1)})
            print(f"episode {ep:3d}  {block.name:<10} -> {slot}: {'OK  ' if ok else 'FAIL'} {msg:<26} "
                  f"perception error {e_xy:4.1f} mm / {e_yaw:3.1f} deg")
        renderer.close()

    total = len(rows)
    print(f"\nCamera-driven success: {n_ok}/{total} = {100 * n_ok / total:.1f} %")
    for b in BLOCKS:
        r = [x for x in rows if x["block"] == b.name]
        print(f"  {b.name:<10}: {sum(x['success'] for x in r)}/{len(r)}")
    pe = [x["perception_err_xy_mm"] for x in rows if "perception_err_xy_mm" in x]
    if pe:
        print(f"perception error (xy) median {np.median(pe):.2f} mm, max {np.max(pe):.2f} mm")
    out = ROOT / a.csv
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", default="outputs/yolo/blocks_obb/weights/best.pt")
    p.add_argument("--handeye", default="perception/calibration/handeye_sim.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--csv", default="experiments/sim_camera_pick_place.csv")
    p.add_argument("--width-tol-mm", type=float, default=1.5,
                   help="max difference between measured and nominal block width")
    main(p.parse_args())
