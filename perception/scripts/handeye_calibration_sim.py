"""Hand-eye calibration in simulation, with the answer known.

  python perception/scripts/handeye_calibration_sim.py --views 20
  python perception/scripts/handeye_calibration_sim.py --study       # error vs. views and joint noise

1. Move the arm (with a checkerboard on a bracket) through random poses in front of the camera.
2. Per pose: record the gripper pose from FK (A_i) and detect the board in the image (B_i).
3. Solve X B_i = A_i Y for the camera pose X, refine on reprojection error.
4. Compare with the true camera pose, and convert the error into what matters: how far off a
   block on the table would be located.
Joint noise simulates servo encoder/backlash error in the FK (the real arm is not perfect).
"""
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from so101_perception.calibration import find_corners
from so101_perception.handeye import calibrate_eye_to_hand, pose_error
from so101_perception.handeye_sim import HANDEYE_BOARD, build_handeye_model, set_arm
from so101_perception.pose_estimation import Camera, camera_from_mujoco
from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.scene import BLOCK_HEIGHT, TABLE_Z, scene_kinematics

ROOT = Path(__file__).resolve().parents[2]
W, H = 960, 720


def collect_views(n, rng, model, data, renderer, kin, opt, max_tries=400):
    """Random reachable poses where the whole board is visible. Returns (q list, corners list)."""
    qs, pts, tries = [], [], 0
    while len(qs) < n and tries < max_tries:
        tries += 1
        p = [rng.uniform(0.17, 0.27), rng.uniform(-0.08, 0.06), TABLE_Z + rng.uniform(0.12, 0.18)]
        yaw = rng.choice([0.0, np.pi / 2, -np.pi / 2]) + rng.uniform(-0.4, 0.4)
        res = solve_ik(kin, top_down_pose(p, yaw), restarts=2)
        if not res.success:
            continue
        q = res.q + np.radians(rng.normal(0, 8, 5))          # tilt the gripper: rotation diversity
        q = np.clip(q, kin.joint_limits[:, 0], kin.joint_limits[:, 1])
        set_arm(model, data, q)
        renderer.update_scene(data, camera="overhead", scene_option=opt)
        img = renderer.render().astype(np.float32)
        img += rng.normal(0, 2.0, img.shape)                  # sensor noise
        gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        c = find_corners(gray, HANDEYE_BOARD)
        if c is not None:
            qs.append(q)
            pts.append(c)
    return qs, pts


def table_error_mm(cam_est: Camera, cam_true: Camera):
    """Locate a grid of block tops with the estimated camera: (mean, max) xy error in mm."""
    z = TABLE_Z + BLOCK_HEIGHT
    g = np.array([[x, y, z] for x in np.linspace(0.10, 0.34, 9) for y in np.linspace(-0.17, 0.17, 9)])
    est = cam_est.backproject_to_plane(cam_true.project(g), z)
    e = np.linalg.norm(est[:, :2] - g[:, :2], axis=1) * 1000
    return e.mean(), e.max()


def run(n_views, joint_noise_deg, seed, setup, verbose=True):
    model, data, renderer, kin, opt, cam_true = setup
    rng = np.random.default_rng(seed)
    qs, pts = collect_views(n_views, rng, model, data, renderer, kin, opt)
    # what the robot REPORTS: true angles + encoder/backlash noise
    q_meas = [q + np.radians(rng.normal(0, joint_noise_deg, 5)) for q in qs]
    A = [kin.link_frames(q)[-1] for q in q_meas]             # T_base_gripper from FK
    res = calibrate_eye_to_hand(A, pts, HANDEYE_BOARD, cam_true.K, cam_true.dist)
    dt, dr = pose_error(res.T_base_cam, cam_true.T_base_cam)
    mean_e, max_e = table_error_mm(Camera(cam_true.K, cam_true.dist, res.T_base_cam), cam_true)
    if verbose:
        print(f"views used: {res.n_views} (rejected as outliers: {res.rejected}), "
              f"reprojection RMS: {res.rms_px:.3f} px")
        print(f"camera position error : {dt:.2f} mm")
        print(f"camera rotation error : {dr:.3f} deg")
        print(f"=> block localisation error on the table: mean {mean_e:.2f} mm, max {max_e:.2f} mm")
    return res, dt, dr, mean_e, max_e


def main(a):
    model, _ = build_handeye_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    mujoco.mj_forward(model, data)
    cam_true = camera_from_mujoco(model, data, "overhead", W, H)
    setup = (model, data, renderer, scene_kinematics(), opt, cam_true)

    if a.study:
        print(f"{'views':>5} {'joint noise':>11} {'cam pos':>9} {'cam rot':>9} {'table mean':>11} {'table max':>10}")
        for noise in (0.0, 0.1, 0.3):
            for n in (5, 10, 20):
                r = [run(n, noise, s, setup, verbose=False)[1:] for s in range(3)]
                dt, dr, me, mx = np.mean(r, axis=0)
                print(f"{n:5d} {noise:9.1f} deg {dt:7.2f} mm {dr:6.3f} deg {me:8.2f} mm {mx:7.2f} mm")
        return

    res, *_ = run(a.views, a.joint_noise, a.seed, setup)
    out = ROOT / "perception" / "calibration" / "handeye_sim.yaml"
    T = res.T_base_cam
    out.write_text(yaml.safe_dump({
        "description": "T_base_cam: overhead camera pose in the robot base frame (OpenCV axes)",
        "T_base_cam": [[float(v) for v in row] for row in T],
        "reprojection_rms_px": round(res.rms_px, 4), "n_views": res.n_views,
    }, sort_keys=False))
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, default=20)
    p.add_argument("--joint-noise", type=float, default=0.1, help="deg, std of joint angle error")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study", action="store_true", help="error vs. number of views and joint noise")
    main(p.parse_args())
