"""Drive the simulated SO-101 through top-down grasp poses with your own IK.

For each target: solve IK -> interpolate joint commands -> let MuJoCo's position
servos track them -> compare where the TCP really ended up with the target.
Run:  python sim/scripts/ik_demo.py            (opens the viewer)
      python sim/scripts/ik_demo.py --headless (prints numbers only)
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.kinematics import ARM_JOINTS, MODEL_DIR, TCP_SITE, SO101Kinematics

# (x, y, z, yaw): a square 5 cm above the table, the same point with different yaws,
# and a last target that is DELIBERATELY unreachable pointing straight down:
# at 10 cm height wrist_flex would need more than its 95 deg limit.
TARGETS = [
    (0.20, -0.08, 0.05, 0.0), (0.28, -0.08, 0.05, 0.0), (0.28, 0.08, 0.05, 0.0),
    (0.20, 0.08, 0.05, 0.0), (0.24, 0.00, 0.03, 0.0), (0.24, 0.00, 0.03, 0.8),
    (0.24, 0.00, 0.03, -0.8), (0.24, 0.00, 0.10, 0.0),
]
MOVE_TIME = 1.5     # s, joint-space interpolation per move
SETTLE_TIME = 0.5   # s, wait for the servos before measuring


def add_marker(viewer, pos, rgba=(1, 0, 0, 0.6)):
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                        np.asarray(pos, dtype=float), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def main(headless: bool):
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    kin = SO101Kinematics()
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    act = [model.actuator(j).id for j in ARM_JOINTS]
    sid = model.site(TCP_SITE).id
    dt = model.opt.timestep

    viewer = None if headless else mujoco.viewer.launch_passive(model, data)

    def step(n):
        for _ in range(n):
            t0 = time.time()
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))   # run at real-time speed

    q_cmd = data.qpos[qadr].copy()
    print(f"{'target (x, y, z) [m], yaw [deg]':<36} {'IK':<4} {'iters':>5} "
          f"{'IK err [mm]':>11} {'sim err [mm]':>12}")
    for x, y, z, yaw in TARGETS:
        res = solve_ik(kin, top_down_pose([x, y, z], yaw), q0=q_cmd)
        if viewer is not None:
            add_marker(viewer, [x, y, z])
        if not res.success:
            print(f"({x:.2f}, {y:+.2f}, {z:.2f}), {np.degrees(yaw):5.0f}   FAIL")
            continue
        q_start = q_cmd.copy()
        n_move = int(MOVE_TIME / dt)
        for k in range(1, n_move + 1):
            s = k / n_move
            s = 3 * s**2 - 2 * s**3                       # smoothstep: zero velocity at both ends
            data.ctrl[act] = q_start + s * (res.q - q_start)
            step(1)
        q_cmd = res.q
        step(int(SETTLE_TIME / dt))
        sim_err = np.linalg.norm(data.site_xpos[sid] - [x, y, z]) * 1000
        print(f"({x:.2f}, {y:+.2f}, {z:.2f}), {np.degrees(yaw):5.0f}           OK   {res.iterations:5d} "
              f"{res.pos_error * 1000:11.3f} {sim_err:12.2f}")

    if viewer is not None:
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            step(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    main(parser.parse_args().headless)
