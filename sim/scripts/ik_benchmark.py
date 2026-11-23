"""Benchmark: own IK vs. MoveIt 2 (pick_ik) on the same top-down grasp poses.

  1. python sim/scripts/ik_benchmark.py make            (venv)  poses + own IK results
  2. ros2 run so101_moveit_config compare_ik_moveit.py   (ROS)   MoveIt IK on the same poses
  3. python sim/scripts/ik_benchmark.py report          (venv)  both scored with the SAME FK

The two halves run in different Python environments (venv vs. ROS) and talk through JSON files,
which avoids mixing their packages. Both solvers start from the same seed (the 'ready' pose).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from so101_sim.ik import pose_error, rot_to_quat, solve_ik, top_down_pose
from so101_sim.scene import BLOCK_HEIGHT, RAIL_TOP_Z, TABLE_Z, ready_q, scene_kinematics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "ik_benchmark"
POS_OK_MM, ROT_OK_DEG = 1.0, 1.0


def make(a):
    rng = np.random.default_rng(a.seed)
    kin = scene_kinematics()
    q0 = ready_q()
    poses, own = [], []
    for _ in range(a.n):
        # grasp and pre-grasp heights over the whole table area, any yaw
        on_rail = rng.random() < 0.3
        top = (RAIL_TOP_Z if on_rail else TABLE_Z) + BLOCK_HEIGHT
        z = top - 0.012 + (0.06 if rng.random() < 0.5 else 0.0)
        x, y = rng.uniform(0.12, 0.31), rng.uniform(-0.17, 0.19)
        T = top_down_pose([x, y, z], rng.uniform(-np.pi, np.pi))
        q = rot_to_quat(T[:3, :3])                                  # w, x, y, z
        poses.append({"position": T[:3, 3].tolist(), "quat_xyzw": [q[1], q[2], q[3], q[0]]})
        t0 = time.perf_counter()
        res = solve_ik(kin, T, q0=q0)
        own.append({"q": res.q.tolist(), "time_ms": (time.perf_counter() - t0) * 1000})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poses.json").write_text(json.dumps({"seed_q": q0.tolist(), "poses": poses}, indent=1))
    (OUT / "own_ik.json").write_text(json.dumps(own))
    print(f"wrote {len(poses)} poses and own-IK results to {OUT}")


def score(kin, poses, results):
    """Per pose: success flag, position error [mm], rotation error [deg], solve time [ms]."""
    from so101_sim.kinematics import make_transform, quat_to_rot
    ok, pe, re, t = [], [], [], []
    for p, r in zip(poses, results):
        t.append(r["time_ms"])
        if r.get("q") is None:
            ok.append(False), pe.append(np.nan), re.append(np.nan)
            continue
        x, y, z, w = p["quat_xyzw"]
        T_goal = make_transform(quat_to_rot([w, x, y, z]), p["position"])
        e = pose_error(T_goal, kin.forward_kinematics(np.array(r["q"][:5])))
        pe.append(np.linalg.norm(e[:3]) * 1000)
        re.append(np.degrees(np.linalg.norm(e[3:])))
        ok.append(pe[-1] < POS_OK_MM and re[-1] < ROT_OK_DEG)
    return np.array(ok), np.array(pe), np.array(re), np.array(t)


def report(a):
    kin = scene_kinematics()
    poses = json.loads((OUT / "poses.json").read_text())["poses"]
    runs = {}
    for name, f in (("own DLS IK", "own_ik.json"), ("MoveIt 2 + pick_ik", "moveit_ik.json")):
        if (OUT / f).exists():
            runs[name] = score(kin, poses, json.loads((OUT / f).read_text()))
        else:
            print(f"(no results for {name}: {f})")
    solvable = np.any([r[0] for r in runs.values()], axis=0)
    print(f"{len(poses)} top-down grasp poses; {solvable.sum()} solvable by at least one solver "
          f"(the rest lie outside the dexterous workspace)")
    print(f"success = pos < {POS_OK_MM} mm and rot < {ROT_OK_DEG} deg\n")
    print(f"{'solver':<20}{'solved':>14}{'pos err med':>13}{'rot err med':>13}"
          f"{'time (solved)':>15}{'time (fail)':>13}")
    for name, (ok, pe, re, t) in runs.items():
        print(f"{name:<20}{ok.sum():>6}/{solvable.sum():<4}{ok.sum() / solvable.sum():6.1%}"
              f"{np.nanmedian(pe[ok]):10.3f} mm{np.nanmedian(re[ok]):9.3f} deg"
              f"{np.median(t[ok]):11.1f} ms{np.median(t[~ok]) if (~ok).any() else 0:9.1f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "report"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make(a) if a.cmd == "make" else report(a)
