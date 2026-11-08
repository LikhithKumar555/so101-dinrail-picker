"""Print the kinematic structure of the SO-101 MuJoCo model."""
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/robotstudio_so101/scene.xml"
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def name(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, idx) or f"<unnamed {idx}>"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    np.set_printoptions(precision=4, suppress=True)

    print(f"bodies={model.nbody} joints={model.njnt} dofs={model.nv} "
          f"actuators={model.nu} timestep={model.opt.timestep}s")

    print("\n--- Bodies (pos/quat relative to parent; quat is w,x,y,z) ---")
    for b in range(model.nbody):
        parent = name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
        print(f"[{b}] {name(model, mujoco.mjtObj.mjOBJ_BODY, b):<22} parent={parent:<22} "
              f"pos={model.body_pos[b]} quat={model.body_quat[b]}")

    print("\n--- Joints (axis is in the child body's frame) ---")
    for j in range(model.njnt):
        body = name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j])
        lo, hi = np.degrees(model.jnt_range[j])
        print(f"[{j}] {name(model, mujoco.mjtObj.mjOBJ_JOINT, j):<15} "
              f"type={JOINT_TYPES[model.jnt_type[j]]:<6} body={body:<22} "
              f"axis={model.jnt_axis[j]} range=[{lo:7.1f}, {hi:7.1f}] deg")

    print("\n--- Actuators ---")
    for a in range(model.nu):
        target = name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0])
        lo, hi = model.actuator_ctrlrange[a]
        print(f"[{a}] {name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<15} -> {target:<15} "
              f"ctrlrange=[{lo:.3f}, {hi:.3f}]")

    print("\n--- Sites (candidate end-effector reference points) ---")
    for s in range(model.nsite):
        body = name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[s])
        print(f"[{s}] {name(model, mujoco.mjtObj.mjOBJ_SITE, s):<22} body={body:<22} "
              f"pos={model.site_pos[s]}")

    print("\n--- Keyframes ---")
    for k in range(model.nkey):
        print(f"[{k}] {name(model, mujoco.mjtObj.mjOBJ_KEY, k)} qpos={model.key_qpos[k]}")

    mujoco.mj_forward(model, data)  # computes FK for qpos = 0
    print("\n--- World positions of bodies at qpos = 0 (MuJoCo's FK) ---")
    for b in range(model.nbody):
        print(f"{name(model, mujoco.mjtObj.mjOBJ_BODY, b):<22} xpos={data.xpos[b]}")


if __name__ == "__main__":
    main()
