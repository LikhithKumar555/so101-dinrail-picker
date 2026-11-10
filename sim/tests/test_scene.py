import mujoco
import numpy as np
import pytest

from so101_sim.ik import solve_ik, top_down_pose
from so101_sim.kinematics import ARM_JOINTS, TCP_SITE
from so101_sim.scene import (BLOCK_HEIGHT, BLOCKS, RAIL_TOP_Z, SLOT_SITES, TABLE_Z, is_placed,
                             load_scene, reset, scene_kinematics, set_block_pose, slot_pose)

rng = np.random.default_rng(2)


@pytest.fixture()
def scene():
    model, data = load_scene()
    reset(model, data)
    return model, data


def simulate(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_scene_kinematics_matches_mujoco(scene):
    model, data = scene
    kin = scene_kinematics()
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    sid = model.site(TCP_SITE).id
    for _ in range(50):
        q = rng.uniform(kin.joint_limits[:, 0], kin.joint_limits[:, 1])
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(kin.forward_kinematics(q)[:3, 3], data.site_xpos[sid], atol=1e-9)


def test_blocks_rest_stably_in_supply_area(scene):
    model, data = scene
    start = {b.name: data.body(b.name).xpos.copy() for b in BLOCKS}
    simulate(model, data, 2.0)
    for b in BLOCKS:
        assert np.linalg.norm(data.body(b.name).xpos - start[b.name]) < 1e-3, b.name
        assert abs(data.body(b.name).xpos[2] - (TABLE_Z + BLOCK_HEIGHT / 2)) < 1e-3, b.name


def test_block_seated_on_rail_is_detected_as_placed(scene):
    model, data = scene
    for block, slot in zip(BLOCKS, SLOT_SITES):
        p = slot_pose(model, data, slot)[:3, 3]
        set_block_pose(model, data, block.name, p)
    simulate(model, data, 1.0)
    for block, slot in zip(BLOCKS, SLOT_SITES):
        assert is_placed(model, data, block.name, slot), (block.name, slot)
    # A block that is 1 cm off along the rail must NOT count as placed
    p = slot_pose(model, data, SLOT_SITES[0])[:3, 3] + [0.0, 0.01, 0.0]
    set_block_pose(model, data, BLOCKS[0].name, p)
    simulate(model, data, 0.5)
    assert not is_placed(model, data, BLOCKS[0].name, SLOT_SITES[0])


def test_pick_and_place_poses_are_reachable_top_down(scene):
    model, data = scene
    kin = scene_kinematics()
    targets = []
    for b in BLOCKS:                                          # grasp + 6 cm above, in the supply area
        top = TABLE_Z + BLOCK_HEIGHT
        targets += [(*b.supply_xy, top - 0.012), (*b.supply_xy, top - 0.012 + 0.06)]
    for s in SLOT_SITES:                                      # same above each rail slot
        x, y, _ = slot_pose(model, data, s)[:3, 3]
        top = RAIL_TOP_Z + BLOCK_HEIGHT
        targets += [(x, y, top - 0.012), (x, y, top - 0.012 + 0.06)]
    for x, y, z in targets:
        res = solve_ik(kin, top_down_pose([x, y, z], yaw=np.pi / 2))
        assert res.success, f"not reachable top-down: {(round(x, 3), round(y, 3), round(z, 3))}"
