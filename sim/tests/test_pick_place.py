import numpy as np

from so101_sim.pick_place import SimRobot, gripper_angle_for_gap, pick_and_place
from so101_sim.scene import BLOCKS, SLOT_SITES, is_placed, load_scene, reset


def test_gripper_gap_lookup_is_monotonic():
    gaps = np.linspace(0.005, 0.05, 20)
    angles = [gripper_angle_for_gap(g) for g in gaps]
    assert np.all(np.diff(angles) > 0)


def run_episode(seed):
    model, data = load_scene()
    reset(model, data, rng=None if seed is None else np.random.default_rng(seed))
    robot = SimRobot(model, data)
    results = [pick_and_place(robot, b, s)[0] for b, s in zip(BLOCKS, SLOT_SITES)]
    # every block must still be in its slot at the end (later moves must not disturb earlier ones)
    final = [is_placed(model, data, b.name, s) for b, s in zip(BLOCKS, SLOT_SITES)]
    return results, final


def test_nominal_episode_places_all_three_blocks():
    results, final = run_episode(None)
    assert all(results) and all(final)


def test_randomised_episodes():
    n_ok = n = 0
    for seed in range(5):
        _, final = run_episode(seed)
        n_ok += sum(final)
        n += len(final)
    assert n_ok / n >= 0.9, f"only {n_ok}/{n} placed"
