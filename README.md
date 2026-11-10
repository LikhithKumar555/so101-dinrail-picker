# SO-101 DIN-Rail Terminal Block Picker

Vision-guided pick-and-place of DIN-rail terminal blocks (3 widths) with the low-cost
SO-101 6-DoF arm: classic pipeline (calibrated camera + own IK) vs. imitation learning (ACT).

## Status
- [x] Phase 1: Setup (WSL2, ROS 2 Jazzy, MuJoCo, LeRobot)
- [x] Phase 2: Simulation (own FK / Jacobian IK, scripted pick-and-place)
- [ ] Phase 3: ROS 2 (URDF/RViz, ros2_control, C++ safety monitor)
- [ ] Phase 4: Perception (camera + hand-eye calibration, detection, 3D pick poses)
- [ ] Phase 5: Hardware (real SO-101, classic pipeline, 20 trials)
- [ ] Phase 6: Imitation learning (ACT, 20 trials, comparison across 3 widths)
- [ ] Phase 7: Write-up, results, video

## Simulation result (Phase 2)
![Scripted pick-and-place in MuJoCo](media/sim_pick_place.gif)

Classic pipeline (own FK + damped-least-squares IK, Cartesian straight-line motions),
block poses from the simulator, ±1 cm / ±10° randomised supply positions:
**60/60 blocks seated** (20 episodes × 3 widths: 5.2 / 8.2 / 12.2 mm).

## Simulation result (Phase 2)
![Scripted pick-and-place in MuJoCo](media/sim_pick_place.gif)

Classic pipeline (own FK + damped-least-squares IK, Cartesian straight-line motions),
block poses from the simulator, ±1 cm / ±10° randomised supply positions:
**60/60 blocks seated** (20 episodes × 3 widths: 5.2 / 8.2 / 12.2 mm),
max placement error 1.4 mm along / 0.7 mm across the rail.
