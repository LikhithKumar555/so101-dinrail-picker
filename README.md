# SO-101 DIN-Rail Terminal Block Picker

Vision-guided pick-and-place of DIN-rail terminal blocks (3 widths) with the low-cost
SO-101 6-DoF arm: classic pipeline (calibrated camera + own IK) vs. imitation learning (ACT).

## Status
- [x] Phase 1: Setup (WSL2, ROS 2 Jazzy, MuJoCo, LeRobot)
- [ ] Phase 2: Simulation (own FK / Jacobian IK, scripted pick-and-place)
- [ ] Phase 3: ROS 2 (URDF/RViz, ros2_control, C++ safety monitor)
- [ ] Phase 4: Perception (camera + hand-eye calibration, detection, 3D pick poses)
- [ ] Phase 5: Hardware (real SO-101, classic pipeline, 20 trials)
- [ ] Phase 6: Imitation learning (ACT, 20 trials, comparison across 3 widths)
- [ ] Phase 7: Write-up, results, video
