# SO-101 DIN-Rail Picker — Roadmap

Vision-guided SO-101 arm that picks DIN-rail terminal blocks of different widths
(5.2 / 8.2 / 12.2 mm) and places them on a TS35 rail.

**Strategy:** build a *hardware-realistic digital twin* entirely on the laptop (€0).
Every software layer is written so that it runs unchanged on a real arm.
Real hardware is an **optional final phase** (Phase 8), only if time and budget allow.

Pace: ~8 h/week. After every phase: README result row + short GIF/clip + commit.

---

## Status overview

| Phase | Content | Status | Target |
|---|---|---|---|
| 0 | Setup: WSL2, GPU rendering, uv venv, Git/SSH, ROS 2 Jazzy | ✅ Done | – |
| 1 | MuJoCo model, forward kinematics, tests | ✅ Done | – |
| 2 | Jacobian, DLS IK, workspace analysis, DIN-rail scene, scripted pick & place | ✅ Done | – |
| 3 | ROS 2: URDF/xacro, TF, ros2_control (mock), JTC, custom msg, C++ safety node + gtest | ✅ Done | – |
| 4.1 | Real laptop-webcam calibration (Zhang, robust outlier rejection) | ✅ Done | – |
| 4.2 | Synthetic dataset + YOLO26n-OBB training/evaluation | ✅ Done | – |
| 4.3 | Block pose via ray–plane back-projection | ✅ Done | – |
| 4.4 | Eye-to-hand calibration (AX=XB, LM refinement, RANSAC, diversity study) | ✅ Done | – |
| 4.5a | Camera-guided pick & place in sim — 60/60 | ✅ Done | – |
| 4.5b | MoveIt 2 (pick_ik) + benchmark vs own IK — own 157/157, 0.07 mm; pick_ik 154/157, 0.56 mm | ✅ Done | – |
| 4.2b | Real-camera test with printed 1:1 sheet | 🟡 Code ready, dry run + print pending | 24 Nov – 13 Dec |
| 7v | Virtual hardware: servo model, Feetech emulator, C++ ros2_control driver | ⬜ | 14 Dec – 3 Jan |
| R | Realism layer + sim-to-real gap study | ⬜ | 4 – 17 Jan |
| 5 | ACT imitation learning (sim demos + real public SO-101 data) | ⬜ | 18 Jan – 7 Feb |
| 6 | Release: Docker, GitHub Actions CI, video, final README | ⬜ | 8 – 21 Feb |
| 8 | *Optional:* real SO-101 follower arm | ⬜ optional | from March, if possible |

Applications can start in **January** with the results from 4.x and 7v.

---

## 4.5b — MoveIt 2 benchmark ✅

1. Launch `ros2 launch so101_moveit_config demo.launch.py`; Plan & Execute in RViz; screenshot.
2. `python sim/scripts/ik_benchmark.py make`
3. `ros2 run so101_moveit_config compare_ik_moveit.py --repo ~/projects/so101-dinrail-picker` (with demo running)
4. `python sim/scripts/ik_benchmark.py report` → README table; commit.

**Result:** 157 of 200 poses reachable. Own DLS IK 157/157, 0.071 mm, 4.6 ms (fails in 1046 ms);
MoveIt 2 + pick_ik 154/157, 0.563 mm, 7.5 ms (fails in 52 ms).
Side findings: scoped launch include (`GroupAction`), WSL2 clock jumps fixed (Windows clock −9.19 s).
**Be able to explain:** why KDL fails on a 5-DoF arm; why failing IK is slow with restarts.

## 4.2b — Real-camera perception test

1. `make_test_sheet.py`: A4 PDF with a 7×4 checkerboard (15 mm) + 12 block top views (4 per width) + ground-truth JSON.
2. `evaluate_sheet_test.py --dry-run 30`: synthetic photos of the same sheet with the real webcam calibration (baseline).
3. Print at 100 % (colour), measure scale bar and square, tape flat.
4. `capture_sheet_images.py` (Windows): 15–20 photos at 1.8–2.4 px/mm (detector input), varied tilt and light.
5. `evaluate_sheet_test.py --images …`: PnP (IPPE + LM) → plane pose; detector + back-projection → errors vs printed truth.
6. Compare real vs dry run = sim-to-real gap of perception; README row + overlay image.

**Deliverable:** measured perception error with a real lens, noise and lighting.
**Be able to explain:** what a flat print tests (camera model, detector) and what it cannot (height, shadows, occlusion).

## 7v — Virtual hardware (the core of the "realistic" version)

1. **Servo physics model (STS3215):** DC motor + 1:345 gearbox from datasheet values
   (Kt, R, rated current), Coulomb + viscous friction, backlash, speed and torque limits,
   12-bit encoder quantization, internal P-controller, I²R thermal model, stall/overheat protection.
2. **Feetech protocol emulator:** Python, virtual serial port (pty), 6 IDs, real register map,
   PING / READ / WRITE / SYNC_WRITE, checksums, bus timing at 1 Mbps, injectable latency and packet loss.
   Its motors are the MuJoCo joints.
3. **C++ protocol library:** packet encode/decode + checksum, gtest unit tests.
4. **C++ ros2_control `SystemInterface` (`so101_hardware`):** lifecycle, SyncWrite goals,
   read position / load / temperature.
5. **Bringup:** launch argument `hardware:=mock|emulator|feetech`.
6. **Safety:** safety node reacts to emulator-injected over-temperature and comms timeouts.
7. **Integration test:** send a trajectory, check tracking within tolerance.

**Deliverable:** the same driver that would run a real arm, tested byte-for-byte against an emulator.
**Be able to explain:** half-duplex bus timing and the control-loop rate limit; ros2_control lifecycle;
why a P-controlled geared servo has a load-dependent position error (e = τ_load / K_p, plus backlash);
what an emulator cannot prove (real timing, compliance, gravity load on real links).

## R — Realism layer + sim-to-real gap study

1. Camera: real webcam intrinsics/distortion, noise, motion blur, 30–100 ms latency, lighting changes.
2. Calibration errors: joint offset (homing) errors, hand-eye errors.
3. Object variation: block mass and friction randomization; alignment tolerance for rail mounting.
4. Run the same tasks in **ideal** vs **realistic** sim.

| Test | Ideal sim | Realistic sim |
|---|---|---|
| Scripted pick & place success | 60/60 | ? |
| Placement error (mm) | ? | ? |
| ACT trained in ideal sim | ? | ? |
| ACT trained with randomization | ? | ? |

**Be able to explain:** which error source hurts most and why; domain randomization vs system identification.

## 5 — ACT imitation learning

1. Demos in the realistic sim: keyboard teleop through own IK + scripted demos with noise; LeRobot dataset format.
2. Train ACT on the RTX 3050 4 GB (small batch, mixed precision).
3. Evaluate: 50 episodes, ACT vs scripted planner; 25/50/100 demos; with/without randomization.
4. Real data: `lerobot/svla_so101_pickplace` (50 real SO-101 episodes; check licence first):
   replay real trajectories in the MuJoCo twin (real-to-sim) and train ACT offline (action-prediction error).
5. Failure analysis with clips (grasp, alignment, timing).

**Be able to explain:** action chunking and compounding error; behaviour cloning vs classical planning; demo quality vs quantity.

## 6 — Release

1. Dockerfile: ROS 2 Jazzy + MuJoCo + Python env; one command runs the demo.
2. GitHub Actions: pytest, colcon build + test, driver-vs-emulator test on every push.
3. 60–90 s video: scene → detection → pick & place → MoveIt → ACT → gap table.
4. Final README: results, architecture diagram, limitations, "ready for hardware" section.

---

## 8 — Optional: real SO-101 follower arm (only if time and budget allow)

Everything above transfers directly: `hardware:=feetech` replaces `hardware:=emulator`.

**Parts (follower only, ~€215–260 incl. shipping, prices from Nov 2026):**

| Part | Qty | Where |
|---|---|---|
| ST3215 / STS3215-C001 servo, 7.4 V, 1:345 | 6 (+1 spare) | DigiKey (DFRobot SER0069) or OpenELAB (Feetech C001) |
| Waveshare Bus Servo Adapter (A) | 1 | TinyTronics |
| 5 V 4 A power supply, 5.5/2.1 mm | 1 | TinyTronics |
| USB-C data cable, 2 table clamps, screwdriver set | 1 each | Amazon (official SO-101 BOM links) |
| Phoenix PT 2,5-TWIN / PT 6 / PT 16 N + 2× E/NS 35 N | 4 each | elektromax24.de |
| TS35 rail 1 m + USB webcam (C270) | 1 each | Galaxus.de |

**3D printing:** BITS SPACE Make (book via medienpraxis@uni-bielefeld.de, Mon–Fri 9–16); ask about filament cost.

**Steps (~4 weeks):**
1. Order parts, book printer slots, print the follower parts.
2. Set servo IDs one at a time, assemble, calibrate; first motion.
3. Own C++ driver on the real arm; MoveIt Plan & Execute on hardware.
4. Real hand-eye calibration, real picks; compare real vs realistic-sim numbers (validates the twin).
5. Optional: servo health monitoring (load, current, temperature, safe stop).

---

## Rules for this project

- All code written by me with help; no code, data or models from university projects (FlowDPG, RoboCup, Real2Sim).
- Every result is measured and reported with numbers; limitations stated honestly.
- Commit after every step; README row + clip after every phase.
