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
| 4.2b | Real-camera test with printed 1:1 sheet — phone 84.9 % graspable, 0.75 mm; laptop 51.5 % | ✅ Done | – |
| 7v | Virtual hardware: servo model, Feetech emulator, C++ ros2_control driver — 100 Hz, 0.66° rms tracking, MoveIt on the virtual bus | ✅ Done (28 Nov, ahead of plan) | – |
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

## 4.2b — Real-camera perception test ✅

1. `make_test_sheet.py`: A4 PDF with a 7×4 checkerboard (15 mm) + 12 block top views (4 per width) + ground-truth JSON.
2. `evaluate_sheet_test.py --dry-run 30`: synthetic photos of the same sheet with the real webcam calibration (baseline).
3. Print at 100 % (colour), measure scale bar and square, tape flat.
4. `capture_sheet_images.py` (Windows): 15–20 photos at 1.8–2.4 px/mm (detector input), varied tilt and light.
5. `evaluate_sheet_test.py --images …`: PnP (IPPE + LM) → plane pose; detector + back-projection → errors vs printed truth.
6. Compare real vs dry run = sim-to-real gap of perception; README row + overlay image.

**Result:** phone 84.9 % graspable (xy 0.75 mm), laptop 51.5 % (1.15 mm), synthetic baseline 91–92 %.
Fused class 100 % everywhere; size check removes all false detections. Remaining gaps: printed blue
(colour randomisation needed, Realism phase) and ~0.8 mm systematic error from the print scale.
Lessons: pre-registered threshold, calibration coverage + distortion model check, binned phone modes.
**Be able to explain:** what a flat print tests (camera model, detector) and what it cannot (height, shadows, occlusion).

## 7v — Virtual hardware ✅

1. **7v.1 Servo physics model (STS3215):** DC motor + 1:345 gearbox from datasheet values
   (k_t 0.785 N·m/A, R 3.04 Ω), dry + viscous friction, backlash, reflected inertia, 12-bit encoder,
   internal P controller, I²R thermal model, overload/overheat protection. **Feetech protocol emulator:**
   real register map, PING / READ / WRITE / REG_WRITE / ACTION / SYNC_WRITE / SYNC_READ, EEPROM lock.
2. **7v.2 Virtual serial port:** pty in raw mode, wire timing at 1 Mbps, real-time physics, injectable
   latency and lost replies; its motors are the MuJoCo joints. Python client: 502 Hz, p99 5.8 ms.
3. **7v.3 C++17 bus library `feetech_bus` + `feetech_cli`:** status codes instead of exceptions, `poll()`
   timeouts, stale-reply handling; byte-compatible with the emulator: 499 Hz, 0 timeouts.
4. **7v.4 ros2_control plugin `so101_hardware`:** lifecycle, one SYNC_READ (registers 56–63) + one
   SYNC_WRITE per cycle, safe start with read-back, setup retries, one driver per bus (`flock`),
   telemetry thread for the safety monitor; bringup `hardware:=mock|feetech`, `timeout_ms`;
   homing tool `go_to_pose.py`; tracking recorder `record_tracking.py`.

**Result:** MoveIt 2 Plan & Execute over the serial protocol; 100 Hz, tracking 0.66° rms while moving
(max 2.2°); bus loss → driver stops in 50 ms, safety monitor stops motion on telemetry timeout;
43 C++ + 64 Python tests.
**Found on the way (none of it visible with mock hardware):** a 5 ms reply timeout was too tight for
the Python bus (p99 ~6 ms) → setup retries + 10 ms; a joint limit between two encoder steps made MoveIt
reject the resting arm → limit with 3 steps of margin; stick-slip of the gripper on the table during
unfolding → lift-off-first homing.
**Be able to explain:** half-duplex bus timing and the cycle budget (0.98 + 0.26 ms of 10 ms);
ros2_control lifecycle and read → update → write; why a P-controlled geared servo has a load-dependent
position error (e = τ_load / K_p); why setup may retry but the loop may not; what an emulator cannot prove
(real USB latency, compliance and backlash of real links, real friction).

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

Everything above transfers directly: the same `hardware:=feetech` driver, only with `port:=/dev/ttyACM0`,
`timeout_ms:=5` and the arm's own joint calibration (`zero_steps`, `direction` per joint).

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
3. Own C++ driver on the real arm (`feetech_cli scan` first, then bringup); MoveIt Plan & Execute on hardware.
4. Real hand-eye calibration, real picks; compare real vs realistic-sim numbers (validates the twin).
5. Optional: servo health monitoring (load, current, temperature, safe stop).

---

## Rules for this project

- All code written by me with help; no code, data or models from university projects (FlowDPG, RoboCup, Real2Sim).
- Every result is measured and reported with numbers; limitations stated honestly.
- Commit after every step; README row + clip after every phase.
