# SO-101 DIN-Rail Terminal Block Picker

Vision-guided pick-and-place of DIN-rail terminal blocks in **three widths (5.2 / 8.2 / 12.2 mm)**
with the low-cost **SO-101** arm (5 arm joints + gripper, Feetech STS3215 servos, LeRobot ecosystem).
The project compares a **classic pipeline** (calibrated camera, my own kinematics and IK, MoveIt 2)
with **imitation learning** (ACT).

The robot is built as a **hardware-realistic digital twin** on a laptop: every software layer is
written so that it runs unchanged on a real arm, and the gap between ideal and realistic simulation is
measured. A real arm is an optional final phase. See [ROADMAP.md](ROADMAP.md).

![Scripted pick-and-place in MuJoCo](media/sim_pick_place.gif)

> **Status:** work in progress. All numbers below come from **simulation and synthetic images**
> unless marked **(real)** - the real-camera rows use real photos of a printed test sheet.

---

## Results so far

| Area | Result |
|---|---|
| Forward kinematics (own, NumPy) | Matches MuJoCo to **< 1e-9 m** on 500 random poses; matches ROS `robot_state_publisher` and the official URDF |
| Jacobian + damped-least-squares IK | Jacobian validated against `mj_jacSite` and finite differences |
| Scripted pick-and-place (ground-truth poses) | **60/60 blocks seated**: 20 episodes × 3 widths, ±1 cm / ±10° randomisation; max error 1.4 mm along / 0.7 mm across the rail |
| C++ safety monitor (ROS 2) | Overheat, sustained overload, under-voltage and telemetry dropout all stop motion; latched fault, reset only when safe; 9 gtest unit tests |
| Camera calibration **(real laptop webcam)** | **0.305 px** RMS reprojection error, 24 images (1 outlier rejected automatically) |
| Block detector, YOLO26n-OBB (synthetic test set, 300 images / 1383 blocks) | Recall **99.2 %**, precision **98.8 %**, width class correct **99.7 %**, centre error **0.12 mm** median (0.51 mm p95), rotation error **0.14°** median (0.59° p95), 5.9 ms per image on an RTX 3050 Ti |
| Pixel → 3D block pose (100 new scenes, 415 blocks, true camera pose) | Found **99.5 %** (1 false detection), width class correct **99.8 %**, xy error **0.12 / 0.46 mm** (median / p95), yaw error **0.12°** median (0.53° p95); **98.6 % graspable** (≤ 3 mm, ≤ 5°) |
| Eye-to-hand calibration (answer known in sim) | **20 views**, 0 outliers, 1.34 px RMS: camera position error **2.19 mm**, rotation error **0.345°** → block localisation error on the table **1.00 mm** mean (1.38 mm max) |
| **Camera-driven pick-and-place** (detector + hand-eye-calibrated camera + own IK) | **60/60 blocks seated** (20/20 per width), nothing about the blocks read from the simulator; perception error (xy) **0.69 mm** median, 1.01 mm max |
| **Real-camera test (real)**: printed 1:1 sheet, 12 blocks, threshold fixed before the photos | **Phone (31 photos): 84.9 % graspable**, fused class **100 %**, xy error **0.75 / 1.72 mm** (median / p95) · laptop webcam (28 photos): 51.5 % graspable, 1.15 / 3.08 mm · synthetic baseline of the same sheet: 91–92 % · 0 false detections after the size check |
| IK benchmark, 200 top-down poses (157 reachable) | **Own DLS IK:** 157/157 (100 %), 0.071 mm median error, 4.6 ms · **MoveIt 2 + pick_ik:** 154/157 (98.1 %), 0.563 mm, 7.5 ms, but fails in 52 ms vs 1046 ms |
| STS3215 servo model (from the datasheet) | DC motor + 1:345 gearbox: k_t 0.785 N·m/A, R 3.04 Ω, dry + viscous friction, reflected rotor inertia, 12-bit encoder, internal P controller, I²R heating; reproduces the datasheet's stall torque, no-load speed and rated current; overheat protection switches torque off at 70 °C |
| Virtual Feetech bus (pseudo-terminal, 1 Mbps wire timing, MuJoCo behind it) | Python client **502 Hz**, p99 5.8 ms · own **C++17 bus library** against it: **499 Hz, 0 timeouts**, p99 6.0 ms |
| **Own ros2_control driver** (`so101_hardware`) on the virtual bus | **100 Hz**, 2 bus transactions per cycle (1.2 ms wire time) · trajectory tracking **0.66° rms** while moving (max 2.2°), ≤ 0.72° at rest (gravity sag) · **MoveIt 2 Plan & Execute over the serial protocol** · bus loss → driver stops in 50 ms, safety monitor stops motion independently · 43 C++ (gtest) / 64 Python tests in total |

## Project status

- [x] **Phase 0, Setup:** WSL2 Ubuntu 24.04, ROS 2 Jazzy, MuJoCo, LeRobot, GPU rendering
- [x] **Phases 1–2, Simulation:** own FK, Jacobian, IK; workspace analysis; DIN-rail scene; scripted pick-and-place
- [x] **Phase 3, ROS 2:** URDF + RViz, ros2_control (mock hardware), C++ safety monitor
- [ ] **Phase 4, Perception**
  - [x] 4.1 Camera intrinsic calibration (Zhang's method, robust outlier rejection)
  - [x] 4.2 Synthetic, domain-randomised dataset + YOLO26-OBB detector
  - [x] 4.3 Pixel → 3D grasp pose (ray–plane back-projection)
  - [x] 4.4 Eye-to-hand calibration in simulation (with RANSAC and a view/noise study)
  - [x] 4.5a Pick-and-place driven by the camera instead of ground truth
  - [x] 4.5b MoveIt 2 configuration + benchmark against my own IK
  - [x] 4.2b Real-camera test with a printed 1:1 test sheet (laptop webcam and phone)
- [x] **Phase 7v, Virtual hardware**
  - [x] 7v.1 STS3215 servo physics model + Feetech protocol emulator
  - [x] 7v.2 Virtual serial port in real time, coupled to the MuJoCo arm
  - [x] 7v.3 C++17 Feetech bus library + command-line tool
  - [x] 7v.4 ros2_control hardware plugin, `hardware:=mock|feetech`, MoveIt on the virtual bus
- [ ] **Realism layer + sim-to-real gap study:** camera noise and latency, calibration errors, ideal vs realistic sim
- [ ] **Phase 5, Imitation learning:** ACT on sim demonstrations and on a public real SO-101 dataset, compared with the classic pipeline
- [ ] **Phase 6, Release:** Docker, GitHub Actions CI, video
- [ ] **Phase 8 (optional), Real SO-101 follower arm:** same driver with `hardware:=feetech`

---

## How it works

### 1. Kinematics: `sim/so101_sim/`
- **Forward kinematics** as a product of homogeneous transforms (fixed link offset × joint rotation
  via Rodrigues' formula), with the parameters read from the MJCF. The math is my own code.
- **Geometric Jacobian** (`J_v = z_i × (p_tcp − p_i)`, `J_w = z_i`) and **damped-least-squares IK**
  with step clamping, joint-limit clipping and random restarts.
- **5-DoF insight:** top-down grasps at any yaw are still possible, because the three pitch joints
  are parallel (3 position + 1 pitch + 1 yaw = 5 constraints).
- **Workspace-driven cell design:** I mapped the dexterous workspace for top-down grasps. The TCP
  can only reach about 8 cm above the base (wrist-flex limit), so the arm stands on an **8 cm riser**
  to gain about 15 cm of usable height.

### 2. Scripted pick-and-place: `sim/so101_sim/pick_place.py`
Approach, descent and retreat are **straight Cartesian lines**, with IK solved every 5 mm and a
smoothstep timing law. The gripper **pre-shapes** to the block width + 10 mm so the swinging jaw
can't hit neighbouring blocks, uses a **measured in-hand offset** (the fixed jaw sits 21 mm from
the TCP), and releases gently (open slightly, back off 2 mm, lift).
Debugging lessons are documented in the commits: contact logging found a sign error, and too few
solver iterations made blocks "walk" on the rail (a simulation artefact, not a controller bug).

### 3. ROS 2 stack (ROS 2 Jazzy): `ros2_ws/src/`
| Package | What it does |
|---|---|
| `so101_description` | URDF imported reproducibly from SO-ARM100, own `tcp_link` matching the sim TCP (the official frame differs by 2 cm), RViz launch |
| `so101_bringup` | ros2_control with `hardware:=mock` or `hardware:=feetech` (virtual or real bus), joint trajectory controller and gripper controller; `go_to_pose.py` (homing), `record_tracking.py` |
| `feetech_bus` | **C++17** Feetech STS protocol library: packets, checksums, sign-magnitude registers, raw `termios` serial port, timeouts, stale-reply handling; `feetech_cli` |
| `so101_hardware` | **C++** ros2_control `SystemInterface` plugin for the SO-101 on a Feetech bus (see section 10) |
| `so101_interfaces` | `ServoTelemetry` message: load, temperature and voltage per servo |
| `so101_safety` | **C++** safety monitor: debounced overload, temperature, voltage, fail-safe telemetry timeout; latched fault; stop-category-2-style stop through the controller manager; reset only when safe. The rules live in a ROS-free class covered by gtest |
| `so101_moveit_config` | MoveIt 2: SRDF with a generated collision matrix, OMPL planning, **pick_ik** as IK plugin, `compare_ik_moveit.py` benchmark client |

The fake telemetry node uses a first-order thermal model (temperature rise ∝ load²), so faults can
be injected and tested before any hardware exists.

### 4. Perception: `perception/so101_perception/`
- **Calibration:** pinhole + Brown-Conrady model, results saved in **ROS camera_info YAML** format.
  Bad photos are removed by a homography pre-filter and per-view outlier rejection.
- **Synthetic data:** 3000 images of the MuJoCo scene, rendered at 960 × 720, with **domain
  randomisation**: lighting, table colour, camera pose and FOV, block count, position, rotation and
  colour, arm pose and occlusions, blur and noise. Labels are exact **oriented boxes of each block's
  top face**, which give the grasp centre and angle directly.
- **Detector:** YOLO26n-OBB (2.4 M parameters), fine-tuned from pretrained weights, 60 epochs
  (about 2 h on a 4 GB laptop GPU).
- **Evaluation for grasping:** besides mAP, it reports recall, the width-class confusion matrix,
  and **centre error in mm and rotation error in degrees**, converted per block from its known 45 mm length.

![Synthetic training images with automatic labels](media/synthetic_examples.png)

### 5. Pixel → 3D pose: `pose_estimation.py`
Each detected box corner is turned into a **viewing ray** (undistorted with the calibrated intrinsics)
and intersected with the **plane of the block's top face**, at a known height above the table.
That gives the grasp centre, yaw and the **measured width in mm**. Using the wrong plane (the table
instead of the top face) would shift every block by about 7.5 mm, so the plane choice matters.

![Detections back-projected to 3D poses](media/pose_estimation_example.png)

### 6. Eye-to-hand calibration: `handeye.py`, `handeye_sim.py`
The arm carries a 7 × 4 checkerboard (15 mm squares) on a bracket through random poses in front of
the fixed camera. For each pose, FK gives the gripper pose `A_i` and solvePnP gives the board pose
`B_i`; the unknown camera pose `X` and board offset `Y` satisfy **`X B_i = A_i Y`**.
A closed-form start (Park–Martin) is refined with **Levenberg–Marquardt on reprojection error**,
and **RANSAC** rejects bad views. A study sweeps the number of views and the joint noise
(encoder / backlash error), and reports the error that matters: **block position error on the table**.
Low rotation diversity between views makes the problem degenerate, so the poses are chosen to vary orientation.

### 7. Camera-driven pick-and-place: `run_camera_pick_place.py`
Arm to a look pose → overhead image → detector → 3D pose with the **hand-eye-calibrated** camera →
own IK → pick-and-place. The block type comes from **two independent cues**: the detector's class and
the measured width (accepted if within ±1.5 mm of a nominal width). The simulator is only used to
score the result.

**Error budget:** with the true camera pose the detector + back-projection error is 0.12 mm (median);
with the *calibrated* camera it grows to 0.69 mm. So the **hand-eye calibration dominates** the
perception error (about 1 mm on the table), and it is still well inside the gripper's
tolerance (3 mm, 5°). This is where to improve first on a real arm: more views, more rotation diversity.
The median hides outliers: the worst pose estimate was 8.5 mm off (max over 415 blocks), which is
why the graspable rate (98.6 %) is reported as well.

### 8. MoveIt 2 vs own IK: `so101_moveit_config/`, `sim/scripts/ik_benchmark.py`
KDL, MoveIt's default IK plugin, fails on this arm because it expects full 6-DoF poses; **pick_ik**
handles the 5-DoF arm. Both solvers get the same 200 random top-down poses, and both are scored by
the same judge: my FK applied to each returned solution.
- 43 of 200 poses are unreachable for both (outside the dexterous workspace of a 5-DoF arm).
- My solver iterates to a tighter tolerance (0.07 mm vs 0.56 mm) and is faster when it succeeds.
- pick_ik fails **20× faster** (52 ms vs 1046 ms, my solver runs 10 restarts before giving up),
  which matters for reachability checks in grasp planning.

![MoveIt 2 Plan & Execute in RViz](media/moveit_rviz.png)

### 9. Real-camera test: `print_sheet.py`, `evaluate_sheet_test.py`, `analyze_sheet_errors.py`
A printed A4 sheet carries a 7 x 4 checkerboard and 12 block top views at known positions (4 per width).
Per photo: checkerboard + `solvePnP` (IPPE + LM, mirrored corner order detected) with the **real
intrinsics** gives the paper plane; detector + back-projection give each block's pose; the result is
scored against the printed layout. A **synthetic dry run** of the same sheet with the same camera model
is the baseline, and the detection threshold was **fixed from the dry run before any real photo**
(`experiments/sheet_test_decision.md`).

| | Laptop dry run | Laptop real | Phone dry run | Phone real |
|---|---|---|---|---|
| Found | 90.3 % | 54.8 % | 91.4 % | **85.2 %** |
| Detector class correct | 86.8 % | 46.7 % | 86.3 % | 89.3 % |
| Fused class (metric width) | 100 % | 100 % | 100 % | **100 %** |
| xy error median / p95 | 0.36 / 1.07 mm | 1.15 / 3.08 mm | 0.34 / 1.10 mm | **0.75 / 1.72 mm** |
| Graspable (<= 3 mm, <= 5 deg, right class) | 90.3 % | 51.5 % | 91.4 % | **84.9 %** |

What the test showed, and what was done about it:
- **Metric size check:** every detection is in millimetres, so boxes that cannot be a block
  (45 mm long, 5-12 mm wide) are rejected - it removed **all** checkerboard false detections and lost no block.
- **The detector's class is biased on a flat print:** every class error is "one size too small"
  (28/28 in the dry run), because a print lacks the side faces that made 3D blocks look wider in training.
  The **metric width** is unaffected and decides the class: 100 % correct in every run.
- **Image quality was most of the laptop gap:** with a phone the real-vs-synthetic gap shrank from
  -39 to -7 points; dark-grey blocks went from 0-46 % to 97-100 % found.
- **What is left is colour:** printed blue blocks are found in only 6-23 % (97-100 % in the dry run) -
  printer ink is not the rendered blue. Fix: wider colour randomisation in the training data.
- **Position error is mostly systematic** (phone: 0.77 mm per block, same in every photo, vs 0.59 mm
  scatter) and grows with the distance from the checkerboard. A best-fit print scale of about -0.7 % in
  y (the same direction with both cameras) explains 60 % of it - the printer, not the pipeline.
- **Calibration lesson:** with board photos only in the image centre, a full distortion model moved the
  image corners by 1600 px (extrapolation); `calibrate_camera.py` now checks the model over the whole
  image, and the phone uses `--dist-model k1` (9 px). The sheet result barely changed (planar target:
  board and blocks share the plane, so intrinsic errors largely cancel).

![Real phone photo: printed truth (green) and detections with their error](media/sheet_test_phone_example.png)

**Honest limitation:** a flat print cannot test block height, side faces, shadows or shiny plastic;
the detector, pose and hand-eye numbers above it come from simulation.

### 10. Virtual hardware: `sim/so101_sim/feetech/`, `feetech_bus`, `so101_hardware`
The goal: the software that would drive a real arm, tested byte for byte, without buying one.
```
MoveIt 2 / demo scripts
  -> JointTrajectoryController (ros2_control, 100 Hz)
  -> so101_hardware  (C++ plugin: SYNC_READ state, SYNC_WRITE goals)
  -> feetech_bus     (C++: Feetech STS protocol over a raw serial port)
  -> /tmp/so101_bus  (pseudo-terminal)      | a real arm: /dev/ttyACM0, same code
  -> Python emulator of 6 STS3215 servos (register map, protections, 1 Mbps wire timing)
  -> servo physics model -> torque on the MuJoCo joints
```
- **Servo model (7v.1):** each STS3215 is a DC motor behind a 1:345 gearbox. The motor constants come
  from the datasheet (stall torque, no-load speed, rated current), friction from the rated operating
  point, the reflected rotor inertia (J·N²) enters MuJoCo as joint armature. The internal P controller
  explains a load-dependent position error **e = τ_load / K_p**; I²R heating and the overload and
  overheat protections behave like the firmware's.
- **Virtual bus (7v.2):** a pseudo-terminal in raw mode, so bytes like 0x03 and 0x0D pass unchanged;
  every reply is delayed by its wire time at 1 Mbps; the physics runs against the wall clock.
  Latency and lost replies can be injected (`--latency-ms`, `--drop-rate`).
- **C++ bus library (7v.3):** no exceptions in the loop (status codes), `poll()` timeouts, and before each
  request the input is flushed, so a late reply to an earlier request is never taken as the answer.
- **ros2_control plugin (7v.4):** one control cycle = **one SYNC_READ** of registers 56–63 (position,
  velocity, load, voltage, temperature of all 6 servos, 0.98 ms) + **one SYNC_WRITE** of the goals
  (0.26 ms). Lifecycle: *configure* checks every servo; *activate* is a **safe start** (goal := present
  position, read back, then torque on); *deactivate* and bus errors leave the servos holding their last
  goal. Telemetry for the safety monitor is published from its own thread, so the control loop never
  waits on DDS. The joint calibration (`zero_steps`, `direction`) is a parameter per joint, so a real
  arm only needs its own values and `port:=/dev/ttyACM0`.

![Servo model against the datasheet, P-control sag and overheat protection](media/servo_model.png)

**Tracking over the virtual bus** (demo trajectory, 100 Hz): the error grows with joint speed
(following error of a P-controlled servo plus one cycle of delay), shoulder and elbow keep a small
static offset under gravity, and steps of 0.088° show the 12-bit encoder.
With mock hardware, the same plot is a flat zero line.

| Joint | rms error | max error | final error |
|---|---|---|---|
| shoulder_pan | 0.20° | 0.65° | 0.09° |
| shoulder_lift | 0.74° | 2.17° | 0.04° |
| elbow_flex | 0.83° | 2.11° | 0.72° |
| wrist_flex | 0.26° | 0.85° | 0.16° |
| wrist_roll | 0.45° | 1.38° | 0.00° |

![Reference vs measured joint angles over the virtual Feetech bus](media/ros2_control_feetech_tracking.png)

Run it (three terminals):
```bash
python sim/scripts/run_virtual_bus.py --viewer                              # venv: 6 virtual servos
ros2 launch so101_moveit_config demo.launch.py hardware:=feetech            # ROS: driver + MoveIt
ros2 run so101_bringup go_to_pose.py park                                    # homing after power-up
```

---

## Engineering notes

**Scoped launch includes.** `demo.launch.py` includes the bringup launch with `rviz:=false`.
Arguments of an included launch file are *not* scoped in ROS 2, so this also switched off the MoveIt
RViz. Fix: wrap the include in a `GroupAction`.

**WSL2 clock jumps (`TF_OLD_DATA`).** Symptom: `robot_state_publisher` warned "Moved backwards in
time", and MoveIt dropped TF data. Measured: 11–13 backward jumps of the Linux wall clock per minute,
about 9.2 s each. Cause: the Windows clock was off by −9.19 s, so Hyper-V time sync and
`systemd-timesyncd` (NTP) kept stepping the clock back and forth. Fix: Windows time resync with the
Windows Time service on auto start, `systemd-timesyncd` disabled (one clock master), clock source
`hyperv_clocksource_tsc_page`.

**Safe start needs a read-back.** A servo keeps its last goal in RAM even with torque off; switching
torque on would make it jump there. So the driver writes the *present* position as the goal first. But
SYNC_WRITE is a broadcast without a reply, so the goals are **read back** before torque goes on - and
write + read-back are repeated *together*, because if the write was lost, reading again never helps.
A test shows the hazard: without safe start the (fake) servo jumps 97°.

**Setup may retry, the control loop may not wait.** On the laptop the first start failed: one bus reply
took longer than the 5 ms timeout (the Python bus has a p99 of about 6 ms under WSL), and safe start
gave up correctly. Fix: start-up steps retry up to 3 times (nothing moves yet), the 100 Hz loop never
retries (a late cycle delays every controller; 5 lost replies in a row = 50 ms blind → stop), and the
reply timeout became a launch argument (`timeout_ms`, 10 ms for the virtual bus; a real USB adapter answers faster, so 5 ms there).
Checked with injected faults: 4 ms extra latency + 2 % lost replies → every start succeeds after 1 retry.

**One program per serial bus.** Two programs on one port take each other's replies, and both see random
timeouts. The driver, the CLI and the Python client lock the port with `flock()`; a second program gets
"in use by another program". The kernel drops the lock when the process ends, even after a crash
(`TIOCEXCL` would stay set while the virtual bus keeps its end of the pty open).

**A joint limit the encoder cannot show.** MoveIt refused to plan on the virtual bus: "Joint
'shoulder_lift' ... outside bounds by [-1.74567], should be in the range [-1.74533]". The arm rests on its
mechanical stop at -100°, and -1.74533 rad lies *between* two encoder steps (1.534 mrad each): the nearest
reading is step -1138 = -1.74567 rad. MoveIt does not tolerate any start state outside the limits. Fix:
the software limit has to contain every pose the arm can rest in, *plus the measurement resolution* →
lower limit -1.7500 rad (3 steps). Moving 'park' off the stop instead was rejected: the shoulder then
holds the arm at ~47 % PWM all the time (heat) and sags 1.3°. Mock hardware cannot show any of this.

**Stick-slip when unfolding.** After power-up the gripper lies on the table. A straight joint move to
'park' turns the wrist up first and presses the gripper into the table: friction holds it, the wrist servo
sits at 100 % duty for 1.6 s, then it breaks loose and jumps 0.66 rad - the trajectory controller aborts
on > 0.2 rad path error (a tracking error is a contact detector). `go_to_pose.py` therefore lifts the
shoulder first and only then moves to the pose, like a real robot unfolding.

---

## Repository layout
```
sim/            MuJoCo models (SO-101 from MuJoCo Menagerie + DIN-rail scenes) and so101_sim package
  so101_sim/    kinematics.py, ik.py, scene.py, pick_place.py
    feetech/    STS3215 servo model, Feetech protocol + emulator, virtual serial bus, MuJoCo coupling, client
  scripts/      demos (IK, scene viewer, pick-and-place runs), IK benchmark, virtual bus, servo model report
  tests/        pytest: FK, Jacobian/IK, scene, pick-and-place, URDF<->MJCF consistency, servo model, protocol, bus
perception/     so101_perception package: calibration, synthetic data, pose estimation, hand-eye
  scripts/      checkerboard PDF, capture (Windows), calibration, dataset generation, YOLO train/eval,
                real-camera sheet test (PDF, capture, phone photo preparation, evaluation, error analysis),
                pose evaluation, hand-eye calibration, camera-driven pick-and-place
  calibration/  camera and hand-eye calibration results (YAML)
  tests/        pytest: calibration, synthetic data, pose estimation, hand-eye
ros2_ws/src/    ROS 2 packages (description, bringup, interfaces, safety, moveit_config, feetech_bus, so101_hardware)
experiments/    trial logs (CSV) and benchmark results (JSON)
media/          GIFs and images for this README
ROADMAP.md      plan and status of all phases
```

## Quick start
Tested on Windows 11 + WSL2 (Ubuntu 24.04), Python 3.12, ROS 2 Jazzy, RTX 3050 Ti 4 GB.
```bash
# Python side (simulation, perception)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-lock.txt && uv pip install -e .
pytest -q                                                  # 64 tests
python sim/scripts/run_pick_place.py --viewer              # watch the scripted pick-and-place
python perception/scripts/run_camera_pick_place.py --episodes 20   # camera-driven pipeline

# ROS 2 side (separate terminal, without the venv)
source /opt/ros/jazzy/setup.bash && cd ros2_ws
colcon build --symlink-install && source install/setup.bash
colcon test --packages-select so101_safety feetech_bus so101_hardware   # C++ unit tests
ros2 launch so101_bringup bringup.launch.py                # robot + controllers + RViz (mock hardware)
ros2 launch so101_safety safety.launch.py                  # safety monitor + simulated telemetry
ros2 launch so101_moveit_config demo.launch.py             # MoveIt 2 + RViz (Plan & Execute)
```

**Virtual hardware** (venv terminal + ROS terminals):
```bash
python sim/scripts/run_virtual_bus.py --viewer                       # venv: virtual servos at /tmp/so101_bus
ros2 launch so101_moveit_config demo.launch.py hardware:=feetech     # ROS: own driver + MoveIt
ros2 launch so101_safety safety.launch.py fake_telemetry:=false      # ROS: safety monitor on real telemetry
ros2 run so101_bringup go_to_pose.py park                             # ROS: homing after power-up
ros2 run so101_bringup demo_trajectory.py                             # ROS: demo trajectory
```

**IK benchmark** (needs both environments):
```bash
python sim/scripts/ik_benchmark.py make                    # venv: poses + own IK
ros2 run so101_moveit_config compare_ik_moveit.py          # ROS, with demo.launch.py running
python sim/scripts/ik_benchmark.py report                  # venv: comparison table
```

## Tech stack
Python, NumPy, SciPy, MuJoCo · ROS 2 Jazzy, ros2_control (own hardware plugin, pluginlib), MoveIt 2
(OMPL, pick_ik), rclcpp (C++17), gtest · Feetech STS serial protocol, POSIX termios / pseudo-terminals · OpenCV · Ultralytics YOLO26-OBB, PyTorch · LeRobot (ACT, coming next) · WSL2

## Credits and licences
- SO-101 URDF and meshes: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0)
- SO-101 MJCF: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0)
- Ultralytics YOLO: AGPL-3.0
- MoveIt 2 and pick_ik: BSD-3-Clause
- Code in this repository: see [LICENSE](LICENSE)
