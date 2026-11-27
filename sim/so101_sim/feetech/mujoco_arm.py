"""Six emulated STS3215 servos driving the simulated SO-101 in MuJoCo.

Standalone, a servo needs an assumed load. Coupled to MuJoCo, the load is real physics:
gravity on every link, the inertia of the whole chain (which changes with the pose), contacts
with the table and the blocks. Each step (default 1 ms):

    joint angle q, velocity qd  (MuJoCo)  -->  encoder, controller, H-bridge, motor  (servo model)
    motor torque tau            (servo)   -->  joint torque actuator                 (MuJoCo)

How the servo maps onto MuJoCo:
  * the Menagerie position actuators (an ideal spring per joint) become plain TORQUE motors:
    the servo's own controller and motor model now decide the torque
  * joint `armature` = reflected rotor inertia (J_rotor * N^2): MuJoCo adds it to the joint's
    inertia exactly like a geared motor does
  * joint `frictionloss` = dry friction, `damping` = viscous friction; the back-EMF damping comes
    from the motor model itself (current depends on qd)
  * encoder: 0 rad = 2048 steps, one step = 2 pi / 4096
Backlash is not in the coupled model (it needs one extra degree of freedom per joint; MuJoCo
Menagerie has a prepared 'backlash' joint class for that) - left for the realism phase.
"""
from pathlib import Path

import mujoco
import numpy as np

from so101_sim.feetech.emulator import FeetechBusEmulator
from so101_sim.feetech.servo_model import ServoParams

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SERVO_IDS = (1, 2, 3, 4, 5, 6)                        # same order as LeRobot's SO-101
DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "models" / "dinrail_scene.xml"


class ServoArm:
    def __init__(self, scene=DEFAULT_SCENE, params: ServoParams | None = None, dt: float = 1e-3,
                 supply_voltage: float = 7.4, q0=None):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.m.opt.timestep = dt
        self.d = mujoco.MjData(self.m)
        self.bus = FeetechBusEmulator(SERVO_IDS, params, supply_voltage=supply_voltage, dt=dt)
        p = self.bus.model.p
        self.jnt = [self.m.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jnt])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.jnt])
        self.act = np.array([self.m.actuator(n).id for n in ARM_JOINTS])
        # position servos -> torque motors driven by the servo model
        self.m.actuator_gaintype[self.act] = mujoco.mjtGain.mjGAIN_FIXED
        self.m.actuator_gainprm[self.act, :] = 0.0
        self.m.actuator_gainprm[self.act, 0] = 1.0
        self.m.actuator_biastype[self.act] = mujoco.mjtBias.mjBIAS_NONE
        self.m.actuator_biasprm[self.act, :] = 0.0
        self.m.actuator_ctrllimited[self.act] = 1
        self.m.actuator_ctrlrange[self.act] = [-p.i_max * p.k_t, p.i_max * p.k_t]
        self.m.actuator_forcelimited[self.act] = 0
        self.m.dof_armature[self.dadr] = p.j_reflected
        self.m.dof_damping[self.dadr] = p.b_viscous
        self.m.dof_frictionloss[self.dadr] = p.tau_coulomb
        if q0 is not None:                                   # 5 arm joints (+ optional gripper)
            self.d.qpos[self.qadr[:len(q0)]] = q0
        mujoco.mj_forward(self.m, self.d)
        md = self.bus.model
        md.theta[:] = md.phi[:] = self.d.qpos[self.qadr]
        md.goal_steps[:] = md.to_steps(md.theta)
        for k, sid in enumerate(SERVO_IDS):                 # goal registers = where the arm is
            self.bus._put(self.bus.mem[sid], "Goal_Position", int(md.goal_steps[k]))
        self.bus._sync_registers_to_model()
        self.bus.physics_step = self._physics_step          # MuJoCo now does the mechanics

    def _physics_step(self, tau_load=None):
        md = self.bus.model
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.dadr]
        self.d.ctrl[self.act] = md.drive_torque(q, qd, self.m.opt.timestep)
        mujoco.mj_step(self.m, self.d)
        md.theta[:] = md.phi[:] = self.d.qpos[self.qadr]
        md.omega[:] = self.d.qvel[self.dadr]

    def advance(self, seconds: float):
        self.bus.advance(seconds)

    @property
    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def gravity_torque(self) -> np.ndarray:
        """Torque gravity puts on each servo right now (MuJoCo's bias force at zero velocity)."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        mujoco.mj_forward(self.m, d)
        return -d.qfrc_bias[self.dadr]
