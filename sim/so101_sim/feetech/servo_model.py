"""Physics model of a Feetech STS3215 servo (7.4 V, 1:345 gearbox), vectorised over N servos.

    supply V --> H-bridge (PWM duty d) --> DC motor --> 1:345 gearbox --(backlash)--> output shaft
                     ^                                                                 |
                     +---- internal PD position controller <---- 12-bit magnetic encoder +

Everything is expressed at the OUTPUT shaft:
  electrical   i  = (d * V - k_e * w) / R(T)              (inductance neglected: L/R << 1 ms)
  current limit |i| <= i_max                              (the driver's protection current)
  torque       tau_m = k_t * i
  mechanics    (J_rotor_reflected + J_load) * dw/dt = tau_m + tau_load - tau_friction
  thermal      C_th * dT/dt = i^2 R(T) - (T - T_amb) / R_th,   R(T) = R_25 (1 + 0.0039 (T - 25))

k_t, k_e and R are identified from three datasheet numbers (see ServoParams.from_datasheet).
Why two constants: for an ideal motor k_t = k_e (in SI). At the output, k_e grows with the gear
ratio N, but k_t only with eta * N - the gearbox loses torque, not speed. So eta = k_t / k_e.

Backlash (play between the gears) is modelled quasi-statically: the output shaft sits at the
end of the play that the transmitted torque pushes it to, and stays put inside the play while
almost no torque is transmitted. The encoder measures the OUTPUT, so the controller sees it.

Assumptions marked [ASSUMED] are not in the datasheet; they are plausible values to be
identified on real hardware (Phase 8): rotor inertia, friction, backlash, thermal values and the
scaling of the internal P/D gains.
"""
from dataclasses import dataclass

import numpy as np

from so101_sim.feetech.protocol import STEPS_PER_REV

KGCM = 0.0980665                      # 1 kg*cm in N*m
RAD_PER_STEP = 2 * np.pi / STEPS_PER_REV
COPPER_ALPHA = 0.0039                 # resistance rise of copper per kelvin


@dataclass(frozen=True)
class ServoParams:
    k_t: float = 0.0          # torque constant at the output [N*m/A]
    k_e: float = 0.0          # back-EMF constant at the output [V*s/rad]
    r_25: float = 0.0         # winding + driver resistance at 25 degC [ohm]
    i_max: float = 3.0        # driver current limit [A]
    j_reflected: float = 0.0  # rotor inertia seen at the output = J_rotor * N^2 [kg m^2]
    tau_coulomb: float = 0.0  # dry friction at the output [N*m]
    b_viscous: float = 0.0    # viscous friction at the output [N*m*s/rad]
    backlash: float = 0.0     # total play at the output [rad]
    r_th: float = 20.0        # thermal resistance winding -> ambient [K/W]       [ASSUMED]
    c_th: float = 4.0         # thermal capacity [J/K] (time constant r_th*c_th)  [ASSUMED]
    t_ambient: float = 25.0
    gear_ratio: float = 345.0

    @property
    def efficiency(self) -> float:
        return self.k_t / self.k_e

    @classmethod
    def sts3215_7v4(cls, **overrides):
        return cls.from_datasheet(**overrides)

    @classmethod
    def from_datasheet(cls, v_nominal=7.4, stall_kgcm=19.5, no_load_rpm=52.0, kt_kgcm_per_a=8.0,
                       rated_kgcm=5.0, rated_current=0.65, b_viscous=0.001,
                       rotor_inertia=0.8e-7, gear_ratio=345.0, backlash_deg=0.3, i_max=3.0, **overrides):
        """STS3215 C001 datasheet, 7.4 V: 19.5 kg*cm stall, ~52 rpm no-load, torque constant
        8 kg*cm/A, rated load 5 kg*cm at 650 mA.  rotor_inertia, the viscous share of the
        friction and the backlash are [ASSUMED].

        friction from the rated point:  k_t * i_rated = tau_rated + tau_friction
        R from the stall point (w = 0): tau_stall = k_t * V / R            -> R = k_t V / tau_stall
        k_e from the no-load point:     k_e w0 = V - R * tau_friction / k_t
        """
        k_t = kt_kgcm_per_a * KGCM
        tau_stall = stall_kgcm * KGCM
        r = k_t * v_nominal / tau_stall
        w0 = no_load_rpm * 2 * np.pi / 60
        tau_f_rated = k_t * rated_current - rated_kgcm * KGCM            # total friction at the rated point
        w_rated = w0 * (1 - rated_kgcm / stall_kgcm)                     # speed there (linear curve)
        tau_coulomb = tau_f_rated - b_viscous * w_rated
        tau_f = tau_coulomb + b_viscous * w0
        k_e = (v_nominal - r * tau_f / k_t) / w0
        p = dict(k_t=k_t, k_e=k_e, r_25=r, i_max=i_max, j_reflected=rotor_inertia * gear_ratio ** 2,
                 tau_coulomb=tau_coulomb, b_viscous=b_viscous, backlash=np.radians(backlash_deg),
                 gear_ratio=gear_ratio)
        p.update(overrides)
        return cls(**p)


# Internal controller scaling [ASSUMED]: P = 32 gives full duty at 32 steps (2.8 deg) error,
# D = 32 gives 0.26 duty per rad/s - together about critically damped with no load.
P_FULL_DUTY_ERROR_STEPS = 32.0
D_DUTY_PER_RAD_S = 0.26


class ServoArray:
    """N servos integrated together with a fixed time step (default 0.5 ms)."""

    def __init__(self, n: int, params: ServoParams | None = None, j_load=0.0, dt: float = 5e-4):
        self.p = params or ServoParams.sts3215_7v4()
        self.n, self.dt = n, dt
        self.j_load = np.broadcast_to(np.asarray(j_load, float), (n,)).copy()
        self.phi = np.zeros(n)             # motor-side angle, expressed at the output [rad]
        self.theta = np.zeros(n)           # output shaft angle [rad] (0 = encoder 2048)
        self.omega = np.zeros(n)           # [rad/s]
        self.current = np.zeros(n)         # [A]
        self.duty = np.zeros(n)            # PWM duty -1..1 (what the servo reports as "load")
        self.temperature = np.full(n, self.p.t_ambient)
        # register-like inputs
        self.goal_steps = np.full(n, STEPS_PER_REV // 2)
        self.torque_enable = np.zeros(n, bool)
        self.torque_limit = np.ones(n)     # 0..1 (Torque_Limit / 1000)
        self.kp = np.full(n, 32.0)         # P_Coefficient
        self.kd = np.full(n, 32.0)         # D_Coefficient
        self.dead_zone = np.ones(n)        # steps (CW/CCW_Dead_Zone)
        self.supply_voltage = 7.4

    # ---------------------------------------------------------------- measurements
    @staticmethod
    def to_steps(theta) -> np.ndarray:
        """Encoder reading for an output angle: 0..4095, 2048 = centre (quantised, wraps)."""
        return (np.floor(np.asarray(theta) / RAD_PER_STEP + 0.5).astype(int) + STEPS_PER_REV // 2) % STEPS_PER_REV

    @property
    def position_steps(self) -> np.ndarray:
        return self.to_steps(self.theta)

    @property
    def velocity_steps(self) -> np.ndarray:
        return np.round(self.omega / RAD_PER_STEP).astype(int)

    def set_position(self, theta) -> None:
        self.theta[:] = theta
        self.phi[:] = theta
        self.omega[:] = 0.0

    # ---------------------------------------------------------------- physics
    def drive_torque(self, theta, omega, dt: float) -> np.ndarray:
        """Controller + H-bridge + motor for a given output state -> motor torque at the output.

        Also integrates the winding temperature. Used by step() (own mechanics) and by the
        MuJoCo coupling (mechanics from the simulator, so theta/omega come from the joints).
        """
        p = self.p
        # the internal controller works on the QUANTISED encoder reading, like the real one
        err = self.goal_steps - self.to_steps(theta)
        err = np.where(err > STEPS_PER_REV // 2, err - STEPS_PER_REV, err)
        err = np.where(err < -STEPS_PER_REV // 2, err + STEPS_PER_REV, err)
        err = np.where(np.abs(err) <= self.dead_zone, 0, err)
        duty = (self.kp / 32.0) * err / P_FULL_DUTY_ERROR_STEPS - (self.kd / 32.0) * D_DUTY_PER_RAD_S * omega
        duty = np.clip(duty, -self.torque_limit, self.torque_limit)
        duty = np.where(self.torque_enable, duty, 0.0)
        r = p.r_25 * (1 + COPPER_ALPHA * (self.temperature - 25.0))
        i = (duty * self.supply_voltage - p.k_e * omega) / r
        i = np.where(self.torque_enable, np.clip(i, -p.i_max, p.i_max), 0.0)    # H-bridge off: coast
        self.temperature += dt * (i * i * r - (self.temperature - p.t_ambient) / p.r_th) / p.c_th
        self.current, self.duty = i, duty
        return p.k_t * i

    def step(self, tau_load=0.0, n_steps: int = 1) -> None:
        """Advance n_steps * dt with the model's own mechanics. tau_load: torque the load applies
        to the output [N*m] (e.g. gravity on the link: -m g r cos(theta))."""
        p, dt = self.p, self.dt
        tau_load = np.broadcast_to(np.asarray(tau_load, float), (self.n,))
        j = p.j_reflected + self.j_load
        for _ in range(n_steps):
            tau_m = self.drive_torque(self.theta, self.omega, dt)
            # friction: viscous + dry; a shaft at rest stays at rest until the drive beats dry friction
            drive = tau_m + tau_load - p.b_viscous * self.omega
            stuck = (np.abs(self.omega) < 1e-3) & (np.abs(drive) <= p.tau_coulomb)
            tau_net = np.where(stuck, 0.0, drive - p.tau_coulomb * np.sign(
                np.where(np.abs(self.omega) < 1e-3, drive, self.omega)))
            alpha = np.where(stuck, 0.0, tau_net / j)
            omega_new = self.omega + alpha * dt
            # dry friction must not reverse the motion within one step
            crossed = (np.sign(omega_new) != np.sign(self.omega)) & (np.abs(self.omega) >= 1e-3) \
                & (np.abs(drive) <= p.tau_coulomb)
            omega_new = np.where(stuck | crossed, 0.0, omega_new)
            self.phi += 0.5 * (self.omega + omega_new) * dt
            self.omega = omega_new
            # backlash: the transmitted torque decides which end of the play the output rests on
            tau_trans = self.j_load * alpha - tau_load
            half = p.backlash / 2
            lo, hi = self.phi - half, self.phi + half
            target = np.where(tau_trans > 1e-3, lo, np.where(tau_trans < -1e-3, hi, self.theta))
            self.theta = np.clip(target, lo, hi)

    # ---------------------------------------------------------------- helpers for tests and plots
    def steady_state_error_steps(self, tau_load: float) -> float:
        """Expected holding error under a constant load (P only, before the current limit):
        the controller must produce duty = tau_load * R / (k_t V) from an error e."""
        p = self.p
        duty = abs(tau_load) * p.r_25 / (p.k_t * self.supply_voltage)
        return duty * P_FULL_DUTY_ERROR_STEPS / (self.kp[0] / 32.0)
