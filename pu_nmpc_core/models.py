"""Mecanum robot model used for prediction in the NMPC.

The wheel controllers (STM32) accept wheel-speed set-points, so the prediction model is the Mecanum kinematics
plus a first-order lag of the body velocity (identified from step tests, see
identification.py). A Newton-Euler torque model would require wheel current/torque measurements and cannot be
identified with the available hardware; the paper states this choice.

State x = [X, Y, theta, vx, vy, omega] (vx, vy, omega in the body frame).
Input u = [vx_c, vy_c, omega_c] (commanded body velocity, mapped to wheel speeds by wheel_matrix).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class RobotParams:
    wheel_radius: float = 0.05          # m, mechanical measurement
    lx: float = 0.20                    # m, half wheelbase (centre -> wheel axle)
    ly: float = 0.22                    # m, half track width
    wheel_speed_max: float = 20.0       # rad/s, wheel-speed limit used in the constraints
    body_acc_max: tuple = (1.0, 1.0, 2.0)  # limit on |du|/dt for vx, vy, omega
    tau: tuple = (0.15, 0.15, 0.12)     # s, time constants of the velocity loops (identified)
    radius: float = 0.42                # m, radius of the circle enclosing the robot
    omega_max: float | None = None      # rad/s, body yaw-rate limit (Table 5: 1.5); None = wheel limits only
    # Defaults are for software tests only; real values come from config/robot.yaml.
    source: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)["robot"]
        src = {k: v.get("source", "") for k, v in d.items() if isinstance(v, dict)}
        val = {k: (v["value"] if isinstance(v, dict) else v) for k, v in d.items()}
        for k in ("body_acc_max", "tau"):
            if k in val:
                val[k] = tuple(val[k])
        return cls(**val, source=src)


def wheel_matrix(p: RobotParams) -> np.ndarray:
    """Wheel speeds [FL, FR, RL, RR] = W @ [vx, vy, omega] (X-roller convention seen from above)."""
    L = p.lx + p.ly
    return np.array([[1.0, -1.0, -L],
                     [1.0, 1.0, L],
                     [1.0, 1.0, -L],
                     [1.0, -1.0, L]]) / p.wheel_radius


def body_from_wheels(p: RobotParams, w: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(wheel_matrix(p)) @ np.asarray(w, float)


def f_cont(x, u, p: RobotParams, lib=np):
    """State derivative; lib = numpy or casadi."""
    th, vx, vy, om = x[2], x[3], x[4], x[5]
    c, s = lib.cos(th), lib.sin(th)
    tau = p.tau
    out = [c * vx - s * vy,
           s * vx + c * vy,
           om,
           (u[0] - vx) / tau[0],
           (u[1] - vy) / tau[1],
           (u[2] - om) / tau[2]]
    if lib is np:
        return np.array(out)
    return lib.vertcat(*out)


def rk4(x, u, dt, p: RobotParams, lib=np):
    k1 = f_cont(x, u, p, lib)
    k2 = f_cont(x + dt / 2 * k1, u, p, lib)
    k3 = f_cont(x + dt / 2 * k2, u, p, lib)
    k4 = f_cont(x + dt * k3, u, p, lib)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
