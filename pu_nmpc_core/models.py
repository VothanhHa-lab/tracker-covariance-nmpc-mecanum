"""Mô hình robot Mecanum dùng cho dự báo trong NMPC.

Bộ điều khiển bánh (STM32) nhận tốc độ bánh đặt, nên mô hình dự báo là động học Mecanum
cộng một khâu quán tính bậc nhất cho vận tốc thân (nhận dạng từ thí nghiệm bước, xem
identification.py). Mô hình mô-men Newton–Euler cần đo dòng/mô-men bánh nên không nhận dạng
được với phần cứng hiện có; bài báo phải nói rõ lựa chọn này.

Trạng thái x = [X, Y, theta, vx, vy, omega] (vx, vy, omega trong khung thân).
Đầu vào u = [vx_c, vy_c, omega_c] (vận tốc thân đặt, chuyển sang tốc độ bánh bằng wheel_matrix).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class RobotParams:
    wheel_radius: float = 0.05          # m, đo cơ khí
    lx: float = 0.20                    # m, nửa chiều dài cơ sở (tâm → trục bánh)
    ly: float = 0.22                    # m, nửa vệt bánh
    wheel_speed_max: float = 20.0       # rad/s, giới hạn tốc độ bánh dùng trong ràng buộc
    body_acc_max: tuple = (1.0, 1.0, 2.0)  # giới hạn |Δu|/dt cho vx, vy, omega
    tau: tuple = (0.15, 0.15, 0.12)     # s, hằng số thời gian vòng vận tốc (nhận dạng)
    radius: float = 0.42                # m, bán kính đường tròn bao robot
    omega_max: float | None = None      # rad/s, giới hạn tốc độ quay thân (Bảng 5: 1.5); None = chỉ giới hạn bánh
    # Các giá trị mặc định chỉ để chạy thử phần mềm; giá trị thật lấy từ config/robot.yaml.
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
    """Tốc độ bánh [FL, FR, RL, RR] = W @ [vx, vy, omega] (quy ước con lăn chữ X nhìn từ trên)."""
    L = p.lx + p.ly
    return np.array([[1.0, -1.0, -L],
                     [1.0, 1.0, L],
                     [1.0, 1.0, -L],
                     [1.0, -1.0, L]]) / p.wheel_radius


def body_from_wheels(p: RobotParams, w: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(wheel_matrix(p)) @ np.asarray(w, float)


def f_cont(x, u, p: RobotParams, lib=np):
    """Đạo hàm trạng thái; lib = numpy hoặc casadi."""
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
