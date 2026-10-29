"""Ràng buộc an toàn theo hiệp phương sai (đóng góp C2) và hai phương án đối chứng.

Mệnh đề (đặt trong bài, chứng minh ở phụ lục). Giả sử tại bước k vị trí thật của người
p_h ~ (p̂_h, P_h) và sai số vị trí robot e_r ~ (0, P_r) độc lập, n là vectơ đơn vị được chọn
chỉ từ các ước lượng (không phụ thuộc giá trị thật). Nếu vị trí robot dự kiến p_r thỏa
    n^T (p_r − p̂_h) ≥ d0 + κ(ε) · sqrt(n^T (P_h + P_r) n),
thì Pr(‖p_r,thật − p_h‖ < d0) ≤ ε, với
    κ = Φ^{-1}(1 − ε)            nếu các sai số là Gauss (kind="gauss"),
    κ = sqrt((1 − ε)/ε)          với mọi phân bố có hiệp phương sai đó (Cantelli, kind="cantelli").
Chứng minh: ‖a‖ ≥ n^T a, rồi áp dụng bất đẳng thức một phía cho biến vô hướng n^T(e_r − e_h).
Bảo đảm chỉ đúng tới mức hiệp phương sai của bộ bám là đúng (kiểm tra NEES/độ phủ trên dữ liệu
thật, tools/analyze_tracking.py) và mô hình chuyển động CV đúng trong tầm dự báo.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .tracker import cv_matrices


def kappa(eps: float, kind: str = "gauss") -> float:
    if not 0 < eps < 1:
        raise ValueError("eps phải trong (0, 1)")
    if kind == "gauss":
        return float(norm.ppf(1 - eps))
    if kind == "cantelli":
        return float(np.sqrt((1 - eps) / eps))
    raise ValueError(kind)


def propagate(x, P, dt, N, q):
    """Dự báo CV của người: tâm [N,2] và hiệp phương sai vị trí [N,2,2] tại k = 1..N."""
    F, Q = cv_matrices(dt, q)
    C, S = np.zeros((N, 2)), np.zeros((N, 2, 2))
    for k in range(N):
        x = F @ x
        P = F @ P @ F.T + Q
        C[k], S[k] = x[:2], P[:2, :2]
    return C, S


def steady_state_cov(dt, q, r_var, n_iter=2000):
    """Hiệp phương sai hậu nghiệm dừng của bộ lọc CV cập nhật vị trí mỗi dt với nhiễu đẳng hướng r_var (m²)."""
    F, Q = cv_matrices(dt, q)
    Hm = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = r_var * np.eye(2)
    P = np.eye(4)
    for _ in range(n_iter):
        Pm = F @ P @ F.T + Q
        K = Pm @ Hm.T @ np.linalg.inv(Hm @ Pm @ Hm.T + R)
        P = (np.eye(4) - K @ Hm) @ Pm
    return P


def normals(robot_pred, centers):
    """Pháp tuyến đơn vị từ người tới vị trí robot dự kiến (điểm tuyến tính hóa)."""
    d = robot_pred - centers
    n = np.linalg.norm(d, axis=1, keepdims=True)
    fallback = np.array([[1.0, 0.0]])
    return np.where(n > 1e-6, d / np.maximum(n, 1e-9), fallback)


def cov_margins(nvec, S, P_r, d0, eps, kind="gauss"):
    k = kappa(eps, kind)
    sig = np.sqrt(np.einsum("ki,kij,kj->k", nvec, S + P_r[None], nvec))
    return d0 + k * sig, sig


def heuristic_risk(robot_xy, robot_v, person_xy, person_v, robot_heading, person_heading, hp):
    """Chỉ số rủi ro của bản CA-NMPC, cài đúng như phương trình (27)–(34) để làm đối chứng.

    Lưu ý: với các trọng số không âm có tổng bằng 1 và các thành phần ≥ 0, sigmoid cho giá trị
    trong [0.5, σ(1)] ≈ [0.5, 0.73]; điều này được báo cáo nguyên trạng, không chỉnh công thức.
    """
    d = np.linalg.norm(np.asarray(person_xy) - np.asarray(robot_xy))
    rho_d = np.exp(-d / hp["d_s"])
    rho_v = min(np.linalg.norm(np.asarray(person_v) - np.asarray(robot_v)) / hp["v_rel_max"], 1.0)
    rho_t = (1 - np.cos(person_heading - robot_heading)) / 2
    z = hp["w_d"] * rho_d + hp["w_v"] * rho_v + hp["w_t"] * rho_t
    return 1.0 / (1.0 + np.exp(-z))
