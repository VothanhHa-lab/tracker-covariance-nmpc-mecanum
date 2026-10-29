"""Nhận dạng tham số từ dữ liệu đo (đóng góp C3).

1. Khâu vận tốc bậc nhất có trễ thuần: y(t) = K·u(t − d) lọc qua 1/(τs + 1), cho từng trục thân.
   Ước lượng (K, τ, d) bằng bình phương tối thiểu trên lưới d, kiểm chứng trên đoạn riêng.
2. Mô hình nhiễu của cảm biến phát hiện người: sigma_r = a + b·r, sigma_b, độ lệch hệ thống
   theo khoảng cách. Dùng phép đo đã ghép với ground truth.
"""
from __future__ import annotations

import numpy as np


def simulate_first_order(t, u, K, tau, delay):
    y = np.zeros_like(u, dtype=float)
    ud = np.interp(t - delay, t, u, left=u[0])
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        a = np.exp(-dt / tau)
        y[i] = a * y[i - 1] + (1 - a) * K * ud[i - 1]
    return y


def fit_first_order(t, u, y, delays=np.arange(0.0, 0.31, 0.01), y0=None):
    """Khớp (K, τ, d) cho mô hình rời rạc y[i] = a·y[i−1] + b·u(t[i−1] − d).

    Giả thiết bước lấy mẫu gần đều; trả về dict có RMSE trên chính dữ liệu khớp.
    """
    t, u, y = map(lambda v: np.asarray(v, float), (t, u, y))
    dt = float(np.median(np.diff(t)))
    best = None
    for d in delays:
        ud = np.interp(t - d, t, u, left=u[0])
        A = np.column_stack([y[:-1], ud[:-1]])
        coef, *_ = np.linalg.lstsq(A, y[1:], rcond=None)
        a, b = coef
        if not 0 < a < 1:
            continue
        tau = -dt / np.log(a)
        K = b / (1 - a)
        yh = simulate_first_order(t, u, K, tau, d)
        rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = dict(K=float(K), tau=float(tau), delay=float(d), rmse=rmse)
    return best


def validate_first_order(t, u, y, fit):
    yh = simulate_first_order(np.asarray(t, float), np.asarray(u, float), fit["K"], fit["tau"], fit["delay"])
    e = yh - np.asarray(y, float)
    span = np.ptp(y) if np.ptp(y) > 0 else 1.0
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), nrmse=float(np.sqrt(np.mean(e ** 2)) / span))


def fit_sensor_noise(r_true, b_true, r_meas, b_meas, bins=np.arange(0.5, 8.01, 0.5)):
    """Mô hình nhiễu cực của một cảm biến.

    Trả về tham số (a, b, sigma_b) và bảng theo khoảng cách. Độ lệch chuẩn dùng ước lượng bền
    1.4826·MAD sau khi trừ độ lệch hệ thống của từng ô.
    """
    r_true, b_true, r_meas, b_meas = map(lambda v: np.asarray(v, float), (r_true, b_true, r_meas, b_meas))
    er = r_meas - r_true
    eb = (b_meas - b_true + np.pi) % (2 * np.pi) - np.pi
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r_true >= lo) & (r_true < hi)
        if m.sum() < 20:
            continue
        bias_r = np.median(er[m])
        sr = 1.4826 * np.median(np.abs(er[m] - bias_r))
        sb = 1.4826 * np.median(np.abs(eb[m] - np.median(eb[m])))
        rows.append(dict(r_lo=lo, r_hi=hi, n=int(m.sum()), bias_r=float(bias_r), sigma_r=float(sr),
                         sigma_b=float(sb)))
    if len(rows) >= 2:
        rc = np.array([(x["r_lo"] + x["r_hi"]) / 2 for x in rows])
        sr = np.array([x["sigma_r"] for x in rows])
        w = np.array([x["n"] for x in rows], float)
        b_, a_ = np.polyfit(rc, sr, 1, w=np.sqrt(w))
        a_, b_ = max(float(a_), 1e-3), max(float(b_), 0.0)
        sig_b = float(np.average([x["sigma_b"] for x in rows], weights=w))
    else:
        a_ = b_ = sig_b = np.nan
    return dict(sigma_r_a=a_, sigma_r_b=b_, sigma_b=sig_b, table=rows,
                n_total=int(len(r_true)))


def fit_cv_q(t, xy, r_meas, q_grid=np.geomspace(0.01, 5.0, 60)):
    """Ước lượng hợp lý cực đại mật độ phổ gia tốc q của mô hình CV từ quỹ đạo người (ground truth).

    t [T], xy [T,2] của MỘT người, r_meas: phương sai nhiễu đo của hệ ground truth (m²).
    Trả về q tốt nhất và log-likelihood trên lưới. Ghép nhiều người: cộng log-likelihood.
    """
    from .tracker import H, cv_matrices
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    R = r_meas * np.eye(2)
    ll = []
    for q in q_grid:
        x = np.r_[xy[0], 0.0, 0.0]
        P = np.diag([r_meas, r_meas, 1.0, 1.0])
        L = 0.0
        for i in range(1, len(t)):
            F, Q = cv_matrices(t[i] - t[i - 1], q)
            x, P = F @ x, F @ P @ F.T + Q
            S = H @ P @ H.T + R
            v = xy[i] - H @ x
            L += -0.5 * (v @ np.linalg.solve(S, v) + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ v
            P = (np.eye(4) - K @ H) @ P
        ll.append(L)
    ll = np.array(ll)
    return float(q_grid[int(np.argmax(ll))]), q_grid, ll
