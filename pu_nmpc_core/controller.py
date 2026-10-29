"""Bộ điều khiển: bộ bám → tham số ràng buộc theo phương pháp → NMPC → lệnh vận tốc.

Ba phương pháp dùng chung NMPC, chung chi phí và giới hạn bánh; chỉ khác cách tính biên:
- "cov"       (đề xuất): biên d0 + κ(ε)·σ_n từ hiệp phương sai dự báo của người và robot.
- "fixed"     : biên d0 + d_fixed, người dự báo bằng trung bình CV.
- "heuristic" : tái hiện CA-NMPC: d_safe = d0 + k_d·ρ_h và giới hạn tốc độ v0(1 − k_v·ρ_h).
  Khác bản gốc: bỏ hạng chi phí rào cản 1/(d − d_safe) vì không xác định khi d < d_safe;
  độ lệch này phải được nêu trong bài.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constraint import cov_margins, heuristic_risk, normals, propagate
from .models import RobotParams, rk4, wrap
from .nmpc import NMPC, NMPCParams


@dataclass
class MethodParams:
    method: str = "cov"
    d0: float = 0.95                 # m, khoảng cách tâm–tâm tối thiểu (r_robot + r_người + khoảng hở)
    eps: float = 0.01                # mức rủi ro mỗi người, mỗi bước
    kappa_kind: str = "gauss"
    d_fixed: float = 0.30            # m, đệm cố định của phương pháp "fixed"
    q_acc_pred: float = 0.5          # m²/s³, nhiễu quá trình khi dự báo người (nhận dạng từ dữ liệu GT)
    sigma_freeze_after: float = 1.0  # s; sau mốc này giữ nguyên σ (bảo đảm chỉ phát biểu cho [0, mốc])
    const_cov: tuple | None = None   # "ccconst": hiệp phương sai người hằng (4×4, làm phẳng), tính trước
    cov_inflate: float = 1.0         # hệ số phóng hiệp phương sai người (hiệu chỉnh trên hạt dev; 1 = không phóng)
    heuristic: dict = field(default_factory=lambda: dict(
        d_s=1.0, v_rel_max=2.0, w_d=0.5, w_v=0.3, w_t=0.2, k_d=0.8, k_v=0.5))
    max_people: int = 4
    max_static: int = 8
    static_clearance: float = 0.10   # m, khoảng hở với điểm tĩnh (cộng bán kính robot)
    static_range: float = 2.5        # m, chỉ xét điểm tĩnh trong bán kính này
    stale_after: float = 0.5         # s, dữ liệu người cũ hơn → dừng an toàn


@dataclass
class RefParams:
    v_ref: float = 0.6               # m/s
    heading_mode: str = "fixed"      # "fixed" (giữ hướng ban đầu) hoặc "tangent"
    goal_tol: float = 0.15           # m


class PathReference:
    def __init__(self, waypoints, rp: RefParams, heading0=0.0):
        self.w = np.asarray(waypoints, float)
        seg = np.diff(self.w, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s_cum = np.r_[0.0, np.cumsum(self.seg_len)]
        self.length = self.s_cum[-1]
        self.rp, self.heading0 = rp, heading0
        self.s = 0.0

    def point(self, s):
        s = np.clip(s, 0.0, self.length)
        i = min(np.searchsorted(self.s_cum, s, side="right") - 1, len(self.seg_len) - 1)
        a = (s - self.s_cum[i]) / max(self.seg_len[i], 1e-9)
        p = self.w[i] + a * (self.w[i + 1] - self.w[i])
        tang = np.arctan2(*(self.w[i + 1] - self.w[i])[::-1])
        return p, tang

    def project(self, p):
        best = (np.inf, self.s)
        for i in range(len(self.seg_len)):
            a, b = self.w[i], self.w[i + 1]
            t = np.clip(np.dot(p - a, b - a) / max(self.seg_len[i] ** 2, 1e-12), 0, 1)
            d = np.linalg.norm(a + t * (b - a) - p)
            s = self.s_cum[i] + t * self.seg_len[i]
            if d < best[0] and s >= self.s - 0.5:       # tiến đơn điệu, cho phép lùi nhỏ
                best = (d, s)
        self.s = max(self.s, best[1])
        return self.s

    def horizon(self, p, N, dt):
        s0 = self.project(p)
        ref = np.zeros((N, 3))
        for k in range(N):
            q, tang = self.point(s0 + self.rp.v_ref * dt * (k + 1))
            ref[k, :2] = q
            ref[k, 2] = tang if self.rp.heading_mode == "tangent" else self.heading0
        return ref

    def reached(self, p):
        return np.linalg.norm(p - self.w[-1]) < self.rp.goal_tol


class Controller:
    def __init__(self, rp: RobotParams, mp: MethodParams, np_: NMPCParams | None = None):
        np_ = np_ or NMPCParams(n_slots=mp.max_people + mp.max_static)
        assert np_.n_slots == mp.max_people + mp.max_static
        np_.n_people_slots = mp.max_people
        self.rp, self.mp, self.np = rp, mp, np_
        self.nmpc = NMPC(rp, np_)
        self.u_prev = np.zeros(3)
        self.pred = None

    def reset(self):
        self.nmpc.reset()
        self.u_prev = np.zeros(3)
        self.pred = None

    # --------------------------------------------------------------------------------------------
    def _slots(self, x, P_r, people, static_pts):
        mp, N, dt = self.mp, self.np.N, self.np.dt
        robot_pred = self.pred if self.pred is not None else np.tile(x[:2], (N, 1))
        c, n, m = self.nmpc.inactive_slots(robot_pred)
        info = dict(rho=[], sigma_n=[], speed_scale=1.0)
        people = sorted(people, key=lambda e: np.linalg.norm(e["x"][:2] - x[:2]))[:mp.max_people]
        speed_scale = 1.0
        for s, e in enumerate(people):
            P0 = np.reshape(mp.const_cov, (4, 4)) if mp.method == "ccconst" else e["P"]
            Cs, Ss = propagate(e["x"], P0, dt, N, mp.q_acc_pred)
            ns = normals(robot_pred, Cs)
            if mp.method in ("cov", "ccconst"):
                kf = max(1, min(N, int(round(mp.sigma_freeze_after / dt))))
                Ss = Ss * mp.cov_inflate
                Ss[kf:] = Ss[kf - 1]
                ms, sig = cov_margins(ns, Ss, P_r[:2, :2], mp.d0, mp.eps, mp.kappa_kind)
                info["sigma_n"].append(float(sig.max()))
            elif mp.method == "fixed":
                ms = np.full(N, mp.d0 + mp.d_fixed)
            elif mp.method == "heuristic":
                h = mp.heuristic
                v_world = np.array([np.cos(x[2]) * x[3] - np.sin(x[2]) * x[4],
                                    np.sin(x[2]) * x[3] + np.cos(x[2]) * x[4]])
                rho = heuristic_risk(x[:2], v_world, e["x"][:2], e["x"][2:], np.arctan2(v_world[1], v_world[0]),
                                     np.arctan2(e["x"][3], e["x"][2]), h)
                info["rho"].append(float(rho))
                ms = np.full(N, mp.d0 + h["k_d"] * rho)
                speed_scale = min(speed_scale, 1 - h["k_v"] * rho)
            else:
                raise ValueError(mp.method)
            c[s], n[s], m[s] = Cs, ns, ms
        # điểm tĩnh gần nhất từ LiDAR (tường, đồ vật), đã loại điểm thuộc người ở node ROS
        if static_pts is not None and len(static_pts):
            # điểm gần nhất trong mỗi cung góc quanh robot (tránh chọn toàn bộ điểm trên cùng một bức tường)
            rel = static_pts - x[:2]
            d = np.linalg.norm(rel, axis=1)
            sector = ((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (2 * np.pi) * mp.max_static).astype(int) % mp.max_static
            idx = []
            for sct in range(mp.max_static):
                m_ = np.where((sector == sct) & (d < mp.static_range))[0]
                if len(m_):
                    idx.append(m_[np.argmin(d[m_])])
            for j, i in enumerate(idx):
                s = mp.max_people + j
                Cs = np.tile(static_pts[i], (N, 1))
                c[s], n[s], m[s] = Cs, normals(robot_pred, Cs), self.rp.radius + mp.static_clearance
        info["speed_scale"] = speed_scale
        return c, n, m, speed_scale, info

    def step(self, t, x, P_r, people, static_pts, ref, people_data_age):
        """Trả về (u, diag). u là vận tốc thân đặt [vx, vy, omega]."""
        diag = dict(t=t, method=self.mp.method, mode="nmpc", n_people=len(people))
        if people_data_age > self.mp.stale_after:
            diag["mode"] = "stale_stop"
            u = self._brake()
        else:
            x = np.asarray(x, float).copy()
            # bù trễ chấp hành: lệnh trước đó vẫn đang được thực thi trong khoảng trễ
            if self.np.input_delay > 0:
                n_sub = max(1, int(round(self.np.input_delay / 0.01)))
                for _ in range(n_sub):
                    x = rk4(x, self.u_prev, self.np.input_delay / n_sub, self.rp)
            ref = ref.copy()
            ref[:, 2] = x[2] + wrap(ref[:, 2] - x[2])
            c, n, m, sv, info = self._slots(x, P_r, people, static_pts)
            r = self.nmpc.solve(x, self.u_prev, ref, c, n, m, speed_scale=sv)
            diag.update(info)
            diag.update(ok=r["ok"], status=r["status"], solve_ms=r["solve_ms"], iters=r["iters"],
                        max_slack=r["max_slack"])
            if r["ok"]:
                u = r["u"]
                self.pred = np.vstack([r["X"][:2, 2:].T, r["X"][:2, -1:].T])
            else:
                diag["mode"] = "solver_fail_brake"
                u = self._brake()
                self.nmpc.reset()
        self.u_prev = np.asarray(u, float)
        return self.u_prev.copy(), diag

    def _brake(self):
        a = np.array(self.rp.body_acc_max) * self.np.dt
        return self.u_prev - np.clip(self.u_prev, -a, a)
