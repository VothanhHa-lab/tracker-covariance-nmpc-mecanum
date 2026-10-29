"""Bộ bám người hợp nhất bất đồng bộ LiDAR + RGB-D (đóng góp C1).

- Mỗi lô phát hiện mang nhãn thời gian đo (stamp) của cảm biến, không phải thời điểm nhận.
- Các lô được giữ trong hàng đợi trễ cố định `lag` (≥ trễ lớn nhất của cảm biến) rồi xử lý theo
  đúng thứ tự stamp; như vậy phép đo đến muộn vẫn được dùng chính xác. Lô đến muộn hơn `lag`
  bị loại và được đếm (late_dropped).
- Ước lượng gửi cho bộ điều khiển được dự báo từ stamp đã xử lý tới thời điểm hiện tại, nên trễ
  cảm biến làm hiệp phương sai lớn lên đúng theo mô hình (bù trễ). Ablation "no_compensation"
  trả về ước lượng tại stamp cuối.
- Nhiễu đo theo tọa độ cực của từng cảm biến (sigma_r = a + b·r, sigma_b), nhận dạng từ dữ liệu
  ground truth (tools/analyze_sensor_noise.py); sai số tư thế robot được cộng vào qua Jacobian.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class SensorModel:
    name: str
    sigma_r_a: float            # m
    sigma_r_b: float            # m/m (tăng theo khoảng cách)
    sigma_b: float              # rad
    fov_half: float | None      # rad; None = 360°
    r_max: float                # m
    mount: tuple = (0.0, 0.0, 0.0)   # (x, y, yaw) của cảm biến trong base_link

    def covariance_polar(self, r):
        return np.diag([(self.sigma_r_a + self.sigma_r_b * r) ** 2, self.sigma_b ** 2])

    def in_fov(self, bearing, r):
        ok = r <= self.r_max
        if self.fov_half is not None:
            ok &= abs(bearing) <= self.fov_half
        return ok


@dataclass
class Detection:
    rng: float                  # m, trong khung cảm biến
    bearing: float              # rad, trong khung cảm biến


@dataclass
class DetectionBatch:
    stamp: float                         # s, thời điểm đo
    sensor: str
    detections: list
    robot_pose: tuple                    # (x, y, yaw) trong map tại stamp
    robot_cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # cov (x, y, yaw)


def to_map(det: Detection, sm: SensorModel, pose, pose_cov):
    """Phép đo → vị trí trong map và hiệp phương sai 2×2 (cảm biến + tư thế robot)."""
    xs, ys, yaws = sm.mount
    b = det.bearing + yaws
    q = np.array([xs + det.rng * np.cos(b), ys + det.rng * np.sin(b)])       # trong base_link
    J = np.array([[np.cos(b), -det.rng * np.sin(b)],
                  [np.sin(b), det.rng * np.cos(b)]])
    C_base = J @ sm.covariance_polar(det.rng) @ J.T
    x, y, th = pose
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = np.array([x, y]) + R @ q
    dth = np.array([-np.sin(th) * q[0] - np.cos(th) * q[1], np.cos(th) * q[0] - np.sin(th) * q[1]])
    C = R @ C_base @ R.T + pose_cov[:2, :2] + pose_cov[2, 2] * np.outer(dth, dth)
    return p, C


def cv_matrices(dt, q):
    I = np.eye(2)
    F = np.block([[I, dt * I], [np.zeros((2, 2)), I]])
    Q = q * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I], [dt ** 2 / 2 * I, dt * I]])
    return F, Q


H = np.hstack([np.eye(2), np.zeros((2, 2))])


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    t: float
    t_created: float
    hits: int = 1
    confirmed: bool = False
    last_update: float = 0.0
    camera_confirmed: bool = False
    n_updates: dict = field(default_factory=dict)

    def predict(self, t, q):
        dt = t - self.t
        if dt <= 0:
            return
        F, Q = cv_matrices(dt, q)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t


@dataclass
class TrackerParams:
    q_acc: float = 0.5          # m²/s³, mật độ phổ gia tốc trắng
    lag: float = 0.20           # s, cửa sổ sắp xếp lại (≥ trễ cảm biến lớn nhất)
    gate_chi2: float = 9.21     # 99 %, 2 bậc tự do
    init_vel_std: float = 1.0   # m/s
    confirm_hits: int = 3
    confirm_window: float = 1.0  # s
    delete_tentative: float = 0.5
    delete_confirmed: float = 1.5
    max_pos_std: float = 1.5    # m
    birth_sensors: tuple = ("lidar", "camera")
    merge_chi2: float = 5.99    # gộp hai vệt khi d² giữa hai vị trí < ngưỡng (95 %, 2 bậc tự do)


class Tracker:
    def __init__(self, sensors: dict, params: TrackerParams = TrackerParams(), enabled=None):
        self.sensors = sensors
        self.p = params
        self.enabled = set(enabled) if enabled is not None else set(sensors)
        self.tracks: list[Track] = []
        self._queue = []
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self.processed_until = -np.inf
        self.stats = dict(batches=0, late_dropped=0, disabled=0, updates=0, births=0, deletions=0)

    # ------------------------------------------------------------------ input
    def add(self, batch: DetectionBatch):
        if batch.sensor not in self.enabled:
            self.stats["disabled"] += 1
            return
        if batch.stamp <= self.processed_until:
            self.stats["late_dropped"] += 1
            return
        heapq.heappush(self._queue, (batch.stamp, next(self._seq), batch))

    def step(self, now: float):
        """Xử lý mọi lô có stamp ≤ now − lag theo thứ tự thời gian."""
        t_proc = now - self.p.lag
        while self._queue and self._queue[0][0] <= t_proc:
            _, _, b = heapq.heappop(self._queue)
            self._process(b)
            self.processed_until = b.stamp
        self._lifecycle(t_proc)

    # ------------------------------------------------------------------ core
    def _process(self, b: DetectionBatch):
        self.stats["batches"] += 1
        sm = self.sensors[b.sensor]
        Z = [to_map(d, sm, b.robot_pose, b.robot_cov) for d in b.detections]
        for tr in self.tracks:
            tr.predict(b.stamp, self.p.q_acc)
        # gán bằng Hungarian trên khoảng cách Mahalanobis, có cổng chi-bình phương
        nT, nZ = len(self.tracks), len(Z)
        assigned_z, in_gate = set(), set()
        if nT and nZ:
            cost = np.full((nT, nZ), 1e6)
            for i, tr in enumerate(self.tracks):
                S0 = H @ tr.P @ H.T
                for j, (z, R) in enumerate(Z):
                    S = S0 + R
                    v = z - H @ tr.x
                    d2 = float(v @ np.linalg.solve(S, v))
                    if d2 <= self.p.gate_chi2:
                        cost[i, j] = d2
                        in_gate.add(j)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                self._update(self.tracks[i], *Z[j], b)
                assigned_z.add(j)
        if b.sensor in self.p.birth_sensors:
            for j, (z, R) in enumerate(Z):
                # không sinh vệt mới từ phép đo đã nằm trong cổng của một vệt có sẵn (tránh vệt trùng)
                if j not in assigned_z and j not in in_gate:
                    self._birth(z, R, b)

    def _update(self, tr: Track, z, R, b):
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        I_KH = np.eye(4) - K @ H
        tr.P = I_KH @ tr.P @ I_KH.T + K @ R @ K.T          # dạng Joseph, giữ P đối xứng xác định dương
        tr.hits += 1
        tr.last_update = b.stamp
        tr.n_updates[b.sensor] = tr.n_updates.get(b.sensor, 0) + 1
        if b.sensor == "camera":
            tr.camera_confirmed = True
        if not tr.confirmed and tr.hits >= self.p.confirm_hits and b.stamp - tr.t_created <= self.p.confirm_window:
            tr.confirmed = True
        self.stats["updates"] += 1

    def _birth(self, z, R, b):
        P = np.zeros((4, 4))
        P[:2, :2] = R
        P[2:, 2:] = self.p.init_vel_std ** 2 * np.eye(2)
        tr = Track(id=next(self._ids), x=np.r_[z, 0.0, 0.0], P=P, t=b.stamp, t_created=b.stamp,
                   last_update=b.stamp, camera_confirmed=(b.sensor == "camera"), n_updates={b.sensor: 1})
        self.tracks.append(tr)
        self.stats["births"] += 1

    def _merge(self):
        """Gộp các vệt trùng (cùng một người do hai cảm biến khởi tạo gần như đồng thời)."""
        self.tracks.sort(key=lambda tr: (-tr.hits, tr.t_created))
        keep = []
        for tr in self.tracks:
            dup = False
            for k in keep:
                if abs(k.t - tr.t) > 1e-6:
                    continue
                v = tr.x[:2] - k.x[:2]
                S = tr.P[:2, :2] + k.P[:2, :2]
                if float(v @ np.linalg.solve(S, v)) < self.p.merge_chi2:
                    dup = True
                    k.camera_confirmed |= tr.camera_confirmed
                    break
            if dup:
                self.stats["merges"] = self.stats.get("merges", 0) + 1
            else:
                keep.append(tr)
        self.tracks = keep

    def _lifecycle(self, t):
        self._merge()
        keep = []
        for tr in self.tracks:
            age = t - tr.last_update
            lim = self.p.delete_confirmed if tr.confirmed else self.p.delete_tentative
            too_old = age > lim or (not tr.confirmed and t - tr.t_created > self.p.confirm_window)
            too_uncertain = np.sqrt(np.max(np.linalg.eigvalsh(tr.P[:2, :2]))) > self.p.max_pos_std
            if too_old or too_uncertain:
                self.stats["deletions"] += 1
            else:
                keep.append(tr)
        self.tracks = keep

    # ------------------------------------------------------------------ output
    def estimates(self, now: float, compensate: bool = True):
        """Danh sách ước lượng của các vệt đã xác nhận.

        compensate=True: dự báo tới `now` (bù trễ). False: giữ tại thời điểm vệt (ablation).
        """
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            x, P = tr.x.copy(), tr.P.copy()
            t_est = tr.t
            if compensate and now > tr.t:
                F, Q = cv_matrices(now - tr.t, self.p.q_acc)
                x, P = F @ x, F @ P @ F.T + Q
                t_est = now
            out.append(dict(id=tr.id, x=x, P=P, t=t_est, age=now - tr.last_update,
                            camera_confirmed=tr.camera_confirmed))
        return out
