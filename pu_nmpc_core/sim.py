"""Closed-loop simulation for SOFTWARE TESTING (software-in-the-loop).

Do NOT use results of this module as numbers in the paper. Its sensor parameters are assumed values
for testing; reported results come from simstudy.py campaigns or from data measured on the robot (tools/analyze_*.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import Controller, MethodParams, PathReference, RefParams
from .models import RobotParams, rk4
from .tracker import Detection, DetectionBatch, SensorModel, Tracker, TrackerParams


@dataclass
class SimSensor:
    model: SensorModel
    rate: float        # Hz
    delay: float       # s
    p_detect: float
    clutter: float     # mean number of false detections per scan


@dataclass
class Person:
    waypoints: np.ndarray
    speed: float
    t_start: float = 0.0

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, float)
        seg = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.s_cum = np.r_[0, np.cumsum(seg)]

    def state(self, t):
        s = np.clip((t - self.t_start) * self.speed, 0, self.s_cum[-1])
        i = min(np.searchsorted(self.s_cum, s, side="right") - 1, len(self.s_cum) - 2)
        a = (s - self.s_cum[i]) / max(self.s_cum[i + 1] - self.s_cum[i], 1e-9)
        p = self.waypoints[i] + a * (self.waypoints[i + 1] - self.waypoints[i])
        moving = 0 < (t - self.t_start) * self.speed < self.s_cum[-1]
        d = self.waypoints[i + 1] - self.waypoints[i]
        v = self.speed * d / max(np.linalg.norm(d), 1e-9) if moving else np.zeros(2)
        return p, v


def default_sensors():
    lidar = SimSensor(SensorModel("lidar", 0.03, 0.01, 0.01, None, 8.0), rate=10, delay=0.03,
                      p_detect=0.9, clutter=0.3)
    cam = SimSensor(SensorModel("camera", 0.05, 0.03, 0.02, np.deg2rad(43.5), 6.0, mount=(0.1, 0.0, 0.0)),
                    rate=15, delay=0.12, p_detect=0.95, clutter=0.0)
    return dict(lidar=lidar, camera=cam)


SCENARIOS = {
    # the robot goes from (0,0) to (8,0); the person follows a polyline
    "crossing": dict(path=[[0, 0], [8, 0]], people=[Person([[4.5, -4.0], [4.5, 4.0]], 1.0, 1.0)]),
    "head_on": dict(path=[[0, 0], [8, 0]], people=[Person([[8.5, 0.15], [-1.0, 0.15]], 1.0, 0.5)]),
    "side": dict(path=[[0, 0], [8, 0]], people=[Person([[2.0, 3.5], [3.8, 0.0], [3.8, -3.0]], 1.0, 0.5)]),
}


@dataclass
class SimConfig:
    dt_ctrl: float = 0.1
    t_max: float = 25.0
    actuation_delay: float = 0.05
    tau_true: tuple = (0.18, 0.18, 0.14)      # deliberate model mismatch
    loc_std: tuple = (0.02, 0.02, 0.01)       # m, m, rad
    r_person: float = 0.25
    compensate: bool = True
    sensors_enabled: tuple = ("lidar", "camera")
    seed: int = 0


def run(scenario: str, mp: MethodParams, cfg: SimConfig = SimConfig(), rp: RobotParams | None = None,
        sensors=None, log=False):
    rng = np.random.default_rng(cfg.seed)
    rp = rp or RobotParams()
    rp_true = RobotParams(**{**rp.__dict__, "tau": cfg.tau_true})
    sensors = sensors or default_sensors()
    sc = SCENARIOS[scenario]
    people = sc["people"]
    tracker = Tracker({k: s.model for k, s in sensors.items()}, TrackerParams(), enabled=cfg.sensors_enabled)
    ctrl = Controller(rp, mp)
    ref = PathReference(sc["path"], RefParams())
    x = np.zeros(6)
    P_r = np.diag(np.square(cfg.loc_std))
    pose_hist = []                       # (t, pose estimate)
    pending = []                         # (t_arrival, batch)
    u_queue = []                         # (t_apply, u)
    u_applied = np.zeros(3)
    next_scan = {k: 0.0 for k in sensors}
    dt_int = 0.01
    t = 0.0
    out = dict(t=[], robot=[], people=[], est=[], diag=[], outcome="timeout")
    last_ctrl = -np.inf
    while t < cfg.t_max:
        # sensors: measure at the TRUE pose, send the ESTIMATED pose at measurement time
        est_pose = x[:3] + rng.normal(0, cfg.loc_std)
        pose_hist.append((t, est_pose))
        for k, s in sensors.items():
            if t + 1e-9 >= next_scan[k]:
                next_scan[k] += 1.0 / s.rate
                dets = []
                for per in people:
                    p, _ = per.state(t)
                    d = p - x[:2]
                    c, sn = np.cos(x[2]), np.sin(x[2])
                    q = np.array([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]]) - np.array(s.model.mount[:2])
                    r, b = np.hypot(*q), np.arctan2(q[1], q[0]) - s.model.mount[2]
                    if s.model.in_fov(b, r) and rng.random() < s.p_detect:
                        C = s.model.covariance_polar(r)
                        dets.append(Detection(r + rng.normal(0, np.sqrt(C[0, 0])),
                                              b + rng.normal(0, np.sqrt(C[1, 1]))))
                for _ in range(rng.poisson(s.clutter)):
                    r = rng.uniform(0.5, s.model.r_max)
                    b = rng.uniform(-np.pi, np.pi) if s.model.fov_half is None else rng.uniform(-s.model.fov_half, s.model.fov_half)
                    dets.append(Detection(r, b))
                pending.append((t + s.delay, DetectionBatch(t, k, dets, tuple(est_pose), np.diag(np.square(cfg.loc_std)))))
        for item in [p for p in pending if p[0] <= t + 1e-9]:
            tracker.add(item[1])
            pending.remove(item)
        # control
        if t - last_ctrl >= cfg.dt_ctrl - 1e-9:
            last_ctrl = t
            tracker.step(t)
            est = tracker.estimates(t, compensate=cfg.compensate)
            age = 0.0 if tracker.processed_until > -np.inf and t - tracker.processed_until < 1.0 else 1.0
            xe = np.r_[est_pose, x[3:]]
            rh = ref.horizon(xe[:2], ctrl.np.N, ctrl.np.dt)
            u, dg = ctrl.step(t, xe, P_r, est, None, rh, age)
            u_queue.append((t + cfg.actuation_delay, u))
            if log:
                out["est"].append([(e["id"], e["x"].copy(), e["P"].copy()) for e in est])
                out["diag"].append(dg)
            if ref.reached(x[:2]):
                out["outcome"] = "success"
                break
        while u_queue and u_queue[0][0] <= t + 1e-9:
            u_applied = u_queue.pop(0)[1]
        x = rk4(x, u_applied, dt_int, rp_true)
        out["t"].append(t)
        out["robot"].append(x.copy())
        out["people"].append([per.state(t)[0] for per in people])
        dmin = min(np.linalg.norm(per.state(t)[0] - x[:2]) for per in people)
        if dmin < rp.radius + cfg.r_person:
            out["outcome"] = "collision"
            break
        t += dt_int
    out["t"] = np.array(out["t"])
    out["robot"] = np.array(out["robot"])
    out["people"] = np.array(out["people"])
    d = np.linalg.norm(out["people"] - out["robot"][:, None, :2], axis=2)
    out["min_dist"] = float(d.min())
    out["tracker_stats"] = dict(tracker.stats)
    return out
