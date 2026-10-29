"""Simulation environment of the simulation study (Sensors paper).

Unlike sim.py (software tests only), this module models:
- robot: Mecanum kinematics, first-order velocity loops with TRUE time constants drawn per seed
  (different from the controller model), actuation delay, saturation of wheel speeds and yaw rate;
- localisation: AR(1) pose error;
- people: waypoint following with Ornstein-Uhlenbeck velocity noise, random speed, optional avoidance of the robot
  (simple social force);
- sensors: 360-deg LiDAR and a camera with finite FOV, range limits, detection probability, clutter, delay,
  occlusion by obstacles (line of sight), range-dependent noise; static scan points of walls/obstacles;
- hard stop on the raw scan (as scan_stop_guard).
All parameters are in config/simstudy.yaml with their source (Table 5 / data sheet / assumed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .controller import Controller, MethodParams, PathReference, RefParams
from .models import RobotParams, rk4, wheel_matrix
from .nmpc import NMPCParams
from .tracker import Detection, DetectionBatch, SensorModel, Tracker, TrackerParams, cv_matrices

C95 = chi2.ppf(0.95, 2)
GAMMAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)   # inflation factors tested when calibrating coverage


# --------------------------------------------------------------------------------------------- geometry
def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))


def box(x0, y0, x1, y1):
    c = [np.array(v, float) for v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return [(c[i], c[(i + 1) % 4]) for i in range(4)]


@dataclass
class World:
    segments: list = field(default_factory=list)          # [(a, b)] walls and obstacle edges

    def visible(self, p, q):
        return not any(segments_intersect(p, q, a, b) for a, b in self.segments)

    def static_points(self, step=0.1):
        pts = []
        for a, b in self.segments:
            n = max(2, int(np.linalg.norm(b - a) / step) + 1)
            pts.extend(a + (b - a) * s for s in np.linspace(0, 1, n))
        return np.array(pts) if pts else np.zeros((0, 2))

    def min_dist(self, p):
        return min((seg_dist(p, a, b) for a, b in self.segments), default=np.inf)


# --------------------------------------------------------------------------------------------- pedestrians
@dataclass
class Walker:
    waypoints: np.ndarray
    speed: float
    t_start: float
    ou_sigma: float = 0.10
    ou_tau: float = 1.0
    reactive: bool = False
    sf_A: float = 3.0            # m/s^2, social-force strength towards the robot (assumed)
    sf_B: float = 0.5            # m, social-force range (assumed)
    y_bounds: tuple | None = None
    p: np.ndarray = None
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    w: np.ndarray = field(default_factory=lambda: np.zeros(2))
    i: int = 1
    active: bool = False
    done: bool = False

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, float)
        self.p = self.waypoints[0].copy()

    def step(self, t, dt, rng, robot_xy, r_sum):
        if t < self.t_start or self.done:
            self.v = np.zeros(2)
            return
        self.active = True
        tgt = self.waypoints[self.i]
        d = tgt - self.p
        if np.linalg.norm(d) < 0.3:
            self.i += 1
            if self.i >= len(self.waypoints):
                self.done = True
                self.v = np.zeros(2)
                return
            d = self.waypoints[self.i] - self.p
        v_des = self.speed * d / max(np.linalg.norm(d), 1e-9)
        # Ornstein-Uhlenbeck velocity noise around the desired velocity
        self.w += -self.w / self.ou_tau * dt + self.ou_sigma * np.sqrt(2 * dt / self.ou_tau) * rng.normal(size=2)
        # social-force model (Helbing & Molnar): relaxation to the desired velocity + anisotropic repulsion from the robot
        acc = (v_des + self.w - self.v) / 0.5
        if self.reactive:
            rel = self.p - robot_xy
            dist = np.linalg.norm(rel)
            if dist < 4.0:
                n = rel / max(dist, 1e-6)
                e = self.v / max(np.linalg.norm(self.v), 1e-6) if np.linalg.norm(self.v) > 0.05 else v_des / max(np.linalg.norm(v_des), 1e-6)
                cos_phi = float(-n @ e)                       # robot ahead -> cos_phi > 0
                lam = 0.35
                acc = acc + self.sf_A * np.exp((r_sum - dist) / self.sf_B) * (lam + (1 - lam) * (1 + cos_phi) / 2) * n
        self.v = self.v + acc * dt
        sp = np.linalg.norm(self.v)
        if sp > 1.8:
            self.v *= 1.8 / sp
        self.p = self.p + self.v * dt
        if self.y_bounds is not None:
            self.p[1] = np.clip(self.p[1], *self.y_bounds)


# --------------------------------------------------------------------------------------------- scenarios
def make_scenario(name, rng, reactive=False):
    """Return (world, robot waypoints, walkers, timeout). The robot starts at (0,0), heading +x."""
    sp = lambda m: float(np.clip(rng.normal(m, 0.15), 0.6, 1.6))  # noqa: E731
    w = World()
    if name == "H1_crossing":
        xc = rng.uniform(3.5, 5.0)
        s = sp(1.1)
        side = rng.choice([-1, 1])
        t0 = xc / 0.6 - 4.0 / s + rng.uniform(-1.0, 1.0)
        walkers = [Walker([[xc, -4.0 * side], [xc, 4.0 * side]], s, max(t0, 0.0), reactive=reactive)]
        return w, [[0, 0], [8, 0]], walkers, 40.0
    if name == "H2_head_on":
        # corridor 2.6 m wide (hospital corridors are typically >= 2.4 m)
        w.segments = [(np.array([-1.0, -1.3]), np.array([10.0, -1.3])), (np.array([-1.0, 1.3]), np.array([10.0, 1.3]))]
        y0 = rng.uniform(-0.3, 0.3)
        walkers = [Walker([[9.5, y0], [-1.5, y0]], sp(1.1), rng.uniform(0.0, 1.0), reactive=reactive, y_bounds=(-1.05, 1.05))]
        return w, [[0, 0], [8, 0]], walkers, 40.0
    if name == "H3_narrow":
        # corridor 3.0 m wide with a 1.6-m opening at x in [3.5, 4.5]; two people walk head-on through it
        H, h = 1.5, 0.8
        w.segments = []
        for sgn in (-1, 1):
            pts = [(-1.0, sgn * H), (3.5, sgn * H), (3.5, sgn * h), (4.5, sgn * h), (4.5, sgn * H), (10.0, sgn * H)]
            w.segments += [(np.array(a, float), np.array(b, float)) for a, b in zip(pts[:-1], pts[1:])]
        t1 = rng.uniform(0.0, 1.0)
        walkers = [Walker([[9.5, 0.25], [-1.5, 0.25]], sp(1.0), t1, reactive=reactive, y_bounds=(-1.25, 1.25)),
                   Walker([[9.5, -0.25], [-1.5, -0.25]], sp(1.0), t1 + rng.uniform(1.0, 2.5), reactive=reactive, y_bounds=(-1.25, 1.25))]
        return w, [[0, 0], [8, 0]], walkers, 45.0
    if name == "H4_occlusion":
        w.segments = box(3.6, 0.8, 4.6, 1.8)
        s = sp(1.1)
        t0 = 3.5 + rng.uniform(-0.5, 1.0)
        walkers = [Walker([[4.1, 2.6], [4.1, -3.5]], s, t0, reactive=reactive)]
        return w, [[0, 0], [8, 0]], walkers, 40.0
    if name == "H5_side":
        s = float(np.clip(rng.normal(1.4, 0.15), 1.0, 1.8))
        walkers = [Walker([[-2.5, 1.0 + rng.uniform(-0.2, 0.3)], [3.5, 0.9], [9.0, -2.5]], s, rng.uniform(0.0, 1.0), reactive=reactive)]
        return w, [[0, 0], [8, 0]], walkers, 40.0
    raise ValueError(name)


SCENARIOS = ["H1_crossing", "H2_head_on", "H3_narrow", "H4_occlusion", "H5_side"]


# --------------------------------------------------------------------------------------------- configuration
@dataclass
class SensorSim:
    model: SensorModel           # model used by the tracker (nominal)
    rate: float
    delay: float
    p_detect: float
    clutter: float               # mean number of false detections per scan
    noise_scale: float = 1.0     # TRUE noise = noise_scale x nominal model (robustness check)


@dataclass
class StudyConfig:
    dt_sim: float = 0.01
    actuation_delay: float = 0.05
    tau_true_range: tuple = (0.12, 0.25)
    loc_std: tuple = (0.03, 0.03, 0.01)
    loc_rho: float = 0.95
    r_person: float = 0.25
    guard_margin: float = 0.30
    footprint: tuple = (0.60, 0.55)
    reactive: bool = True            # nominal: people avoid the robot (social force); False = worst case
    compensate: bool = True
    sensors_enabled: tuple = ("lidar", "camera")
    delay_scale: float = 1.0
    eval_horizons: tuple = (0.0, 0.5, 1.0)


def run_episode(scenario, seed, mp: MethodParams, rp: RobotParams, npar: NMPCParams, refp: RefParams,
                tp: TrackerParams, sensors: dict, cfg: StudyConfig, ctrl: Controller | None = None,
                trace: dict | None = None):
    rng = np.random.default_rng(seed)
    world, path, walkers, timeout = make_scenario(scenario, rng, cfg.reactive)
    for wk in walkers:
        wk.reactive = cfg.reactive
    tau_true = tuple(rng.uniform(*cfg.tau_true_range, size=3))
    rp_true = RobotParams(**{**rp.__dict__, "tau": tau_true})
    ctrl = ctrl or Controller(rp, mp, npar)
    ctrl.mp = mp
    ctrl.reset()
    tracker = Tracker({k: s.model for k, s in sensors.items()}, tp, enabled=cfg.sensors_enabled)
    ref = PathReference(path, refp, heading0=0.0)
    static_all = world.static_points()
    contact_pts = world.static_points(step=0.02)          # wall contact checked against the rectangular footprint
    Lf, Wf = cfg.footprint
    W = wheel_matrix(rp_true)
    x = np.zeros(6)
    loc_err = np.zeros(3)
    P_r = np.diag(np.square(cfg.loc_std))
    next_scan = {k: rng.uniform(0, 1 / s.rate) for k, s in sensors.items()}
    pending, u_queue = [], []
    u_applied = np.zeros(3)
    last_ctrl = -np.inf
    t = 0.0
    log_t, log_r, log_p = [], [], []
    diags, cov_rows = [], []
    est_hist = []
    outcome = "timeout"
    guard_steps = 0
    r_sum = rp.radius + cfg.r_person
    while t < timeout:
        # estimated localisation: AR(1) error
        loc_err = cfg.loc_rho * loc_err + np.sqrt(1 - cfg.loc_rho ** 2) * rng.normal(0, cfg.loc_std)
        est_pose = x[:3] + loc_err
        robot_xy = x[:2]
        # sensors
        for k, s in sensors.items():
            if t + 1e-9 < next_scan[k]:
                continue
            next_scan[k] += 1.0 / s.rate
            sm = s.model
            dets = []
            c, sn = np.cos(x[2]), np.sin(x[2])
            sensor_xy = robot_xy + np.array([c * sm.mount[0] - sn * sm.mount[1], sn * sm.mount[0] + c * sm.mount[1]])
            for wk in walkers:
                if not wk.active or wk.done:
                    continue
                d = wk.p - robot_xy
                q = np.array([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]]) - np.array(sm.mount[:2])
                r, b = np.hypot(*q), np.arctan2(q[1], q[0]) - sm.mount[2]
                if r < 0.3 or not sm.in_fov(b, r) or not world.visible(sensor_xy, wk.p):
                    continue
                if rng.random() > s.p_detect:
                    continue
                C = sm.covariance_polar(r) * s.noise_scale ** 2
                dets.append(Detection(r + rng.normal(0, np.sqrt(C[0, 0])), b + rng.normal(0, np.sqrt(C[1, 1]))))
            for _ in range(rng.poisson(s.clutter)):
                r = rng.uniform(0.8, sm.r_max)
                b = rng.uniform(-np.pi, np.pi) if sm.fov_half is None else rng.uniform(-sm.fov_half, sm.fov_half)
                dets.append(Detection(r, b))
            pending.append((t + s.delay * cfg.delay_scale, DetectionBatch(t, k, dets, tuple(est_pose), np.diag(np.square(cfg.loc_std)))))
        for item in [p for p in pending if p[0] <= t + 1e-9]:
            tracker.add(item[1])
            pending.remove(item)
        # control at 1/dt Hz
        if t - last_ctrl >= npar.dt - 1e-9:
            last_ctrl = t
            tracker.step(t)
            est = tracker.estimates(t, compensate=cfg.compensate)
            age = t - tracker.processed_until if np.isfinite(tracker.processed_until) else (0.0 if t < 1.0 else np.inf)
            age = min(age, 0.0) if t < 0.5 else age
            # static points within LiDAR range and line of sight (map frame, from the estimated pose)
            if len(static_all):
                dd = np.linalg.norm(static_all - robot_xy, axis=1)
                cand = static_all[dd < 4.0]
                vis = [p for p in cand if world.visible(robot_xy, p + 0.02 * (robot_xy - p) / max(np.linalg.norm(robot_xy - p), 1e-9))]
                st = (np.array(vis) + (est_pose[:2] - robot_xy)) if vis else None
            else:
                st = None
            xe = np.r_[est_pose, x[3:]]
            rh = ref.horizon(xe[:2], npar.N, npar.dt)
            u, dg = ctrl.step(t, xe, P_r, est, st, rh, age)
            # hard stop on the raw scan
            L, Wd = cfg.footprint
            blocked = False
            c, sn = np.cos(x[2]), np.sin(x[2])
            obs = [wk.p for wk in walkers if wk.active and not wk.done]
            for p in obs:
                d = p - robot_xy
                qx, qy = c * d[0] + sn * d[1], -sn * d[0] + c * d[1]
                if abs(qx) < L / 2 + cfg.guard_margin + cfg.r_person and abs(qy) < Wd / 2 + cfg.guard_margin + cfg.r_person:
                    blocked = True
            if blocked:
                u = np.zeros(3)
                guard_steps += 1
                ctrl.u_prev = u
            dg["guard"] = blocked
            diags.append(dg)
            u_queue.append((t + cfg.actuation_delay, u))
            est_hist.append((t, [(e["x"].copy(), e["P"].copy()) for e in est]))
            if ref.reached(x[:2]):
                outcome = "success"
                break
        while u_queue and u_queue[0][0] <= t + 1e-9:
            u_applied = u_queue.pop(0)[1]
        # true saturation: wheel speeds and yaw rate
        wv = W @ u_applied
        sat = np.max(np.abs(wv)) / rp.wheel_speed_max
        ua = u_applied / sat if sat > 1 else u_applied.copy()
        ua[2] = np.clip(ua[2], -rp.omega_max, rp.omega_max) if rp.omega_max else ua[2]
        x = rk4(x, ua, cfg.dt_sim, rp_true)
        for wk in walkers:
            wk.step(t, cfg.dt_sim, rng, x[:2], r_sum)
        ppl = np.array([wk.p if wk.active and not wk.done else [np.nan, np.nan] for wk in walkers])
        log_t.append(t); log_r.append(x.copy()); log_p.append(ppl)
        dmin = np.nanmin(np.linalg.norm(ppl - x[:2], axis=1)) if np.isfinite(ppl).any() else np.inf
        if dmin < r_sum:
            speed = float(np.hypot(*ua[:2]))
            outcome = "contact_active" if speed > 0.1 else "contact_passive"
            break
        if len(contact_pts):
            cth, sth = np.cos(x[2]), np.sin(x[2])
            dd = contact_pts - x[:2]
            near = dd[np.abs(dd).max(axis=1) < 0.5]
            if len(near):
                qx = cth * near[:, 0] + sth * near[:, 1]
                qy = -sth * near[:, 0] + cth * near[:, 1]
                if np.any((np.abs(qx) < Lf / 2) & (np.abs(qy) < Wf / 2)):
                    outcome = "wall_contact"
                    break
        t += cfg.dt_sim
    T = np.array(log_t); R = np.array(log_r); Pp = np.array(log_p)
    d = np.linalg.norm(Pp - R[:, None, :2], axis=2)
    dmin_t = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    dt_ = np.diff(T, append=T[-1])
    sm_ = [g.get("solve_ms", np.nan) for g in diags if g.get("mode") == "nmpc"]
    v = R[:, 3:5]
    jerk = np.gradient(np.gradient(v, T, axis=0), T, axis=0) if len(T) > 5 else np.zeros((1, 2))
    res = dict(scenario=scenario, seed=seed, method=mp.method, outcome=outcome, time=float(T[-1]) if len(T) else np.nan,
               min_clearance=float(np.min(dmin_t) - r_sum) if np.isfinite(dmin_t).any() else np.nan,
               zone_time=float(np.sum(dt_[dmin_t < r_sum + 0.5])),
               path_len=float(np.sum(np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1))) if len(R) > 1 else 0.0,
               mean_jerk=float(np.mean(np.linalg.norm(jerk, axis=1))),
               solve_p50=float(np.nanmedian(sm_)) if sm_ else np.nan, solve_p95=float(np.nanpercentile(sm_, 95)) if sm_ else np.nan,
               solve_max=float(np.nanmax(sm_)) if sm_ else np.nan,
               max_slack=float(max((g.get("max_slack", 0.0) for g in diags), default=0.0)),
               frac_non_nmpc=float(np.mean([g.get("mode") != "nmpc" for g in diags])) if diags else 0.0,
               guard_steps=guard_steps, late_dropped=tracker.stats["late_dropped"],
               tau_true_x=tau_true[0], wall_min=float(min(world.min_dist(p) for p in R[::10, :2])) - rp.radius if world.segments else np.nan)
    # coverage of the predicted covariance against the true position (closed-loop tracker evaluation)
    def truth_at(tq):
        i = np.searchsorted(T, tq)
        return Pp[min(i, len(T) - 1)] if i < len(T) else None
    for h in cfg.eval_horizons:
        n_in = n = 0
        se = 0.0
        n_g = {g: 0 for g in GAMMAS}
        F, Q = cv_matrices(h, mp.q_acc_pred) if h > 0 else (np.eye(4), np.zeros((4, 4)))
        for te, es in est_hist[::2]:
            if not es:
                continue
            tru = truth_at(te + h)
            if tru is None or te + h > T[-1]:
                continue
            ok = ~np.isnan(tru[:, 0])
            if not ok.any():
                continue
            tr_ = tru[ok]
            X = np.array([F @ e[0] for e in es])
            D = np.linalg.norm(X[:, None, :2] - tr_[None], axis=2)
            ri, ci = linear_sum_assignment(D)
            for a, b in zip(ri, ci):
                if D[a, b] > 1.0:
                    continue
                Pk = F @ es[a][1] @ F.T + Q
                e = X[a, :2] - tr_[b]
                n += 1
                se += float(e @ e)
                nees = float(e @ np.linalg.solve(Pk[:2, :2], e))
                n_in += int(nees <= C95 * mp.cov_inflate)
                for g in GAMMAS:
                    n_g[g] += int(nees <= C95 * g)
        res[f"cov_n_{h}"] = n
        res[f"cov_in_{h}"] = n_in
        res[f"cov_se_{h}"] = se
        for g in GAMMAS:
            res[f"cov_in_{h}_g{g}"] = n_g[g]
    if trace is not None:   # time series for illustrative figures (does not affect results)
        trace.update(T=T, R=R, Pp=Pp, clearance=dmin_t - r_sum, walls=list(world.segments), diags=diags)
    return res
