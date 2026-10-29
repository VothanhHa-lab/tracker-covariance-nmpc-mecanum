"""Kiểm thử phần lõi (chạy: python3 tests/test_core.py). Không cần ROS."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pu_nmpc_core import constraint, identification, metrics  # noqa: E402
from pu_nmpc_core.models import RobotParams  # noqa: E402
from pu_nmpc_core.nmpc import NMPC, NMPCParams  # noqa: E402
from pu_nmpc_core.tracker import (Detection, DetectionBatch, SensorModel, Tracker,  # noqa: E402
                                  TrackerParams, to_map)

RNG = np.random.default_rng(0)
LIDAR = SensorModel("lidar", 0.03, 0.01, 0.01, None, 10.0, mount=(0.1, 0.05, 0.2))


def test_to_map_covariance():
    pose, pc = (1.0, 2.0, 0.7), np.diag([0.02 ** 2, 0.03 ** 2, 0.02 ** 2])
    det = Detection(4.0, 0.3)
    p, C = to_map(det, LIDAR, pose, pc)
    S = []
    for _ in range(20000):
        Cp = LIDAR.covariance_polar(4.0)
        d = Detection(4.0 + RNG.normal(0, np.sqrt(Cp[0, 0])), 0.3 + RNG.normal(0, np.sqrt(Cp[1, 1])))
        ps = tuple(np.array(pose) + RNG.multivariate_normal(np.zeros(3), pc))
        S.append(to_map(d, LIDAR, ps, np.zeros((3, 3)))[0])
    Cmc = np.cov(np.array(S).T)
    assert np.allclose(C, Cmc, rtol=0.1, atol=2e-4), (C, Cmc)


def _walk(T=12.0, rate=10.0):
    t = np.arange(0, T, 1 / rate)
    p = np.column_stack([1.0 + 0.8 * t, 3.0 + 0.3 * np.sin(0.4 * t)])
    return t, p


def _batches(t, p, sensor=LIDAR, name="lidar", seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for ti, pi in zip(t, p):
        q = pi - np.array(sensor.mount[:2])
        r, b = np.hypot(*q), np.arctan2(q[1], q[0]) - sensor.mount[2]
        C = sensor.covariance_polar(r)
        d = Detection(r + rng.normal(0, np.sqrt(C[0, 0])), b + rng.normal(0, np.sqrt(C[1, 1])))
        out.append(DetectionBatch(float(ti), name, [d], (0.0, 0.0, 0.0)))
    return out


def test_tracker_consistency_and_order():
    t, p = _walk()
    res = []
    for order in ("sorted", "shuffled"):
        tr = Tracker({"lidar": LIDAR}, TrackerParams(lag=0.35, q_acc=0.3))
        bs = _batches(t, p)
        errs, Ps = [], []
        for i in range(0, len(bs), 3):
            chunk = bs[i:i + 3]
            if order == "shuffled":
                chunk = chunk[::-1]            # đến ngược thứ tự trong cửa sổ trễ
            for b in chunk:
                tr.add(b)
            now = max(c.stamp for c in chunk) + 0.45
            tr.step(now)
            for e in tr.estimates(now):
                truth = np.interp(e["t"], t, p[:, 0]), np.interp(e["t"], t, p[:, 1])
                errs.append(e["x"][:2] - np.array(truth))
                Ps.append(e["P"][:2, :2])
        res.append((np.array(errs), np.array(Ps), tr.stats["late_dropped"]))
    (e1, P1, l1), (e2, P2, l2) = res
    assert l1 == 0 and l2 == 0
    assert np.allclose(e1, e2)                 # thứ tự đến không ảnh hưởng khi trễ < lag
    cov = metrics.ellipse_coverage(e1[20:], P1[20:], 0.95)
    assert 0.85 < cov <= 1.0, cov


def test_late_batch_dropped():
    tr = Tracker({"lidar": LIDAR}, TrackerParams(lag=0.1))
    bs = _batches(*_walk(2.0))
    tr.add(bs[5]); tr.step(bs[5].stamp + 0.2)
    tr.add(bs[2])
    assert tr.stats["late_dropped"] == 1


def test_kappa_gauss_and_cantelli():
    eps = 0.02
    k = constraint.kappa(eps, "gauss")
    z = RNG.normal(size=200000)
    assert abs(np.mean(z < -k) - eps) < 0.003
    kc = constraint.kappa(eps, "cantelli")
    assert kc > k
    # phân bố lệch, cùng phương sai: Cantelli vẫn là cận trên
    zz = (RNG.exponential(size=200000) - 1.0)
    assert np.mean(zz < -kc) <= eps and np.mean(-zz < -kc) <= eps


def test_chance_constraint_monte_carlo():
    eps, d0 = 0.05, 1.0
    ph, Ph = np.array([0.0, 0.0]), np.array([[0.09, 0.02], [0.02, 0.04]])
    Pr = np.diag([0.01, 0.01])
    n = np.array([1.0, 0.0])
    m, _ = constraint.cov_margins(n[None], Ph[None], Pr, d0, eps, "gauss")
    pr = ph + n * m[0]                         # robot đặt đúng trên biên ràng buộc
    eh = RNG.multivariate_normal(np.zeros(2), Ph, 200000)
    er = RNG.multivariate_normal(np.zeros(2), Pr, 200000)
    viol = np.mean(np.linalg.norm((pr + er) - (ph + eh), axis=1) < d0)
    assert viol <= eps + 0.003, viol


def test_heuristic_range():
    hp = dict(d_s=1.0, v_rel_max=2.0, w_d=0.5, w_v=0.3, w_t=0.2)
    vals = [constraint.heuristic_risk(RNG.normal(size=2), RNG.normal(size=2), RNG.normal(size=2),
                                      RNG.normal(size=2), RNG.uniform(-3, 3), RNG.uniform(-3, 3), hp)
            for _ in range(2000)]
    assert min(vals) >= 0.5 and max(vals) <= 1 / (1 + np.exp(-1)) + 1e-9


def test_first_order_fit():
    t = np.arange(0, 20, 0.02)
    u = np.where((t % 4) < 2, 0.5, -0.3)
    y = identification.simulate_first_order(t, u, 0.95, 0.17, 0.06) + RNG.normal(0, 0.005, len(t))
    f = identification.fit_first_order(t, u, y)
    assert abs(f["K"] - 0.95) < 0.03 and abs(f["tau"] - 0.17) < 0.03 and abs(f["delay"] - 0.06) < 0.021


def test_sensor_noise_fit():
    r = RNG.uniform(0.5, 8, 20000)
    b = RNG.uniform(-1, 1, 20000)
    rm = r + RNG.normal(0, 0.02 + 0.015 * r)
    bm = b + RNG.normal(0, 0.012, len(b))
    f = identification.fit_sensor_noise(r, b, rm, bm)
    assert abs(f["sigma_r_a"] - 0.02) < 0.01 and abs(f["sigma_r_b"] - 0.015) < 0.004
    assert abs(f["sigma_b"] - 0.012) < 0.002


def test_cv_q_fit():
    from pu_nmpc_core.tracker import cv_matrices
    q_true, dt = 0.3, 0.05
    F, Q = cv_matrices(dt, q_true)
    x, X = np.array([0, 0, 1.0, 0.0]), []
    for _ in range(1200):
        x = F @ x + RNG.multivariate_normal(np.zeros(4), Q)
        X.append(x[:2] + RNG.normal(0, 0.01, 2))
    q, *_ = identification.fit_cv_q(np.arange(1200) * dt, np.array(X), 1e-4)
    assert 0.15 < q < 0.6, q


def test_stats():
    assert abs(metrics.upper_bound_zero(20) - 0.1391) < 1e-3
    lo, hi = metrics.clopper_pearson(0, 20)
    assert lo == 0 and abs(hi - 0.1684) < 1e-3
    m, lo, hi, n = metrics.paired_bootstrap(np.ones(30), np.zeros(30))
    assert m == 1 and lo == 1 and hi == 1 and n == 30
    assert metrics.ospa([[0, 0]], [[0, 0]]) == 0 and metrics.ospa([], [[1, 1]], c=2.0) == 2.0


def test_nmpc_respects_halfspace():
    rp = RobotParams()
    mpc = NMPC(rp, NMPCParams(N=15, n_slots=1))
    ref = np.column_stack([np.linspace(0.1, 3.0, 15), np.zeros(15), np.zeros(15)])
    c = np.tile(np.array([1.5, 0.0]), (1, 15, 1))
    n = np.tile(np.array([0.0, 1.0]), (1, 15, 1))          # phải ở phía y ≥ 0.6
    m = np.full((1, 15), 0.6)
    r = mpc.solve(np.zeros(6), np.zeros(3), ref, c, n, m)
    assert r["ok"], r["status"]
    y = r["X"][1, 1:]
    assert np.all(y + r["max_slack"] >= 0.6 - 1e-4) and r["max_slack"] > 0   # đầu tầm: không khả thi → slack


def test_cluster_scan_two_legs():
    from pu_nmpc_core.detect import cluster_scan
    inc = np.deg2rad(0.5)
    ang = -np.pi + inc * np.arange(720)
    r = np.full(720, np.inf)
    for cx, cy in [(3.0, 0.12), (3.0, -0.12)]:        # hai chân bán kính 0.06 m tại x = 3 m
        for i, a in enumerate(ang):
            d = np.array([np.cos(a), np.sin(a)])
            b = d @ np.array([cx, cy]); c = cx ** 2 + cy ** 2 - 0.06 ** 2
            disc = b * b - c
            if disc > 0:
                r[i] = min(r[i], b - np.sqrt(disc))
    out = cluster_scan(r, -np.pi, inc, min_pts=2)
    assert len(out) == 1 and abs(out[0][0] - 2.95) < 0.08 and abs(out[0][1]) < 0.02, out


def test_bbox_range_bearing():
    from pu_nmpc_core.detect import bbox_to_range_bearing
    depth = np.full((480, 640), 5.0)
    depth[100:400, 400:500] = 3.0
    fx = fy = 600.0; cx, cy = 320.0, 240.0
    rb = bbox_to_range_bearing((400, 100, 500, 400), depth, fx, fy, cx, cy)
    X = (450 - cx) / fx * 3.0
    assert abs(rb[0] - np.hypot(X, 3.0)) < 1e-6 and rb[1] < 0          # bên phải → bearing âm


def test_fusion_no_duplicate_tracks():
    cam = SensorModel("camera", 0.05, 0.03, 0.02, np.deg2rad(43.5), 6.0, mount=(0.1, 0.0, 0.0))
    tr = Tracker({"lidar": LIDAR, "camera": cam}, TrackerParams(lag=0.2))
    t, p = _walk(8.0, rate=30.0)
    bl = _batches(t[::3], p[::3], LIDAR, "lidar", seed=2)
    bc = _batches(t[::2], p[::2], cam, "camera", seed=3)
    allb = sorted(bl + bc, key=lambda b: b.stamp)
    counts = []
    for b in allb:
        tr.add(b)
        tr.step(b.stamp + 0.25)
        counts.append(len(tr.estimates(b.stamp + 0.25)))
    assert max(counts[30:]) == 1, set(counts)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:300])
    sys.exit(1 if fails else 0)
