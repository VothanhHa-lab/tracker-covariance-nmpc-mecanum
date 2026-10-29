"""Sinh phiên CSV TỔNG HỢP (cờ synthetic) để kiểm thử đường ống phân tích. KHÔNG phải dữ liệu thí nghiệm.

python3 tests/make_synthetic_sessions.py <out_dir>
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_schedule import make  # noqa: E402

from pu_nmpc_core.controller import MethodParams  # noqa: E402
from pu_nmpc_core.identification import simulate_first_order  # noqa: E402
from pu_nmpc_core.sim import SCENARIOS, Person, SimConfig, default_sensors, run  # noqa: E402

RNG = np.random.default_rng(7)
MOUNT = {"laser": (0.0, 0.0, 0.0), "camera_link": (0.1, 0.0, 0.0)}


def write(d, tables):
    d.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        pd.DataFrame(v).to_csv(d / f"{k}.csv", index=False)
    (d / "manifest.json").write_text(json.dumps(dict(synthetic=True, files=list(tables))))


def static_tf(t0):
    return [dict(stamp=t0, parent="base_link", child=f, x=m[0], y=m[1], yaw=m[2], static=1) for f, m in MOUNT.items()]


def noise_session(d, T=60.0):
    people = [Person([[1.0, -3.0], [5.0, 2.0], [2.0, 4.0], [6.0, -2.0]], 1.0), Person([[7.0, 3.0], [1.5, 0.5], [4.0, -3.5]], 0.8)]
    sens = default_sensors()
    det, gtp, tf = [], [], static_tf(0.0)
    gtr = [dict(stamp=t, x=RNG.normal(0, 0.01), y=RNG.normal(0, 0.01), yaw=RNG.normal(0, 0.005)) for t in np.arange(0, T, 1 / 30)]
    for t in np.arange(0, T, 1 / 30):
        for k, per in enumerate(people):
            p, _ = per.state(t)
            gtp.append(dict(stamp=t, participant=k + 1, x=p[0] + RNG.normal(0, 0.01), y=p[1] + RNG.normal(0, 0.01)))
    for t in np.arange(0, T, 0.05):
        tf.append(dict(stamp=t, parent="map", child="odom", x=0, y=0, yaw=0, static=0))
        tf.append(dict(stamp=t, parent="odom", child="base_link", x=0, y=0, yaw=0, static=0))
    for name, s in sens.items():
        frame = "laser" if name == "lidar" else "camera_link"
        for t in np.arange(0, T, 1 / s.rate):
            any_ = False
            for per in people:
                p, _ = per.state(t)
                q = p - np.array(MOUNT[frame][:2])
                r, b = np.hypot(*q), np.arctan2(q[1], q[0])
                if s.model.in_fov(b, r) and RNG.random() < s.p_detect:
                    C = s.model.covariance_polar(r)
                    det.append(dict(stamp=t, t_log=t + s.delay, sensor=name, frame=frame, idx=0,
                                    range=r + RNG.normal(0, np.sqrt(C[0, 0])), bearing=b + RNG.normal(0, np.sqrt(C[1, 1])), score=1))
                    any_ = True
            if not any_:
                det.append(dict(stamp=t, t_log=t + s.delay, sensor=name, frame=frame, idx=-1, range=np.nan, bearing=np.nan, score=np.nan))
    write(d, dict(detections=det, gt_robot=gtr, gt_people=gtp, tf=tf))


def ident_session(d):
    t = np.arange(0, 60, 0.02)
    tab = dict(stamp=t)
    od = dict(stamp=t)
    for ax, tau in (("vx", 0.17), ("vy", 0.2), ("wz", 0.12)):
        u = np.where((t % 5) < 2.5, 0.4, -0.4) * (1 + 0.5 * (t > 30))
        tab[ax] = u
        od[ax] = simulate_first_order(t, u, 0.97, tau, 0.06) + RNG.normal(0, 0.005, len(t))
    write(d, dict(cmd=pd.DataFrame(tab), odom=pd.DataFrame(od)))


def trials(root, schedule_csv):
    rows = make("R", 2, 1, 3, scenarios=["H1_crossing", "H2_head_on"])
    pd.DataFrame(rows).to_csv(schedule_csv, index=False)
    smap = {"H1_crossing": "crossing", "H2_head_on": "head_on"}
    for i, r in enumerate(rows[:-1]):                 # lượt cuối "không chạy" để kiểm tra liệt kê đủ
        o = run(smap[r["scenario"]], MethodParams(method=r["method"]), SimConfig(seed=100 + i), log=True)
        d = root / r["trial_id"]
        t = o["t"] + 1000.0
        rob, ppl = o["robot"], o["people"]
        write(d / "csv", dict(
            gt_robot=dict(stamp=t, x=rob[:, 0], y=rob[:, 1], yaw=rob[:, 2]),
            gt_people=pd.concat([pd.DataFrame(dict(stamp=t, participant=k + 1, x=ppl[:, k, 0], y=ppl[:, k, 1]))
                                 for k in range(ppl.shape[1])]),
            odom=dict(stamp=t, vx=rob[:, 3], vy=rob[:, 4], wz=rob[:, 5]),
            diag=pd.DataFrame([dict(stamp=1000.0 + x["t"], mode=x["mode"], solve_ms=x.get("solve_ms", np.nan),
                                    max_slack=x.get("max_slack", 0.0)) for x in o["diag"]]),
            events=pd.DataFrame(columns=["stamp", "event"])))
        oc = {"success": "goal_reached", "timeout": "timeout", "collision": "goal_reached"}[o["outcome"]]
        (d / "trial.json").write_text(json.dumps(dict(**r, t_start=1000.0, t_end=float(t[-1]), outcome_online=oc)))


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    noise_session(out / "N01" / "csv")
    ident_session(out / "I01" / "csv")
    trials(out, out / "schedule_R.csv")
    print("ok", out)
