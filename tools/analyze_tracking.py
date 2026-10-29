"""P4: đánh giá bộ bám so với ground truth, chạy lại OFFLINE trên phát hiện đã ghi.

python3 tools/analyze_tracking.py <data>/N01/csv <data>/R001/csv ...
Cấu hình: fusion / lidar_only / camera_only × bù trễ (on/off). Vì chạy lại trên cùng phát hiện,
các ablation cảm biến không cần thêm lượt thử với người tham gia.
Độ phủ elip 95 % được đánh giá ở các tầm h = 0, 0.5, 1.0 s: đây là kiểm chứng trực tiếp giả thiết
của ràng buộc hiệp phương sai (C2).
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bag_to_csv import TFSeries  # noqa: E402
from common import RESULTS, ROOT, interp_pose, is_synthetic, load, provenance, write_json  # noqa: E402

from pu_nmpc_core import config  # noqa: E402
from pu_nmpc_core.metrics import ospa  # noqa: E402
from pu_nmpc_core.tracker import Detection, DetectionBatch, Tracker, cv_matrices  # noqa: E402

CONFIGS = {"fusion": ("lidar", "camera"), "lidar_only": ("lidar",), "camera_only": ("camera",)}
HORIZONS = (0.0, 0.5, 1.0)
GATE = 1.0
C95 = chi2.ppf(0.95, 2)


def gt_at(gp, t):
    out = {}
    for pid, g in gp.groupby("participant"):
        p = interp_pose(g.assign(yaw=0.0), np.array([t]))[0]
        if not np.isnan(p).any():
            out[pid] = p[:2]
    return out


def run_session(d, cfg_name, compensate, sensors_yaml, tracker_yaml, q_pred, rate=20.0):
    det, tf, gp, pc = load(d, "detections"), load(d, "tf"), load(d, "gt_people"), load(d, "pose_cov")
    if det.empty or tf.empty or gp.empty:
        return None
    tfs = TFSeries(tf)
    sensors = config.sensors(sensors_yaml)
    for s in sensors:
        fr = det[det.sensor == s].frame
        if len(fr):
            sensors[s].mount = tuple(tfs.lookup(["base_link", fr.iloc[0]], np.array([det.stamp.iloc[0]]))[0])
    tr = Tracker(sensors, config.tracker(tracker_yaml), enabled=CONFIGS[cfg_name])
    batches = []
    for (st, s), g in det.groupby(["stamp", "sensor"]):
        pose = tfs.lookup(["map", "odom", "base_link"], np.array([st]))[0]
        if np.isnan(pose).any():
            continue
        cov = np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2])
        if not pc.empty:
            r = pc[pc.stamp <= st].tail(1)
            if len(r):
                cov = np.array([[r.sxx.iloc[0], r.sxy.iloc[0], 0], [r.sxy.iloc[0], r.syy.iloc[0], 0], [0, 0, r.sth.iloc[0]]])
        dets = [Detection(x.range, x.bearing) for x in g.itertuples() if np.isfinite(x.range)]
        batches.append((g.t_log.iloc[0], DetectionBatch(float(st), s, dets, tuple(pose), cov)))
    batches.sort(key=lambda b: b[0])
    t0, t1 = batches[0][0], batches[-1][0]
    rows, i = [], 0
    last_match = {}
    for now in np.arange(t0 + 1.0, t1, 1.0 / rate):
        while i < len(batches) and batches[i][0] <= now:
            tr.add(batches[i][1]); i += 1
        tr.step(now)
        est = tr.estimates(now, compensate=compensate)
        truth = gt_at(gp, now)
        X = [e["x"][:2] for e in est]
        rows.append(dict(t=now, kind="ospa", ospa=ospa(X, list(truth.values()), c=1.0), n_gt=len(truth), n_est=len(est)))
        if not est or not truth:
            for pid in truth:
                rows.append(dict(t=now, kind="miss", participant=pid))
            continue
        pids = list(truth)
        D = np.array([[np.linalg.norm(e["x"][:2] - truth[p]) for p in pids] for e in est])
        r_, c_ = linear_sum_assignment(D)
        matched = set()
        for a, b in zip(r_, c_):
            if D[a, b] > GATE:
                continue
            e, pid = est[a], pids[b]
            matched.add(pid)
            sw = int(pid in last_match and last_match[pid] != e["id"])
            last_match[pid] = e["id"]
            row = dict(t=now, kind="match", participant=pid, id=e["id"], switch=sw)
            for h in HORIZONS:
                F, Q = cv_matrices(h, q_pred) if h > 0 else (np.eye(4), np.zeros((4, 4)))
                x, P = F @ e["x"], F @ e["P"] @ F.T + Q
                g = gt_at(gp, now + h).get(pid)
                if g is None:
                    continue
                err = x[:2] - g
                n2 = float(err @ np.linalg.solve(P[:2, :2], err))
                row[f"err_{h}"] = float(np.linalg.norm(err))
                row[f"nees_{h}"] = n2
                row[f"in95_{h}"] = int(n2 <= C95)
            rows.append(row)
        for pid in pids:
            if pid not in matched:
                rows.append(dict(t=now, kind="miss", participant=pid))
    df = pd.DataFrame(rows)
    df["session"], df["config"], df["compensate"] = str(d), cfg_name, int(compensate)
    return df


def summarize(df):
    m = df[df.kind == "match"]
    s = dict(n_match=int(len(m)), recall=len(m) / max(len(m) + (df.kind == "miss").sum(), 1),
             ospa=float(df[df.kind == "ospa"].ospa.mean()), id_switches=int(m.switch.sum()))
    for h in HORIZONS:
        if f"err_{h}" in m:
            s[f"rmse_{h}"] = float(np.sqrt(np.nanmean(m[f"err_{h}"] ** 2)))
            s[f"cov95_{h}"] = float(np.nanmean(m[f"in95_{h}"]))
            s[f"nees_{h}"] = float(np.nanmean(m[f"nees_{h}"]))
    return s


def cluster_bootstrap(df, key, n_boot=2000, seed=0):
    """CI theo cụm phiên: lấy mẫu lại các phiên, tránh coi các bước thời gian tương quan là độc lập."""
    rng = np.random.default_rng(seed)
    ses = df.session.unique()
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ses, len(ses))
        sub = pd.concat([df[df.session == s] for s in pick])
        vals.append(summarize(sub).get(key, np.nan))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def main(dirs, sensors_yaml, tracker_yaml, q_pred):
    synth = any(is_synthetic(d) for d in dirs)
    frames = []
    for d in dirs:
        for cfg in CONFIGS:
            for comp in (True, False):
                df = run_session(d, cfg, comp, sensors_yaml, tracker_yaml, q_pred)
                if df is not None:
                    frames.append(df)
    if not frames:
        print("không có dữ liệu")
        return
    all_ = pd.concat(frames)
    all_.to_csv(RESULTS / "tracking_steps.csv", index=False)
    out = dict(configs={})
    for (cfg, comp), g in all_.groupby(["config", "compensate"]):
        s = summarize(g)
        for key in ("rmse_0.0", "cov95_0.0", "cov95_1.0"):
            if key in s:
                s[key + "_ci"] = cluster_bootstrap(g, key, n_boot=500)
        out["configs"][f"{cfg}{'_comp' if comp else '_nocomp'}"] = s
    out["n_sessions"] = len(dirs)
    out["provenance"] = provenance(dirs, synthetic=synth, extra=dict(sensors_yaml=str(sensors_yaml),
                                                                     tracker_yaml=str(tracker_yaml), q_pred=q_pred))
    write_json(RESULTS / "tracking.json", out)
    for k, v in out["configs"].items():
        print(k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors-yaml", default=str(ROOT / "config" / "sensors.yaml"))
    ap.add_argument("--tracker-yaml", default=str(ROOT / "config" / "tracker.yaml"))
    ap.add_argument("--q-pred", type=float, default=0.5)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    a = ap.parse_args()
    main(a.dirs, a.sensors_yaml, a.tracker_yaml, a.q_pred)
