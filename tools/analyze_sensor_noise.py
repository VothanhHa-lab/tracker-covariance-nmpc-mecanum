"""P2: mô hình nhiễu, tỉ lệ phát hiện, trễ hiệu dụng của từng cảm biến; q của người đi bộ.

python3 tools/analyze_sensor_noise.py --gt-std 0.03 <data>/N01/csv [...]
--gt-std: độ lệch chuẩn vị trí của hệ ground truth đo ở P1 (m), dùng khi ước lượng q.
Ghi results/sensor_noise.json, results/sensor_noise_bins.csv và config/sensors_identified.yaml
(không ghi đè config/sensors.yaml; người phụ trách sao chép sau khi kiểm tra).
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bag_to_csv import TFSeries  # noqa: E402
from common import RESULTS, ROOT, interp_pose, is_synthetic, load, provenance, write_json  # noqa: E402

from pu_nmpc_core import config  # noqa: E402
from pu_nmpc_core.identification import fit_cv_q, fit_sensor_noise  # noqa: E402
from pu_nmpc_core.metrics import clopper_pearson  # noqa: E402

MATCH = 0.7    # m, cổng ghép phát hiện ↔ GT


def truth_in_sensor(det, gt_r, gt_p, mount, tau=0.0, model=None):
    """Với mỗi lô phát hiện: vị trí thật của từng người trong khung cảm biến tại stamp − tau."""
    rows = []
    for st, g in det.groupby("stamp"):
        t = st - tau
        rp = interp_pose(gt_r, np.array([t]))[0]
        if np.isnan(rp).any():
            continue
        for pid, gp in gt_p.groupby("participant"):
            pp = interp_pose(gp.assign(yaw=0.0), np.array([t]))[0]
            if np.isnan(pp).any():
                continue
            c, s = np.cos(rp[2]), np.sin(rp[2])
            d = pp[:2] - rp[:2]
            q = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - np.array(mount[:2])
            cy, sy = np.cos(mount[2]), np.sin(mount[2])
            q = np.array([cy * q[0] + sy * q[1], -sy * q[0] + cy * q[1]])
            r_, b_ = np.hypot(*q), np.arctan2(q[1], q[0])
            if model is not None and not model.in_fov(b_, r_):
                continue                 # tỉ lệ phát hiện chỉ tính trong vùng nhìn danh định
            rows.append(dict(stamp=st, participant=pid, r_true=r_, b_true=b_))
    return pd.DataFrame(rows)


def match(det, tru):
    """Ghép tham lam theo khoảng cách Euclid trong khung cảm biến cho từng lô."""
    out, n_fp = [], 0
    for st, g in det.dropna(subset=["range"]).groupby("stamp"):
        tg = tru[tru.stamp == st]
        used = set()
        for _, d in g.iterrows():
            pd_ = np.array([d.range * np.cos(d.bearing), d.range * np.sin(d.bearing)])
            best, bj = MATCH, None
            for j, t in tg.iterrows():
                if j in used:
                    continue
                pt = np.array([t.r_true * np.cos(t.b_true), t.r_true * np.sin(t.b_true)])
                e = np.linalg.norm(pd_ - pt)
                if e < best:
                    best, bj = e, j
            if bj is None:
                n_fp += 1
            else:
                used.add(bj)
                t = tg.loc[bj]
                out.append(dict(stamp=st, participant=t.participant, r_true=t.r_true, b_true=t.b_true,
                                r_meas=d.range, b_meas=d.bearing))
    return pd.DataFrame(out), n_fp


def main(dirs, gt_std):
    models = config.sensors(ROOT / "config" / "sensors.yaml")
    synth = any(is_synthetic(d) for d in dirs)
    res, bins, ident = dict(sensors={}), [], {}
    q_ll, q_grid = None, None
    for sensor in ("lidar", "camera"):
        M, TR, fp, scans = [], [], 0, 0
        taus = []
        for d in dirs:
            det, gr, gp, tf = load(d, "detections"), load(d, "gt_robot"), load(d, "gt_people"), load(d, "tf")
            if det.empty or gr.empty or gp.empty:
                continue
            det = det[det.sensor == sensor]
            if det.empty:
                continue
            frame = det.frame.iloc[0]
            mount = tuple(TFSeries(tf).lookup(["base_link", frame], np.array([det.stamp.iloc[0]]))[0]) if not tf.empty else (0, 0, 0)
            # trễ hiệu dụng: τ cực tiểu hóa sai số ghép (người phải di chuyển trong phiên)
            grid = np.arange(-0.1, 0.31, 0.02)
            errs = []
            for tau in grid:
                m_, _ = match(det, truth_in_sensor(det, gr, gp, mount, tau))
                errs.append(np.nan if m_.empty else np.mean((m_.r_meas * np.cos(m_.b_meas) - m_.r_true * np.cos(m_.b_true)) ** 2
                                                            + (m_.r_meas * np.sin(m_.b_meas) - m_.r_true * np.sin(m_.b_true)) ** 2))
            tau_hat = float(grid[int(np.nanargmin(errs))]) if np.isfinite(errs).any() else 0.0
            taus.append(tau_hat)
            tru = truth_in_sensor(det, gr, gp, mount, tau_hat, models[sensor])
            m_, f_ = match(det, tru)
            M.append(m_); TR.append(tru); fp += f_; scans += det.stamp.nunique()
        if not M:
            continue
        M, TR = pd.concat(M), pd.concat(TR)
        fit = fit_sensor_noise(M.r_true, M.b_true, M.r_meas, M.b_meas)
        # tỉ lệ phát hiện theo khoảng cách (người thật trong vùng nhìn)
        rec = []
        for lo in np.arange(0.5, 8.0, 1.0):
            sel = (TR.r_true >= lo) & (TR.r_true < lo + 1.0)
            k = int(M[(M.r_true >= lo) & (M.r_true < lo + 1.0)].shape[0]); n = int(sel.sum())
            if n:
                rec.append(dict(r_lo=lo, n=n, k=k, recall=k / n, ci=clopper_pearson(min(k, n), n)))
        res["sensors"][sensor] = dict(fit={k: v for k, v in fit.items() if k != "table"}, recall=rec,
                                      false_pos_per_scan=fp / max(scans, 1), n_matched=int(len(M)),
                                      latency_tau=taus, latency_median=float(np.median(taus)))
        for row in fit["table"]:
            bins.append(dict(sensor=sensor, **row))
        ident[sensor] = dict(sigma_r_a=fit["sigma_r_a"], sigma_r_b=fit["sigma_r_b"], sigma_b=fit["sigma_b"])
    # q của người đi bộ từ quỹ đạo GT
    for d in dirs:
        gp = load(d, "gt_people")
        for _, g in (gp.groupby("participant") if not gp.empty else []):
            g = g.sort_values("stamp")
            if len(g) < 50:
                continue
            q, grid, ll = fit_cv_q(g.stamp.values, g[["x", "y"]].values, gt_std ** 2)
            q_grid, q_ll = grid, (ll if q_ll is None else q_ll + ll)
    if q_ll is not None:
        res["pedestrian_q"] = float(q_grid[int(np.argmax(q_ll))])
    res["provenance"] = provenance(dirs, synthetic=synth, extra=dict(gt_std=gt_std))
    write_json(RESULTS / "sensor_noise.json", res)
    pd.DataFrame(bins).to_csv(RESULTS / "sensor_noise_bins.csv", index=False)
    base = yaml.safe_load((ROOT / "config" / "sensors.yaml").read_text())
    for s, v in ident.items():
        base["sensors"][s].update({k: float(x) for k, x in v.items()})
    base["identified_from"] = [str(d) for d in dirs]
    (ROOT / "config" / "sensors_identified.yaml").write_text(yaml.safe_dump(base, allow_unicode=True))
    print({s: v["fit"] for s, v in res["sensors"].items()}, res.get("pedestrian_q"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-std", type=float, required=True)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    a = ap.parse_args()
    main(a.dirs, a.gt_std)
