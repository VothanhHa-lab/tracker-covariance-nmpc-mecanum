"""P6: chỉ số từng lượt (từ GROUND TRUTH) và tổng hợp theo phương pháp × kịch bản.

python3 tools/analyze_closed_loop.py --phase R <data_dir>
Chỉ các lượt có trial.json với phase khớp được đưa vào; MỌI lượt của lịch đều được liệt kê,
kể cả lượt hỏng/bỏ dở (outcome = missing_data / aborted), không loại lượt nào sau khi xem kết quả.
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, ROOT, interp_pose, is_synthetic, load, provenance, write_json  # noqa: E402

from pu_nmpc_core import config  # noqa: E402
from pu_nmpc_core.metrics import clopper_pearson, paired_bootstrap, upper_bound_zero  # noqa: E402

R_PERSON = 0.25       # m, bán kính người (chọn trước)
ZONE = 0.5            # m, vùng cá nhân tính từ mép robot tới mép người
METHODS = ["cov", "fixed", "heuristic"]


def trial_metrics(d: pathlib.Path, rp):
    tj = json.loads((d / "trial.json").read_text())
    csv_dir = d / "csv"
    row = dict(trial_id=tj["trial_id"], phase=tj["phase"], participant=tj["participant"], scenario=tj["scenario"],
               method=tj["method"], rep=int(tj["rep"]), outcome_online=tj.get("outcome_online"))
    if not csv_dir.exists() or "t_start" not in tj:
        row["outcome"] = "missing_data"
        return row, False
    synth = is_synthetic(csv_dir)
    gr, gp, od, dg, ev, gd = (load(csv_dir, k) for k in ("gt_robot", "gt_people", "odom", "diag", "events", "guard"))
    t0, t1 = tj["t_start"], tj.get("t_end", np.inf)
    if gr.empty:
        row["outcome"] = "missing_data"
        return row, synth
    gr = gr[(gr.stamp >= t0) & (gr.stamp <= t1)].sort_values("stamp")
    t = gr.stamp.values
    R = gr[["x", "y"]].values
    dmin_t = np.full(len(t), np.inf)
    for _, g in gp.groupby("participant"):
        P = interp_pose(g.assign(yaw=0.0), t)[:, :2]
        dmin_t = np.fmin(dmin_t, np.linalg.norm(P - R, axis=1))
    contact = rp.radius + R_PERSON
    dt = np.diff(t, append=t[-1])
    row["min_dist"] = float(np.nanmin(dmin_t)) if np.isfinite(dmin_t).any() else np.nan
    row["min_clearance"] = row["min_dist"] - contact
    row["zone_time"] = float(np.sum(dt[dmin_t < contact + ZONE]))
    row["contact"] = int(np.nanmin(dmin_t) < contact) if np.isfinite(dmin_t).any() else 0
    row["notes"] = ";".join(e for e in (ev.event if not ev.empty else []) if str(e).startswith("note:"))
    row["duration"] = float(t[-1] - t[0]) if len(t) else np.nan
    row["path_len"] = float(np.sum(np.linalg.norm(np.diff(R, axis=0), axis=1))) if len(R) > 1 else np.nan
    if not od.empty:
        o = od[(od.stamp >= t0) & (od.stamp <= t1)]
        if len(o) > 5:
            v = o[["vx", "vy"]].values
            a = np.gradient(v, o.stamp.values, axis=0)
            j = np.gradient(a, o.stamp.values, axis=0)
            row["mean_jerk"] = float(np.mean(np.linalg.norm(j, axis=1)))
            row["mean_speed"] = float(np.mean(np.linalg.norm(v, axis=1)))
    if not dg.empty:
        g = dg[(dg.stamp >= t0) & (dg.stamp <= t1)]
        if len(g):
            sm = g.solve_ms[g["mode"] == "nmpc"]
            row.update(solve_p50=float(sm.median()) if len(sm) else np.nan,
                       solve_p95=float(sm.quantile(0.95)) if len(sm) else np.nan,
                       solve_max=float(sm.max()) if len(sm) else np.nan,
                       frac_fallback=float(np.mean(g["mode"] != "nmpc")), max_slack=float(g.max_slack.max()))
    if not gd.empty:
        gg = gd[(gd.stamp >= t0) & (gd.stamp <= t1)]
        row["guard_frac"] = float(gg.active.mean()) if len(gg) else 0.0
        row["guard_any"] = int(gg.active.max()) if len(gg) else 0
    oo = tj.get("outcome_online")
    if row["contact"]:
        row["outcome"] = "contact"
    elif oo == "intervention":
        row["outcome"] = "intervention"
    elif oo == "goal_reached":
        row["outcome"] = "success"
    else:
        row["outcome"] = oo or "unknown"
    return row, synth


def aggregate(df):
    out = {}
    for (m, sc), g in list(df.groupby(["method", "scenario"])) + [((m, "all"), df[df.method == m]) for m in df.method.unique()]:
        n = len(g)
        ks, kc = int((g.outcome == "success").sum()), int((g.outcome == "contact").sum())
        ki = int((g.outcome == "intervention").sum())
        out[f"{m}|{sc}"] = dict(
            n=n, success=ks / n, success_ci=clopper_pearson(ks, n), contact_k=kc,
            contact_ci=clopper_pearson(kc, n), contact_ub0=upper_bound_zero(n) if kc == 0 else None,
            intervention_k=ki, timeout_k=int((g.outcome == "timeout").sum()),
            missing_k=int((g.outcome == "missing_data").sum()),
            guard_trials=int(g.guard_any.sum()) if "guard_any" in g else None,
            min_clear_mean=float(g.min_clearance.mean()), min_clear_p05=float(g.min_clearance.quantile(0.05)),
            zone_time=float(g.zone_time.mean()), duration_success=float(g[g.outcome == "success"].duration.mean()),
            mean_jerk=float(g.mean_jerk.mean()) if "mean_jerk" in g else np.nan,
            solve_p95=float(g.solve_p95.median()) if "solve_p95" in g else np.nan,
            solve_max=float(g.solve_max.max()) if "solve_max" in g else np.nan)
    pairs = {}
    key = ["participant", "scenario", "rep"]
    for a, b in (("cov", "fixed"), ("cov", "heuristic")):
        A, B = df[df.method == a].set_index(key), df[df.method == b].set_index(key)
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        for met in ("min_clearance", "zone_time", "duration"):
            pairs[f"{a}-{b}|{met}"] = paired_bootstrap(j[f"{met}_a"], j[f"{met}_b"])
        sa, sb = (j.outcome_a == "success").astype(float), (j.outcome_b == "success").astype(float)
        pairs[f"{a}-{b}|success"] = paired_bootstrap(sa, sb)
    return out, pairs


def main(data_dir, phase, schedule=None):
    rp = config.robot(ROOT / "config" / "robot.yaml")
    sched = pathlib.Path(schedule) if schedule else ROOT / "protocol" / f"schedule_{phase}.csv"
    planned = list(csv.DictReader(open(sched))) if sched.exists() else []
    rows, synth, seen = [], False, set()
    for d in sorted(pathlib.Path(data_dir).glob(f"{phase}*")):
        if (d / "trial.json").exists():
            r, s = trial_metrics(d, rp)
            rows.append(r); synth |= s; seen.add(r["trial_id"])
    for p in planned:
        if p["trial_id"] not in seen:
            rows.append(dict(**{k: p[k] for k in ("trial_id", "phase", "participant", "scenario", "method")},
                             rep=int(p["rep"]), outcome="not_run"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"closed_loop_trials_{phase}.csv", index=False)
    ran = df[df.outcome != "not_run"]
    agg, pairs = aggregate(ran) if len(ran) else ({}, {})
    write_json(RESULTS / f"closed_loop_{phase}.json",
               dict(aggregate=agg, paired=pairs, n_planned=len(planned), n_run=int(len(ran)),
                    n_not_run=int((df.outcome == "not_run").sum()),
                    provenance=provenance([pathlib.Path(data_dir) / t / "csv" for t in sorted(seen)], synthetic=synth)))
    print(ran.groupby(["method", "outcome"]).size() if len(ran) else "chưa có lượt nào")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["D", "R"], default="R")
    ap.add_argument("--schedule", default=None)
    ap.add_argument("data_dir", type=pathlib.Path)
    a = ap.parse_args()
    main(a.data_dir, a.phase, a.schedule)
