"""Analysis of the simulation campaigns -> results/sim_summary.json (plan fixed before seeing the reporting results).

- Proportions (success, contacts): Clopper-Pearson 95 %; 0 events -> also the one-sided upper bound.
- Continuous means: 95 % percentile bootstrap over runs (10 000 resamples).
- Comparisons: paired by (scenario, seed), 95 % bootstrap.
- Covariance coverage: sum inside / sum total; bootstrap CI over runs.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, sha, write_json  # noqa: E402

from pu_nmpc_core.metrics import clopper_pearson, paired_bootstrap, upper_bound_zero  # noqa: E402

HS = ("0.0", "0.5", "1.0")
CONTACT = ("contact_active", "contact_passive", "wall_contact")


def boot_mean(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def coverage(g, h, n_boot=2000, seed=0):
    n, k = g[f"cov_n_{h}"].values.astype(float), g[f"cov_in_{h}"].values.astype(float)
    if n.sum() == 0:
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    b = k[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    return [float(k.sum() / n.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarize(g):
    n = len(g)
    s = dict(n=n)
    ks = int((g.outcome == "success").sum())
    s["success"] = [ks / n, *clopper_pearson(ks, n)]
    for c in CONTACT + ("timeout", "error"):
        k = int((g.outcome == c).sum())
        s[c] = dict(k=k, ci=clopper_pearson(k, n), ub0=upper_bound_zero(n) if k == 0 else None)
    s["clearance"] = boot_mean(g.min_clearance)
    s["clearance_p05"] = float(np.nanpercentile(g.min_clearance, 5))
    s["zone"] = boot_mean(g.zone_time)
    s["time_success"] = boot_mean(g[g.outcome == "success"].time)
    s["jerk"] = boot_mean(g.mean_jerk)
    s["solve_p95_median"] = float(np.nanmedian(g.solve_p95))
    s["solve_max"] = float(np.nanmax(g.solve_max))
    s["guard_runs"] = int((g.guard_steps > 0).sum())
    s["slack_median"] = float(np.nanmedian(g.max_slack))
    s["late_dropped"] = int(g.late_dropped.sum())
    for h in HS:
        s[f"cov_{h}"] = coverage(g, h)
    return s


def paired(df, a, b, keys=("scenario", "seed")):
    A = df[df.method_v == a].set_index(list(keys))
    B = df[df.method_v == b].set_index(list(keys))
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    out = {}
    for m in ("min_clearance", "zone_time"):
        out[m] = list(paired_bootstrap(j[f"{m}_a"], j[f"{m}_b"]))
    both = (j.outcome_a == "success") & (j.outcome_b == "success")
    out["time_both_success"] = list(paired_bootstrap(j.time_a[both], j.time_b[both]))
    out["success"] = list(paired_bootstrap((j.outcome_a == "success").astype(float), (j.outcome_b == "success").astype(float)))
    return out


def load(name):
    f = RESULTS / f"sim_{name}.csv"
    if not f.exists():
        return None, None
    meta = json.loads((RESULTS / f"sim_{name}.meta.json").read_text())
    return pd.read_csv(f), meta


def main():
    out, prov = {}, []
    main_df, meta = load("main")
    alt, meta_alt = load("altmain")
    if main_df is not None:
        prov.append(dict(file="sim_main.csv", sha256=sha(RESULTS / "sim_main.csv"), meta=meta))
        if alt is not None:   # ccconst baseline (added later), same seeds 1000-1099
            prov.append(dict(file="sim_altmain.csv", sha256=sha(RESULTS / "sim_altmain.csv"), meta=meta_alt))
            main_df = pd.concat([main_df, alt], ignore_index=True)
        main_df["method_v"] = main_df.method
        out["main"] = {}
        for m, g in main_df.groupby("method"):
            out["main"][f"{m}|all"] = summarize(g)
            for sc, gs in g.groupby("scenario"):
                out["main"][f"{m}|{sc}"] = summarize(gs)
        bl = [b for b in ("fixed", "heuristic", "ccconst") if (main_df.method == b).any()]
        out["paired_main"] = {f"cov-{b}": paired(main_df, "cov", b) for b in bl}
        out["paired_main_scen"] = {f"cov-{b}|{sc}": paired(main_df[main_df.scenario == sc], "cov", b)
                                   for b in bl for sc in main_df.scenario.unique()}
    rob, meta = load("robust")
    if rob is not None:
        rob["method_v"] = rob.method
        prov.append(dict(file="sim_robust.csv", sha256=sha(RESULTS / "sim_robust.csv"), meta=meta))
        out["robust"] = {}
        out["paired_robust"] = {}
        for cond, gc in rob.groupby("condition"):
            for m, g in gc.groupby("method"):
                out["robust"][f"{cond}|{m}"] = summarize(g)
            out["paired_robust"].update({f"{cond}|cov-{b}": paired(gc, "cov", b) for b in ("fixed", "heuristic")})
    abl, meta = load("ablation")
    if abl is not None and main_df is not None:
        prov.append(dict(file="sim_ablation.csv", sha256=sha(RESULTS / "sim_ablation.csv"), meta=meta))
        out["ablation"] = {}
        base = main_df[main_df.method == "cov"].assign(method_v="cov")
        for cond, g in abl.groupby("condition"):
            out["ablation"][cond] = summarize(g)
            both = pd.concat([base, g.assign(method_v=cond)])
            out["ablation"][cond]["paired_vs_cov"] = paired(both, cond, "cov")
        out["ablation"]["cov_nominal"] = summarize(base)
    rev, meta = load("revision")
    if rev is not None:
        # post hoc analyses (new seeds 3000-3049): arm = condition if not nominal, otherwise = method
        prov.append(dict(file="sim_revision.csv", sha256=sha(RESULTS / "sim_revision.csv"), meta=meta))
        ar, meta_ar = load("altrev")
        if ar is not None:
            prov.append(dict(file="sim_altrev.csv", sha256=sha(RESULTS / "sim_altrev.csv"), meta=meta_ar))
            rev = pd.concat([rev, ar], ignore_index=True)
        rev["method_v"] = np.where(rev.condition == "nominal", rev.method, rev.condition)
        out["revision"] = {a: summarize(g) for a, g in rev.groupby("method_v")}
        out["paired_revision"] = {}
        for a, b in [("covcal", x) for x in ("cov", "fixed", "fix045", "fix060", "fix075", "heuristic")] + \
                    [("cov", x) for x in ("fix045", "fix060", "fix075")] + [("covcal", "ccconst"), ("cov", "ccconst")]:
            if (rev.method_v == b).any():
                out["paired_revision"][f"{a}-{b}"] = paired(rev, a, b)
    tim, meta = load("timing")
    at, meta_at = load("alttiming")
    if tim is not None and at is not None:
        prov.append(dict(file="sim_alttiming.csv", sha256=sha(RESULTS / "sim_alttiming.csv"), meta=meta_at))
        tim = pd.concat([tim, at], ignore_index=True)
    if tim is not None:
        prov.append(dict(file="sim_timing.csv", sha256=sha(RESULTS / "sim_timing.csv"), meta=meta))
        out["timing"] = {m: dict(n=len(g), p50=float(np.median(g.solve_p50)), p95=float(np.median(g.solve_p95)),
                                 max=float(g.solve_max.max()), runs_over_period=int((g.solve_max > 100).sum()),
                                 runs_p95_over_period=int((g.solve_p95 > 100).sum()),
                                 wall_median=float(g.wall_s.median()), wall_max=float(g.wall_s.max()),
                                 wall_outliers=int((g.wall_s > 10 * g.wall_s.median()).sum()),
                                 p95_max=float(g.solve_p95.max()))
                         for m, g in tim.groupby("method")}
    out["provenance"] = dict(kind="simulation", synthetic=False, inputs=prov)
    write_json(RESULTS / "sim_summary.json", out)
    if "main" in out:
        for k in ("cov|all", "fixed|all", "heuristic|all"):
            s = out["main"][k]
            print(k, "succ %.3f" % s["success"][0], "clr %.3f" % s["clearance"][0], "zone %.2f" % s["zone"][0],
                  "time %.1f" % s["time_success"][0], "cov1 %.3f" % s["cov_1.0"][0])


if __name__ == "__main__":
    main()
