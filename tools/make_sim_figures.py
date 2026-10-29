"""Hình cho nghiên cứu mô phỏng → ../paper_sensors/figures/{scenarios,sim_grid}.pdf.

Lưới mờ, vector, phân biệt bằng hoa văn để in đen trắng. Chỉ đọc results/sim_*.csv (dữ liệu mô phỏng báo cáo).
"""
import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402

from pu_nmpc_core.simstudy import SCENARIOS, make_scenario  # noqa: E402

plt.rcParams.update({"lines.linewidth": 1.5, "patch.linewidth": 1.5, "boxplot.boxprops.linewidth": 1.5,
                     "boxplot.whiskerprops.linewidth": 1.5, "boxplot.capprops.linewidth": 1.5,
                     "boxplot.medianprops.linewidth": 1.5, "boxplot.flierprops.linewidth": 1.5,
                     "errorbar.capsize": 2})
plt.rcParams.update({"font.size": 7.5, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
METH = [("cov", "Covariance (proposed)", "#1b9e77", "//"), ("fixed", "Fixed margin", "#7570b3", ".."),
        ("heuristic", "Heuristic risk", "#d95f02", "xx"), ("ccconst", "Constant-covariance CC", "#e7298a", "\\\\")]
SLAB = ["H1", "H2", "H3", "H4", "H5"]


def fig_scenarios(out):
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
    for a, sc, lab in zip(ax, SCENARIOS, ["H1 crossing", "H2 head-on", "H3 narrow passage", "H4 occlusion", "H5 side/overtake"]):
        rng = np.random.default_rng(0)
        w, path, walkers, _ = make_scenario(sc, rng)
        for s0, s1 in w.segments:
            a.plot([s0[0], s1[0]], [s0[1], s1[1]], "k-", lw=1.5)
        p = np.array(path)
        a.plot(p[:, 0], p[:, 1], "k--", lw=1.5)
        a.add_patch(plt.Rectangle((-0.3, -0.275), 0.6, 0.55, fill=False, lw=1.5))
        for wk in walkers:
            q = wk.waypoints
            a.annotate("", xy=q[-1], xytext=q[0], arrowprops=dict(arrowstyle="->", lw=1.5, color="0.3"))
            a.plot(q[:, 0], q[:, 1], ":", color="0.3", lw=1.5)
        a.set_title(lab, fontsize=7)
        a.set_aspect("equal")
        a.set_xlim(-2.8, 9.8); a.set_ylim(-4.2, 4.2)
        a.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    fig.savefig(out / "scenarios.pdf")
    plt.close(fig)


def grouped_box(a, df, col, ylabel, title):
    w = 0.8 / len(METH)
    for i, (m, lab, colr, hatch) in enumerate(METH):
        data = [df[(df.method == m) & (df.scenario == s)][col].dropna().values for s in SCENARIOS]
        bp = a.boxplot(data, positions=np.arange(5) + (i - (len(METH) - 1) / 2) * w, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=colr, edgecolor=colr, alpha=0.75, hatch=hatch, lw=1.5)
        for ln in bp["whiskers"] + bp["caps"]:
            ln.set(color=colr, lw=1.5)
        for med in bp["medians"]:
            med.set(color="k", lw=1.5)
    a.set_xticks(range(5), SLAB); a.set_ylabel(ylabel); a.set_title(title, loc="left")


def fig_grid(main, rob, abl, out):
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.4))
    grouped_box(ax[0, 0], main, "min_clearance", "Min. clearance (m)", "(a)")
    grouped_box(ax[0, 1], main, "zone_time", "Time in 0.5-m zone (s)", "(b)")
    grouped_box(ax[0, 2], main[main.outcome == "success"], "time", "Time to goal (s)", "(c)")
    # (d) độ bền: khoảng hở trung bình ± CI theo điều kiện
    conds = [("nominal", "Nominal", main)]
    if rob is not None:
        conds += [(c, l, rob[rob.condition == c]) for c, l in (("noise2", "Noise ×2"), ("delay15", "Delay ×1.5"), ("nonreactive", "Non-react."))]
    w = 0.8 / len(METH)
    for i, (m, lab, colr, hatch) in enumerate(METH):
        mu, lo, hi = [], [], []
        for _, _, d in conds:
            x = d[d.method == m].min_clearance.dropna().values
            bs = np.random.default_rng(0).choice(x, (2000, len(x))).mean(axis=1) if len(x) else np.array([np.nan])
            mu.append(x.mean() if len(x) else np.nan); lo.append(mu[-1] - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - mu[-1])
        ax[1, 0].bar(np.arange(len(conds)) + (i - (len(METH) - 1) / 2) * w, mu, w, yerr=[lo, hi], color=colr, alpha=0.6, hatch=hatch,
                     edgecolor="k", lw=1.5, capsize=1.5, label=lab)
    ax[1, 0].set_xticks(range(len(conds)), [c[1] for c in conds], fontsize=6.5, rotation=20)
    ax[1, 0].set_ylabel("Mean min. clearance (m)"); ax[1, 0].set_title("(d)", loc="left")
    # (e) độ phủ elip 95 % theo tầm dự báo
    hs = [0.0, 0.5, 1.0]
    series = [("Fusion, compensated", main[main.method == "cov"], "-", "o")]
    if abl is not None:
        series += [("No delay compensation", abl[abl.condition == "nocomp"], "--", "s"),
                   ("LiDAR only", abl[abl.condition == "lidar_only"], "-.", "^")]
    if rob is not None:
        series += [("Sensor noise ×2", rob[(rob.condition == "noise2") & (rob.method == "cov")], ":", "d")]
    for lab, d, ls, mk in series:
        y = [100 * d[f"cov_in_{h}"].sum() / max(d[f"cov_n_{h}"].sum(), 1) for h in hs]
        ax[1, 1].plot(hs, y, ls, marker=mk, mfc="none", color="k", lw=1.5, label=lab)
    ax[1, 1].axhline(95, color="0.5", lw=1.5, ls=":")
    ax[1, 1].set_xlabel("Prediction horizon (s)"); ax[1, 1].set_ylabel("95% ellipse coverage (%)")
    ax[1, 1].legend(fontsize=5.5, loc="lower left", bbox_to_anchor=(0.02, 0.36)); ax[1, 1].set_ylim(0, 102); ax[1, 1].set_title("(e)", loc="left")
    data = [main[main.method == m].solve_p95.dropna().values for m, *_ in METH]
    bp = ax[1, 2].boxplot(data, patch_artist=True, showfliers=True, flierprops=dict(markersize=2),
                          medianprops=dict(color="k", lw=1.5))
    for b, (_, _, colr, hatch) in zip(bp["boxes"], METH):
        b.set(facecolor=colr, edgecolor=colr, alpha=0.75, hatch=hatch, lw=1.5)
    ax[1, 2].set_xticks(range(1, len(METH) + 1), ["Cov.", "Fixed", "Heur.", "Const."][:len(METH)])
    ax[1, 2].axhline(100, color="0.4", ls=":", lw=1.5)
    ax[1, 2].set_ylabel("Per-run p95 solve time (ms)"); ax[1, 2].set_title("(f)", loc="left")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "sim_grid.pdf")
    plt.close(fig)


def fig_revision(S, out):
    """Chiến dịch bổ sung (hạt 3000–3049): khoảng cách và thời gian trong vùng theo thời gian tới đích."""
    R = S["revision"]
    fx = [("fixed", "0.30"), ("fix045", "0.45"), ("fix060", "0.60"), ("fix075", "0.75")]
    pts = [("cov", "Covariance", "o", "#1b9e77"), ("covcal", "Covariance, calibrated", "s", "#1b9e77"),
           ("heuristic", "Heuristic risk", "^", "#d95f02"), ("ccconst", "Constant-covariance CC", "v", "#e7298a")]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for a, key, lab in ((ax[0], "clearance", "Mean min. clearance (m)"), (ax[1], "zone", "Mean time in 0.5-m zone (s)")):
        x = [R[k]["time_success"][0] for k, _ in fx]
        y = [R[k][key][0] for k, _ in fx]
        a.errorbar(x, y, yerr=[[y_ - R[k][key][1] for y_, (k, _) in zip(y, fx)], [R[k][key][2] - y_ for y_, (k, _) in zip(y, fx)]],
                   fmt="--D", color="#7570b3", mfc="white", ms=4, lw=1.5, capsize=2, label="Fixed margin (buffer, m)")
        for (k, t), x_, y_ in zip(fx, x, y):
            a.annotate(t, (x_, y_), textcoords="offset points", xytext=(4, 3), fontsize=6)
        for k, lab2, mk, colr in pts:
            if k not in R:
                continue
            r = R[k]
            a.errorbar(r["time_success"][0], r[key][0], yerr=[[r[key][0] - r[key][1]], [r[key][2] - r[key][0]]],
                       xerr=[[r["time_success"][0] - r["time_success"][1]], [r["time_success"][2] - r["time_success"][0]]],
                       fmt=mk, color=colr, mfc="white" if k == "cov" else colr, ms=5, capsize=2, lw=1.5, label=lab2)
        a.set_xlabel("Mean time to goal, successful runs (s)"); a.set_ylabel(lab)
    ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "revision.pdf")
    plt.close(fig)


def fig_drow(D, ss, out):
    """Dữ liệu thật DROW v2: tỉ lệ phát hiện và độ lệch chuẩn sai số khoảng cách theo khoảng cách."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    keys = list(D["recall_by_range"])
    lo_hi = [tuple(float(x) for x in k.split("-")) for k in keys]
    xc = [0.5 * (a + b) for a, b in lo_hi]
    ci = np.array([D["recall_by_range"][k]["ci"] for k in keys]) * 100
    ax[0].errorbar(xc, ci[:, 0], yerr=[ci[:, 0] - ci[:, 1], ci[:, 2] - ci[:, 0]], fmt="o-", color="k", mfc="white",
                   lw=1.5, capsize=2, label="Leg detector on DROW v2 (real)")
    ax[0].axhline(100 * ss["lidar"]["p_detect"]["value"], color="#1b9e77", ls="--", lw=1.5, label="Assumed in simulation")
    ax[0].set_xlabel("Range (m)"); ax[0].set_ylabel("Detection rate (%)"); ax[0].set_ylim(0, 100)
    ax[0].legend(fontsize=6.5, loc="lower left"); ax[0].set_title("(a)", loc="left")
    sd = D["range_error"]["robust_sd_by_range"]
    rr = np.array([float(k) for k in sd]); yy = 100 * np.array(list(sd.values()))
    ax[1].plot(rr, yy, "o-", color="k", mfc="white", lw=1.5, label="DROW v2 (incl. label error)")
    a_, b_ = ss["_fit"]
    x = np.linspace(0.3, 8, 50)
    ax[1].plot(x, 100 * (a_ + b_ * x), "--", color="#1b9e77", lw=1.5, label="Simulation model")
    ax[1].set_xlabel("Range (m)"); ax[1].set_ylabel("Range error std. (cm)"); ax[1].set_ylim(0, None)
    ax[1].legend(fontsize=6.5, loc="upper left"); ax[1].set_title("(b)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "drow.pdf"); plt.close(fig)


def load(name):
    f = RESULTS / f"sim_{name}.csv"
    if not f.exists():
        return None
    meta = json.loads((RESULTS / f"sim_{name}.meta.json").read_text())
    if meta.get("synthetic"):
        sys.exit(f"TỪ CHỐI: sim_{name} có cờ synthetic")
    return pd.read_csv(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=pathlib.Path, required=True)
    a = ap.parse_args()
    out = a.paper / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_scenarios(out)
    main = load("main")
    alt = load("altmain")
    if main is not None and alt is not None:
        main = pd.concat([main, alt], ignore_index=True)
    if main is not None:
        fig_grid(main, load("robust"), load("ablation"), out)
    dj = RESULTS / "drow_validation.json"
    if dj.exists():
        import yaml
        from common import ROOT
        ss = yaml.safe_load((ROOT / "config" / "simstudy.yaml").read_text())
        mm = json.loads((RESULTS / "sim_main.meta.json").read_text())
        ss["_fit"] = mm["sensor_fit"]["lidar_ab"]
        fig_drow(json.loads(dj.read_text()), ss, out)
    sj = RESULTS / "sim_summary.json"
    if sj.exists() and "revision" in json.loads(sj.read_text()):
        fig_revision(json.loads(sj.read_text()), out)
    print("đã vẽ:", sorted(p.name for p in out.glob("*.pdf")))
