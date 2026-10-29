"""Hình cho bài báo, vẽ từ results/ (dữ liệu đo). Lưới mờ, dạng vector, đọc được khi in đen trắng.

python3 tools/make_figures.py --paper ../paper_sensors            # dữ liệu thật
python3 tools/make_figures.py --test-out /tmp/x --allow-synthetic  # chỉ kiểm thử
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

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
METH = [("cov", "Covariance", "#1b9e77", "//"), ("fixed", "Fixed", "#7570b3", ".."), ("heuristic", "Heuristic", "#d95f02", "xx")]
SCEN = [("H1_crossing", "H1"), ("H2_head_on", "H2"), ("H3_narrow", "H3"), ("H4_occlusion", "H4"), ("H5_side", "H5")]


def load_json(name):
    f = RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


def synthetic_flags(*objs):
    return [o.get("provenance", {}).get("synthetic", False) for o in objs if o]


def fig_sensors(sn, bins, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for (s, lab, mk) in (("lidar", "2D LiDAR", "o"), ("camera", "RGB-D camera", "s")):
        b = bins[bins.sensor == s]
        if b.empty:
            continue
        rc = (b.r_lo + b.r_hi) / 2
        ax[0].plot(rc, 100 * b.sigma_r, mk, mfc="none", color="k", label=f"{lab} (bins)")
        f = sn["sensors"][s]["fit"]
        r = np.linspace(rc.min(), rc.max(), 50)
        ax[0].plot(r, 100 * (f["sigma_r_a"] + f["sigma_r_b"] * r), "-" if s == "lidar" else "--", color="k", label=f"{lab} (fit)")
        rec = sn["sensors"][s]["recall"]
        x = [q["r_lo"] + 0.5 for q in rec]
        y = [100 * q["recall"] for q in rec]
        lo = [100 * (q["recall"] - q["ci"][0]) for q in rec]
        hi = [100 * (q["ci"][1] - q["recall"]) for q in rec]
        ax[1].errorbar(x, y, yerr=[lo, hi], fmt=mk + ("-" if s == "lidar" else "--"), mfc="none", color="k", capsize=2, label=lab)
    ax[0].set_xlabel("Range (m)"); ax[0].set_ylabel(r"$\sigma_\rho$ (cm)"); ax[0].set_title("(a)", loc="left")
    ax[1].set_xlabel("Range (m)"); ax[1].set_ylabel("Detection rate (%)"); ax[1].set_title("(b)", loc="left")
    ax[1].set_ylim(0, 105)
    for a in ax:
        a.legend(fontsize=6)
    fig.savefig(out / "sensors.pdf")
    plt.close(fig)


def fig_grid(trials, tk, out):
    df = trials[trials.outcome != "not_run"]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    labels = [m[1] for m in METH]
    data = [df[df.method == m].min_clearance.dropna().values for m, *_ in METH]
    ax[0, 0].boxplot(data, showfliers=True)
    ax[0, 0].set_xticks(range(1, 4), labels, fontsize=7)
    ax[0, 0].set_ylabel("Min. clearance (m)"); ax[0, 0].set_title("(a)", loc="left")
    data = [df[df.method == m].zone_time.dropna().values for m, *_ in METH]
    ax[0, 1].boxplot(data, showfliers=True)
    ax[0, 1].set_xticks(range(1, 4), labels, fontsize=7)
    ax[0, 1].set_ylabel("Time in personal zone (s)"); ax[0, 1].set_title("(b)", loc="left")
    w = 0.26
    for i, (m, lab, col, hatch) in enumerate(METH):
        v = [100 * (df[(df.method == m) & (df.scenario == s)].outcome == "success").mean() for s, _ in SCEN]
        ax[0, 2].bar(np.arange(5) + (i - 1) * w, v, w, color=col, hatch=hatch, edgecolor="k", lw=0.4, label=lab)
    ax[0, 2].set_xticks(range(5), [s for _, s in SCEN]); ax[0, 2].set_ylabel("Success (%)")
    ax[0, 2].set_title("(c)", loc="left")
    data = [df[(df.method == m) & (df.outcome == "success")].duration.dropna().values for m, *_ in METH]
    ax[1, 0].boxplot(data, showfliers=True)
    ax[1, 0].set_xticks(range(1, 4), labels, fontsize=7)
    ax[1, 0].set_ylabel("Time to goal (s)"); ax[1, 0].set_title("(d)", loc="left")
    if tk:
        hs = [0.0, 0.5, 1.0]
        for key, lab, ls in (("fusion_comp", "Fusion, comp.", "-"), ("fusion_nocomp", "Fusion, no comp.", "--"),
                             ("lidar_only_comp", "LiDAR, comp.", "-."), ("camera_only_comp", "Camera, comp.", ":")):
            s = tk["configs"].get(key)
            if s:
                ax[1, 1].plot(hs, [100 * s.get(f"cov95_{h}", np.nan) for h in hs], ls, marker="o", mfc="none", color="k", label=lab)
        ax[1, 1].axhline(95, color="0.5", lw=0.8, ls=":")
        ax[1, 1].set_xlabel("Horizon (s)"); ax[1, 1].set_ylabel("95% ellipse coverage (%)"); ax[1, 1].legend(fontsize=6)
    ax[1, 1].set_title("(e)", loc="left")
    if "solve_p95" in df:
        data = [df[df.method == m].solve_p95.dropna().values for m, *_ in METH]
        ax[1, 2].boxplot(data, showfliers=True)
        ax[1, 2].set_xticks(range(1, 4), labels, fontsize=7)
        ax[1, 2].axhline(100, color="0.4", ls=":", lw=0.8)
    ax[1, 2].set_ylabel("p95 solve time (ms)"); ax[1, 2].set_title("(f)", loc="left")
    h, l = ax[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "results_grid.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=pathlib.Path)
    ap.add_argument("--test-out", type=pathlib.Path)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--phase", default="R")
    a = ap.parse_args()
    sn, tk, cl = load_json("sensor_noise.json"), load_json("tracking.json"), load_json(f"closed_loop_{a.phase}.json")
    syn = any(synthetic_flags(sn, tk, cl))
    if a.paper:
        if syn:
            sys.exit("TỪ CHỐI: có kết quả synthetic; không vẽ hình cho bài báo.")
        out = a.paper / "figures"
    else:
        if syn and not a.allow_synthetic:
            sys.exit("có kết quả synthetic; dùng --allow-synthetic với --test-out")
        out = a.test_out
    out.mkdir(parents=True, exist_ok=True)
    made = []
    bins_f = RESULTS / "sensor_noise_bins.csv"
    if sn and bins_f.exists():
        fig_sensors(sn, pd.read_csv(bins_f), out); made.append("sensors.pdf")
    tr_f = RESULTS / f"closed_loop_trials_{a.phase}.csv"
    if cl and tr_f.exists():
        fig_grid(pd.read_csv(tr_f), tk, out); made.append("results_grid.pdf")
    print("đã vẽ:", made or "không có dữ liệu")
