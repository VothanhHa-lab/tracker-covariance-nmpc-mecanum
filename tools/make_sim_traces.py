"""Hình minh họa chuỗi thời gian: chạy lại hạt báo cáo đầu tiên (1000) của mỗi kịch bản cho ba bộ điều khiển.

Hạt được chọn cố định (hạt đầu tiên của chiến dịch chính), không chọn theo kết quả. Mỗi lượt là tất định theo hạt,
nên kết quả trùng với dòng tương ứng của results/sim_main.csv; script kiểm tra điều này.
→ ../paper_sensors/figures/{trajectories,timeseries}.pdf
"""
import argparse
import pickle
import pathlib
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402
from make_sim_figures import METH, plt as _plt  # noqa: E402,F401
from sim_campaign import build  # noqa: E402

from pu_nmpc_core.controller import Controller, MethodParams  # noqa: E402
from pu_nmpc_core.constraint import kappa  # noqa: E402
from pu_nmpc_core.simstudy import SCENARIOS, run_episode  # noqa: E402

SEED = 1000
LAB = ["H1 crossing", "H2 head-on", "H3 narrow passage", "H4 occlusion", "H5 side/overtake"]
LS = {"cov": "-", "fixed": "--", "heuristic": ":", "ccconst": "-."}


def run_all():
    s, rp, sensors, cfg, mp, npar, refp, tp, _ = build()
    out = {}
    for sc in SCENARIOS:
        for m, *_ in METH:
            mpm = replace(mp, method=m)
            tr = {}
            r = run_episode(sc, SEED, mpm, rp, npar, refp, tp, sensors, cfg, ctrl=Controller(rp, MethodParams(**mpm.__dict__), npar), trace=tr)
            out[(sc, m)] = (r, tr)
    return out, mp


def check(out):
    d = pd.concat([pd.read_csv(RESULTS / f) for f in ("sim_main.csv", "sim_altmain.csv") if (RESULTS / f).exists()])
    bad = []
    for (sc, m), (r, _) in out.items():
        row = d[(d.scenario == sc) & (d.seed == SEED) & (d.method == m)].iloc[0]
        if row.outcome != r["outcome"] or abs(row.min_clearance - r["min_clearance"]) > 1e-6:
            bad.append((sc, m, row.outcome, r["outcome"], row.min_clearance, r["min_clearance"]))
    return bad


def margin_series(tr, m, mp):
    t, y = [], []
    for g in tr["diags"]:
        if g.get("mode") != "nmpc" or not g.get("n_people"):
            continue
        if m in ("cov", "ccconst") and g.get("sigma_n"):
            t.append(g["t"]); y.append(mp.d0 + kappa(mp.eps, mp.kappa_kind) * g["sigma_n"][0])
        elif m == "heuristic" and g.get("rho"):
            t.append(g["t"]); y.append(mp.d0 + mp.heuristic["k_d"] * g["rho"][0])
        elif m == "fixed":
            t.append(g["t"]); y.append(mp.d0 + mp.d_fixed)
    return np.array(t), np.array(y)


def fig_traj(out, dst):
    fig, ax = plt.subplots(5, 1, figsize=(7.0, 7.4), sharex=True)
    for a, sc, lab in zip(ax, SCENARIOS, LAB):
        _, tr0 = out[(sc, "cov")]
        for s0, s1 in tr0["walls"]:
            a.plot([s0[0], s1[0]], [s0[1], s1[1]], "k-", lw=1.5)
        for m, name, colr, _ in METH:
            r, tr = out[(sc, m)]
            a.plot(tr["R"][:, 0], tr["R"][:, 1], LS[m], color=colr, lw=1.5, label=name)
            k = int(np.argmin(tr["clearance"]))          # thời điểm khoảng cách nhỏ nhất
            a.plot(*tr["R"][k, :2], "o", color=colr, ms=3.5, mfc="white")
            pj = tr["Pp"][k]
            j = int(np.nanargmin(np.linalg.norm(pj - tr["R"][k, :2], axis=1)))
            a.add_patch(plt.Circle(pj[j], 0.25, fill=False, ls=LS[m], color=colr, lw=1.5))
        Pp = tr0["Pp"]
        for j in range(Pp.shape[1]):
            a.plot(Pp[:, j, 0], Pp[:, j, 1], color="0.5", lw=1.5)
        a.set_ylim(-2.0, 2.0); a.set_xlim(-0.8, 9.0); a.set_ylabel("y (m)")
        a.text(0.005, 0.96, lab, transform=a.transAxes, va="top", fontsize=7,
               bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax[-1].set_xlabel("x (m)")
    h, l = ax[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="0.5", lw=1.5)); l.append("Pedestrian paths (run of proposed controller)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(dst / "trajectories.pdf"); plt.close(fig)


def fig_ts(out, mp, dst):
    sc_sel = [("H4_occlusion", "H4 occlusion"), ("H5_side", "H5 side/overtake")]
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 5.0), sharex="col")
    for col, (sc, lab) in enumerate(sc_sel):
        for m, name, colr, _ in METH:
            r, tr = out[(sc, m)]
            T, R = tr["T"], tr["R"]
            ax[0, col].plot(T, np.clip(tr["clearance"], None, 4), LS[m], color=colr, lw=1.5, label=name)
            ax[1, col].plot(T, np.hypot(R[:, 3], R[:, 4]), LS[m], color=colr, lw=1.5)
            tm, ym = margin_series(tr, m, mp)
            ax[2, col].plot(tm, ym, LS[m], color=colr, lw=1.5, drawstyle="steps-post")
        ax[0, col].axhline(0.5, color="0.4", ls=":", lw=1.5)
        ax[0, col].set_title(f"{lab}, seed {SEED}", fontsize=7.5)
        ax[2, col].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Clearance (m, clipped at 4)"); ax[1, 0].set_ylabel("Robot speed (m/s)")
    ax[2, 0].set_ylabel("Margin, nearest person (m)")
    for i, a in enumerate(ax.flat):
        a.set_title(f"({'abcdef'[i]})", loc="left", fontsize=7.5)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(dst / "timeseries.pdf"); plt.close(fig)


def path_of(sc):
    """Đường tham chiếu của kịch bản: tạo lại như run_episode (cùng hạt, tất định)."""
    from pu_nmpc_core.simstudy import make_scenario
    _, path, _, _ = make_scenario(sc, np.random.default_rng(SEED), True)
    return np.asarray(path, float)


def lateral(R, path):
    """Sai lệch ngang có dấu so với đường tham chiếu gấp khúc (m)."""
    e = np.full(len(R), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        d = b - a; L = np.linalg.norm(d); t_ = d / L; nrm = np.array([-t_[1], t_[0]])
        s_ = np.clip((R[:, :2] - a) @ t_, 0, L)
        foot = a + s_[:, None] * t_
        dist = np.linalg.norm(R[:, :2] - foot, axis=1)
        sgn = np.sign((R[:, :2] - a) @ nrm)
        better = dist < np.abs(e)
        e[better] = (sgn * dist)[better]
    return e


def fig_tracking(out, mp, dst, vref):
    sc_sel = [("H2_head_on", "H2 head-on"), ("H4_occlusion", "H4 occlusion")]
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 5.6), sharex="col")
    for col, (sc, lab) in enumerate(sc_sel):
        path = path_of(sc)
        Lp = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        for m, name, colr, _ in METH:
            _, tr = out[(sc, m)]
            T, R = tr["T"], tr["R"]
            ax[0, col].plot(T, R[:, 0], LS[m], color=colr, lw=1.5, label=name)
            ax[1, col].plot(T, lateral(R, path), LS[m], color=colr, lw=1.5)
            ax[2, col].plot(T, np.degrees(R[:, 2]), LS[m], color=colr, lw=1.5)
        Tm = max(out[(sc, m)][1]["T"][-1] for m, *_ in METH)
        tt = np.linspace(0, Tm, 200)
        ax[0, col].plot(tt, np.minimum(vref * tt, Lp), color="k", ls=(0, (1, 1)), lw=1.5, label="Set point at $v_{\\mathrm{ref}}$")
        ax[1, col].axhline(0, color="k", ls=(0, (1, 1)), lw=1.5)
        ax[2, col].axhline(np.degrees(0.0), color="k", ls=(0, (1, 1)), lw=1.5)
        ax[0, col].set_title(f"{lab}, seed {SEED}", fontsize=7.5)
        ax[2, col].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("$X$ (m)"); ax[1, 0].set_ylabel("Lateral deviation (m)"); ax[2, 0].set_ylabel("Heading $\\theta$ (deg)")
    for i, a in enumerate(ax.flat):
        a.set_title(f"({'abcdef'[i]})", loc="left", fontsize=7.5)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(dst / "tracking.pdf"); plt.close(fig)


def tracking_macros(out):
    SCm = dict(zip(SCENARIOS, ["HA", "HB", "HC", "HD", "HE"]))
    MN = {"cov": "Cov", "fixed": "Fixed", "heuristic": "Heur", "ccconst": "Const"}
    L = []
    for (sc, m), (_, tr) in sorted(out.items()):
        e = lateral(tr["R"], path_of(sc))
        th = np.degrees(tr["R"][:, 2])
        L.append(rf"\newcommand{{\Trk{MN[m]}{SCm[sc]}Ey}}{{{np.max(np.abs(e)):.2f}}}")
        L.append(rf"\newcommand{{\Trk{MN[m]}{SCm[sc]}Th}}{{{np.sqrt(np.mean(th ** 2)):.1f}}}")
        L.append(rf"\newcommand{{\Trk{MN[m]}{SCm[sc]}Thmax}}{{{np.max(np.abs(th)):.1f}}}")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=pathlib.Path, required=True)
    a = ap.parse_args()
    ap2 = RESULTS / "traces_seed1000.pkl"   # lưu lại để vẽ lại mà không chạy lại (lượt chạy phụ thuộc tải máy)
    if ap2.exists():
        out, mp = pickle.loads(ap2.read_bytes())
    else:
        out, mp = run_all()
        ap2.write_bytes(pickle.dumps((out, mp)))
    bad = check(out)
    print("lệch so với sim_main.csv:", bad if bad else "không")
    dst = a.paper / "figures"
    fig_traj(out, dst); fig_ts(out, mp, dst)
    import yaml
    from common import ROOT
    vref = yaml.safe_load((ROOT / "config" / "controller.yaml").read_text())["reference"]["v_ref"]
    fig_tracking(out, mp, dst, vref)
    SC = dict(zip(SCENARIOS, ["HA", "HB", "HC", "HD", "HE"]))
    MN = {"cov": "Cov", "fixed": "Fixed", "heuristic": "Heur", "ccconst": "Const"}
    lines = ["% Sinh tự động bởi pu_nmpc/tools/make_sim_traces.py (hạt 1000) — KHÔNG sửa tay"]
    for (sc, m), (r, _) in sorted(out.items()):
        lines.append(rf"\newcommand{{\Tr{MN[m]}{SC[sc]}Clear}}{{{r['min_clearance']:.2f}}}")
    lines.append(rf"\newcommand{{\TrTotal}}{{{len(out)}}}")
    lines.append(rf"\newcommand{{\TrMatch}}{{{len(out) - len(bad)}}}")
    for i, b in enumerate(bad[:2]):   # lượt lệch: kịch bản, bộ điều khiển, khoảng cách lưu / chạy lại
        tag = "AB"[i]
        lines.append(rf"\newcommand{{\TrMis{tag}Scen}}{{{b[0].split('_')[0]}}}")
        lines.append(rf"\newcommand{{\TrMis{tag}Stored}}{{{b[4]:.2f}}}")
        lines.append(rf"\newcommand{{\TrMis{tag}Rerun}}{{{b[5]:.2f}}}")
    lines += tracking_macros(out)
    (a.paper / "tables" / "trace_macros.tex").write_text("\n".join(lines) + "\n")
    for (sc, m), (r, _) in sorted(out.items()):
        print(sc, m, r["outcome"], round(r["min_clearance"], 3))
