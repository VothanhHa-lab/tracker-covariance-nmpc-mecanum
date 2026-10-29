"""Sinh macro LaTeX cho bài báo từ results/*.json. Mục thiếu → \\TBD{}. Không bao giờ gõ số tay vào bài.

python3 tools/make_macros.py --paper ../paper_sensors           # dữ liệu thật
python3 tools/make_macros.py --test-out /tmp/x --allow-synthetic # chỉ để kiểm thử đường ống
Từ chối ghi vào thư mục bài báo nếu bất kỳ kết quả nào có provenance.synthetic = true.
"""
import argparse
import json
import math
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, ROOT  # noqa: E402

TBD = r"\TBD{}"


class Macros:
    def __init__(self):
        self.lines = ["% Sinh tự động bởi pu_nmpc/tools/make_macros.py — KHÔNG sửa tay"]
        self.names = set()

    def add(self, name, value, fmt="{:.2f}"):
        assert name.isalpha(), name
        assert name not in self.names, name
        self.names.add(name)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            s = TBD
        elif isinstance(value, str):
            s = value
        else:
            s = fmt.format(value)
        self.lines.append(rf"\newcommand{{\{name}}}{{{s}}}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def jload(name):
    f = RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


def g(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def pct(x):
    return None if x is None else 100 * x


def build(phase="R"):
    M = Macros()
    synthetic = []
    # tham số thiết kế (không phải kết quả) — đọc từ config để bài và mã luôn khớp
    from scipy.stats import norm
    c = yaml.safe_load((ROOT / "config" / "controller.yaml").read_text())
    tkc = yaml.safe_load((ROOT / "config" / "tracker.yaml").read_text())["tracker"]
    me, nm, rf = c["method"], c["nmpc"], c["reference"]
    M.add("CfgDzero", me["d0"]); M.add("CfgEps", me["eps"], "{:g}")
    M.add("CfgKappa", float(norm.ppf(1 - me["eps"])), "{:.2f}")
    M.add("CfgKappaCant", float(((1 - me["eps"]) / me["eps"]) ** 0.5), "{:.2f}")
    M.add("CfgDfixed", me["d_fixed"]); M.add("CfgFreeze", me["sigma_freeze_after"], "{:.1f}")
    M.add("CfgMaxPeople", me["max_people"], "{:d}"); M.add("CfgMaxStatic", me["max_static"], "{:d}")
    M.add("CfgStaticClear", me["static_clearance"]); M.add("CfgStale", me["stale_after"], "{:.1f}")
    M.add("CfgHeurKd", me["heuristic"]["k_d"], "{:.1f}"); M.add("CfgHeurKv", me["heuristic"]["k_v"], "{:.1f}")
    M.add("CfgN", nm["N"], "{:d}"); M.add("CfgDt", nm["dt"], "{:.1f}"); M.add("CfgHorizon", nm["N"] * nm["dt"], "{:.1f}")
    M.add("CfgMaxCpu", 1e3 * nm["max_cpu_time"], "{:.0f}")
    M.add("CfgVref", rf["v_ref"], "{:.1f}"); M.add("CfgGoalTol", rf["goal_tol"])
    M.add("CfgLag", 1e3 * tkc["lag"], "{:.0f}"); M.add("CfgGate", tkc["gate_chi2"], "{:.2f}")
    M.add("CfgMerge", tkc["merge_chi2"] if "merge_chi2" in tkc else 5.99, "{:.2f}")
    M.add("CfgConfirm", tkc["confirm_hits"], "{:d}"); M.add("CfgDelete", tkc["delete_confirmed"], "{:.1f}")
    M.add("CfgStopMargin", c.get("guard", {}).get("stop_margin"))
    M.add("CfgRperson", 0.25); M.add("CfgZone", 0.5, "{:.1f}")
    # robot: chỉ in giá trị đã đo / nhận dạng
    rob = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())["robot"]
    for key, name in (("wheel_radius", "RobWheelRadius"), ("lx", "RobLx"), ("ly", "RobLy"),
                      ("wheel_speed_max", "RobWheelMax"), ("radius", "RobRadius")):
        v = rob[key]
        ok = isinstance(v, dict) and "TO_" not in str(v.get("source", ""))
        M.add(name, v["value"] if ok else None, "{:.3f}")
    idn = jload("identification.json")
    if idn and g(idn, "provenance", "synthetic"):
        synthetic.append("identification")
    for ax, n in (("vx", "X"), ("vy", "Y"), ("wz", "W")):
        a = g(idn, "axes", ax)
        M.add(f"IdTau{n}", g(a, "tau"), "{:.3f}")
        M.add(f"IdGain{n}", g(a, "K"), "{:.2f}")
        M.add(f"IdDelay{n}", None if g(a, "delay") is None else 1e3 * a["delay"], "{:.0f}")
        M.add(f"IdNrmse{n}", pct(g(a, "val_nrmse_max")), "{:.1f}")
    gt = jload("gt_validation.json")
    if gt and g(gt, "provenance", "synthetic"):
        synthetic.append("gt_validation")
    M.add("GtRmse", None if g(gt, "rmse") is None else 1e2 * gt["rmse"], "{:.1f}")      # cm
    M.add("GtMax", None if g(gt, "max_rmse") is None else 1e2 * gt["max_rmse"], "{:.1f}")
    M.add("GtPoints", g(gt, "n_points"), "{:d}")
    sn = jload("sensor_noise.json")
    if sn and g(sn, "provenance", "synthetic"):
        synthetic.append("sensor_noise")
    for s, n in (("lidar", "Lidar"), ("camera", "Camera")):
        f = g(sn, "sensors", s, "fit")
        M.add(f"Noise{n}Sra", None if g(f, "sigma_r_a") is None else 1e2 * f["sigma_r_a"], "{:.1f}")   # cm
        M.add(f"Noise{n}Srb", None if g(f, "sigma_r_b") is None else 1e2 * f["sigma_r_b"], "{:.2f}")   # cm/m
        M.add(f"Noise{n}Sb", None if g(f, "sigma_b") is None else math.degrees(f["sigma_b"]), "{:.2f}")  # độ
        lat = g(sn, "sensors", s, "latency_median")
        M.add(f"Noise{n}Lat", None if lat is None else 1e3 * lat, "{:.0f}")
        M.add(f"Noise{n}Fp", g(sn, "sensors", s, "false_pos_per_scan"), "{:.2f}")
        M.add(f"Noise{n}N", g(sn, "sensors", s, "n_matched"), "{:d}")
    M.add("PedQ", g(sn, "pedestrian_q"), "{:.2f}")
    tk = jload("tracking.json")
    if tk and g(tk, "provenance", "synthetic"):
        synthetic.append("tracking")
    M.add("TrkSessions", g(tk, "n_sessions"), "{:d}")
    for cfg, n in (("fusion", "Fusion"), ("lidar_only", "Lidar"), ("camera_only", "Camera")):
        for comp, c in (("_comp", "Comp"), ("_nocomp", "Nocomp")):
            s = g(tk, "configs", cfg + comp)
            p = f"Trk{n}{c}"
            M.add(p + "Rmse", None if g(s, "rmse_0.0") is None else 1e2 * s["rmse_0.0"], "{:.1f}")
            M.add(p + "CovZero", pct(g(s, "cov95_0.0")), "{:.1f}")
            M.add(p + "CovHalf", pct(g(s, "cov95_0.5")), "{:.1f}")
            M.add(p + "CovOne", pct(g(s, "cov95_1.0")), "{:.1f}")
            ci = g(s, "cov95_1.0_ci")
            M.add(p + "CovOneLo", pct(ci[0]) if ci else None, "{:.1f}")
            M.add(p + "CovOneHi", pct(ci[1]) if ci else None, "{:.1f}")
            M.add(p + "Ospa", g(s, "ospa"), "{:.2f}")
            M.add(p + "Recall", pct(g(s, "recall")), "{:.1f}")
            M.add(p + "Switch", g(s, "id_switches"), "{:d}")
    cl = jload(f"closed_loop_{phase}.json")
    if cl and g(cl, "provenance", "synthetic"):
        synthetic.append("closed_loop")
    M.add("ClPlanned", g(cl, "n_planned"), "{:d}")
    M.add("ClRun", g(cl, "n_run"), "{:d}")
    M.add("ClNotRun", g(cl, "n_not_run"), "{:d}")
    scen = {"all": "All", "H1_crossing": "HA", "H2_head_on": "HB", "H3_narrow": "HC", "H4_occlusion": "HD", "H5_side": "HE"}
    meth = {"cov": "Cov", "fixed": "Fixed", "heuristic": "Heur"}
    for m, mn in meth.items():
        for sc, sn_ in scen.items():
            a = g(cl, "aggregate", f"{m}|{sc}")
            p = f"Cl{mn}{sn_}"
            M.add(p + "N", g(a, "n"), "{:d}")
            M.add(p + "Succ", pct(g(a, "success")), "{:.0f}")
            ci = g(a, "success_ci")
            M.add(p + "SuccLo", pct(ci[0]) if ci else None, "{:.0f}")
            M.add(p + "SuccHi", pct(ci[1]) if ci else None, "{:.0f}")
            M.add(p + "Contact", g(a, "contact_k"), "{:d}")
            cci = g(a, "contact_ci")
            M.add(p + "ContactHi", pct(cci[1]) if cci else None, "{:.0f}")
            M.add(p + "Interv", g(a, "intervention_k"), "{:d}")
            M.add(p + "Timeout", g(a, "timeout_k"), "{:d}")
            M.add(p + "Guard", g(a, "guard_trials"), "{:d}")
            M.add(p + "Clear", g(a, "min_clear_mean"), "{:.2f}")
            M.add(p + "ClearLow", g(a, "min_clear_p05"), "{:.2f}")
            M.add(p + "Zone", g(a, "zone_time"), "{:.1f}")
            M.add(p + "Time", g(a, "duration_success"), "{:.1f}")
            if sc == "all":
                M.add(p + "Jerk", g(a, "mean_jerk"), "{:.2f}")
                M.add(p + "SolveP", g(a, "solve_p95"), "{:.0f}")
                M.add(p + "SolveMax", g(a, "solve_max"), "{:.0f}")
    for pair, pn in (("cov-fixed", "CovFixed"), ("cov-heuristic", "CovHeur")):
        for met, mn in (("success", "Succ"), ("min_clearance", "Clear"), ("zone_time", "Zone"), ("duration", "Time")):
            v = g(cl, "paired", f"{pair}|{met}")
            sc_ = 100 if met == "success" else 1
            f = "{:+.0f}" if met == "success" else "{:+.2f}"
            M.add(f"Pair{pn}{mn}", None if v is None else sc_ * v[0], f)
            M.add(f"Pair{pn}{mn}Lo", None if v is None else sc_ * v[1], f)
            M.add(f"Pair{pn}{mn}Hi", None if v is None else sc_ * v[2], f)
            M.add(f"Pair{pn}{mn}N", None if v is None else v[3], "{:d}")
    return M, synthetic


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=pathlib.Path)
    ap.add_argument("--test-out", type=pathlib.Path)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--phase", default="R")
    a = ap.parse_args()
    M, syn = build(a.phase)
    if a.paper:
        if syn:
            sys.exit(f"TỪ CHỐI: kết quả tổng hợp (synthetic) trong {syn}; không ghi vào bài báo.")
        out = a.paper / "tables" / "macros.tex"
    else:
        if syn and not a.allow_synthetic:
            sys.exit("có kết quả synthetic; dùng --allow-synthetic với --test-out")
        out = a.test_out / "macros.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(M.text())
    n_tbd = sum(TBD in l for l in M.lines)
    print(f"đã ghi {out}: {len(M.names)} macro, {n_tbd} còn \\TBD")
