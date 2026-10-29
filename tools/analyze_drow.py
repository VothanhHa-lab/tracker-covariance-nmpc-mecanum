"""Kiểm chứng một phần trên dữ liệu thật công khai: DROW v2 (Beyer và cs., RA-L 2018; giấy phép MIT).

Cảm biến KHÁC robot của bài: SICK S300, 225°, 450 tia, 12,5 Hz, cao 37 cm (RPLIDAR A2 của robot: 360°, 10 Hz).
Nhãn là vị trí người do người gán tay (r, φ), không phải ground truth độc lập: sai lệch vị trí đo được gồm cả sai
số gán nhãn. Không có bản đồ tĩnh nên bộ phát hiện chạy không có mặt nạ tĩnh: số phát hiện giả là cận trên.

Đo bằng chính bộ phát hiện chân của bài (pu_nmpc_core.detect.cluster_scan, tham số mặc định):
- tỉ lệ phát hiện theo khoảng cách (người đi bộ, nhãn .wp), ghép Hungarian với bán kính 0,5 m;
- số phát hiện giả mỗi lần quét (không khớp người, xe lăn hay khung tập đi nào);
- sai lệch hướng tâm và tiếp tuyến của phát hiện so với nhãn → σ_r(r) = a + b·r, σ_b.

python3 tools/analyze_drow.py external_data/DROWv2/DROWv2-data  →  results/drow_validation.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RESULTS, sha, write_json  # noqa: E402

from pu_nmpc_core.detect import cluster_scan  # noqa: E402

FOV = np.radians(225.0)
N_BEAMS = 450
R_MATCH = 0.5
R_MAX = 8.0
BINS = [(0.3, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)]


def load_dets(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        seq, js = line.split(",", 1)
        out[int(seq)] = [tuple(x) for x in json.loads(js)]
    return out


def polar_to_xy(rp):
    rp = np.asarray(rp, float).reshape(-1, 2)
    return np.column_stack([rp[:, 0] * np.cos(rp[:, 1]), rp[:, 0] * np.sin(rp[:, 1])])


def analyze(root):
    a_min, a_inc = -FOV / 2, FOV / (N_BEAMS - 1)
    n_scan, rows, fp_counts = 0, [], []
    per_split = {}
    files = sorted(root.glob("*/*.bag.csv"))
    for f in files:
        split = f.parent.name
        wp = load_dets(f.with_suffix(".wp"))
        others = {}
        for ext in (".wc", ".wa"):
            for k, v in load_dets(f.with_suffix(ext)).items():
                others.setdefault(k, []).extend(v)
        if not wp:
            continue
        for line in f.read_text().splitlines():
            parts = line.split(",")
            seq = int(parts[0])
            if seq not in wp:              # chỉ dùng các lần quét có nhãn
                continue
            ranges = np.array(parts[2:], float)
            if len(ranges) != N_BEAMS:
                continue
            det = cluster_scan(ranges, a_min, a_inc, r_max=R_MAX)
            D = polar_to_xy(det) if det else np.zeros((0, 2))
            P = [p for p in wp[seq] if p[0] <= R_MAX]
            O = [p for p in others.get(seq, []) if p[0] <= R_MAX]
            G = polar_to_xy(P) if P else np.zeros((0, 2))
            used, matched_g = set(), set()
            if len(G) and len(D):
                C = np.linalg.norm(G[:, None] - D[None], axis=2)
                for g, d in zip(*linear_sum_assignment(C)):
                    if C[g, d] <= R_MATCH:
                        used.add(d); matched_g.add(g)
                        rows.append(dict(split=split, r=P[g][0], hit=1, dr=det[d][0] - P[g][0],
                                         db=float(np.angle(np.exp(1j * (det[d][1] - P[g][1]))))))
            for g in range(len(G)):
                if g not in matched_g:
                    rows.append(dict(split=split, r=P[g][0], hit=0, dr=np.nan, db=np.nan))
            # phát hiện còn lại: khớp xe lăn / khung tập đi thì không tính là giả
            rest = [d for d in range(len(D)) if d not in used]
            if O and rest:
                Oxy = polar_to_xy(O)
                rest = [d for d in rest if np.min(np.linalg.norm(Oxy - D[d], axis=1)) > R_MATCH]
            fp_counts.append(len(rest))
            n_scan += 1
            per_split[split] = per_split.get(split, 0) + 1
    return rows, np.array(fp_counts), n_scan, per_split, files


def cp(k, n, a=0.05):
    from scipy.stats import beta
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [k / n if n else np.nan, float(lo), float(hi)]


def main(root):
    rows, fp, n_scan, per_split, files = analyze(root)
    r = np.array([x["r"] for x in rows]); hit = np.array([x["hit"] for x in rows])
    dr = np.array([x["dr"] for x in rows]); db = np.array([x["db"] for x in rows])
    out = dict(kind="external real data (DROW v2)", synthetic=False, sensor="SICK S300, 225 deg, 450 beams, 12.5 Hz, 0.37 m",
               license="MIT (reha subset CC-BY-NC-SA 3.0)", n_files=len(files), n_scans=int(n_scan), scans_per_split=per_split,
               n_person_labels=int(len(rows)), match_radius=R_MATCH, r_max=R_MAX)
    out["recall_all"] = cp(int(hit.sum()), len(hit))
    out["recall_by_range"] = {f"{lo}-{hi}": dict(n=int(((r >= lo) & (r < hi)).sum()),
                                                  ci=cp(int(hit[(r >= lo) & (r < hi)].sum()), int(((r >= lo) & (r < hi)).sum())))
                              for lo, hi in BINS}
    rng = np.random.default_rng(0)
    b = rng.choice(fp, (2000, len(fp))).mean(axis=1)
    out["fp_per_scan"] = [float(fp.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    m = hit == 1
    # σ_r(r): độ lệch chuẩn bền (1,4826·MAD) theo khoảng, rồi khớp a + b·r
    centers, sds = [], []
    for lo, hi in BINS:
        s_ = m & (r >= lo) & (r < hi)
        if s_.sum() >= 30:
            x = dr[s_]
            centers.append(float(np.median(r[s_]))); sds.append(float(1.4826 * np.median(np.abs(x - np.median(x)))))
    bb, aa = np.polyfit(centers, sds, 1) if len(centers) >= 2 else (np.nan, np.nan)
    out["range_error"] = dict(bias=float(np.median(dr[m])), robust_sd_by_range=dict(zip([f"{c:.2f}" for c in centers], sds)),
                              fit_a=float(aa), fit_b=float(bb))
    out["bearing_error_deg"] = dict(bias=float(np.degrees(np.median(db[m]))),
                                    robust_sd=float(np.degrees(1.4826 * np.median(np.abs(db[m] - np.median(db[m]))))))
    out["provenance"] = dict(zip_sha256=sha(root.parent / "DROWv2-data.zip") if (root.parent / "DROWv2-data.zip").exists() else None,
                             detector="pu_nmpc_core.detect.cluster_scan (default parameters)")
    write_json(RESULTS / "drow_validation.json", out)
    print(json.dumps({k: out[k] for k in ("n_scans", "n_person_labels", "recall_all", "fp_per_scan", "range_error", "bearing_error_deg")}, indent=1))
    for k, v in out["recall_by_range"].items():
        print(k, v["n"], [round(x, 3) for x in v["ci"]])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
