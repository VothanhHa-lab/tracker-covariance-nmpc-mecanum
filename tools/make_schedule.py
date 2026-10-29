"""Lịch thử ngẫu nhiên hóa có cân bằng (chạy MỘT lần, lưu cả hạt ngẫu nhiên; không sửa sau khi bắt đầu).

python3 tools/make_schedule.py --phase R --participants 5 --reps 4 --seed 20261101 --out protocol/schedule_R.csv
Pha D (pilot/phát triển) và pha R (báo cáo) dùng hạt khác nhau; chỉ pha R được báo cáo trong bài.
Thứ tự phương pháp trong mỗi khối kịch bản xoay vòng theo hình vuông Latin (người tham gia, lặp).
"""
import argparse
import csv

import numpy as np

SCEN = ["H1_crossing", "H2_head_on", "H3_narrow", "H4_occlusion", "H5_side"]
METH = ["cov", "fixed", "heuristic"]


def make(phase, n_part, reps, seed, scenarios=SCEN, methods=METH):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for pi in range(n_part):
        pid = f"P{pi + 1:02d}"
        for sc in rng.permutation(scenarios):
            for r in range(reps):
                shift = (pi + r) % len(methods)
                for m in methods[shift:] + methods[:shift]:
                    k += 1
                    rows.append(dict(trial_id=f"{phase}{k:03d}", phase=phase, participant=pid, scenario=str(sc),
                                     method=m, rep=r + 1, seed=seed))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["D", "R"], required=True)
    ap.add_argument("--participants", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = make(a.phase, a.participants, a.reps, a.seed)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_cell = a.participants * a.reps
    print(f"{len(rows)} lượt; {n_cell} lượt mỗi ô kịch bản×phương pháp; "
          f"cận trên 95 % khi 0 va chạm/ô: {1 - 0.05 ** (1 / n_cell):.3f}")
