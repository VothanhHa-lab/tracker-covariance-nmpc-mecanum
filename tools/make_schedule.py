"""Balanced randomised trial schedule (run ONCE, store the random seed; do not modify after the start).

python3 tools/make_schedule.py --phase R --participants 5 --reps 4 --seed 20261101 --out protocol/schedule_R.csv
Phase D (pilot/development) and phase R (reporting) use different seeds; only phase R is reported in the paper.
The method order within each scenario block rotates as a Latin square (participant, repetition).
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
    print(f"{len(rows)} trials; {n_cell} trials per scenario x method cell; "
          f"95 % upper bound with 0 contacts per cell: {1 - 0.05 ** (1 / n_cell):.3f}")
