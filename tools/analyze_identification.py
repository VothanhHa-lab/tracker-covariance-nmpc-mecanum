"""P3: nhận dạng khâu vận tốc bậc nhất có trễ từ /cmd_vel và vận tốc odometry.

python3 tools/analyze_identification.py <data>/I01/csv [<data>/I02/csv ...]
Khớp trên 2/3 đầu mỗi phiên, kiểm chứng trên 1/3 cuối (không chồng lấn). Ghi results/identification.json.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, is_synthetic, load, provenance, write_json  # noqa: E402

from pu_nmpc_core.identification import fit_first_order, simulate_first_order  # noqa: E402

AX = [("vx", "vx"), ("vy", "vy"), ("wz", "wz")]


def main(dirs):
    out, synth = dict(axes={}), any(is_synthetic(d) for d in dirs)
    for ax_cmd, ax_meas in AX:
        T, U, Y = [], [], []
        for d in dirs:
            cmd, od = load(d, "cmd"), load(d, "odom")
            if cmd.empty or od.empty:
                continue
            t = od.stamp.values
            idx = np.searchsorted(cmd.stamp.values, t, side="right") - 1
            u = cmd[ax_cmd].values[np.clip(idx, 0, len(cmd) - 1)]
            T.append(t - t[0]); U.append(u); Y.append(od[ax_meas].values)
        if not T:
            continue
        fits = []
        for t, u, y in zip(T, U, Y):
            n = int(len(t) * 2 / 3)
            f = fit_first_order(t[:n], u[:n], y[:n])
            if f is None:
                continue
            yh = simulate_first_order(t, u, f["K"], f["tau"], f["delay"])   # mô phỏng toàn phiên, chấm điểm 1/3 cuối
            e = yh[n:] - y[n:]
            rmse = float(np.sqrt(np.mean(e ** 2)))
            fits.append(dict(**f, val_rmse=rmse, val_nrmse=rmse / max(np.ptp(y[n:]), 1e-9), n_train=n, n_val=len(t) - n))
        if fits:
            out["axes"][ax_cmd] = dict(
                sessions=fits, tau=float(np.median([f["tau"] for f in fits])),
                K=float(np.median([f["K"] for f in fits])), delay=float(np.median([f["delay"] for f in fits])),
                val_nrmse_max=float(max(f["val_nrmse"] for f in fits)))
    out["provenance"] = provenance(dirs, synthetic=synth)
    write_json(RESULTS / "identification.json", out)
    print({k: (round(v["tau"], 3), round(v["K"], 3), round(v["delay"], 3)) for k, v in out["axes"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    main(ap.parse_args().dirs)
