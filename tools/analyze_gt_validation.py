"""P1: accuracy of the ground-truth system - tags placed at measured points (config/gt_validation_points.yaml).

python3 tools/analyze_gt_validation.py <data>/G01/csv
YAML: points: {participant_tag_id_as_participant: [x, y, h]} - record one session per tag placement,
or use several tags at once. Writes results/gt_validation.json.
"""
import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, ROOT, is_synthetic, load, provenance, write_json  # noqa: E402


def main(dirs):
    pts = yaml.safe_load((ROOT / "config" / "gt_validation_points.yaml").read_text())["points"]
    errs = []
    for d in dirs:
        gp = load(d, "gt_people")
        for pid, g in gp.groupby("participant"):
            key = f"{d.parent.name}:{pid}"
            if key not in pts:
                continue
            x, y = pts[key][:2]
            e = np.hypot(g.x.values - x, g.y.values - y)
            errs.append(dict(point=key, n=len(e), bias=float(np.hypot(g.x.mean() - x, g.y.mean() - y)),
                             std=float(np.std(np.column_stack([g.x, g.y]), axis=0).mean()), rmse=float(np.sqrt(np.mean(e ** 2)))))
    if not errs:
        print("no matching validation points"); return
    rm = np.array([e["rmse"] for e in errs])
    write_json(RESULTS / "gt_validation.json", dict(points=errs, rmse=float(np.sqrt(np.mean(rm ** 2))), max_rmse=float(rm.max()),
                                                   n_points=len(errs),
                                                   provenance=provenance(dirs, synthetic=any(is_synthetic(d) for d in dirs))))
    print(f"GT RMSE {np.sqrt(np.mean(rm ** 2)):.3f} m over {len(errs)} points")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    main(ap.parse_args().dirs)
