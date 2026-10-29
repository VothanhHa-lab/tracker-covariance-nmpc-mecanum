"""Hiệu chỉnh hệ số phóng hiệp phương sai γ trên hạt phát triển (results/sim_devcal.csv).

Quy tắc chốt trước khi chạy chiến dịch revision: γ là giá trị nhỏ nhất trong lưới GAMMAS sao cho độ phủ
elip 95 % gộp trên mọi lượt dev đạt ≥ 95 % ở cả ba tầm 0, 0.5 và 1 s. Kết quả ghi vào
config/simstudy.yaml (khối revision) và results/calibration.json.
"""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import RESULTS, ROOT  # noqa: E402

from pu_nmpc_core.simstudy import GAMMAS  # noqa: E402

d = pd.read_csv(RESULTS / "sim_devcal.csv")
cov = {h: {g: d[f"cov_in_{h}_g{g}"].sum() / d[f"cov_n_{h}"].sum() for g in GAMMAS} for h in ("0.0", "0.5", "1.0")}
gamma = next(g for g in GAMMAS if all(cov[h][g] >= 0.95 for h in cov))
(RESULTS / "calibration.json").write_text(json.dumps(dict(rule="min gamma with pooled coverage >= 0.95 at h=0,0.5,1 s",
                                                          n_runs=len(d), coverage=cov, gamma=gamma), indent=1))
y = ROOT / "config" / "simstudy.yaml"
t = y.read_text()
if "\nrevision:" not in t:
    t = t.rstrip() + (f"\nrevision:\n  cov_inflate: {{value: {gamma}, source: \"calibrate_inflation.py on dev seeds 0-9 "
                      f"(sim_devcal.csv, {len(d)} runs)\"}}\n")
    y.write_text(t)
print("gamma =", gamma, {h: round(cov[h][gamma], 3) for h in cov})
