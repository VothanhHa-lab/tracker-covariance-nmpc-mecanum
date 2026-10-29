"""Tiện ích chung cho công cụ phân tích: nguồn gốc dữ liệu (provenance) và đọc phiên."""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
import os

RESULTS = pathlib.Path(os.environ.get("PU_NMPC_RESULTS", ROOT / "results"))


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "pu_nmpc_core").glob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def provenance(inputs, synthetic=False, extra=None):
    """Mỗi tệp kết quả mang theo: tệp đầu vào + mã băm, mã băm mã nguồn, cờ synthetic."""
    files = []
    for p in inputs:
        p = pathlib.Path(p)
        m = p / "manifest.json" if p.is_dir() else p
        files.append(dict(path=str(p), sha256=sha(m) if m.exists() else None))
    return dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), code=code_hash(), synthetic=bool(synthetic),
                inputs=files, **(extra or {}))


def is_synthetic(csv_dir: pathlib.Path) -> bool:
    m = csv_dir / "manifest.json"
    return m.exists() and json.loads(m.read_text()).get("synthetic", False)


def load(csv_dir: pathlib.Path, name: str) -> pd.DataFrame:
    f = csv_dir / f"{name}.csv"
    if not f.exists() or f.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def interp_pose(df: pd.DataFrame, t, max_gap=0.2):
    """Nội suy tư thế (x, y, yaw) tại các thời điểm t; NaN nếu khoảng trống dữ liệu > max_gap."""
    df = df.sort_values("stamp")
    st = df.stamp.values
    out = np.column_stack([np.interp(t, st, df.x.values), np.interp(t, st, df.y.values),
                           np.interp(t, st, np.unwrap(df.yaw.values)) if "yaw" in df else np.zeros(len(t))])
    i = np.clip(np.searchsorted(st, t), 1, len(st) - 1)
    gap = np.minimum(np.abs(t - st[i - 1]), np.abs(st[i] - t))
    out[(gap > max_gap) | (t < st[0]) | (t > st[-1])] = np.nan
    return out


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
