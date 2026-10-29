"""Phát hiện người từ quét LiDAR 2D và từ hộp YOLO + ảnh độ sâu (hàm thuần, kiểm thử được)."""
from __future__ import annotations

import numpy as np


def cluster_scan(ranges, angle_min, angle_inc, jump=0.13, min_pts=3, leg_width=(0.05, 0.35),
                 merge_dist=0.45, r_min=0.3, r_max=8.0, mask=None):
    """Trả về danh sách (range, bearing) của ứng viên người trong khung LiDAR.

    1. Tách cụm khi hai điểm liền kề cách nhau > jump (m).
    2. Giữ cụm có ≥ min_pts điểm và bề rộng trong leg_width (một chân hoặc thân người hẹp).
    3. Gộp các cụm có tâm cách nhau < merge_dist (hai chân) thành một ứng viên.
    mask: mảng bool cùng độ dài ranges, True = điểm thuộc bản đồ tĩnh (bị bỏ).
    """
    r = np.asarray(ranges, float)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r >= r_min) & (r <= r_max)
    if mask is not None:
        ok &= ~np.asarray(mask, bool)
    rr = np.where(ok, r, 0.0)
    xy = np.column_stack([rr * np.cos(a), rr * np.sin(a)])
    clusters, cur = [], []
    for i in range(len(r)):
        if not ok[i]:
            if cur:
                clusters.append(cur)
            cur = []
            continue
        if cur and np.linalg.norm(xy[i] - xy[cur[-1]]) > jump:
            clusters.append(cur)
            cur = []
        cur.append(i)
    if cur:
        clusters.append(cur)
    cents = []
    for c in clusters:
        if len(c) < min_pts:
            continue
        w = np.linalg.norm(xy[c[-1]] - xy[c[0]])
        if leg_width[0] <= w <= leg_width[1]:
            cents.append(xy[c].mean(axis=0))
    merged, used = [], set()
    for i, ci in enumerate(cents):
        if i in used:
            continue
        group = [ci]
        for j in range(i + 1, len(cents)):
            if j not in used and np.linalg.norm(cents[j] - ci) < merge_dist:
                group.append(cents[j])
                used.add(j)
        m = np.mean(group, axis=0)
        merged.append((float(np.hypot(*m)), float(np.arctan2(m[1], m[0]))))
    return merged


def bbox_to_range_bearing(bbox, depth, fx, fy, cx, cy, shrink=0.4, z_range=(0.3, 8.0)):
    """Hộp (u1, v1, u2, v2) + ảnh độ sâu (m) → (range, bearing) trong khung thân camera (x tới, y trái).

    Lấy trung vị độ sâu trong phần trung tâm của hộp (tỉ lệ `shrink`) để tránh nền phía sau.
    """
    u1, v1, u2, v2 = bbox
    du, dv = (u2 - u1) * (1 - shrink) / 2, (v2 - v1) * (1 - shrink) / 2
    a, b = int(max(u1 + du, 0)), int(min(u2 - du, depth.shape[1] - 1))
    c, d = int(max(v1 + dv, 0)), int(min(v2 - dv, depth.shape[0] - 1))
    patch = depth[c:d + 1, a:b + 1]
    vals = patch[np.isfinite(patch) & (patch > z_range[0]) & (patch < z_range[1])]
    if vals.size < 10:
        return None
    z = float(np.median(vals))
    u = (u1 + u2) / 2
    X = (u - cx) / fx * z               # khung quang học: X phải, Z tới
    return float(np.hypot(X, z)), float(np.arctan2(-X, z))
