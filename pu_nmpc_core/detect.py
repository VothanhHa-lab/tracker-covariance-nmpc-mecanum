"""People detection from 2D LiDAR scans and from YOLO boxes + depth images (pure, testable functions)."""
from __future__ import annotations

import numpy as np


def cluster_scan(ranges, angle_min, angle_inc, jump=0.13, min_pts=3, leg_width=(0.05, 0.35),
                 merge_dist=0.45, r_min=0.3, r_max=8.0, mask=None):
    """Return a list of (range, bearing) person candidates in the LiDAR frame.

    1. Split clusters where two consecutive points are more than `jump` (m) apart.
    2. Keep clusters with >= min_pts points and a width within leg_width (one leg or a narrow body).
    3. Merge clusters whose centres are closer than merge_dist (two legs) into one candidate.
    mask: bool array of the same length as ranges, True = point of the static map (discarded).
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
    """Box (u1, v1, u2, v2) + depth image (m) -> (range, bearing) in the camera body frame (x forward, y left).

    Uses the median depth over the central part of the box (fraction `shrink`) to avoid the background.
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
    X = (u - cx) / fx * z               # optical frame: X right, Z forward
    return float(np.hypot(X, z)), float(np.arctan2(-X, z))
