"""Chỉ số đánh giá và thống kê (dùng chung cho kiểm thử phần mềm và phân tích dữ liệu thật)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta, chi2


def clopper_pearson(k, n, conf=0.95):
    """Khoảng hai phía cho tỉ lệ k/n."""
    if n == 0:
        return (np.nan, np.nan)
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def upper_bound_zero(n, conf=0.95):
    """Cận trên một phía khi quan sát 0/n sự kiện: 1 − (1 − conf)^(1/n)."""
    return 1 - (1 - conf) ** (1 / n)


def paired_bootstrap(a, b, n_boot=10000, seed=0, conf=0.95):
    """Trung bình chênh lệch ghép cặp a − b và khoảng bootstrap percentile."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(d.mean()), float(lo), float(hi), int(len(d))


def nees(err, P):
    """NEES cho từng mẫu: err [T,d], P [T,d,d]."""
    return np.einsum("ti,ti->t", err, np.linalg.solve(P, err[..., None])[..., 0])


def nees_consistency(e, d, conf=0.95):
    """Trung bình NEES và khoảng chấp nhận chi-bình phương cho trung bình của T mẫu độc lập.

    Lưu ý: các mẫu liên tiếp của một vệt tương quan với nhau; khoảng này chỉ mang tính tham khảo,
    kết luận chính dựa trên độ phủ elip (ellipse_coverage) theo từng người/lượt.
    """
    T = len(e)
    a = 1 - conf
    return float(np.mean(e)), chi2.ppf(a / 2, d * T) / T, chi2.ppf(1 - a / 2, d * T) / T


def ellipse_coverage(err, P, level=0.95):
    """Tỉ lệ sai số rơi vào elip mức `level` của hiệp phương sai (2D)."""
    return float(np.mean(nees(err, P) <= chi2.ppf(level, err.shape[1])))


def ospa(X, Y, c=1.0, p=1):
    """Khoảng cách OSPA giữa hai tập điểm 2D (Schuhmacher và cộng sự, 2008)."""
    X, Y = np.asarray(X, float).reshape(-1, 2), np.asarray(Y, float).reshape(-1, 2)
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)
    if m > n:
        X, Y, m, n = Y, X, n, m
    D = np.minimum(np.linalg.norm(X[:, None] - Y[None], axis=2), c) ** p
    r, k = linear_sum_assignment(D)
    return float(((D[r, k].sum() + c ** p * (n - m)) / n) ** (1 / p))


def time_offset(t_a, x_a, t_b, x_b, search=(-0.5, 0.5), step=0.005):
    """Độ trễ τ cực tiểu hóa sai số giữa x_a(t) và x_b(t − τ) (ước lượng trễ hiệu dụng của bộ bám).

    τ > 0 nghĩa là chuỗi a chậm hơn chuỗi b.
    """
    taus = np.arange(search[0], search[1] + step / 2, step)
    best = (np.inf, np.nan)
    for tau in taus:
        xb = np.column_stack([np.interp(t_a - tau, t_b, x_b[:, j], left=np.nan, right=np.nan)
                              for j in range(x_b.shape[1])])
        e = np.nanmean(np.sum((x_a - xb) ** 2, axis=1))
        if e < best[0]:
            best = (e, tau)
    return float(best[1]), float(np.sqrt(best[0]))


def min_distance_and_intrusion(t, robot_xy, people_xy, zone):
    """Khoảng cách tâm–tâm nhỏ nhất và tổng thời gian ở trong vùng `zone` (m).

    people_xy: [T, M, 2] (NaN khi người không có mặt).
    """
    d = np.linalg.norm(people_xy - robot_xy[:, None, :], axis=2)
    dmin_t = np.nanmin(d, axis=1) if d.size else np.full(len(t), np.inf)
    dt = np.diff(t, append=t[-1])
    return float(np.nanmin(dmin_t)), float(np.sum(dt[dmin_t < zone]))
