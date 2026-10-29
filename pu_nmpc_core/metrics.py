"""Evaluation metrics and statistics (shared by software tests and the analysis of real data)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta, chi2


def clopper_pearson(k, n, conf=0.95):
    """Two-sided interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def upper_bound_zero(n, conf=0.95):
    """One-sided upper bound when 0/n events are observed: 1 - (1 - conf)^(1/n)."""
    return 1 - (1 - conf) ** (1 / n)


def paired_bootstrap(a, b, n_boot=10000, seed=0, conf=0.95):
    """Mean paired difference a - b and its percentile-bootstrap interval."""
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
    """NEES per sample: err [T,d], P [T,d,d]."""
    return np.einsum("ti,ti->t", err, np.linalg.solve(P, err[..., None])[..., 0])


def nees_consistency(e, d, conf=0.95):
    """Mean NEES and the chi-square acceptance interval for the mean of T independent samples.

    Note: consecutive samples of a track are correlated; this interval is indicative only, and the main
    conclusion rests on the ellipse coverage (ellipse_coverage) per person/run.
    """
    T = len(e)
    a = 1 - conf
    return float(np.mean(e)), chi2.ppf(a / 2, d * T) / T, chi2.ppf(1 - a / 2, d * T) / T


def ellipse_coverage(err, P, level=0.95):
    """Fraction of errors inside the covariance ellipse of level `level` (2D)."""
    return float(np.mean(nees(err, P) <= chi2.ppf(level, err.shape[1])))


def ospa(X, Y, c=1.0, p=1):
    """OSPA distance between two sets of 2D points (Schuhmacher et al., 2008)."""
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
    """Delay tau minimising the error between x_a(t) and x_b(t - tau) (effective delay of the tracker).

    tau > 0 means that series a lags series b.
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
    """Minimum centre distance and total time spent inside the zone `zone` (m).

    people_xy: [T, M, 2] (NaN when a person is absent).
    """
    d = np.linalg.norm(people_xy - robot_xy[:, None, :], axis=2)
    dmin_t = np.nanmin(d, axis=1) if d.size else np.full(len(t), np.inf)
    dt = np.diff(t, append=t[-1])
    return float(np.nanmin(dmin_t)), float(np.sum(dt[dmin_t < zone]))
