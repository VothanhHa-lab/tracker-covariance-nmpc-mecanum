"""Covariance-based safety constraint (contribution C2) and the two baseline margins.

Proposition (stated in the paper, proof in the appendix). Suppose that at step k the true position of the person
p_h ~ (p_hat_h, P_h) and the robot position error e_r ~ (0, P_r) are independent, and n is a unit vector chosen
from the estimates only (not from the true values). If the planned robot position p_r satisfies
    n^T (p_r − p̂_h) ≥ d0 + κ(ε) · sqrt(n^T (P_h + P_r) n),
then Pr(||p_r,true - p_h|| < d0) <= eps, with
    kappa = Phi^{-1}(1 - eps)        if the errors are Gaussian (kind="gauss"),
    kappa = sqrt((1 - eps)/eps)      for any distribution with that covariance (Cantelli, kind="cantelli").
Proof: ||a|| >= n^T a, then apply the one-sided inequality to the scalar n^T(e_r - e_h).
The guarantee holds only as far as the tracker covariance is correct (check NEES/coverage on real
data, tools/analyze_tracking.py) and the constant-velocity model holds over the prediction horizon.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .tracker import cv_matrices


def kappa(eps: float, kind: str = "gauss") -> float:
    if not 0 < eps < 1:
        raise ValueError("eps must lie in (0, 1)")
    if kind == "gauss":
        return float(norm.ppf(1 - eps))
    if kind == "cantelli":
        return float(np.sqrt((1 - eps) / eps))
    raise ValueError(kind)


def propagate(x, P, dt, N, q):
    """Constant-velocity prediction of a person: centres [N,2] and position covariances [N,2,2] at k = 1..N."""
    F, Q = cv_matrices(dt, q)
    C, S = np.zeros((N, 2)), np.zeros((N, 2, 2))
    for k in range(N):
        x = F @ x
        P = F @ P @ F.T + Q
        C[k], S[k] = x[:2], P[:2, :2]
    return C, S


def steady_state_cov(dt, q, r_var, n_iter=2000):
    """Steady-state posterior covariance of a CV filter updated with positions every dt with isotropic noise r_var (m^2)."""
    F, Q = cv_matrices(dt, q)
    Hm = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = r_var * np.eye(2)
    P = np.eye(4)
    for _ in range(n_iter):
        Pm = F @ P @ F.T + Q
        K = Pm @ Hm.T @ np.linalg.inv(Hm @ Pm @ Hm.T + R)
        P = (np.eye(4) - K @ Hm) @ Pm
    return P


def normals(robot_pred, centers):
    """Unit normals from the person to the planned robot position (linearisation point)."""
    d = robot_pred - centers
    n = np.linalg.norm(d, axis=1, keepdims=True)
    fallback = np.array([[1.0, 0.0]])
    return np.where(n > 1e-6, d / np.maximum(n, 1e-9), fallback)


def cov_margins(nvec, S, P_r, d0, eps, kind="gauss"):
    k = kappa(eps, kind)
    sig = np.sqrt(np.einsum("ki,kij,kj->k", nvec, S + P_r[None], nvec))
    return d0 + k * sig, sig


def heuristic_risk(robot_xy, robot_v, person_xy, person_v, robot_heading, person_heading, hp):
    """Risk index of the CA-NMPC formulation, implemented as in its Eqs. (27)-(34), used as a baseline.

    Note: with non-negative weights summing to one and non-negative terms, the sigmoid lies in
    [0.5, sigmoid(1)] ~ [0.5, 0.73]; this is reported as is, the formula is not modified.
    """
    d = np.linalg.norm(np.asarray(person_xy) - np.asarray(robot_xy))
    rho_d = np.exp(-d / hp["d_s"])
    rho_v = min(np.linalg.norm(np.asarray(person_v) - np.asarray(robot_v)) / hp["v_rel_max"], 1.0)
    rho_t = (1 - np.cos(person_heading - robot_heading)) / 2
    z = hp["w_d"] * rho_d + hp["w_v"] * rho_v + hp["w_t"] * rho_t
    return 1.0 / (1.0 + np.exp(-z))
