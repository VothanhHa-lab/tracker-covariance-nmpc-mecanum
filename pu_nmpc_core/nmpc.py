"""Multiple-shooting NMPC with CasADi + IPOPT.

Collision constraints are half-planes that are linear in the robot position:
    n_{s,k}^T (p_{k+1} − c_{s,k}) + slack_{s,k} ≥ m_{s,k}
for every "slot" s (a person or a static LiDAR point) and every step k. The parameters (c, n, m) are
computed by controller.py for each method; the NMPC does not distinguish methods, which keeps the comparison
fair (same solver, same cost, same wheel limits).

On a Raspberry Pi 4, IPOPT may not reach 10 Hz; then use acados (SQP-RTI) with the same model,
or run the controller on a carried laptop, and state the configuration in the paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .models import RobotParams, rk4, wheel_matrix


@dataclass
class NMPCParams:
    N: int = 20
    dt: float = 0.1
    n_slots: int = 12              # number of constraint slots (people + static points)
    n_people_slots: int | None = None  # slots from this index on are static points (walls)
    static_slack_mult: float = 100.0   # heavier penalty on wall-constraint violation -> nearly hard constraint
    input_delay: float = 0.0           # s, actuation delay (identified); the initial state is predicted over the delay
    q_pos: float = 10.0
    q_th: float = 2.0
    q_term: float = 5.0            # multiplier for the last step
    r_u: float = 0.1
    r_du: float = 2.0
    w_slack_l1: float = 1e3
    w_slack_l2: float = 1e4
    max_iter: int = 60
    max_cpu_time: float = 0.08     # s


class NMPC:
    def __init__(self, rp: RobotParams, mp: NMPCParams = NMPCParams()):
        self.rp, self.mp = rp, mp
        N, S = mp.N, mp.n_slots
        opti = ca.Opti()
        X = opti.variable(6, N + 1)
        U = opti.variable(3, N)
        SL = opti.variable(S, N)
        x0 = opti.parameter(6)
        u_prev = opti.parameter(3)
        ref = opti.parameter(3, N)          # X, Y, theta at k = 1..N
        C = opti.parameter(2 * S, N)        # slot centres
        Nn = opti.parameter(2 * S, N)       # slot normals
        M = opti.parameter(S, N)            # slot margins
        sv = opti.parameter()               # speed-limit scale (1 = nominal)

        W = wheel_matrix(rp)
        wmax = rp.wheel_speed_max
        amax = np.array(rp.body_acc_max) * mp.dt
        J = 0
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == rk4(X[:, k], U[:, k], mp.dt, rp, lib=ca))
            wk = ca.mtimes(W, U[:, k])
            opti.subject_to(opti.bounded(-sv * wmax, wk, sv * wmax))
            if rp.omega_max:
                opti.subject_to(opti.bounded(-rp.omega_max, U[2, k], rp.omega_max))
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            opti.subject_to(opti.bounded(-amax, du, amax))
            w = mp.q_term if k == N - 1 else 1.0
            e = X[0:2, k + 1] - ref[0:2, k]
            J += w * (mp.q_pos * ca.sumsqr(e) + mp.q_th * 2 * (1 - ca.cos(X[2, k + 1] - ref[2, k])))
            J += mp.r_u * ca.sumsqr(U[:, k]) + mp.r_du * ca.sumsqr(du)
            for s in range(S):
                n_sk = Nn[2 * s:2 * s + 2, k]
                c_sk = C[2 * s:2 * s + 2, k]
                opti.subject_to(ca.dot(n_sk, X[0:2, k + 1] - c_sk) + SL[s, k] >= M[s, k])
        opti.subject_to(ca.vec(SL) >= 0)
        ns = S if mp.n_people_slots is None else mp.n_people_slots
        for s in range(S):
            w = 1.0 if s < ns else mp.static_slack_mult
            J += w * (mp.w_slack_l1 * ca.sum1(SL[s, :].T) + mp.w_slack_l2 * ca.sumsqr(SL[s, :]))
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": mp.max_iter, "max_cpu_time": mp.max_cpu_time,
                     "warm_start_init_point": "yes", "sb": "yes"})
        self.opti = opti
        self.v = dict(X=X, U=U, SL=SL)
        self.p = dict(x0=x0, u_prev=u_prev, ref=ref, C=C, N=Nn, M=M, sv=sv)
        self._guess = None

    def inactive_slots(self, robot_pred):
        """Unused slots: centre 100 m away from the linearisation point, margin 0 (always satisfied)."""
        S, N = self.mp.n_slots, self.mp.N
        n = np.tile(np.array([1.0, 0.0]), (S, N, 1))
        c = robot_pred[None, :, :] - 100.0 * n
        return c, n, np.zeros((S, N))

    def solve(self, x0, u_prev, ref, c, n, m, speed_scale=1.0):
        """c, n: [S, N, 2]; m: [S, N]; ref: [N, 3]. Returns a result dict."""
        o, p, mp = self.opti, self.p, self.mp
        o.set_value(p["x0"], x0)
        o.set_value(p["u_prev"], u_prev)
        o.set_value(p["ref"], ref.T)
        o.set_value(p["C"], c.transpose(0, 2, 1).reshape(2 * mp.n_slots, mp.N))
        o.set_value(p["N"], n.transpose(0, 2, 1).reshape(2 * mp.n_slots, mp.N))
        o.set_value(p["M"], m)
        o.set_value(p["sv"], speed_scale)
        if self._guess is not None:
            Xg, Ug = self._guess
            o.set_initial(self.v["X"], Xg)
            o.set_initial(self.v["U"], Ug)
        else:
            o.set_initial(self.v["X"], np.tile(np.asarray(x0)[:, None], (1, mp.N + 1)))
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            ok, stats = True, sol.stats()
            Xs, Us, SLs = sol.value(self.v["X"]), sol.value(self.v["U"]), sol.value(self.v["SL"])
        except RuntimeError:
            # time limit / no convergence: take the current iterate (debug); the controller decides what to do
            ok, stats = False, o.stats()
            Xs, Us, SLs = o.debug.value(self.v["X"]), o.debug.value(self.v["U"]), o.debug.value(self.v["SL"])
        dt_ms = 1e3 * (time.perf_counter() - t0)
        Us = np.atleast_2d(Us)
        self._guess = (np.hstack([Xs[:, 1:], Xs[:, -1:]]), np.hstack([Us[:, 1:], Us[:, -1:]]))
        return dict(ok=ok, status=stats.get("return_status", ""), u=Us[:, 0].copy(), X=Xs, U=Us,
                    max_slack=float(np.max(SLs)) if np.size(SLs) else 0.0, solve_ms=dt_ms,
                    iters=int(stats.get("iter_count", -1)))

    def reset(self):
        self._guess = None
