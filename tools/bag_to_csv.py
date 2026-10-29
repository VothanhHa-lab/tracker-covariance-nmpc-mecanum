"""Trích rosbag2 (mcap/sqlite3) → CSV chuẩn hóa cho phân tích. Không cần cài ROS (dùng thư viện rosbags).

python3 tools/bag_to_csv.py <data_dir>/<id>/bag   → ghi <data_dir>/<id>/csv/*.csv và manifest.json

Thời gian: dùng header.stamp khi có (thời điểm đo), nếu không thì thời điểm ghi của bag (log_time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

MSG_DIR = pathlib.Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "pu_nmpc_msgs" / "msg"


def typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    for f in sorted(MSG_DIR.glob("*.msg")):
        add.update(get_types_from_msg(f.read_text(), f"pu_nmpc_msgs/msg/{f.stem}"))
    ts.register(add)
    return ts


def _stamp(msg, t_log):
    h = getattr(msg, "header", None)
    if h is not None and (h.stamp.sec or h.stamp.nanosec):
        return h.stamp.sec + 1e-9 * h.stamp.nanosec
    return t_log * 1e-9


def _yaw(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def extract(bag: pathlib.Path, out: pathlib.Path):
    from rosbags.highlevel import AnyReader
    out.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("detections", "tracks", "diag", "tf", "odom", "cmd", "gt_robot", "gt_people",
                            "pose_cov", "events", "enable", "guard")}
    with AnyReader([bag], default_typestore=typestore()) as r:
        for conn, t_log, raw in r.messages():
            m = r.deserialize(raw, conn.msgtype)
            tp, ts = conn.topic, _stamp(m, t_log)
            tl = t_log * 1e-9
            if tp.startswith("/people/detections/"):
                for i, d in enumerate(m.detections):
                    rows["detections"].append(dict(stamp=ts, t_log=tl, sensor=m.sensor, frame=m.header.frame_id,
                                                   idx=i, range=d.range, bearing=d.bearing, score=d.score))
                if not m.detections:
                    rows["detections"].append(dict(stamp=ts, t_log=tl, sensor=m.sensor, frame=m.header.frame_id,
                                                   idx=-1, range=np.nan, bearing=np.nan, score=np.nan))
            elif tp == "/people/tracks":
                for p in m.people:
                    rows["tracks"].append(dict(stamp=ts, t_log=tl, id=p.id, x=p.state[0], y=p.state[1], vx=p.state[2],
                                               vy=p.state[3], age=p.age, cam=int(p.camera_confirmed),
                                               **{f"P{i}": v for i, v in enumerate(p.covariance)}))
            elif tp == "/controller/diagnostics":
                rows["diag"].append(dict(stamp=ts, t_log=tl, trial=m.trial_id, method=m.method, mode=m.mode,
                                         status=m.solver_status, solve_ms=m.solve_ms, iters=m.iters,
                                         max_slack=m.max_slack, speed_scale=m.speed_scale, n_people=m.n_people,
                                         sigma_n=max(m.sigma_n) if len(m.sigma_n) else np.nan,
                                         rho=max(m.rho) if len(m.rho) else np.nan,
                                         cvx=m.cmd[0], cvy=m.cmd[1], cw=m.cmd[2]))
            elif tp in ("/tf", "/tf_static"):
                for tr in m.transforms:
                    s = tr.header.stamp
                    rows["tf"].append(dict(stamp=s.sec + 1e-9 * s.nanosec, parent=tr.header.frame_id,
                                           child=tr.child_frame_id, x=tr.transform.translation.x,
                                           y=tr.transform.translation.y, yaw=_yaw(tr.transform.rotation),
                                           static=int(tp == "/tf_static")))
            elif tp == "/odom":
                tw = m.twist.twist
                rows["odom"].append(dict(stamp=ts, t_log=tl, vx=tw.linear.x, vy=tw.linear.y, wz=tw.angular.z,
                                         x=m.pose.pose.position.x, y=m.pose.pose.position.y,
                                         yaw=_yaw(m.pose.pose.orientation)))
            elif tp == "/cmd_vel":
                rows["cmd"].append(dict(stamp=tl, vx=m.linear.x, vy=m.linear.y, wz=m.angular.z))
            elif tp == "/gt/robot":
                rows["gt_robot"].append(dict(stamp=ts, x=m.pose.position.x, y=m.pose.position.y,
                                             yaw=_yaw(m.pose.orientation)))
            elif tp == "/gt/people":
                for p in m.people:
                    rows["gt_people"].append(dict(stamp=ts, participant=p.id, x=p.state[0], y=p.state[1]))
            elif tp == "/amcl_pose":
                C = np.asarray(m.pose.covariance).reshape(6, 6)
                rows["pose_cov"].append(dict(stamp=ts, sxx=C[0, 0], syy=C[1, 1], sxy=C[0, 1], sth=C[5, 5]))
            elif tp == "/trial/event":
                rows["events"].append(dict(stamp=tl, event=m.data))
            elif tp == "/guard/active":
                rows["guard"].append(dict(stamp=tl, active=int(m.data)))
            elif tp == "/trial/enable":
                rows["enable"].append(dict(stamp=tl, enabled=int(m.data)))
    man = dict(bag=str(bag), files={})
    for k, v in rows.items():
        f = out / f"{k}.csv"
        pd.DataFrame(v).to_csv(f, index=False)
        man["files"][k] = dict(rows=len(v), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


class TFSeries:
    """Tra cứu tư thế 2D parent←child theo thời gian (nội suy tuyến tính), ghép chuỗi khung."""

    def __init__(self, tf: pd.DataFrame):
        self.s = {}
        for (p, c), g in tf.groupby(["parent", "child"]):
            g = g.sort_values("stamp")
            self.s[(p.lstrip("/"), c.lstrip("/"))] = (g.stamp.values, g.x.values, g.y.values, np.unwrap(g.yaw.values),
                                                      bool(g.static.max()))

    def _one(self, p, c, t):
        st, x, y, yaw, static = self.s[(p, c)]
        if static or len(st) == 1:
            return np.array([x[-1], y[-1], yaw[-1]]) * np.ones((len(np.atleast_1d(t)), 1))
        t = np.atleast_1d(t)
        out = np.column_stack([np.interp(t, st, x), np.interp(t, st, y), np.interp(t, st, yaw)])
        out[(t < st[0]) | (t > st[-1])] = np.nan
        return out

    def lookup(self, chain, t):
        """chain = ["map", "odom", "base_link"] → [T,3] tư thế base_link trong map."""
        T = None
        for p, c in zip(chain[:-1], chain[1:]):
            P = self._one(p, c, t)
            if T is None:
                T = P
            else:
                cth, sth = np.cos(T[:, 2]), np.sin(T[:, 2])
                T = np.column_stack([T[:, 0] + cth * P[:, 0] - sth * P[:, 1],
                                     T[:, 1] + sth * P[:, 0] + cth * P[:, 1], T[:, 2] + P[:, 2]])
        return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=pathlib.Path)
    a = ap.parse_args()
    m = extract(a.bag, a.bag.parent / "csv")
    print(json.dumps({k: v["rows"] for k, v in m["files"].items()}))
