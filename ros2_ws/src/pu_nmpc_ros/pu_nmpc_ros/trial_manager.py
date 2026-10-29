"""Quản lý một lượt thử theo lịch đã ngẫu nhiên hóa (tools/make_schedule.py).

Chạy: ros2 run pu_nmpc_ros trial_manager --ros-args -p trial_id:=R017 -p schedule_csv:=... -p scenarios_yaml:=...
Người vận hành dùng `mark_event` để gửi: start | intervention | abort | note:<văn bản>.
Kết quả ghi vào <data_dir>/<trial_id>/trial.json. Kết quả chính được tính lại offline từ ground truth;
trường `outcome_online` chỉ để đối chiếu.
"""
import csv
import json
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from .common import tf_to_pose2d


class TrialManager(Node):
    def __init__(self):
        super().__init__("trial_manager")
        p = self.declare_parameters("", [("trial_id", ""), ("schedule_csv", ""), ("scenarios_yaml", ""),
                                         ("data_dir", "data"), ("map_frame", "map"), ("base_frame", "base_link")])
        self.prm = {x.name: x.value for x in p}
        with open(self.prm["schedule_csv"]) as f:
            rows = {r["trial_id"]: r for r in csv.DictReader(f)}
        self.row = rows[self.prm["trial_id"]]
        with open(self.prm["scenarios_yaml"]) as f:
            self.sc = yaml.safe_load(f)["scenarios"][self.row["scenario"]]
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_path = self.create_publisher(Path, "/trial/path", latched)
        self.pub_id = self.create_publisher(String, "/trial/id", latched)
        self.pub_en = self.create_publisher(Bool, "/trial/enable", 10)
        self.create_subscription(String, "/trial/event", self.event_cb, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.out_dir = os.path.join(self.prm["data_dir"], self.prm["trial_id"])
        os.makedirs(self.out_dir, exist_ok=True)
        self.rec = dict(**self.row, events=[], wall_start=None, outcome_online="not_started",
                        scenario_cfg=self.sc, host=os.uname().nodename)
        self.enabled, self.t_start = False, None
        path = Path()
        path.header.frame_id = self.prm["map_frame"]
        for x, y in self.sc["waypoints"]:
            ps = PoseStamped()
            ps.header.frame_id = self.prm["map_frame"]
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            path.poses.append(ps)
        self.pub_path.publish(path)
        self.pub_id.publish(String(data=self.prm["trial_id"]))
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"Lượt {self.prm['trial_id']}: {self.row['scenario']} / {self.row['method']} / "
                               f"{self.row['participant']} — KIỂM TRA nmpc_controller đang chạy method={self.row['method']}")
        self._save()

    def event_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.rec["events"].append(dict(t=now, wall=time.time(), event=msg.data))
        if msg.data == "start" and not self.enabled:
            self.enabled, self.t_start = True, now
            self.rec["t_start"] = now
            self.rec["wall_start"] = time.time()
            self.rec["outcome_online"] = "running"
        elif msg.data in ("intervention", "abort"):
            self._finish("intervention" if msg.data == "intervention" else "aborted")
        self._save()

    def tick(self):
        self.pub_en.publish(Bool(data=self.enabled))
        if not self.enabled:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            x, y, _ = tf_to_pose2d(self.tf.lookup_transform(self.prm["map_frame"], self.prm["base_frame"], Time()))
            if np.hypot(x - self.sc["waypoints"][-1][0], y - self.sc["waypoints"][-1][1]) < self.sc.get("goal_tol", 0.15):
                self._finish("goal_reached")
        except Exception:  # noqa: BLE001
            pass
        if now - self.t_start > self.sc.get("timeout", 60.0):
            self._finish("timeout")

    def _finish(self, outcome):
        if not self.enabled and self.rec["outcome_online"] not in ("running", "not_started"):
            return
        self.enabled = False
        self.pub_en.publish(Bool(data=False))
        self.rec["outcome_online"] = outcome
        self.rec["t_end"] = self.get_clock().now().nanoseconds * 1e-9
        self._save()
        self.get_logger().info(f"Kết thúc: {outcome}")

    def _save(self):
        with open(os.path.join(self.out_dir, "trial.json"), "w") as f:
            json.dump(self.rec, f, indent=1, ensure_ascii=False)


def main():
    rclpy.init()
    rclpy.spin(TrialManager())


if __name__ == "__main__":
    main()
