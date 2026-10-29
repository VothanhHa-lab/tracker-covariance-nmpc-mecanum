"""NMPC (cov | fixed | heuristic) chạy ở 1/dt Hz; chỉ gửi lệnh khi /trial/enable = True.

An toàn: dừng có kiểm soát khi tư thế, odom hoặc vệt người quá cũ (watchdog); bộ giải lỗi → phanh.
Nút dừng khẩn phần cứng là lớp bảo vệ độc lập, không phụ thuộc node này.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from pu_nmpc_core import config
from pu_nmpc_core.controller import Controller, PathReference
from pu_nmpc_msgs.msg import ControllerDiagnostics, TrackedPersonArray

from .common import pose_cov_3x3, stamp_to_sec, tf_to_pose2d


class NMPCNode(Node):
    def __init__(self):
        super().__init__("nmpc_controller")
        p = self.declare_parameters("", [
            ("robot_yaml", ""), ("controller_yaml", ""), ("method", "cov"), ("map_frame", "map"),
            ("base_frame", "base_link"), ("odom_topic", "/odom"), ("pose_cov_topic", "/amcl_pose"),
            ("scan_topic", "/scan"), ("pose_timeout", 0.3), ("track_exclusion", 0.5),
            ("require_confirmation", "camera_or_moving"), ("moving_speed", 0.2)])
        self.prm = {x.name: x.value for x in p}
        rp = config.robot(self.prm["robot_yaml"])
        mp, npar, self.refp = config.controller(self.prm["controller_yaml"])
        mp.method = self.prm["method"]
        self.ctrl = Controller(rp, mp, npar)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.enabled, self.ref, self.trial_id = False, None, ""
        self.tracks, self.twist, self.scan_pts = None, None, None
        self.pose_cov = np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/trial/path", self.path_cb, latched)
        self.create_subscription(String, "/trial/id", lambda m: setattr(self, "trial_id", m.data), latched)
        self.create_subscription(Bool, "/trial/enable", self.enable_cb, 10)
        self.create_subscription(TrackedPersonArray, "people/tracks", lambda m: setattr(self, "tracks", m), 10)
        self.create_subscription(Odometry, self.prm["odom_topic"], self.odom_cb, 20)
        self.create_subscription(PoseWithCovarianceStamped, self.prm["pose_cov_topic"],
                                 lambda m: setattr(self, "pose_cov", pose_cov_3x3(m.pose.covariance)), 10)
        self.create_subscription(LaserScan, self.prm["scan_topic"], self.scan_cb, qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel_raw", 10)   # qua scan_stop_guard
        self.pub_diag = self.create_publisher(ControllerDiagnostics, "controller/diagnostics", 10)
        self.create_timer(npar.dt, self.tick)

    def path_cb(self, msg):
        wps = [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        try:
            yaw0 = tf_to_pose2d(self.tf.lookup_transform(self.prm["map_frame"], self.prm["base_frame"], Time()))[2]
        except Exception:  # noqa: BLE001
            yaw0 = 0.0
        self.ref = PathReference(wps, self.refp, heading0=yaw0)
        self.ctrl.reset()

    def enable_cb(self, msg):
        if msg.data and not self.enabled:
            self.ctrl.reset()
        self.enabled = msg.data

    def odom_cb(self, msg):
        tw = msg.twist.twist
        self.twist = (stamp_to_sec(msg.header.stamp), np.array([tw.linear.x, tw.linear.y, tw.angular.z]))

    def scan_cb(self, scan):
        try:
            tf = self.tf.lookup_transform(self.prm["map_frame"], scan.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return
        r = np.asarray(scan.ranges)
        a = scan.angle_min + scan.angle_increment * np.arange(len(r))
        ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
        x, y, th = tf_to_pose2d(tf)
        px = x + r[ok] * np.cos(a[ok] + th)
        py = y + r[ok] * np.sin(a[ok] + th)
        self.scan_pts = np.column_stack([px, py])[::2]

    def _people(self, now):
        if self.tracks is None:
            return [], np.inf
        out = []
        for t in self.tracks.people:
            x = np.array(t.state)
            moving = np.hypot(x[2], x[3]) > self.prm["moving_speed"]
            if self.prm["require_confirmation"] == "camera_or_moving" and not (t.camera_confirmed or moving):
                continue
            out.append(dict(id=t.id, x=x, P=np.array(t.covariance).reshape(4, 4)))
        return out, now - stamp_to_sec(self.tracks.header.stamp)

    def tick(self):
        now = stamp_to_sec(self.get_clock().now().to_msg())
        d = ControllerDiagnostics()
        d.header.stamp = self.get_clock().now().to_msg()
        d.trial_id, d.method = self.trial_id, self.ctrl.mp.method
        cmd = np.zeros(3)
        if not self.enabled or self.ref is None:
            d.mode = "disabled"
            self.ctrl.u_prev = np.zeros(3)
        else:
            try:
                tf = self.tf.lookup_transform(self.prm["map_frame"], self.prm["base_frame"], Time())
                pose_age = now - stamp_to_sec(tf.header.stamp)
            except Exception:  # noqa: BLE001
                tf, pose_age = None, np.inf
            odom_age = now - self.twist[0] if self.twist else np.inf
            if pose_age > self.prm["pose_timeout"] or odom_age > self.prm["pose_timeout"]:
                d.mode = "watchdog_brake"
                cmd = self.ctrl._brake()
                self.ctrl.u_prev = cmd
            else:
                x = np.r_[tf_to_pose2d(tf), self.twist[1]]
                people, age = self._people(now)
                pts = self.scan_pts
                if pts is not None and people:
                    P = np.array([e["x"][:2] for e in people])
                    keep = np.min(np.linalg.norm(pts[:, None] - P[None], axis=2), axis=1) > self.prm["track_exclusion"]
                    pts = pts[keep]
                ref = self.ref.horizon(x[:2], self.ctrl.np.N, self.ctrl.np.dt)
                cmd, dg = self.ctrl.step(now, x, self.pose_cov, people, pts, ref, age)
                d.mode = dg["mode"]
                d.solver_status = str(dg.get("status", ""))
                d.solve_ms = float(dg.get("solve_ms", 0.0))
                d.iters = int(dg.get("iters", -1))
                d.max_slack = float(dg.get("max_slack", 0.0))
                d.speed_scale = float(dg.get("speed_scale", 1.0))
                d.n_people = len(people)
                d.sigma_n = [float(v) for v in dg.get("sigma_n", [])]
                d.rho = [float(v) for v in dg.get("rho", [])]
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = map(float, cmd)
        self.pub_cmd.publish(tw)
        d.cmd = [float(v) for v in cmd]
        self.pub_diag.publish(d)


def main():
    rclpy.init()
    rclpy.spin(NMPCNode())


if __name__ == "__main__":
    main()
