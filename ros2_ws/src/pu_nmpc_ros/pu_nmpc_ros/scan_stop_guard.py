"""Lớp dừng cứng độc lập với NMPC: /cmd_vel_raw → /cmd_vel.

Lệnh bị đặt về 0 khi có điểm quét thô nằm trong hình chữ nhật bao robot nới thêm `stop_margin`,
hoặc khi không nhận được quét trong `scan_timeout` s. Mọi lần kích hoạt được phát lên /guard/active
(được ghi vào bag và báo cáo). Đây là lớp phần mềm; e-stop phần cứng vẫn là lớp bảo vệ cuối.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from .common import tf_to_pose2d


class ScanStopGuard(Node):
    def __init__(self):
        super().__init__("scan_stop_guard")
        p = self.declare_parameters("", [("length", 0.60), ("width", 0.55), ("stop_margin", 0.30),
                                         ("scan_timeout", 0.3), ("base_frame", "base_link")])
        self.prm = {x.name: x.value for x in p}
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.blocked, self.t_scan, self.mount = True, None, None
        self.create_subscription(LaserScan, "/scan", self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel_raw", self.cmd_cb, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_active = self.create_publisher(Bool, "/guard/active", 10)

    def scan_cb(self, scan):
        if self.mount is None:
            try:
                self.mount = tf_to_pose2d(self.tf.lookup_transform(self.prm["base_frame"], scan.header.frame_id,
                                                                    rclpy.time.Time()))
            except Exception:  # noqa: BLE001
                return
        r = np.asarray(scan.ranges)
        a = scan.angle_min + scan.angle_increment * np.arange(len(r)) + self.mount[2]
        ok = np.isfinite(r) & (r > scan.range_min)
        x = self.mount[0] + r[ok] * np.cos(a[ok])
        y = self.mount[1] + r[ok] * np.sin(a[ok])
        m = self.prm["stop_margin"]
        inside = (np.abs(x) < self.prm["length"] / 2 + m) & (np.abs(y) < self.prm["width"] / 2 + m)
        # loại điểm thuộc chính thân robot (nằm trong hình chữ nhật thân)
        body = (np.abs(x) < self.prm["length"] / 2) & (np.abs(y) < self.prm["width"] / 2)
        self.blocked = bool(np.any(inside & ~body))
        self.t_scan = self.get_clock().now().nanoseconds * 1e-9

    def cmd_cb(self, cmd):
        now = self.get_clock().now().nanoseconds * 1e-9
        stale = self.t_scan is None or now - self.t_scan > self.prm["scan_timeout"]
        active = self.blocked or stale
        self.pub_active.publish(Bool(data=active))
        self.pub.publish(Twist() if active else cmd)


def main():
    rclpy.init()
    rclpy.spin(ScanStopGuard())


if __name__ == "__main__":
    main()
