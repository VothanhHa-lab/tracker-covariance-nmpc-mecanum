"""Asynchronous fusion tracker (wraps pu_nmpc_core.tracker.Tracker).

For each detection batch: look up TF map<-base_link AT the measurement stamp, attach the latest pose covariance,
and put it in the lag queue. A 20-Hz timer processes the queue and publishes estimates (compensated or not).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from pu_nmpc_core import config
from pu_nmpc_core.tracker import Detection, DetectionBatch, Tracker
from pu_nmpc_msgs.msg import PersonDetectionArray, TrackedPerson, TrackedPersonArray

from .common import pose_cov_3x3, stamp_to_sec, tf_to_pose2d


class FusionTracker(Node):
    def __init__(self):
        super().__init__("fusion_tracker")
        p = self.declare_parameters("", [
            ("sensors_yaml", ""), ("tracker_yaml", ""), ("map_frame", "map"), ("base_frame", "base_link"),
            ("sensors_enabled", ["lidar", "camera"]), ("compensate", True), ("rate", 20.0),
            ("pose_cov_topic", "/amcl_pose"), ("default_pose_std", [0.05, 0.05, 0.02])])
        prm = {x.name: x.value for x in p}
        self.prm = prm
        self.sensors = config.sensors(prm["sensors_yaml"])
        self.tracker = Tracker(self.sensors, config.tracker(prm["tracker_yaml"]), enabled=prm["sensors_enabled"])
        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        TransformListener(self.tf, self)
        self.pose_cov = np.diag(np.square(prm["default_pose_std"]))
        self.mount_known = set()
        self.create_subscription(PoseWithCovarianceStamped, prm["pose_cov_topic"], self.pose_cb, 10)
        for s in self.sensors:
            self.create_subscription(PersonDetectionArray, f"people/detections/{s}", self.det_cb, 20)
        self.pub = self.create_publisher(TrackedPersonArray, "people/tracks", 10)
        self.create_timer(1.0 / prm["rate"], self.tick)
        self.lookup_fail = 0

    def pose_cb(self, msg):
        self.pose_cov = pose_cov_3x3(msg.pose.covariance)

    def _update_mount(self, sensor, frame):
        if sensor in self.mount_known:
            return
        try:
            tf = self.tf.lookup_transform(self.prm["base_frame"], frame, Time())
        except Exception:  # noqa: BLE001
            return
        sm = self.sensors[sensor]
        sm.mount = tf_to_pose2d(tf)
        self.mount_known.add(sensor)
        self.get_logger().info(f"mount {sensor} ({frame}) = {sm.mount}")

    def det_cb(self, msg: PersonDetectionArray):
        self._update_mount(msg.sensor, msg.header.frame_id)
        if msg.sensor not in self.mount_known:
            return
        try:
            tf = self.tf.lookup_transform(self.prm["map_frame"], self.prm["base_frame"], Time.from_msg(msg.header.stamp),
                                          timeout=Duration(seconds=0.05))
        except Exception:  # noqa: BLE001
            self.lookup_fail += 1
            return
        dets = [Detection(d.range, d.bearing) for d in msg.detections]
        self.tracker.add(DetectionBatch(stamp_to_sec(msg.header.stamp), msg.sensor, dets, tf_to_pose2d(tf),
                                        self.pose_cov.copy()))

    def tick(self):
        now_msg = self.get_clock().now().to_msg()
        now = stamp_to_sec(now_msg)
        self.tracker.step(now)
        out = TrackedPersonArray()
        out.header.stamp = now_msg
        out.header.frame_id = self.prm["map_frame"]
        out.processed_until = float(self.tracker.processed_until) if np.isfinite(self.tracker.processed_until) else -1.0
        out.late_dropped = self.tracker.stats["late_dropped"]
        for e in self.tracker.estimates(now, compensate=self.prm["compensate"]):
            out.people.append(TrackedPerson(id=e["id"], state=e["x"].tolist(), covariance=e["P"].ravel().tolist(),
                                            age=float(e["age"]), camera_confirmed=bool(e["camera_confirmed"])))
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(FusionTracker())


if __name__ == "__main__":
    main()
