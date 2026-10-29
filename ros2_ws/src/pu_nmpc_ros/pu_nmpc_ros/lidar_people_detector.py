"""LiDAR scan -> person candidates (PersonDetectionArray, stamp = stamp of the scan)."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from pu_nmpc_core.detect import cluster_scan
from pu_nmpc_msgs.msg import PersonDetection, PersonDetectionArray


class LidarPeopleDetector(Node):
    def __init__(self):
        super().__init__("lidar_people_detector")
        p = self.declare_parameters("", [("scan_topic", "/scan"), ("jump", 0.13), ("min_pts", 3),
                                         ("leg_width_min", 0.05), ("leg_width_max", 0.35),
                                         ("merge_dist", 0.45), ("r_min", 0.3), ("r_max", 8.0)])
        self.prm = {x.name: x.value for x in p}
        self.pub = self.create_publisher(PersonDetectionArray, "people/detections/lidar", 10)
        self.create_subscription(LaserScan, self.prm["scan_topic"], self.cb, qos_profile_sensor_data)

    def cb(self, scan: LaserScan):
        dets = cluster_scan(np.asarray(scan.ranges), scan.angle_min, scan.angle_increment,
                            jump=self.prm["jump"], min_pts=self.prm["min_pts"],
                            leg_width=(self.prm["leg_width_min"], self.prm["leg_width_max"]),
                            merge_dist=self.prm["merge_dist"], r_min=self.prm["r_min"], r_max=self.prm["r_max"])
        msg = PersonDetectionArray()
        msg.header = scan.header            # keep the measurement stamp and frame of the LiDAR
        msg.sensor = "lidar"
        msg.detections = [PersonDetection(range=r, bearing=b, score=1.0) for r, b in dets]
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(LidarPeopleDetector())


if __name__ == "__main__":
    main()
