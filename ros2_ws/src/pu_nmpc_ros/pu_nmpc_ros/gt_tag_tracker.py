"""Ground truth từ camera trần cố định với thẻ AprilTag 36h11 (OpenCV ArUco).

1. Tư thế camera so với sàn: solvePnP trên các thẻ mốc gắn sàn có tọa độ đo bằng thước/máy đo.
2. Mỗi thẻ mục tiêu (robot, người): tia qua tâm thẻ cắt mặt phẳng z = chiều cao gắn thẻ (theo ID),
   tránh sai số thị sai khi thẻ gắn trên vai/mũ người tham gia.
3. Hướng robot: từ hai góc thẻ robot chiếu xuống cùng mặt phẳng.
Độ chính xác của hệ này phải được kiểm chứng riêng (giao thức, bước P1) và báo cáo trong bài.
"""
import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from pu_nmpc_msgs.msg import TrackedPerson, TrackedPersonArray


class GTTagTracker(Node):
    def __init__(self):
        super().__init__("gt_tag_tracker")
        p = self.declare_parameters("", [("gt_yaml", ""), ("image_topic", "/gt_cam/image_raw"),
                                         ("info_topic", "/gt_cam/camera_info")])
        prm = {x.name: x.value for x in p}
        with open(prm["gt_yaml"]) as f:
            self.cfg = yaml.safe_load(f)["ground_truth"]
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                           cv2.aruco.DetectorParameters())
        self.bridge, self.K, self.D, self.R, self.t = CvBridge(), None, None, None, None
        self.create_subscription(CameraInfo, prm["info_topic"], self.info_cb, 10)
        self.create_subscription(Image, prm["image_topic"], self.cb, qos_profile_sensor_data)
        self.pub_robot = self.create_publisher(PoseStamped, "/gt/robot", 10)
        self.pub_people = self.create_publisher(TrackedPersonArray, "/gt/people", 10)

    def info_cb(self, msg):
        self.K, self.D = np.array(msg.k).reshape(3, 3), np.array(msg.d)

    def _floor_pose(self, corners, ids):
        obj, img = [], []
        s = self.cfg["floor_tag_size"] / 2
        local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])   # thứ tự góc của ArUco
        for c, i in zip(corners, ids):
            ft = self.cfg["floor_tags"].get(int(i))
            if ft is None:
                continue
            x, y, yaw = ft
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            obj.extend((Rz @ local.T).T + np.array([x, y, 0]))
            img.extend(c.reshape(4, 2))
        if len(obj) < 8:
            return False
        ok, rvec, tvec = cv2.solvePnP(np.array(obj, np.float64), np.array(img, np.float64), self.K, self.D)
        if ok:
            self.R, self.t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        return ok

    def _ray_plane(self, uv, h):
        pt = cv2.undistortPoints(np.array([[uv]], np.float64), self.K, self.D).reshape(2)
        ray_c = np.array([pt[0], pt[1], 1.0])
        Rw = self.R.T
        origin = -Rw @ self.t
        ray_w = Rw @ ray_c
        lam = (h - origin[2]) / ray_w[2]
        return origin + lam * ray_w

    def cb(self, msg):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        corners, ids, _ = self.det.detectMarkers(img)
        if ids is None:
            return
        ids = ids.ravel()
        if not self._floor_pose(corners, ids) and self.R is None:
            return
        people = TrackedPersonArray()
        people.header = msg.header
        people.header.frame_id = "map"
        for c, i in zip(corners, ids):
            c = c.reshape(4, 2)
            tgt = self.cfg["targets"].get(int(i))
            if tgt is None:
                continue
            h = tgt["height"]
            ctr = self._ray_plane(c.mean(axis=0), h)
            if tgt["kind"] == "robot":
                front = self._ray_plane((c[0] + c[1]) / 2, h) - self._ray_plane((c[3] + c[2]) / 2, h)
                yaw = np.arctan2(front[1], front[0]) + tgt.get("yaw_offset", 0.0)
                ps = PoseStamped()
                ps.header = people.header
                ps.pose.position.x, ps.pose.position.y = float(ctr[0]), float(ctr[1])
                ps.pose.orientation.z, ps.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
                self.pub_robot.publish(ps)
            else:
                people.people.append(TrackedPerson(id=int(tgt["participant"]), state=[float(ctr[0]), float(ctr[1]), 0.0, 0.0]))
        self.pub_people.publish(people)


def main():
    rclpy.init()
    rclpy.spin(GTTagTracker())


if __name__ == "__main__":
    main()
