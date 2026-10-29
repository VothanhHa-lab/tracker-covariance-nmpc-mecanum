"""RGB-D (D435i, depth aligned to colour) + YOLO -> person candidates in camera_link.

Requires `pip install ultralytics`; YOLO weights set with the parameter `model` (e.g. yolov8n.pt).
Stamp of the detection array = stamp of the colour image (exposure time per the realsense driver).
"""
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from pu_nmpc_core.detect import bbox_to_range_bearing
from pu_nmpc_msgs.msg import PersonDetection, PersonDetectionArray


class CameraPeopleDetector(Node):
    def __init__(self):
        super().__init__("camera_people_detector")
        p = self.declare_parameters("", [
            ("color_topic", "/camera/camera/color/image_raw"),
            ("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"),
            ("info_topic", "/camera/camera/color/camera_info"),
            ("model", "yolov8n.pt"), ("conf", 0.5), ("body_frame", "camera_link"),
            ("depth_scale", 0.001), ("device", "cpu")])
        self.prm = {x.name: x.value for x in p}
        from ultralytics import YOLO
        self.model = YOLO(self.prm["model"])
        self.bridge = CvBridge()
        self.K = None
        self.create_subscription(CameraInfo, self.prm["info_topic"], self.info_cb, 10)
        subs = [message_filters.Subscriber(self, Image, self.prm["color_topic"], qos_profile=qos_profile_sensor_data),
                message_filters.Subscriber(self, Image, self.prm["depth_topic"], qos_profile=qos_profile_sensor_data)]
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, 5, 0.03)
        self.sync.registerCallback(self.cb)
        self.pub = self.create_publisher(PersonDetectionArray, "people/detections/camera", 10)

    def info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def cb(self, color, depth):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(color, "bgr8")
        d = self.bridge.imgmsg_to_cv2(depth).astype(np.float32)
        if depth.encoding in ("16UC1", "mono16"):
            d *= self.prm["depth_scale"]
        res = self.model.predict(img, classes=[0], conf=self.prm["conf"], verbose=False, device=self.prm["device"])[0]
        msg = PersonDetectionArray()
        msg.header.stamp = color.header.stamp
        msg.header.frame_id = self.prm["body_frame"]
        msg.sensor = "camera"
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        for box, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            rb = bbox_to_range_bearing(box, d, fx, fy, cx, cy)
            if rb is not None:
                msg.detections.append(PersonDetection(range=rb[0], bearing=rb[1], score=float(sc)))
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CameraPeopleDetector())


if __name__ == "__main__":
    main()
