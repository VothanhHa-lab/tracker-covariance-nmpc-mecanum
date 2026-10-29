import os

import yaml

CONFIG = os.environ.get("PU_NMPC_CONFIG", os.path.expanduser("~/pu_nmpc/config"))
DATA = os.environ.get("PU_NMPC_DATA", os.path.expanduser("~/pu_nmpc_data"))

TOPICS = ["/scan", "/odom", "/imu/data", "/tf", "/tf_static", "/amcl_pose", "/cmd_vel", "/cmd_vel_raw", "/guard/active", "/joint_states",
          "/people/detections/lidar", "/people/detections/camera", "/people/tracks",
          "/controller/diagnostics", "/trial/path", "/trial/id", "/trial/enable", "/trial/event",
          "/gt/robot", "/gt/people", "/camera/camera/color/camera_info"]

with open(os.path.join(CONFIG, "controller.yaml")) as _f:
    GUARD = yaml.safe_load(_f).get("guard", {})
