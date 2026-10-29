"""Record sensor-noise data (robot STATIONARY, people walk through positions/ranges as in protocol P2).
ros2 launch pu_nmpc_ros sensor_noise.launch.py session:=N01
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from _common import CONFIG, DATA, TOPICS  # noqa: E402


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("session"),
        Node(package="pu_nmpc_ros", executable="lidar_people_detector"),
        Node(package="pu_nmpc_ros", executable="camera_people_detector",
             parameters=[{"model": os.path.join(CONFIG, "yolov8n.pt")}]),
        Node(package="pu_nmpc_ros", executable="gt_tag_tracker",
             parameters=[{"gt_yaml": os.path.join(CONFIG, "ground_truth.yaml")}]),
        ExecuteProcess(cmd=["ros2", "bag", "record", "-s", "mcap", "-o", [DATA, "/", LC("session"), "/bag"]] + TOPICS),
    ])
