"""Thí nghiệm bước nhận dạng khâu vận tốc (giao thức P3).
ros2 launch pu_nmpc_ros identification.launch.py session:=I01
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from _common import GUARD, DATA, TOPICS  # noqa: E402


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("session"),
        Node(package="pu_nmpc_ros", executable="scan_stop_guard", parameters=[GUARD]),
        Node(package="pu_nmpc_ros", executable="step_test"),
        ExecuteProcess(cmd=["ros2", "bag", "record", "-s", "mcap", "-o", [DATA, "/", LC("session"), "/bag"]] + TOPICS),
    ])
