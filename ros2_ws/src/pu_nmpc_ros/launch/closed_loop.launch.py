"""One closed-loop trial. Example:
ros2 launch pu_nmpc_ros closed_loop.launch.py trial_id:=R017 method:=cov schedule:=~/pu_nmpc/protocol/schedule.csv
Sensor drivers, localisation (AMCL/robot_localization) and the motor bridge are started separately (robot bringup).
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from _common import GUARD, CONFIG, DATA, TOPICS  # noqa: E402


def generate_launch_description():
    args = [DeclareLaunchArgument("trial_id"), DeclareLaunchArgument("method"),
            DeclareLaunchArgument("schedule"), DeclareLaunchArgument("compensate", default_value="true"),
            DeclareLaunchArgument("record_images", default_value="false")]
    y = lambda n: os.path.join(CONFIG, n)  # noqa: E731
    nodes = [
        Node(package="pu_nmpc_ros", executable="scan_stop_guard", parameters=[GUARD]),
        Node(package="pu_nmpc_ros", executable="lidar_people_detector"),
        Node(package="pu_nmpc_ros", executable="camera_people_detector",
             parameters=[{"model": os.path.join(CONFIG, "yolov8n.pt")}]),
        Node(package="pu_nmpc_ros", executable="fusion_tracker",
             parameters=[{"sensors_yaml": y("sensors.yaml"), "tracker_yaml": y("tracker.yaml"),
                          "compensate": LC("compensate")}]),
        Node(package="pu_nmpc_ros", executable="nmpc_controller",
             parameters=[{"robot_yaml": y("robot.yaml"), "controller_yaml": y("controller.yaml"), "method": LC("method")}]),
        Node(package="pu_nmpc_ros", executable="trial_manager",
             parameters=[{"trial_id": LC("trial_id"), "schedule_csv": LC("schedule"),
                          "scenarios_yaml": y("scenarios.yaml"), "data_dir": DATA}]),
        ExecuteProcess(cmd=["ros2", "bag", "record", "-s", "mcap", "-o", [DATA, "/", LC("trial_id"), "/bag"]] + TOPICS,
                       output="screen"),
    ]
    return LaunchDescription(args + nodes)
