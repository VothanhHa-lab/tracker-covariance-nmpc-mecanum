from glob import glob

from setuptools import setup

package_name = "pu_nmpc_ros"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "lidar_people_detector = pu_nmpc_ros.lidar_people_detector:main",
        "camera_people_detector = pu_nmpc_ros.camera_people_detector:main",
        "fusion_tracker = pu_nmpc_ros.fusion_tracker_node:main",
        "nmpc_controller = pu_nmpc_ros.nmpc_controller_node:main",
        "gt_tag_tracker = pu_nmpc_ros.gt_tag_tracker:main",
        "trial_manager = pu_nmpc_ros.trial_manager:main",
        "step_test = pu_nmpc_ros.step_test:main",
        "mark_event = pu_nmpc_ros.mark_event:main",
        "scan_stop_guard = pu_nmpc_ros.scan_stop_guard:main",
    ]},
)
