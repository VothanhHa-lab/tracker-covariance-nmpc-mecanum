"""Nạp cấu hình YAML thành dataclass của phần lõi."""
from __future__ import annotations

import numpy as np
import yaml

from .controller import MethodParams, RefParams
from .models import RobotParams
from .nmpc import NMPCParams
from .tracker import SensorModel, TrackerParams


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def robot(path) -> RobotParams:
    return RobotParams.from_yaml(path)


def sensors(path) -> dict:
    d = _load(path)["sensors"]
    out = {}
    for name, s in d.items():
        fov = s.get("fov_half_deg")
        out[name] = SensorModel(name, s["sigma_r_a"], s["sigma_r_b"], s["sigma_b"],
                                None if fov is None else np.deg2rad(fov), s["r_max"],
                                mount=tuple(s.get("mount", (0.0, 0.0, 0.0))))
    return out


def tracker(path) -> TrackerParams:
    d = _load(path)["tracker"]
    if "birth_sensors" in d:
        d["birth_sensors"] = tuple(d["birth_sensors"])
    return TrackerParams(**d)


def controller(path):
    d = _load(path)
    return MethodParams(**d["method"]), NMPCParams(**d["nmpc"]), RefParams(**d["reference"])
