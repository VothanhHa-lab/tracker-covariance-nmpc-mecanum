"""Utilities shared by the nodes."""
import math

import numpy as np


def stamp_to_sec(stamp):
    return stamp.sec + 1e-9 * stamp.nanosec


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def tf_to_pose2d(tf):
    t = tf.transform.translation
    return (t.x, t.y, yaw_from_quat(tf.transform.rotation))


def pose_cov_3x3(cov36):
    """Extract (x, y, yaw) from the 6x6 covariance of a PoseWithCovariance."""
    C = np.asarray(cov36, float).reshape(6, 6)
    idx = [0, 1, 5]
    return C[np.ix_(idx, idx)]
