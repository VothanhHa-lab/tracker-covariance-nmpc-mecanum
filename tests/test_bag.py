"""Bag-extraction test: write a small synthetic bag with rosbags and read it back (python3 tests/test_bag.py)."""
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import bag_to_csv  # noqa: E402


def main():
    from rosbags.rosbag2 import StoragePlugin, Writer
    ts = bag_to_csv.typestore()
    T = ts.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    Det, DetArr = T["pu_nmpc_msgs/msg/PersonDetection"], T["pu_nmpc_msgs/msg/PersonDetectionArray"]
    TP, TPA = T["pu_nmpc_msgs/msg/TrackedPerson"], T["pu_nmpc_msgs/msg/TrackedPersonArray"]
    TFM, TS = T["tf2_msgs/msg/TFMessage"], T["geometry_msgs/msg/TransformStamped"]
    Tr, V3, Q = T["geometry_msgs/msg/Transform"], T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
    with tempfile.TemporaryDirectory() as d:
        bag = pathlib.Path(d) / "bag"
        with Writer(bag, version=8, storage_plugin=StoragePlugin.MCAP) as w:
            cd = w.add_connection("/people/detections/lidar", DetArr.__msgtype__, typestore=ts)
            ct = w.add_connection("/people/tracks", TPA.__msgtype__, typestore=ts)
            cf = w.add_connection("/tf", TFM.__msgtype__, typestore=ts)
            for k in range(5):
                t = 100.0 + 0.1 * k
                st = Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
                h = Header(stamp=st, frame_id="laser")
                m = DetArr(header=h, sensor="lidar", detections=[Det(range=2.0 + k, bearing=0.1, score=1.0)])
                w.write(cd, int(t * 1e9) + 30_000_000, ts.serialize_cdr(m, DetArr.__msgtype__))
                tp = TPA(header=Header(stamp=st, frame_id="map"),
                         people=[TP(id=1, state=np.array([1.0, 2.0, 0.5, 0.0]), covariance=np.eye(4).ravel() * 0.01,
                                    age=0.05, camera_confirmed=True)], processed_until=t - 0.2, late_dropped=0)
                w.write(ct, int(t * 1e9), ts.serialize_cdr(tp, TPA.__msgtype__))
                tf = TFM(transforms=[TS(header=Header(stamp=st, frame_id="map"), child_frame_id="odom",
                                        transform=Tr(translation=V3(x=0.0, y=0.0, z=0.0), rotation=Q(x=0, y=0, z=0, w=1))),
                                     TS(header=Header(stamp=st, frame_id="odom"), child_frame_id="base_link",
                                        transform=Tr(translation=V3(x=0.1 * k, y=0.0, z=0.0),
                                                     rotation=Q(x=0, y=0, z=np.sin(0.05), w=np.cos(0.05))))])
                w.write(cf, int(t * 1e9), ts.serialize_cdr(tf, TFM.__msgtype__))
        man = bag_to_csv.extract(bag, pathlib.Path(d) / "csv")
        assert man["files"]["detections"]["rows"] == 5 and man["files"]["tracks"]["rows"] == 5
        det = pd.read_csv(pathlib.Path(d) / "csv" / "detections.csv")
        assert np.allclose(det.stamp - det.t_log, -0.03, atol=1e-6)      # measurement stamp 30 ms before logging time
        tf = bag_to_csv.TFSeries(pd.read_csv(pathlib.Path(d) / "csv" / "tf.csv"))
        P = tf.lookup(["map", "odom", "base_link"], np.array([100.25]))
        assert abs(P[0, 0] - 0.25) < 1e-6 and abs(P[0, 2] - 0.1) < 1e-6
    print("PASS test_bag_roundtrip")


if __name__ == "__main__":
    main()
