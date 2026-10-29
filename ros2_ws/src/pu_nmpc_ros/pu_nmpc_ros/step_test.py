"""Step experiment to identify the velocity loops (robot on blocks or on an empty floor, with an operator at the e-stop).

Sequence: for each axis (vx, vy, omega) and amplitude, hold 0 for `rest` s, hold the amplitude for `hold` s,
then return to 0; repeat `reps` times. Commands are sent only while /trial/enable = True. Record a rosbag in parallel
(launch identification.launch.py) to obtain /cmd_vel, /odom, /joint_states (if available) and /imu.
"""
import itertools

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class StepTest(Node):
    def __init__(self):
        super().__init__("step_test")
        p = self.declare_parameters("", [("amplitudes_v", [0.2, 0.4, 0.6]), ("amplitudes_w", [0.5, 1.0]),
                                         ("hold", 2.0), ("rest", 1.5), ("reps", 3), ("rate", 50.0)])
        prm = {x.name: x.value for x in p}
        seq = []
        for axis, amps in (("vx", prm["amplitudes_v"]), ("vy", prm["amplitudes_v"]), ("w", prm["amplitudes_w"])):
            for a, _ in itertools.product(amps, range(prm["reps"])):
                for sign in (1, -1):
                    seq.append((axis, 0.0, prm["rest"]))
                    seq.append((axis, sign * a, prm["hold"]))
        seq.append(("vx", 0.0, prm["rest"]))
        self.seq, self.i, self.t_seg = seq, 0, None
        self.enabled = False
        self.pub = self.create_publisher(Twist, "/cmd_vel_raw", 10)   # qua scan_stop_guard
        self.create_subscription(Bool, "/trial/enable", lambda m: setattr(self, "enabled", m.data), 10)
        self.create_timer(1.0 / prm["rate"], self.tick)

    def tick(self):
        tw = Twist()
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.enabled and self.i < len(self.seq):
            axis, val, dur = self.seq[self.i]
            self.t_seg = self.t_seg or now
            if now - self.t_seg >= dur:
                self.i, self.t_seg = self.i + 1, now
            if axis == "vx":
                tw.linear.x = val
            elif axis == "vy":
                tw.linear.y = val
            else:
                tw.angular.z = val
        self.pub.publish(tw)
        if self.i >= len(self.seq):
            self.get_logger().info("Step sequence completed", once=True)


def main():
    rclpy.init()
    rclpy.spin(StepTest())


if __name__ == "__main__":
    main()
