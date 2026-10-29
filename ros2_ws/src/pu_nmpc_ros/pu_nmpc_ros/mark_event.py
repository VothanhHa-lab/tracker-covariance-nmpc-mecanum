"""Bàn phím người vận hành: s = start, i = intervention, a = abort, n = ghi chú, q = thoát."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEYS = {"s": "start", "i": "intervention", "a": "abort"}


def main():
    rclpy.init()
    node = Node("mark_event")
    pub = node.create_publisher(String, "/trial/event", 10)
    print(__doc__)
    while rclpy.ok():
        k = input("> ").strip().lower()
        if k == "q":
            break
        if k == "n":
            pub.publish(String(data="note:" + input("ghi chú: ")))
        elif k in KEYS:
            pub.publish(String(data=KEYS[k]))
            print("đã gửi", KEYS[k])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
