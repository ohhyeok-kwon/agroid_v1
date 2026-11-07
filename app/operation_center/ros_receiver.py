"""Operation Center receives measured local joint transforms over ROS 2 TF."""
import argparse
import ctypes
import ipaddress
import json
import os
import signal
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

from pose import normalize_pose
from tf_pose import TfPose

output_lock = threading.Lock()


def emit(value):
    with output_lock:
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


class OperationCenterReceiver(Node):
    def __init__(self, callback=emit):
        super().__init__('operation_center_receiver')
        self.callback = callback
        self.last_stamp = None
        self.model = TfPose()
        self.static = {}
        self.create_subscription(TFMessage, '/tf', self.accept,
                                 QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(TFMessage, '/tf_static', self.accept_static,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def accept_static(self, message):
        # The renderer uses these fixed links from the identical local URDF.
        for transform in message.transforms:
            self.static[transform.child_frame_id] = transform

    def accept(self, message):
        # state_manager sends one /tf message per measurement with every valid
        # joint; joints without a valid measurement are omitted (shown as missing).
        try:
            joints, transforms, stamps = {}, {}, set()
            for transform in message.transforms:
                joint = self.model.joint(transform)
                if joint is None:
                    continue
                stamp = transform.header.stamp
                stamps.add(stamp.sec*1_000_000_000 + stamp.nanosec)
                name, angle, value = joint
                joints[name] = angle
                transforms[transform.child_frame_id] = value
            if not joints:
                return
            if len(stamps) != 1:
                raise ValueError('한 TF 메시지 안의 관절 측정 시각이 서로 다릅니다.')
            key = stamps.pop()
            if key <= 0 or abs(time.time_ns()-key) > 3_000_000_000:
                return
            # Repeated or slightly reordered samples are skipped; a jump back of 1 s or more
            # is an R1 clock adjustment and must not freeze the view.
            if self.last_stamp is not None and 0 <= self.last_stamp-key < 1_000_000_000:
                return
            pose = normalize_pose({'tick': key//1_000_000, 'joints': joints, 'transforms': transforms,
                                   'quaternion': None, 'rpy': None})
            self.callback({'event': 'pose', 'pose': pose})
            self.last_stamp = key
        except (ValueError, KeyError, TypeError, IndexError) as error:
            self.callback({'event': 'error', 'message': str(error)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, type=lambda value: str(ipaddress.IPv4Address(value)))
    parser.parse_args()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() == 1:
        return
    stopped = threading.Event()
    rclpy.init(args=[])
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stopped.set())
    receiver = OperationCenterReceiver()
    executor = SingleThreadedExecutor()
    executor.add_node(receiver)
    try:
        emit({'event': 'status', 'message': 'R1 관절 TF를 기다리는 중입니다.'})
        while not stopped.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.1)
    finally:
        executor.shutdown(timeout_sec=1)
        receiver.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit({'event': 'error', 'message': str(error)})
        raise SystemExit(1)
