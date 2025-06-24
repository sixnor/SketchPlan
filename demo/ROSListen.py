import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
import threading
import queue

class ROSQueueBridge:
    def __init__(self):
        self.subscribers = []
        self.node = None
        self.executor = None
        self.spin_thread = None

    def init(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('ros_queue_bridge')

    def add_subscriber(self, topic, msg_type, qos_profile=None):
        q = queue.Queue(maxsize=1)

        def callback(msg):
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(msg)

        qos_profile = qos_profile or QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.node.create_subscription(msg_type, topic, callback, qos_profile)
        self.subscribers.append((topic, q))
        return q

    def start(self):
        if self.executor is None:
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)

            def spin_exec():
                self.executor.spin()
                self.node.destroy_node()
                rclpy.shutdown()

            self.spin_thread = threading.Thread(target=spin_exec, daemon=True)
            self.spin_thread.start()