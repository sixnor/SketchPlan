import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage, Image
from px4_msgs.msg import VehicleOdometry
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import make_interp_spline
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from threading import Lock, Thread

from demo.data_utils import decompress_image, decode_ros_image_float32



class SyncNode(Node):
    def __init__(self, shared_object):
        super().__init__('sync_node')
        self.shared_object = shared_object

        odom_qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_ALL)


        depth_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # === Subscribers ===
        self.depth_sub = Subscriber(self, CompressedImage, '/zed/zed_node/depth/depth_registered/compressedDepth', qos_profile=depth_qos_profile)
        self.conf_sub = Subscriber(self, Image, '/zed/zed_node/confidence/confidence_map', qos_profile=depth_qos_profile)
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/drone4/fmu/out/vehicle_odometry',
            self.odom_callback,
            odom_qos_profile
        )

        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.conf_sub], queue_size=10, slop=0.01)
        self.ats.registerCallback(self.synced_callback)

        # === Odometry buffer ===
        self.odom_buffer = []
        self.odom_lock = Lock()
        self.odom_window_sec = 4.0

    def odom_callback(self, msg):
        t = msg.timestamp_sample * 1e-6
        with self.odom_lock:
            self.odom_buffer.append(msg)
            self.odom_buffer = [m for m in self.odom_buffer if (m.timestamp_sample * 1e-6) >= t - self.odom_window_sec]

    def synced_callback(self, depth_msg, conf_msg):
        t_depth = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        t_conf = conf_msg.header.stamp.sec + conf_msg.header.stamp.nanosec * 1e-9
        t_mid = 0.5 * (t_depth + t_conf)

       

        with self.odom_lock:
            odom_ts = np.array([m.timestamp_sample * 1e-6 for m in self.odom_buffer])
            if len(odom_ts) < 2 or t_mid < odom_ts[0] or t_mid > odom_ts[-1]:
                print("CLOCK SYNC FUCKED!")
                return  # skip if out of interpolation range

            poses = np.stack([m.position for m in self.odom_buffer])
            quats = np.stack([m.q for m in self.odom_buffer])
            rots = R.from_quat(quats, scalar_first=True)

            pos_interp = make_interp_spline(odom_ts, poses, k=1)
            slerp = Slerp(odom_ts, rots)

            pos = pos_interp(t_mid)
            quat = slerp(t_mid).as_quat(scalar_first=True)

        # === Decode ===
        depth_img = decompress_image(depth_msg)
        conf_img = decode_ros_image_float32(conf_msg)

        # === Apply to shared object ===
        self.shared_object.step(depth_img, conf_img, pos, quat)
        self.get_logger().info("called update on pc.")

def buildsync(shared_object):
    def spin_thread():
        if not rclpy.ok():
            rclpy.init()
        node = SyncNode(shared_object)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
        node.destroy_node()
        rclpy.shutdown()

    t = Thread(target=spin_thread, daemon=True)
    t.start()
