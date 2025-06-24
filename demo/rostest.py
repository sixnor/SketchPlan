import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np

class ROSBridge(Node):
    def __init__(self):
        super().__init__('odometry_bridge')

        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_ALL)

        self.sub = self.create_subscription(
            VehicleOdometry,
            '/drone4/fmu/out/vehicle_odometry',  # or whatever your topic is
            self.odom_callback,
            qos_profile
        )

    def odom_callback(self, msg: VehicleOdometry):
        global posarray, quatarray
        print("position", msg.position)
        print("quat", msg.q)
        posarray.append(msg.position)
        quatarray.append(msg.q)

        np.savetxt("pos.txt",np.stack([np.array(p) for p in posarray]))
        np.savetxt("quat.txt",np.stack([np.array(q) for q in quatarray]))

    
def main(args=None):
    global posarray, quatarray
    posarray, quatarray = [], []
    rclpy.init(args=args)
    node = ROSBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()