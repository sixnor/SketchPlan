import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint
from rclpy.executors import MultiThreadedExecutor
from demo.collsionavoider import CollisionAvoider
import numpy as np
import queue
import threading
from copy import deepcopy

class WaypointController(Node):
    def __init__(
        self,
        pose_queue: queue.Queue,
        waypoint_queue: queue.Queue,
        radius: float,
        topic_name: str = '/fmu/in/trajectory_setpoint',
        frequency: float = 10.0,
        ca: CollisionAvoider | None = None,
        nudge_max = 0.4
    ):
        super().__init__('waypoint_controller_node')
        self.pose_queue = pose_queue
        self.waypoint_queue = waypoint_queue
        self.radius = radius
        self.current_waypoint = None
        self.current_pose = None
        self.ca = ca
        self.nudge_max = nudge_max


        # Custom QoS profile as specified
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publisher with specified QoS
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            topic_name,
            qos_profile
        )

        # Timer for control loop
        self.create_timer(1.0 / frequency, self.control_loop)

    def control_loop(self):
        # Update pose from queue (get latest)   
        # Load next waypoint if needed
        if self.current_waypoint is None:
            try:
                self.current_waypoint = self.waypoint_queue.get_nowait()
                #self.get_logger().info(f"New waypoint: {self.current_waypoint}")
            except queue.Empty:
                #self.get_logger().info("No more waypoints.")
                return

        try:
            while True:
                self.current_pose = self.pose_queue.get_nowait()
        except queue.Empty:
            pass

        if self.current_pose is None:
            self.get_logger().warn("No pose received yet.")
            return
        
         # Get current drone position
        pos = np.array([
            self.current_pose.position[0],
            self.current_pose.position[1],
            self.current_pose.position[2]
        ])

        # Compute adjusted
        candidate_point = deepcopy(self.current_waypoint)
        if self.ca is not None:
            #self.get_logger().info(f"before: {candidate_point}, {self.current_waypoint}")
            candidate_point[:-1], nudgedist = self.ca.safeguard_min(pos, candidate_point[:-1])
            #self.get_logger().info(f"after: {candidate_point}, {self.current_waypoint}")
        if np.linalg.norm(self.current_waypoint[:-1] - pos) < self.radius:
            self.get_logger().info(f"Reached waypoint: {self.current_waypoint} with adjustment: {candidate_point}")
            self.get_logger().info(f"Nudge distance: {nudgedist}")
            self.current_waypoint = None
            return
        
        # Create and publish setpoint
        sp = TrajectorySetpoint()
        sp.timestamp = self.get_clock().now().nanoseconds // 1000
        sp.position = candidate_point[:-1].astype(np.float32)
        sp.velocity = [np.nan] * 3
        sp.acceleration = [np.nan] * 3
        sp.jerk = [np.nan] * 3
        sp.yaw = float(candidate_point[3])
        sp.yawspeed = np.nan

        self.setpoint_pub.publish(sp)

def fill_waypoint_queue(wp_queue: queue.Queue, new_waypoints: np.ndarray):
    try: # Empty queue
        while True:
            wp_queue.get_nowait()
    except queue.Empty:
        pass  # fully emptied
    for wp in new_waypoints: # Restock with new waypoints
        wp_queue.put(wp)

def start_controller_async(pose_queue, 
                           waypoint_queue,
                           topic = "/drone4/fmu/in/trajectory_setpoint",
                           frequency=50, 
                           radius=0.1,
                           ca = None):
    if not rclpy.ok():
        rclpy.init()
    controller = WaypointController(
        pose_queue=pose_queue,
        waypoint_queue=waypoint_queue,
        radius=radius,
        topic_name=topic,
        frequency=frequency,
        ca=ca
    )

    def spin_node():
        executor = MultiThreadedExecutor()
        executor.add_node(controller)
        executor.spin()
        controller.destroy_node()
        rclpy.shutdown()

    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    return controller