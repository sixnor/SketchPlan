import numpy as np
import os
import queue
import threading
import signal
import time
from px4_msgs.msg import VehicleOdometry  # Only for type hints; optional

class PoseLogger:
    def __init__(self, pose_queue: queue.Queue, save_dir: str, poll_rate: float = 20.0, goal_point = None, goal_margin = 1.0):
        self.pose_queue = pose_queue
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.positions = []
        self.quaternions = []
        self.running = True
        self.poll_rate = poll_rate
        self.goal_point = goal_point
        self.goal_margin = goal_margin

        # Start polling thread
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

        # Register shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _poll_loop(self):
        while self.running:
            try:
                msg = self.pose_queue.get(timeout=1.0 / self.poll_rate)
                self.positions.append(msg.position)
                self.quaternions.append(msg.q)

                if self.goal_point is not None:
                    if np.linalg.norm(self.goal_point - msg.position) < self.goal_margin:
                        print("GOAL REACHED!!!")
            except queue.Empty:
                continue

    def _shutdown(self, *args):
        print("Shutting down PoseLogger and writing to disk...")
        self.running = False
        self.thread.join()

        np.savetxt(os.path.join(self.save_dir, "positions.txt"), np.array(self.positions))
        np.savetxt(os.path.join(self.save_dir, "quaternions.txt"), np.array(self.quaternions))

# Example usage:
# pose_logger = PoseLogger(pose_queue, save_dir="sessions/2025-05-20_12-00-00_my_model")
