import os
from px4_msgs.msg import VehicleOdometry
from sensor_msgs.msg import CompressedImage
from demo.ROSListen import ROSQueueBridge
from demo.ROSPoints import WaypointController, fill_waypoint_queue, start_controller_async
import queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from demo.data_utils import decompress_image, pose_to_fli_ocv, compute_yaws_ned, quat_to_yaw
import rclpy
import pygame
import numpy as np
from drawing.preprocessing import interpolatePoints
from drawing.projectUtils import fill_nans_with_nearest, trajAffine
import torch
from torchvision.transforms import Resize
from diffusion.unet import SketchLinearTransform
from demo.syncer import buildsync
from demo.collsionavoider import CollisionAvoider

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize pygame window for displaying images
def init_pygame(width, height):
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Depth Stream")
    return window

# Display a NumPy image in the pygame window
def show_image_pygame(window, image: np.ndarray):
    # Normalize to 0-255 and convert to 3-channel grayscale for display
    norm_img = np.nan_to_num(image, nan=0.0)
    norm_img = (255 * (norm_img - np.min(norm_img)) / (np.max(norm_img) - np.min(norm_img) + 1e-6)).astype(np.uint8)
    rgb_image = np.stack([norm_img]*3, axis=-1)
    surf = pygame.surfarray.make_surface(np.transpose(rgb_image, (1, 0, 2)))  # transpose due to pygame's coord system
    window.blit(surf, (0, 0))
    pygame.display.update()

rclpy.init() # Init RCLPY
bridge = ROSQueueBridge()
bridge.init()

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

odom_queue = bridge.add_subscriber('/drone4/fmu/out/vehicle_odometry', VehicleOdometry, qos_profile=odom_qos_profile)
depth_queue = bridge.add_subscriber('/zed/zed_node/depth/depth_registered/compressedDepth', CompressedImage, qos_profile=depth_qos_profile)

pose_control_queue = bridge.add_subscriber('/drone4/fmu/out/vehicle_odometry', VehicleOdometry, qos_profile=odom_qos_profile)
waypoint_queue = queue.Queue() # Queue for waypoints. First in First out!


bridge.start()


res = [640, 360]
resizeres = (224,224)

wp_radius = 0.5 # In metres
control_freq = 50 # In Hz
robot_radius = 0.3

depthmean = 4.9893 # In metres
depthstd = 4.8041 # In metres
depthmax = 2.250 # Not in metres!

hand_eye = np.array([[ 1.,     0.,     0.,     0.15 ],
                     [ 0.,     1.,     0.,     0.025],
                     [ 0.,     0.,     1.,    -0.03 ],
                     [ 0.,     0.,     0.,     1.   ]])


modelname = "model.pt"
data_folder = "drawing/data/"


ca = CollisionAvoider(robot_radius=robot_radius, hand_eye=hand_eye, n_memory=20)
buildsync(ca)

start_controller_async(pose_control_queue, waypoint_queue, topic="/waypoints", frequency=control_freq, radius=wp_radius,ca=ca)


pygame_window = init_pygame(*res)

drawing_surface = pygame.Surface(pygame_window.get_size(), pygame.SRCALPHA)
drawing_surface.fill((0, 0, 0, 0))

resizer = Resize(resizeres)
model = torch.load(f"{data_folder}{modelname}", weights_only=False)
model.eval()

# Drawing settings
DRAWING_COLOR = (255, 0, 0, 255)
THICKNESS = 5
drawing = False
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    try:
                        # Fetch pose on click
                        odom_msg = odom_queue.get_nowait()
                        print("Position:", odom_msg.position)
                        cur_wayp = np.array([*odom_msg.position, quat_to_yaw(odom_msg.q)])
                        fill_waypoint_queue(waypoint_queue, [cur_wayp]) # Clear wayp que and hover at current
                    except queue.Empty:
                        print("MISSING POSE ON CLICK: ARE YOU SURE POSE ESTIMATE IS AVAILABLE?")
                    drawpoints = []
                    drawing_surface.fill((0, 0, 0, 0))
                    drawing = True
                    last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    last_pos = None
                    # PREDICT TIME!!!
                    interpoints = interpolatePoints(np.array(drawpoints))
                    torchpoints = torch.from_numpy(interpoints[None]).to(device).to(torch.float32)
                    torchpoints = 2*torchpoints / torch.Tensor(res).to(device) - 1 # Normalise Points

                    depth = torch.tensor(depth_img).to(device).to(torch.float32)

                    resizedimage = resizer(depth[None,:,:])
                    # RUN MODEL
                    with torch.no_grad():
                        resizedimage = (resizedimage - 4.9893)/4.8041
                        resizedimage = torch.clamp_max(resizedimage, 2.25)
                        traj = model(resizedimage[None,:,:], torchpoints).detach().clone().cpu().numpy().squeeze().T # Points in rel frame
                    
                    # GET FRAME TRANSFORM
                    fli_ocv = pose_to_fli_ocv(odom_msg.position, odom_msg.q)
                    # GET TRAJ
                    traj_xyz = trajAffine(traj, 1.0, fli_ocv)
                    traj_yaw = compute_yaws_ned(traj_xyz)
                    traj = np.concatenate([traj_xyz, traj_yaw[:,None]],axis=-1)
                    print(np.linalg.norm(odom_msg.position - traj[0,:-1]))
                    fill_waypoint_queue(waypoint_queue, traj)
                    
            elif event.type == pygame.MOUSEMOTION:
                if drawing and last_pos is not None:
                    pygame.draw.line(drawing_surface, DRAWING_COLOR, last_pos, event.pos, THICKNESS)
                    last_pos = event.pos
                    drawpoints.append(last_pos)
        if drawing is False: 
            drawing_surface.fill((0, 0, 0, 0)) # Clear sketch when not drawing
            try:
                depth_msg = depth_queue.get_nowait()
                depth_img = decompress_image(depth_msg)
                #depth_img = fill_nans_with_nearest(depth_img)
                depth_img[np.isnan(depth_img)] = 20.0 # Clamp to maximum depth range of ZED

                if pygame_window is None:
                    h, w = depth_img.shape
                    pygame_window = init_pygame(w, h)
                show_image_pygame(pygame_window, depth_img)
            except queue.Empty:
                pass
        pygame_window.blit(drawing_surface, (0, 0))
        pygame.display.update()

except KeyboardInterrupt:
    print("Shutting down...")
    pygame.quit()
