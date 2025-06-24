import numpy as np
import cv2
import struct
from scipy.spatial.transform import Rotation as R
import logging
import os
from datetime import datetime

def decompress_image(msg):
    img_raw = msg.data[12:]
    header_raw = msg.data[:12]
    np_arr = np.frombuffer(img_raw, np.uint8)
    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32)

    compfmt, depthQuantA, depthQuantB = struct.unpack("iff", header_raw)
    depth_img_scaled = depthQuantA / (depth_image.astype(np.float32)-depthQuantB)
    depth_img_scaled[depth_image == 0] = np.nan
    return depth_img_scaled

def decode_ros_image_float32(msg):
    """Decode a ROS2 Image message encoded as float32."""
    if msg.encoding != '32FC1':
        raise ValueError(f"Expected encoding '32FC1', got '{msg.encoding}'")

    data = np.frombuffer(msg.data, dtype=np.float32)
    img = data.reshape((msg.height, msg.width))
    return img

def pose_to_fli_ocv(pos, quat, hand_eye=np.identity(4)):
    dlf_ocv = np.array([[0,0,1,0],
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1]])
    # Given xyz and quaternion, that takes vectors from the ned drone local frame to the ned flightroom frame
    r = R.from_quat(quat, scalar_first=True)
    fli_dlf = np.identity(4)
    fli_dlf[:3, :3] =  R.as_matrix(r)
    fli_dlf[:3,-1] = pos
    fli_dlf = fli_dlf @ hand_eye
    return fli_dlf @ dlf_ocv

def compute_yaws_ned(points):
    """
    Given (N, 3) array of 3D points in NED frame, compute yaw angles (in radians)
    from each point to the next. Last yaw is copied from the second-to-last.
    
    Yaw is defined as angle from North (x) toward East (y), in the N-E plane.
    """
    diffs = points[1:, :2] - points[:-1, :2]  # Get differences in N-E plane
    yaws = np.arctan2(diffs[:, 1], diffs[:, 0])  # atan2(dE, dN)
    
    # Optionally, append last yaw to maintain array size
    yaws = np.append(yaws, yaws[-1])
    return yaws

def quat_to_yaw(quat):
    r = R.from_quat(quat, scalar_first=True)
    # Extract Euler angles (in radians) in the 'zyx' order: yaw, pitch, roll
    yaw, pitch, roll = r.as_euler('zyx', degrees=False)
    return yaw  # in radians