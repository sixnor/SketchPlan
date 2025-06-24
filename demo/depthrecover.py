import numpy as np
import cv2
import plotly.express as px
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import struct
import matplotlib.pyplot as plt

# Set up type store (for ROS 2 Foxy or whatever distro you're using)
typestore = get_typestore(Stores.ROS2_FOXY)
timestamps = []

bag_path = 'may2_depthpose'

with Reader(bag_path) as reader:
    connections = [x for x in reader.connections if x.topic == '/zed/zed_node/depth/depth_registered/compressedDepth']
    
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        print(connection.msgtype)
        timestamps.append(timestamp * 1e-9)  # Convert from nanoseconds to seconds

        # Decode compressed depth image
        img_raw = msg.data[12:]
        header_raw = msg.data[:12]
        np_arr = np.frombuffer(img_raw, np.uint8)
        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32)

        compfmt, depthQuantA, depthQuantB = struct.unpack("iff", header_raw)
        depth_img_scaled = depthQuantA / (depth_image.astype(np.float32)-depthQuantB)
        depth_img_scaled[depth_image == 0] = np.nan
        
        if depth_image is not None:

            fig = px.imshow(depth_img_scaled, color_continuous_scale='viridis')
            fig.update_layout(title="Depth Image", coloraxis_colorbar=dict(title='Depth (mm)'))
            #fig.show()

timestamps = np.array(timestamps)
dts = np.diff(timestamps)
avg_freq = 1.0 / np.mean(dts)