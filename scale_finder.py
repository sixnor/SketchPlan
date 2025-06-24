import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import open3d as o3d
import plotly.graph_objects as go
import cvxpy as cp

gltocv = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

def convert_opengl_to_opencv(T_c2w_opengl):
    T_c2w_opencv = np.eye(4)
    T_c2w_opencv[:3, :3] = gltocv @ T_c2w_opengl[:3, :3]
    return T_c2w_opencv

def load_transforms(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_camera_params(transforms_data, image_path):
    filename = os.path.basename(image_path)
    for idx, frame in enumerate(transforms_data['frames']):
        if os.path.basename(frame['file_path']) == filename:
            transform_matrix = np.array(frame['transform_matrix'])
            fl_x = transforms_data.get('fl_x')
            fl_y = transforms_data.get('fl_y')
            cx = transforms_data.get('cx', transforms_data.get('w', 0) / 2)
            cy = transforms_data.get('cy', transforms_data.get('h', 0) / 2)

            if fl_x is None and 'camera_angle_x' in transforms_data:
                width = transforms_data.get('w', 0)
                fl_x = width / (2 * np.tan(transforms_data['camera_angle_x'] / 2))
                fl_y = fl_x

            intrinsics = {
                'fl_x': fl_x, 'fl_y': fl_y,
                'cx': cx, 'cy': cy,
                'w': transforms_data.get('w', 0),
                'h': transforms_data.get('h', 0),
                "k1": transforms_data.get("k1", 0),
                "k2": transforms_data.get("k2", 0),
                "p1": transforms_data.get("p1", 0),
                "p2": transforms_data.get("p2", 0),
            }
            return idx, transform_matrix, intrinsics
    raise ValueError(f"Image {filename} not found in transforms.json")

def get_ray(px, py, K, c2w, distort):
    pxx, pyy = cv2.undistortPoints(np.array([[px, py]], dtype=np.float64), K, np.array(distort), P=K).squeeze()
    pixel_h = np.array([pxx, pyy, 1.0])
    ray_dir_camera = np.linalg.inv(K) @ pixel_h
    ray_dir_camera /= np.linalg.norm(ray_dir_camera)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return t, t + R @ ray_dir_camera

def select_points_contrast(image_paths):
    selected_points = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not open image: {path}")

        # --- Enhance contrast using CLAHE ---
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        # ------------------------------------

        points = []
        window_name = f"Image {i+1}"
        cv2.namedWindow(window_name)

        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, img)
                print(f"Selected point {len(points)} in {window_name}: ({x}, {y})")

        cv2.setMouseCallback(window_name, callback)
        cv2.imshow(window_name, img)
        print(f"Select 2 points in {window_name}")
        while len(points) < 2:
            if cv2.waitKey(100) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None
        selected_points.append(points)
        cv2.destroyWindow(window_name)
    return selected_points

def select_points(image_paths):
    selected_points = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not open image: {path}")

        points = []
        window_name = f"Image {i+1}"
        cv2.namedWindow(window_name)
        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, img)
                print(f"Selected point {len(points)} in {window_name}: ({x}, {y})")

        cv2.setMouseCallback(window_name, callback)
        cv2.imshow(window_name, img)
        print(f"Select 2 points in {window_name}")
        while len(points) < 2:
            if cv2.waitKey(100) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None
        selected_points.append(points)
        cv2.destroyWindow(window_name)
    return selected_points

def visualize_3d(points_3d, raybunch):
    """Visualize cameras, frustrums, and 3D points."""


    pcd = o3d.io.read_point_cloud("sparse_pc.ply")
    points = np.asarray(pcd.points)

    # Optional: grab color if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (len(points), 1))  # gray fallback

    # Convert RGB to 0-255
    colors = (colors * 255).astype(np.uint8)

    # Create Plotly scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,
                color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in colors],
                opacity=0.8
            )
        )
    ])

    fig.update_layout(scene=dict(
        xaxis=dict(title='X', range=[np.quantile(points, 0.005), np.quantile(points, 0.995)]),
        yaxis=dict(title='Y', range=[np.quantile(points, 0.005), np.quantile(points, 0.995)]),
        zaxis=dict(title='Z', range=[np.quantile(points, 0.005), np.quantile(points, 0.995)]),
        aspectmode='manual',  # Important!
        aspectratio=dict(x=1, y=1, z=1)),
        title='PLY Point Cloud Visualization',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    pts = points_3d
    highlight_points = go.Scatter3d(
    x=pts[:, 0],
    y=pts[:, 1],
    z=pts[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color=['red', 'blue'],  # easy to distinguish
        symbol='circle'
    ),
    name='Highlight Points'
)

    # Line
    connecting_line = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='lines',
        line=dict(color='lime', width=5),
        name='Line Between Points'
    )

    # Add both to figure
    fig.add_trace(highlight_points)
    fig.add_trace(connecting_line)


    fig.add_trace(
        go.Scatter3d(x=list(raybunch[0,:]), y=list(raybunch[1,:]), z=list(raybunch[2,:]),mode='lines',
        line=dict(color='yellow', width=4),
        name='Rays'))


    #fig.show()

def solve_for_points(rays):
    O, T = list(map(list, zip(*rays)))
    O = np.array(O)
    T = np.array(T)

    t = cp.Variable(len(O))
    x = cp.Variable((1,3))

    line_points = cp.diag(t) @ (T-O)  + O
    obj = cp.sum(cp.norm(x - line_points,axis=1))
    cons = []
    prob = cp.Problem(cp.Minimize(obj),cons)
    prob.solve()
    return x.value.squeeze(0)

def main():
    TRANSFORMS_JSON_PATH = "transforms.json"
    transforms_data = load_transforms(TRANSFORMS_JSON_PATH)

    # Build a mapping from file name to frame info
    frame_lookup = {
        os.path.basename(frame['file_path']): frame
        for frame in transforms_data['frames']
    }

    print("Available images:")
    for frame in transforms_data['frames']:
        print(f"- {os.path.basename(frame['file_path'])}")

    # Prompt for filenames
    filenames = input("Enter image file names to use (space-separated): ")
    filenames = filenames.strip().split()

    # Validate and collect matching paths
    missing = [f for f in filenames if f not in frame_lookup]
    if missing:
        raise ValueError(f"Could not find the following images in transforms.json: {missing}")

    # Resolve full paths
    base_dir = os.path.dirname(TRANSFORMS_JSON_PATH)
    image_paths = [os.path.join(base_dir, frame_lookup[f]['file_path']) for f in filenames]

    selected_points = select_points_contrast(image_paths)
    if selected_points is None:
        print("Point selection cancelled.")
        return

    rays = []
    for path, points in zip(image_paths, selected_points):
        _, cam_matrix, intrinsics = get_camera_params(transforms_data, path)
        cam_matrix = cam_matrix @ np.diag([1, -1, -1, 1])
        K = np.array([[intrinsics['fl_x'], 0, intrinsics['cx']],
                      [0, intrinsics['fl_y'], intrinsics['cy']],
                      [0, 0, 1]])
        distort = [intrinsics[k] for k in ["k1", "k2", "p1", "p2"]]

        for px, py in points:
            ray_origin, ray_tip = get_ray(px, py, K, cam_matrix, distort)
            rays.append((ray_origin, ray_tip))

    raybunch = np.full((3, len(rays)*3), np.nan)
    for i, (o, t) in enumerate(rays):
        raybunch[:, i*3] = o
        raybunch[:, i*3+1] = o + 10 * (t - o)

    # Placeholder triangulated points for visualization
    points_3d = np.stack([solve_for_points(rays[0::2]), solve_for_points(rays[1::2])])

    print(f"\nDistance between points: {np.linalg.norm(points_3d[0]-points_3d[1])} units")

    visualize_3d(points_3d, raybunch)

if __name__ == "__main__":
    main()
