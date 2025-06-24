import numpy as np
from pathlib import Path
import viser
import time
from scipy.spatial import cKDTree
from collections import defaultdict
import json
import os, sys
import trimesh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drawing.projectUtils import loadJSON
from splatnav.splat.splat_utils import GSplatLoader
import torch

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SLIDER_STATE_FILE = "visualization/position_slider_state.json"

splat = "hard"
x_min, x_max = -7.0, 7.0
y_min, y_max = -3.0, 3.0
z_min, z_max = -2.75, np.inf

# GOAL POINTS
goal_points = np.array([
    [4.16,  0.9,  -1.06],
    [4.05,  -0.5,  -1.06],
    [3.45, -1.9,  -1.06],
    [4.16,  0.9,  -1.06],
    [4.32,  2.08, -1.10]
]) # HARD

#goal_points = np.array([[3.363342046737670898e+00, -5.301108211278915405e-02, -1.0]]) # Medium

#goal_points = np.array([[2.25, -0.10, -1.0]]) # Easy 1
#goal_points = np.array([[2.25, -1.5, -1.0]]) Easy 2

splatinfo = loadJSON("drawing/data/splatinfo.json")[splat]
gsplat = GSplatLoader(Path(splatinfo["configYML"]), device)

means = gsplat.means.cpu().numpy()
covs = gsplat.covs.cpu().numpy()

T = np.array(splatinfo["transform"])



method_colors = {
    "convmodelbeeg": (0.121, 0.466, 0.705),  # blue
    "diffraw":       (0.839, 0.152, 0.156),  # red
    "diffbakedmay20":(0.172, 0.627, 0.172),  # green
}

box_aabb = np.array([[-7.92-10,-3.275-10, 0.0],[8.7+10,3.37+10,0.1]])
box_center = [0.0, 0.0, 0.3]
box_size = [30.0, 30.0, 0.2]  # Width, Height, Depth
box_colour = (68/255, 65/255, 60/255)

def transform_gaussian_splat(means, covs, T):
    """
    Apply 4x4 affine transform to Gaussian splats.
    
    Args:
        means: (N, 3) array of Gaussian centers
        covs: (N, 3, 3) array of Gaussian covariances
        T: (4, 4) transformation matrix
    
    Returns:
        means_out: (N, 3) transformed means
        covs_out: (N, 3, 3) transformed covariances
    """
    A = T[:3, :3]
    t = T[:3, 3]

    means_out = means @ A.T + t
    covs_out = A @ covs @ A.T  # Uses broadcasting

    return means_out, covs_out

def load_slider_state():
    if Path(SLIDER_STATE_FILE).exists():
        try:
            with open(SLIDER_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load slider state: {e}")
    return {}

def analyze_paths_against_positions(root_folder: Path):
    print("\n=== Path Analysis ===")
    root_folder = Path(root_folder).resolve()
    path_files = sorted(root_folder.rglob("path_*"))

    # Group per run
    run_max_dists = defaultdict(list)

    for path_file in path_files:
        run_dir = path_file.parent
        run_name = run_dir.relative_to(root_folder)

        try:
            path_data = np.loadtxt(path_file)
            if path_data.ndim != 2 or path_data.shape[1] < 3:
                print(f"Skipping {path_file}: not valid 3D path")
                continue

            z_diff = path_data[0, 2] - path_data[-1, 2]

            # Look for corresponding positions.txt
            pos_file = run_dir / "positions.txt"
            if not pos_file.exists():
                print(f"{path_file.relative_to(root_folder)} — Δz: {z_diff:.3f} — No positions.txt found")
                continue

            pos_data = np.loadtxt(pos_file)
            if pos_data.ndim != 2 or pos_data.shape[1] < 2:
                print(f"Skipping {pos_file}: not valid positions")
                continue
            if pos_data.shape[1] == 2:
                pos_data = np.hstack([pos_data, np.zeros((pos_data.shape[0], 1))])

            tree = cKDTree(pos_data[:, :3])
            dists, _ = tree.query(path_data[:, :3], k=1)
            max_dist = dists.max()
            run_max_dists[run_name].append(max_dist)

            print(f"{path_file.relative_to(root_folder)} — Δz: {z_diff:.3f} — Max dist to positions: {max_dist:.3f}")
        except Exception as e:
            print(f"Error processing {path_file}: {e}")

    print("\n=== Per-Run Maximum Distances ===")
    for run_name, max_dists in run_max_dists.items():
        overall_max = max(max_dists)
        print(f"{run_name} — Max of path max-dists: {overall_max:.3f}")

def save_slider_state(position_slider_data):
    state = {}
    for item in position_slider_data:
        key = item["name"].replace("/positions", "").lstrip("/")
        state[key] = {
            "start": item["start_slider"].value,
            "end": item["end_slider"].value
        }
    try:
        with open(SLIDER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Saved slider state to {SLIDER_STATE_FILE}")
    except Exception as e:
        print(f"Failed to save slider state: {e}")

def plot_paths_from_folder(root_folder: str):
    global position_slider_data
    root_folder = Path(root_folder).resolve()
    path_files = sorted(root_folder.rglob("path_*"))
    print(f"Found {len(path_files)} path files")

    vis = viser.ViserServer()
    vis.scene.set_up_direction("-z")
    all_points = []

    colors = [
        (0.121, 0.466, 0.705), (1.0, 0.498, 0.054), (0.172, 0.627, 0.172),
        (0.839, 0.152, 0.156), (0.580, 0.403, 0.741), (0.549, 0.337, 0.294),
        (0.890, 0.466, 0.760), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133),
        (0.090, 0.745, 0.811)
    ]

    # Plot each path
    for i, path_file in enumerate(path_files):
        try:
            data = np.loadtxt(path_file)
            pts = np.stack([data[:, 0], data[:, 1], data[:, 2]], axis=1)
            all_points.append(pts)

            segments = np.stack([pts[:-1], pts[1:]], axis=1)

            # Get method name
            filename = path_file.stem  # e.g., easy2025-05-20_09-58-30_convmodelbeeg
            if "pcloudeasy" in filename:
                continue  # Skip if it's the stray file
            method = filename.split('_')[-1]

            color = np.array(method_colors.get(method, (0.5, 0.5, 0.5)))  # fallback gray
            colors_arr = np.broadcast_to(color, segments.shape)

            # Structure: /run_name/method/path_file
            run_dir = path_file.parent.relative_to(root_folder)
            name = f"/{run_dir}/{method}/{path_file.stem}"

            vis.scene.add_line_segments(
                name=name,
                points=segments,
                colors=colors_arr,
                line_width=10.0
            )
        except Exception as e:
            print(f"Skipping {path_file}: {e}")

    # Plot positions as connected line segments
    # Plot positions as connected line segments
    # Plot positions with color based on method (inferred from folder name suffix)
    position_slider_data = []

    pos_files = sorted(root_folder.rglob("positions.txt"))
    slider_state = load_slider_state()
    for j, pos_file in enumerate(pos_files):
        try:
            pos_data = np.loadtxt(pos_file)
            if pos_data.ndim != 2:
                print(f"Invalid positions file shape: {pos_file}")
                continue
            pos_x, pos_y = pos_data[:, 0], pos_data[:, 1]
            pos_z = pos_data[:, 2] if pos_data.shape[1] == 3 else np.zeros_like(pos_x)
            pts = np.stack([pos_x, pos_y, pos_z], axis=1)
            all_points.append(pts)

            session_folder = pos_file.parent.name
            method = session_folder.split('_')[-1]
            color = np.array(method_colors.get(method, (0.5, 0.5, 0.5)))

            rel_path = pos_file.parent.relative_to(root_folder)
            name = f"/{rel_path}/positions"

            # Create line segment handle, but use dummy data initially
            handle = vis.scene.add_line_segments(
                name=name,
                points=np.zeros((1, 2, 3)),  # placeholder
                colors=np.broadcast_to(color, (1, 2, 3)),
                line_width=7.5
            )

            # Add sliders to trim the position range
            with vis.gui.add_folder(f"{rel_path}"):
                key = str(rel_path)
                start_default = slider_state.get(key, {}).get("start", 0)
                end_default = slider_state.get(key, {}).get("end", len(pts)-1)

                start_slider = vis.gui.add_slider(f"{rel_path}_start", min=0, max=len(pts)-2, step=1, initial_value=start_default)
                end_slider = vis.gui.add_slider(f"{rel_path}_end", min=1, max=len(pts)-1, step=1, initial_value=end_default)

            position_slider_data.append({
                "name": name,
                "points": pts,
                "handle": handle,
                "start_slider": start_slider,
                "end_slider": end_slider,
                "color": color
            })

        except Exception as e:
            print(f"Skipping {pos_file}: {e}")


    # Equal axis scaling
    print("Serving visualization at http://localhost:8080")
    return vis

def add_box(vis):
    # Create a box mesh using trimesh
    box = trimesh.creation.box(bounds=box_aabb)
    box.apply_translation(box_center)

    # Convert to triangle mesh format
    vertices = np.array(box.vertices, dtype=np.float32)
    faces = np.array(box.faces, dtype=np.uint32)

    # Add to Viser
    vis.scene.add_mesh_simple(
        name="/solid_box",
        vertices=vertices,
        faces=faces,
        color=box_colour,
    )

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else input("Enter folder path: ")
    vis = plot_paths_from_folder(folder)
    analyze_paths_against_positions(Path(folder))
    augmeans, augcovs = transform_gaussian_splat(means, covs, T)

    valid = (augmeans[:,0] > x_min) & (augmeans[:,0] < x_max)
    valid &= (augmeans[:,1] > y_min) & (augmeans[:,1] < y_max)
    valid &= (augmeans[:,2] > z_min) & (augmeans[:,2] < z_max)

    vis.scene.add_gaussian_splats("/gsplat",augmeans[valid], 
                                  augcovs[valid], 
                                  rgbs= gsplat.colors.cpu().numpy()[valid],
                                  opacities= gsplat.opacities.cpu().numpy()[valid])
    
    for i, point in enumerate(goal_points):
        l = vis.scene.add_icosphere(
            name=f"/goal_{i}",
            radius=1.0,
            position=point,
            color=(1.0, 0.0, 1.0),  # green,
            subdivisions=1
        )
        l.opacity = 1.0
        l.wireframe = True
    add_box(vis)
    try:
        while True:
            for item in position_slider_data:
                s, e = item["start_slider"].value, item["end_slider"].value
                if s < e:
                    seg = np.stack([item["points"][s:e], item["points"][s+1:e+1]], axis=1)
                    item["handle"].points = seg
                    item["handle"].colors = np.broadcast_to(item["color"], seg.shape)
    except KeyboardInterrupt:
        save_slider_state(position_slider_data)
