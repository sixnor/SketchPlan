import os
import time
import json
from pathlib import Path
import torch
import numpy as np
import open3d as o3d
import viser
import viser.transforms as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from splatnav.splat.splat_utils import GSplatLoader
from drawing.projectUtils import loadJSON

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape
    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    T = np.eye(4)
    T[:3, :3] = c * R
    T[:3, 3] = t
    return T

# === Known room corners (match order below) ===
room_points = np.array([
    [-7.92,  3.37,   0.00],   # Back-Right-Floor
    [-7.92, -3.275,  0.00],   # Back-Left-Floor
    [ 8.70,  3.37,   0.00],   # Front-Right-Floor
    [ 8.70, -3.275,  0.00],   # Front-Left-Floor
    [-7.92,  3.37,  -3.17],   # Back-Right-Ceiling
    [-7.92, -3.275, -3.17],   # Back-Left-Ceiling
    [ 8.70,  3.37,  -3.17],   # Front-Right-Ceiling
    [ 8.70, -3.275, -3.17],   # Front-Left-Ceiling
])

gsplat_path = Path("outputs/hard/splatfacto/2025-05-29_192120/config.yml")

server = viser.ViserServer()
print("Place 8 points in the following order:")
print("1. Back-Right-Floor")
print("2. Back-Left-Floor")
print("3. Front-Right-Floor")
print("4. Front-Left-Floor")
print("5. Back-Right-Ceiling")
print("6. Back-Left-Ceiling")
print("7. Front-Right-Ceiling")
print("8. Front-Left-Ceiling")

selected_points = []
handles = []

gui_add = server.gui.add_button("Add point at current camera position")
gui_reset = server.gui.add_button("Reset points")

# === GSplat loading ===
gsplat = GSplatLoader(gsplat_path, device)
means = gsplat.means.numpy(force=True)
covs = gsplat.covs.numpy(force=True)
colors = gsplat.colors.numpy(force=True)
opacities = gsplat.opacities.numpy(force=True)

server.scene.add_gaussian_splats("/splat",
                                 centers=means,
                                 covariances=covs,
                                 rgbs=colors,
                                 opacities=opacities)

# === Fine-tuning sliders ===
gui_dx = server.gui.add_slider("dx", -1.0, 1.0, 0.001, 0.0)
gui_dy = server.gui.add_slider("dy", -1.0, 1.0, 0.001, 0.0)
gui_dz = server.gui.add_slider("dz", -1.0, 1.0, 0.001, 0.0)

# Internal state for offsets
fine_offsets = np.zeros(3)

def update_fine_tune(_=None):
    if not handles:
        return
    idx = len(handles) - 1
    base_pos = selected_points[idx]
    offset = np.array([gui_dx.value, gui_dy.value, gui_dz.value])
    handles[idx].position = base_pos + offset

gui_dx.on_update(update_fine_tune)
gui_dy.on_update(update_fine_tune)
gui_dz.on_update(update_fine_tune)

def reset_fine_sliders():
    gui_dx.value = 0.0
    gui_dy.value = 0.0
    gui_dz.value = 0.0

@gui_add.on_click
def add_point(_=None):
    if len(selected_points) >= 8:
        print("Already added 8 points.")
        return
    cam = next(iter(server.get_clients().values())).camera
    pos = np.array(cam.position)
    selected_points.append(pos)

    handle = server.scene.add_icosphere(
        name=f"/pt{len(selected_points)}",
        radius=0.02,
        position=pos,
        color=(0.0, 1.0, 0.0),
    )
    handles.append(handle)
    reset_fine_sliders()
    print(f"Point {len(selected_points)} added at {pos}")

    if len(selected_points) == 8:
        user_pts = np.array([
            p + np.array([gui_dx.value, gui_dy.value, gui_dz.value]) if i == 7 else p
            for i, p in enumerate(selected_points)
        ])
        T = kabsch_umeyama(room_points, user_pts)
        print("Estimated Transformation:\n", T)

@gui_reset.on_click
def reset_points(_=None):
    global selected_points, handles
    for h in handles:
        h.remove()
    selected_points = []
    handles = []
    reset_fine_sliders()
    print("Points reset.")

print("Viser point picker + Kabsch alignment running at http://localhost:8080")

while True:
    time.sleep(1)