import os
import time
from pathlib import Path
from splatnav.splat.splat_utils import GSplatLoader
import torch 
import numpy as np
import trimesh
import json
import viser
import viser.transforms as tf
import matplotlib as mpl
import scipy
from splatnav.polytopes.polytopes_utils import find_interior
from drawing.projectUtils import loadJSON
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'



def xyztheta_to_mesh(xyz, theta):
    """
    Create a hyperrectangle (cuboid) mesh given a center, scales, and a rotation about the z-axis.

    Parameters:
        center (array-like, shape (3,)): The (x,y,z) center of the cuboid.
        xyz (array-like, shape (3,)): The full extents (lengths) of the cuboid along the x, y, and z axes.
        theta (float): Rotation angle (in radians) about the z-axis.
    
    Returns:
        vertices (ndarray): Array of shape (8, 3) with vertex positions.
        faces (ndarray): Array of triangle faces (each row with 3 vertex indices), shape (12, 3).
    """
    dims = np.array(xyz, dtype=float)
    hx, hy, hz = dims / 2.0

    # Define the local vertices of the box centered at origin:
    local_verts = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz]
    ])

    # Rotation matrix about the z-axis:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Rz = np.array([
        [ cos_t, -sin_t, 0],
        [ sin_t,  cos_t, 0],
        [ 0,      0,     1]
    ])

    # Apply rotation to each vertex and then translate by the center.
    vertices = (local_verts @ Rz.T)

    # Define triangle faces (two per each of 6 faces)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],   # Bottom face
        [4, 5, 6], [4, 6, 7],   # Top face
        [0, 1, 5], [0, 5, 4],   # Front face
        [1, 2, 6], [1, 6, 5],   # Right face
        [2, 3, 7], [2, 7, 6],   # Back face
        [3, 0, 4], [3, 4, 7]    # Left face
    ], dtype=int)

    return vertices, faces

def dumpParams(center, xyz, theta):
    center = np.array(center, dtype=float)
    dims = np.array(xyz, dtype=float)
    hx, hy, hz = dims / 2.0

    # Define the local vertices of the box centered at origin:
    local_verts = np.diag([hx,hy,hz])

    # Rotation matrix about the z-axis:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Rz = np.array([
        [ cos_t, -sin_t, 0],
        [ sin_t,  cos_t, 0],
        [ 0,      0,     1]
    ])

    # Apply rotation to each vertex and then translate by the center.
    vertices = (local_verts @ Rz.T)

    return vertices, center


### PARAMETERS###
# NOTE: THIS REQUIRES CHANGING TO THE SCENE YOU WANT TO VISUALIZE
scene_name = 'alameda'      # statues, stonehenge, old_union, flight. If custom scene, specify path to gsplat config file and trajectory data            


# Can visualize SplatPlan and the SFC. Can also visualize the sparse scenario.
jsonname = "drawing/data/splatinfo.json"
splatjson = loadJSON(jsonname)
path_to_gsplat = Path(splatjson[scene_name]["configYML"])

bounds = None
rotation = tf.SO3.from_x_radians(0.0).wxyz      # identity rotation

### ------------------- ###
gsplat = GSplatLoader(path_to_gsplat, device)

server = viser.ViserServer()

### ------------------- ###
# Only visualize the gsplat within some bounding box set by bounds
if bounds is not None:
    mask = torch.all((gsplat.means - bounds[:, 0] >= 0) & (bounds[:, 1] - gsplat.means >= 0), dim=-1)
else:
    mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

means = gsplat.means[mask]
covs = gsplat.covs[mask]
colors = gsplat.colors[mask]
opacities = gsplat.opacities[mask]

# Add splat to the scene
server.scene.add_gaussian_splats(
    name="/splats",
    centers= means.cpu().numpy(),
    covariances= covs.cpu().numpy(),
    rgbs= colors.cpu().numpy(),
    opacities= opacities.cpu().numpy(),
    wxyz=rotation,
)
### ------------------- ###


gui_anchor = server.gui.add_vector3(
            "Anchor",
            initial_value=(1.0, 1.0, 1.0),
            step=0.001)

gui_x = server.gui.add_slider(
    "X",
    initial_value=0.1,
    min=0.0,
    max=3.0,
    step=0.0001
)

gui_y = server.gui.add_slider(
    "Y",
    initial_value=0.1,
    min=0.0,
    max=3.0,
    step=0.0001
)

gui_z = server.gui.add_slider(
    "Z",
    initial_value=0.1,
    min=0.0,
    max=3.0,
    step=0.0001
)

gui_theta = server.gui.add_slider(
    "Theta",
    initial_value=0.1,
    min=0,
    max=2*np.pi,
    step=0.0001
)

gui_anchor_set = server.gui.add_button("Set Anchor to current position")
gui_dump = server.gui.add_button("Save sampling box")

scale = (gui_x.value, gui_y.value, gui_z.value)
theta = gui_theta.value
v, f = xyztheta_to_mesh(scale,theta)
mesh_handle = server.scene.add_mesh_simple(
        "/mesh",
        v,
        f,
        (255,0,0),
        wireframe=True,
        position= gui_anchor.value
    )

@gui_anchor.on_update
@gui_x.on_update
@gui_y.on_update
@gui_z.on_update
@gui_theta.on_update
def _(_) -> None:
    global mesh_handle
    mesh_handle.remove()
    scale = (gui_x.value, gui_y.value, gui_z.value)
    theta = gui_theta.value
    v, f = xyztheta_to_mesh(scale,theta)
    mesh_handle = server.scene.add_mesh_simple(
        "/mesh",
        v,
        f,
        (255,0,0),
        wireframe=True,
        position= gui_anchor.value
    )
    
@gui_anchor_set.on_click
def _(_) -> None:
    for id, client in clients.items():
        gui_anchor.value = client.camera.position

@gui_dump.on_click
def _(_) -> None:
    center = gui_anchor.value
    scale = (gui_x.value, gui_y.value, gui_z.value)
    theta = gui_theta.value
    axes, center= dumpParams(center, scale, theta)
    data = {
    "axes": axes,
    "center": center}
    
    splatjson[scene_name].update({key: value.tolist() for key, value in data.items()})
    with open(jsonname, "w") as f:
        json.dump(splatjson, f, indent=4)



### ------------------- ###
while True:
    clients = server.get_clients()
    for id, client in clients.items():
        #print(f"\tposition: {client.camera.position}")
        pass
    time.sleep(2.0)