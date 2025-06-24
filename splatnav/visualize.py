import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
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
from drawing.projectUtils import loadJSON, bboxSample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# CHANGE SCENE HERE
scene_name = 'alameda'      # statues, stonehenge, old_union, flight. If custom scene, specify path to gsplat config file and trajectory data            

# Some useful helper functions
def create_polytope_trimesh(polytopes, colors=None):
    for i, (A, b) in enumerate(polytopes):
        # Transfer all tensors to numpy
        pt = find_interior(A, b)

        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        output = trimesh.convex.convex_hull(qhull_pts)

        if colors is not None:
            output.visual.face_colors = colors[i]
            output.visual.vertex_colors = colors[i]

        if i == 0:
            mesh = output
        else:
            mesh += output
    
    return mesh

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

### ------------------- ###

### PARAMETERS###
# NOTE: THIS REQUIRES CHANGING TO THE SCENE YOU WANT TO VISUALIZE
sparse = False
method = 'splatplan'        # splatplan or sfc




# Can visualize SplatPlan and the SFC. Can also visualize the sparse scenario.
try:
    if scene_name == 'statues':
        if sparse:
            path_to_gsplat = Path('outputs/statues/sparse-splat/2024-10-25_113753/config.yml')
        else:
            path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

    elif scene_name == 'stonehenge':
        if sparse:
            path_to_gsplat = Path('outputs/stonehenge/sparse-splat/2024-10-25_120323/config.yml')
        else:
            path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

    elif scene_name == 'old_union':
        if sparse:
            path_to_gsplat = Path('outputs/old_union2/sparse-splat/2024-10-25_113753/config.yml')
        else:
            path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

    elif scene_name == 'flight':
        if sparse:
            path_to_gsplat = Path('outputs/flight/sparse-splat/2024-10-25_114702/config.yml')
        else:
            path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

    else:
        splatjson = loadJSON("drawing/data/splatinfo.json")
        content = splatjson[scene_name]
        center = np.array(content["center"])
        axes = np.array(content["axes"])
        scale = content["scale"]
        path_to_gsplat = Path(content["configYML"])
except:
    raise ValueError("Scene or data not found")

if sparse:
    traj_filepath = f'trajs/{scene_name}_sparse_{method}.json'
else:
    traj_filepath = f'trajs/{scene_name}_{method}.json'

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

### ------------------- ###

try:
    voxels = trimesh.load_mesh(f"{scene_name}_voxel.obj")

    # Load the voxel representation
    server.scene.add_mesh_simple(
    name="/voxel",
    vertices=voxels.vertices,
    faces=voxels.faces,
    wireframe=True,
    opacity=1,
    wxyz=rotation,
    color=(255,0,0)
    )
except:
    print("No voxel mesh found")

### ------------------- ###

# Load in trajectories
with open(traj_filepath, 'r') as f:
    meta = json.load(f)

datas = meta['total_data']

splatjson = loadJSON("drawing/data/splatinfo.json")
content = splatjson[scene_name]
#center = np.array(content["center"])
#axes = np.array(content["axes"])

scale = content["scale"]
path_to_gsplat = Path(content["configYML"])
""" OPTIONAL BBOX VIS
server.scene.add_point_cloud(
    "/points",
    bboxSample(center, axes, 1000, scale=scale, rejectDist=3.0)[0],
    (255,0,0)
)
"""
# Visualize each trajectory and corresponding polytope
for i, data in enumerate(datas):
    if i > 25:
        break

    # Visualize the trajectory and series of line segments
    traj = np.array(data['traj'])[:, :3]

    points = np.stack([traj[:-1], traj[1:]], axis=1)
    progress = np.linspace(0, 1, len(points))

    # Safety margin color
    cmap = mpl.cm.get_cmap('jet')
    colors = np.array([cmap(prog) for prog in progress])[..., :3]
    colors = colors.reshape(-1, 1, 3)

    '''
    # (Only implemented for OMPL)
    # Add start & goal markers 
    start = np.array(data['start'])
    goal = np.array(data['goal'])
    server.scene.add_icosphere(
        name=f"/start_point_{i}",
        position=start,
        radius=0.005,  # Adjust radius as needed
        color=np.array([0, 1, 0]),  # Green color for start point
        wxyz=rotation,
    )
    goal = np.array(data['goal'])
    server.scene.add_icosphere(
        name=f"/goal_point_{i}",
        position=goal,
        radius=0.005,  # Adjust radius as needed
        color=np.array([1, 0, 0]),  # Red color for goal point
        wxyz=rotation,
    )
    '''
    # Add trajectory to scene
    server.scene.add_line_segments(
        name=f"/trajs/traj_{i}",
        points=points,
        colors=colors,
        line_width=10,
        wxyz=rotation,
    )
    print(f'Method: {method}')
    if method == 'sfc' or method == 'splatplan':
        # Visualize the polytopes as well
        polytopes = data['polytopes']
        polytopes = [(np.array(polytope)[..., :3], np.array(polytope)[..., 3]) for polytope in polytopes]

        colors = np.array([cmap(i) for i in np.linspace(0, 1, len(polytopes))])[..., :3]
        colors = colors.reshape(-1, 3)
        colors = np.concatenate([colors, 0.1*np.ones((len(polytopes), 1))], axis=-1)
        colors = (255*colors).astype(np.uint8)

        # Create polytope corridor mesh object
        corridor_mesh = create_polytope_trimesh(polytopes, colors=colors)

        # Add the corridor to the scene
        server.scene.add_mesh_trimesh(
        name=f"/corridors/corridor_{i}",
        mesh=corridor_mesh,
        wxyz=rotation,
        visible=False
        )

while True:
    clients = server.get_clients()
    for id, client in clients.items():
        pass
        #print(f"\tposition: {client.camera.position}")
    time.sleep(2.0)
