import pygame
import sys
import os
from pathlib import Path
from splatnav.splat.splat_utils import GSplatLoader
import torch
import numpy as np
import json
import viser
import viser.transforms as tf
import matplotlib as mpl
from splatnav.polytopes.polytopes_utils import find_interior
from drawing.preprocessing import interpolatePoints, arcLength, datasetMaker
import pandas as pd
from nerfstudio.data.dataparsers import nerfstudio_dataparser
from drawing.splinehelper import BSplineEvaluator
from splatnav.ellipsoids.covariance_utils import quaternion_to_rotation_matrix as quat2R
from splatnav.ellipsoids.intersection_utils import gs_sphere_intersection_test
from drawing.projectUtils import checkTrajCollision, transformTraj, trajAffine, c2wFromQuatpos, loadJSON, spoofCameraObj
from torchvision.transforms import Resize
from drawing.adapters import SketchLinearTransform
from matplotlib import cm
from itertools import cycle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['SDL_VIDEO_CENTERED'] = '1'

color_cycle = cycle([tuple(int(255 * c) for c in rgb[:3]) for rgb in cm.tab10.colors])
viser_color_cycle = cycle([rgb[:3] for rgb in cm.tab10.colors])


### ------------------- ###

### PARAMETERS###

# NOTE: THIS REQUIRES CHANGING TO THE SCENE YOU WANT TO VISUALIZE
scene_name = 'flight'      # london, nyc, berlin, alameda, flight or any other scene detailed in splatinfo.json           
sparse = False
method = 'splatplan'        # splatplan or sfc
data_folder = "drawing/data/"
debug = False
camera_make = "zed720prect"

mode = "predict" # Either "label", "results" or "predict"

hold = False # Doesn't remove previous sketches from display

sketchlinewidth = 9

dfname = "test_df.pkl" # For "label" and "results"
modelname = "model_standard.pt" # For "predict"
resizeres = (224,224)
normalisesketch = True
resizer = Resize(resizeres)
savedf = "outputs.pkl" # For "label"

#################



gltocv = np.diag([1,-1,-1,1])
dfnames = [dfname]

intrinsics = loadJSON(f"{data_folder}intrinsics.json")[camera_make]

def placetrajIntoSplat(server,traj, ind, pred=False,cname="", fixed_color=None):
    rotation = tf.SO3.from_x_radians(0.0).wxyz
    points = np.stack([traj[:-1], traj[1:]], axis=1)
    progress = np.linspace(0, 1, len(points))
    # Safety margin color
    if fixed_color is not None:
        colors = np.tile(np.array(fixed_color)[None, None, :], (len(points), 1, 1))
    else:
        cmap = mpl.colormaps['viridis'] if pred else mpl.colormaps['jet']
        colors = np.array([cmap(prog) for prog in progress])[..., :3]
        colors = colors.reshape(-1, 1, 3)

    if pred: 
        name = "pred"
    else: 
        name = "real"

    # Add trajectory to scene
    traj_handle = server.scene.add_line_segments(
        name=f"/traj_{name}{cname}_{ind}",
        points=points,
        colors=colors,
        line_width=10,
        wxyz=rotation,
    )
    return traj_handle

def clearTrajs(handles):
    for handle in handles:
        handle.remove()
    handles = []
    return

def weighted_shuffle(items, weights):
    items = np.array(items)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()  # normalize
    return list(np.random.choice(items, size=len(items), replace=False, p=weights))


def main():
    permanent_sketch = None
    permanent_sketch_color = None
    # HERE BEGINS THE SPLATTING STUFF
    method = 'splatplan'        # splatplan or sfc

    # Can visualize SplatPlan and the SFC. Can also visualize the sparse scenario.
    try:
        if scene_name == 'statues':
            path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

        elif scene_name == 'stonehenge':
            path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

        elif scene_name == 'old_union':
            path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

        elif scene_name == 'flight':
            path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')
            path_to_transforms = Path("data/flight/transforms.json")
            config = nerfstudio_dataparser.NerfstudioDataParserConfig(data=path_to_transforms, train_split_fraction=0.1)
            dataparser = config.setup()
            dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
            cameraParams = dataparser_outputs.cameras[0] # Note that the camera to world matrix may change, and thus not should not be used.
            scale = dataparser_outputs.dataparser_scale

        else:
            # If you want a custom scene, you need to specify the path to the gsplat config file and the trajectory data
            splatjson = loadJSON(f"{data_folder}splatinfo.json")
            content = splatjson[scene_name]

            path_to_gsplat = Path(content["configYML"])
            path_to_transforms = Path(content["transformsJSON"])
            config = nerfstudio_dataparser.NerfstudioDataParserConfig(data=path_to_transforms, train_split_fraction=0.1)
            dataparser = config.setup()
            dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
            cameraParams = dataparser_outputs.cameras[0] # Note that the camera to world matrix may change, and thus not should not be used.
            scale = content["scale"]
            
            
    except:
        raise ValueError("Scene or data not found")

    cameraParams = spoofCameraObj(intrinsics, cameraParams)
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
    scales = gsplat.scales[mask]
    rots = quat2R(gsplat.rots[mask])
    covs = gsplat.covs[mask]
    colors = gsplat.colors[mask]
    opacities = gsplat.opacities[mask]

    server.scene.add_gaussian_splats(
        name="/splats",
        centers= means.cpu().numpy(),
        covariances= covs.cpu().numpy(),
        rgbs= colors.cpu().numpy(),
        opacities= opacities.cpu().numpy(),
        wxyz=rotation,
    )

    # Load in trajectories
    with open(traj_filepath, 'r') as f:
        meta = json.load(f)

    # Visualize each trajectory and corresponding polytope
    # HERE ENDS THE SPLATTING STUFF
    pygame.init()

    df = pd.read_pickle(f"{data_folder}{dfname}").reset_index()
    df["3d_pred"] = df["3d_gt"]

    selection = np.ones(len(df), dtype=bool)
    selection &= (df["env"] == scene_name)

    if mode == "results":
        selection &= df["3d_pred"].notna()

    if mode == "label":
        selection &= df["2d_human"].isna()

    inds = df.index[selection]

    if mode == "results":
        dfs = [pd.read_pickle(f"{data_folder}{dfn}") for dfn in dfnames]

    # Load images from the "images" folder
    """
    image_files = sorted(glob.glob("drawing/images/*.*"))
    if not image_files:
        print("No images found in the 'images' folder.")
        sys.exit()
    """

    rawdepthimage = torch.load(f"{data_folder}depthImages/{df.loc[inds[0]]['depthfile']}").dequantize().numpy()
    imagearray = np.round(255*(rawdepthimage/np.max(rawdepthimage))).astype(int).T
    imagearray = np.stack([imagearray]*3, axis=-1)

    
    # Create window with first image's dimensions
    initial_img = pygame.surfarray.make_surface(imagearray)
    screen = pygame.display.set_mode(initial_img.get_size())
    pygame.display.set_caption("Image Drawing App")
    window_size = screen.get_size()

    current_image_index = 0
    img = pygame.transform.scale(initial_img, window_size)

    # Drawing surface (matches window size)
    drawing_surface = pygame.Surface(window_size, pygame.SRCALPHA)
    drawing_surface.fill((0, 0, 0, 0))

    # Drawing settings
    DRAWING_COLOR = (255, 0, 0, 255)
    THICKNESS = 5

    drawing = False
    last_pos = None
    clock = pygame.time.Clock()

    handles = []

    # Visualize the trajectory and series of line segments
    if mode in ["results", "label"]:
        proj_points = df.at[inds[current_image_index], "2d_projection"]
        pygame.draw.lines(drawing_surface, (0,255,0), False, proj_points,width=sketchlinewidth)
        traj = transformTraj(df,inds[current_image_index])
        handles.append(placetrajIntoSplat(server,traj,inds[current_image_index]))
    if mode == "results":
        human_points = df.at[inds[current_image_index], "2d_human"]
        pygame.draw.lines(drawing_surface, (255,0,0), False, human_points,width=sketchlinewidth)
        
        traj = transformTraj(df,inds[current_image_index], pred=True)
        handles.append(placetrajIntoSplat(server,traj,inds[current_image_index],cname=f"_", pred=True))
    
    if mode == "predict":
        model = torch.load(f"{data_folder}{modelname}")
        predready = False
        model.eval()

    drawpoints = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if mode == "label":
                    pd.to_pickle(df,f"{data_folder}{savedf}")
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mode == "label":
                        pd.to_pickle(df,f"{data_folder}{savedf}")

                elif event.key == pygame.K_SPACE and mode == "label":
                    if len(drawpoints) != 0:
                        # Save drawing
                        drawing_file = f"drawing/data/sketches/sketches_{current_image_index}.png"
                        pygame.image.save(drawing_surface, drawing_file)
                        print(f"Saved drawing to {drawing_file}")

                        drawpoints = np.array(drawpoints)
                        interppoints = interpolatePoints(drawpoints)
                        df.at[inds[current_image_index],"2d_human"] = interppoints

                    drawpoints = []
                
                    # Load next image
                    current_image_index += 1
                    if current_image_index >= len(inds):
                        print("No more images. Exiting.")
                        pygame.quit()
                        sys.exit()

                    # Load and scale new image to window size
                    rawdepthimage = torch.load(f"{data_folder}depthImages/{df.loc[inds[current_image_index]]['depthfile']}").dequantize().numpy()
                    imagearray = np.round(255*(rawdepthimage/np.max(rawdepthimage))).astype(int).T
                    imagearray = np.stack([imagearray]*3, axis=-1)
                    new_img = pygame.surfarray.make_surface(imagearray)
                    img = pygame.transform.scale(new_img, window_size)
                    drawing_surface.fill((0, 0, 0, 0))
                    drawing = False
                    last_pos = None

                    clearTrajs(handles)
                    traj = transformTraj(df,inds[current_image_index])
                    handles.append(placetrajIntoSplat(server,traj,inds[current_image_index]))


                    clients = server.get_clients()
                    for id, client in clients.items():
                        client.camera.look_at = traj[50]

                elif event.key == pygame.K_RETURN and mode == "predict": # Push view from viewer to display
                    clients = server.get_clients()
                    for id, client in clients.items():
                        wxyz = client.camera.wxyz
                        pos = client.camera.position

                    c2w = c2wFromQuatpos(wxyz, pos)

                    camera = cameraParams
                    camera.camera_to_worlds = torch.Tensor(c2w)[None,:] # Kind of a hacky solution, just switch the matrix in one camera object to create new views
                    outputs = gsplat.splat.pipeline.model.get_outputs_for_camera(camera)
                    depth = outputs["depth"].detach().clone().squeeze()/scale

                    imagearray = np.round(255*(depth.cpu().numpy()/np.max(depth.cpu().numpy()))).astype(int).T
                    imagearray = np.stack([imagearray]*3, axis=-1)
                    new_img = pygame.surfarray.make_surface(imagearray)
                    img = pygame.transform.scale(new_img, window_size)
                    drawing_surface.fill((0, 0, 0, 0))
                    if permanent_sketch is not None:
                        pygame.draw.lines(drawing_surface, permanent_sketch_color, False, list(permanent_sketch[::5]) + list(permanent_sketch[-1:1000000]), width=sketchlinewidth)
                    predready = True
                    

                elif event.key == pygame.K_SPACE and len(drawpoints) != 0 and mode == "predict" and predready:
                    # Save drawing

                    drawpoints = np.array(drawpoints)
                    interppoints = interpolatePoints(drawpoints)
                    torchinputs = torch.from_numpy(interppoints[None,:,:]).to(device).to(torch.float32)

                    if normalisesketch:
                        torchinputs = 2*torchinputs / torch.Tensor([1280, 720]).to(device) - 1

                    resizedimage = resizer(depth[None,:,:])
                    with torch.no_grad():
                        #traj = model(torchinputs,depth[None,None,:,:]).detach().clone().cpu().numpy().squeeze().T
                        resizedimage = (resizedimage - 4.9893)/4.8041
                        resizedimage = torch.clamp_max(resizedimage, 2.25)
                        traj = model(resizedimage[None,:,:], torchinputs).detach().clone().cpu().numpy().squeeze().T
                    # Feed points to model to obtain new traj
                    traj = trajAffine(traj,scale,c2w @ gltocv)

                    if hold:
                        color_rgb = next(viser_color_cycle)
                        sketch_color = tuple(int(c * 255) for c in color_rgb) + (255,)
                    else:
                        color_rgb = (0.0, 1.0, 0.0)
                        sketch_color = (0, 255, 0, 255)

                    # Save sketch permanently if it's the first one
                    
                    permanent_sketch = interppoints
                    permanent_sketch_color = sketch_color

                    pygame.draw.lines(drawing_surface, sketch_color, False, list(interppoints[::5]) + list(interppoints[-1:1000000]), width=sketchlinewidth)
                    if not hold:
                        clearTrajs(handles)
                    handles.append(placetrajIntoSplat(server, traj, f"{np.random.randint(10000000000)}", pred=True, fixed_color=color_rgb))

                    #print(checkTrajCollision(gsplat,torch.from_numpy(traj[None,:]).to(device=device).to(torch.float32),robot_radius,scale)) # Check for collision


                elif event.key == pygame.K_SPACE and mode == "results":
                    current_image_index += 1

                    rawdepthimage = torch.load(f"{data_folder}depthImages/{df.loc[inds[current_image_index]]['depthfile']}").dequantize().numpy()
                    imagearray = np.round(255*(rawdepthimage/np.max(rawdepthimage))).astype(int).T
                    imagearray = np.stack([imagearray]*3, axis=-1)
                    new_img = pygame.surfarray.make_surface(imagearray)
                    img = pygame.transform.scale(new_img, window_size)
                    drawing_surface.fill((0, 0, 0, 0))
                    drawing = False
                    last_pos = None

                    human_points = df.at[inds[current_image_index], "2d_human"]
                    #pygame.draw.lines(drawing_surface, (255,0,0), False, human_points, width=sketchlinewidth)

                    proj_points = df.at[inds[current_image_index], "2d_projection"]
                    pygame.draw.lines(drawing_surface, (0,0,255), False, proj_points, width=sketchlinewidth)

                    clearTrajs(handles)
                    traj = transformTraj(df,inds[current_image_index])
                    handles.append(placetrajIntoSplat(server,traj,inds[current_image_index]))

                    
                    traj = transformTraj(df,inds[current_image_index], pred=True)
                    handles.append(placetrajIntoSplat(server,traj,inds[current_image_index],cname=f"_", pred=True))


            elif event.type == pygame.MOUSEBUTTONDOWN and mode in ["label", "predict"]:
                if event.button == 1:
                    drawpoints = []
                    if not (mode == "predict" and hold):
                        drawing_surface.fill((0, 0, 0, 0))
                    drawing = True
                    last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and mode in ["label", "predict"]:
                if event.button == 1:
                    drawing = False
                    last_pos = None
            elif event.type == pygame.MOUSEMOTION and mode in ["label", "predict"]:
                if drawing and last_pos is not None:
                    pygame.draw.line(drawing_surface, DRAWING_COLOR, last_pos, event.pos, THICKNESS)
                    last_pos = event.pos
                    drawpoints.append(last_pos)

        # Display composition
        screen.blit(img, (0, 0))
        screen.blit(drawing_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

        if debug is True:
            clients = server.get_clients()
            for id, client in clients.items():
                print(f"\tposition: {client.camera.position}")

if __name__ == "__main__":
    main()
