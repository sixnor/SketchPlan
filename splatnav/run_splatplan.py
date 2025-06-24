import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import torch
from pathlib import Path    
import time
import numpy as np
import json
from splatnav.SFC.corridor_utils import SafeFlightCorridor
from splatnav.splat.splat_utils import GSplatLoader
from splatnav.splatplan.splatplan import SplatPlan
from splatnav.splatplan.spline_utils import SplinePlanner
from drawing.projectUtils import loadJSON, bboxSample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

# Methods for the simulation
n = 1000        # number of different configurations
n_steps = 100   # number of time discretizations
minimum_path_length = 3.0 # minimum distance between initial and final state in meters

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

# Using sparse representation?
sparse = False
# Accel and velocity limits (m/s^2, m/s)
global_amax = 0.75
global_vmax = 0.75
# Robot collsion radius (m)
global_radius = 0.15

### ----------------- Possible Methods ----------------- ###
# method = 'splatplan'
# method = 'sfc-*' {* can be 1, 2, 3, 4 for different modes, check SFC.corridor_utils.SafeFlightCorridor for more details} 
### ----------------- Possible Distance Types ----------------- ###

for sparse in [False]:
    for scene_name in ['london']:
        for method in ['splatplan']:

            # NOTE: POPULATE THE UPPER AND LOWER BOUNDS FOR OTHER SCENES!!!
            if scene_name == 'old_union':
                radius_z = 0.01     # How far to undulate up and down
                radius_config = 1.35/2  # radius of xy circle
                mean_config = np.array([0.14, 0.23, -0.15]) # mean of the circle
                # Change these to square that inscribes the circle

                if sparse:
                    path_to_gsplat = Path('outputs/old_union2/sparse-splat/2024-10-25_113753/config.yml')
                else:
                    path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

                radius = 0.01       # radius of robot
                amax = 0.1
                vmax = 0.1

                xlow = [-0.535,-0.445,-0.16]
                xhigh = [0.815,0.905,-0.14]
                scale = 0.22264282186595347

                lower_bound = torch.tensor([-.8, -.7, -0.2], device=device)
                upper_bound = torch.tensor([1., 1., -0.1], device=device)

                resolution = 100

                

            elif scene_name == 'stonehenge':
                radius_z = 0.01
                radius_config = 0.784/2
                mean_config = np.array([-0.08, -0.03, 0.05])

                if sparse:
                    path_to_gsplat = Path('outputs/stonehenge/sparse-splat/2024-10-25_120323/config.yml')
                else:
                    path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

                radius = 0.01
                amax = 0.1
                vmax = 0.1

                lower_bound = torch.tensor([-.5, -.5, -0.], device=device)
                upper_bound = torch.tensor([.5, .5, 0.3], device=device)

                resolution = 150

            elif scene_name == 'statues':
                radius_z = 0.03    
                radius_config = 0.475
                mean_config = np.array([-0.064, -0.0064, -0.025])

                if sparse:
                    path_to_gsplat = Path('outputs/statues/sparse-splat/2024-10-25_114702/config.yml')
                else:
                    path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

                radius = 0.03
                amax = 0.1
                vmax = 0.1

                lower_bound = torch.tensor([-.5, -.5, -0.3], device=device)
                upper_bound = torch.tensor([.5, .5, 0.2], device=device)
 
                resolution = 100

            elif scene_name == 'flight':
                radius_z = 0.06
                radius_config = 0.545/2
                mean_config = np.array([0.19, 0.01, -0.02])

                if sparse:
                    path_to_gsplat = Path('outputs/flight/sparse-splat/2024-10-25_115216/config.yml')
                else:
                    path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

                radius = 0.02
                amax = 0.1
                vmax = 0.1
                # Some corners to create a sampling bbox from.
                # [ 0.58827141  0.3886472  -0.09083986]
                # [ 0.58803772 -0.23819628 -0.06762304]
                # [ 0.56630861 -0.21673055  0.15262594]
                # [-0.95453431 -0.26713421 -0.05526212]
                # [-0.87066678  0.31304889 -0.05482201]

                # [ 0.6  0.4  -0.1]
                # [ 0.6 -0.2 -0.1]
                # Sample uniformly from [0.6,-0.2,-0.1] to [-1,0.4,0.15]
                # [ 0.6 -0.2  0.15]
                # [-1 -0.2 -0.1]
                xlow = [-1,-0.2,-0.1]
                xhigh = [0.6,0.4,0.05]
                scale = 0.1390141015176164

                lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
                upper_bound = torch.tensor([1, 0.5, 0.26], device=device)
                # Maybe rejection sample based on eccentricity.

                resolution = 100

            else:
                splatjson = loadJSON("drawing/data/splatinfo.json")
                content = splatjson[scene_name]
                center = np.array(content["center"])
                axes = np.array(content["axes"])
                
                scale = content["scale"]
                path_to_gsplat = Path(content["configYML"])
                upper_bound = torch.tensor(np.sum(np.abs(axes),axis=0)*1.2 + center, device=device)
                lower_bound = torch.tensor(-np.sum(np.abs(axes),axis=0)*1.2 + center, device=device)

                x0, xf = bboxSample(center, axes, n, scale=scale, rejectDist=minimum_path_length)
                
                resolution = torch.tensor([200,100,50]).to(device)
                resolution = 125       
            amax = global_amax * scale
            vmax = global_vmax * scale
            radius = global_radius * scale

            print(f"Running {scene_name} with {method}")

            # Robot configuration
            robot_config = {
                'radius': radius,
                'vmax': vmax,
                'amax': amax,
            }

            # Environment configuration (specifically voxel)
            voxel_config = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'resolution': resolution,
            }

            tnow = time.time()
            gsplat = GSplatLoader(path_to_gsplat, device)
            print('Time to load GSplat:', time.time() - tnow)

            spline_planner = SplinePlanner(spline_deg=6, device=device)
            
            if method == 'splatplan':
                planner = SplatPlan(gsplat, robot_config, voxel_config, spline_planner, device)

                # Creates the voxel grid for visualization
                # if sparse:
                #     planner.gsplat_voxel.create_mesh(f'blender_envs/{scene_name}_sparse_voxel.obj')
                # else:
                #planner.gsplat_voxel.create_mesh(f'{scene_name}_voxel.obj')

            elif method.split("-")[0] == "sfc":
                mode = int(method.split("-")[1])
                planner = SafeFlightCorridor(gsplat, robot_config, voxel_config, spline_planner, device, mode=mode)

            else:
                raise ValueError(f"Method {method} not recognized")

            ### Create configurations in a circle
            x0, xf = bboxSample(center, axes, n, scale=scale, rejectDist=minimum_path_length)

            """
            x0 = np.random.uniform(xlow, xhigh, size=(n,3))
            xf = np.random.uniform(xlow, xhigh, size=(n,3))

            while np.any(np.linalg.norm(x0-xf,axis=1) < minimum_path_length*scale): # Rejection sampling for minimum path length go brrr
                tooshort = np.linalg.norm(x0-xf,axis=1) < minimum_path_length*scale
                x0[tooshort] =  np.random.uniform(xlow, xhigh, size=(np.sum(tooshort),3))
                xf[tooshort] =  np.random.uniform(xlow, xhigh, size=(np.sum(tooshort),3))
            """
            # Run simulation
            total_data = []
            for trial, (start, goal) in enumerate(zip(x0, xf)):

                # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
                x = torch.tensor(start).to(device).to(torch.float32)
                goal = torch.tensor(goal).to(device).to(torch.float32)

                tnow = time.time()
                torch.cuda.synchronize()
                try:
                    output = planner.generate_path(x, goal)
                except: 
                    output = None
                torch.cuda.synchronize()
                if output is None:
                    print(f"TRIAL {trial} FAILED")
                else: 
                    plan_time = time.time() - tnow
                    output['plan_time'] = plan_time

                    total_data.append(output)
                    print(f"Trial {trial} completed")

            radius_z = 0
            radius_z = 0.06
            radius_config = 0.545/2
            mean_config = np.array([-0.064, -0.0064, -0.025])
            # Save trajectory
            data = {
                'scene': scene_name,
                'method': method,
                'radius': radius,
                'amax': amax,
                'vmax': vmax,
                'radius_z': radius_z,
                'radius_config': radius_config,
                'mean_config': mean_config.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'n_steps': n_steps,
                'total_data': total_data,
            }

            # create directory if it doesn't exist
            os.makedirs('trajs', exist_ok=True)
            
            # write to the file
            if sparse:
                save_path = f'trajs/{scene_name}_sparse_{method}.json'
            else:
                save_path = f'trajs/{scene_name}_{method}.json'
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
        
# %%
