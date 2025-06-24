#%% 
import os
import numpy as np
import torch
import json
import time 

#Import utilies
from splatnav.nerfnav.nav.planner import NerfNav
from splatnav.nerfnav.nav.math_utils import vec_to_rot_matrix, sample_sphere
from splatnav.nerfnav.nerf.nerf import NeRFWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

def simulate(planner_cfg, density_fn, iteration, device):
    '''
    Encapsulates planning.
    '''

    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']
    penalty = planner_cfg['penalty']
    
    # Creates a workspace to hold all the trajectory data
    basefolder = f"nerfnav_paths/{planner_cfg['exp_name']}/{penalty}_iter{iteration}"
    try:
        os.makedirs(basefolder)
        os.mkdir(basefolder + "/init_poses")
        os.mkdir(basefolder + "/init_costs")
        print("created", basefolder)
    except:
        pass
  
    # Initialize Planner
    traj = NerfNav(start_state, end_state, planner_cfg, density_fn, device)

    traj.basefolder = basefolder

    # Create a coarse trajectory to initialize the planner by using A*. 
    traj.a_star_init()

    # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
    # that minimizes collision and control effort.
    output = traj.learn_init()

    return output

### Baseline configs

### ----- NERF-NAV PARAMETERS ----- #

g = 1e-6            # Assumes no gravity. gravitational constant
density = 0.28 / (4./3. * np.pi * (0.25)**3) # density of MODAL AI Starling 2 drone used in hardware experiments in kg/m^3, but approximated as a sphere.

### PLANNER CONFIGS, device=device
# X, Y, Z

# Rotation vector
start_R = [0., 0., 0.0]     # Starting orientation (Euler angles)
end_R = [0., 0., 0.0]       # Goal orientation

# Angular and linear velocities
init_rates = torch.zeros(3) # All rates

# Change rotation vector to rotation matrix 3x3
start_R = vec_to_rot_matrix( torch.tensor(start_R))
end_R = vec_to_rot_matrix(torch.tensor(end_R))

# Run
T_final = 10.                # Final time of simulation in seconds

planner_lr = 1e-4          # Learning rate when learning a plan
epochs_init = 100          # Num. Gradient descent steps to perform during initial plan
fade_out_epoch = 0
fade_out_sharpness = 10
epochs_update = 250         # Num. grad descent steps to perform when replanning

penalty = 100       # Penalty for collision avoidance

# Methods for the simulation
n = 100         # number of different configurations

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

### ----------------- Possible Methods ----------------- ###
# method = 'nerfnav'

for scene_name in ['stonehenge', 'statues', 'flight', 'old_union']:

    if scene_name == 'old_union':
        radius_z = 0.01     # How far to undulate up and down
        radius_config = 1.35/2  # radius of xy circle
        mean_config = np.array([0.14, 0.23, -0.15]) # mean of the circle

        path_to_config = 'outputs/old_union2/nerfacto/2024-09-12_203602' # points to where the nerf params are stored

        radius = 0.01       # radius of robot

        lower_bound = torch.tensor([-.8, -.7, -0.2], device=device)
        upper_bound = torch.tensor([1., 1., -0.1], device=device)

        resolution = 100

        mass = (4./3. * np.pi * radius**3) * density          #kg, mass of drone.

    elif scene_name == 'stonehenge':
        radius_z = 0.01
        radius_config = 0.784/2
        mean_config = np.array([-0.08, -0.03, 0.05])

        path_to_config = 'outputs/stonehenge/nerfacto/2024-12-04_153827'
        radius = 0.01

        lower_bound = torch.tensor([-.5, -.5, -0.], device=device)
        upper_bound = torch.tensor([.5, .5, 0.3], device=device)

        resolution = 150

        mass = (4./3. * np.pi * radius**3) * density          #kg, mass of drone.

    elif scene_name == 'statues':
        radius_z = 0.03    
        radius_config = 0.475
        mean_config = np.array([-0.064, -0.0064, -0.025])

        path_to_config = 'outputs/statues/nerfacto/2024-09-12_204832'

        radius = 0.03

        lower_bound = torch.tensor([-.5, -.5, -0.3], device=device)
        upper_bound = torch.tensor([.5, .5, 0.2], device=device)

        resolution = 100

        mass = (4./3. * np.pi * radius**3) * density          #kg, mass of drone.

    elif scene_name == 'flight':
        radius_z = 0.06
        radius_config = 0.545/2
        mean_config = np.array([0.19, 0.01, -0.02])

        path_to_config = 'outputs/flight/nerfacto/2024-09-14_083406'

        radius = 0.02

        lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
        upper_bound = torch.tensor([1, 0.5, 0.26], device=device)

        resolution = 100

        mass = (4./3. * np.pi * radius**3) * density          #kg, mass of drone.

    print(f"Running {scene_name} with NerfNav")
    # Approximate as a sphere, I = 2/5 * m * r^2
    I = np.diag([2/5 * mass * radius**2, 2/5 * mass * radius**2, 2/5 * mass * radius**2]) # Inertia tensor
    robot_pcd = sample_sphere(radius, 10).to(device)

    # Instantiate the NeRF model
    nerf = NeRFWrapper(path_to_config, use_ns_coordinates=True)

    #Store configs in dictionary
    planner_cfg = {
    "T_final": T_final,
    "steps": 100,
    "lr": planner_lr,
    "epochs_init": epochs_init,
    "fade_out_epoch": fade_out_epoch,
    "fade_out_sharpness": fade_out_sharpness,
    "epochs_update": epochs_update,
    'start_state': None,
    'end_state': None,
    'exp_name': scene_name,                  # Experiment name
    'I': torch.tensor(I).float().to(device),
    'g': g,
    'mass': mass,
    'radius': radius,
    'robot_pcd': robot_pcd,
    'resolution': resolution,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound,
    'cutoff': 0.3,      # Density cutoff for collision detection
    'penalty': penalty,
    }

    ### Create configurations in a circle
    x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
    x0 = x0 + mean_config

    xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
    xf = xf + mean_config

    # Run simulation
    total_data = []
    
    for trial, (start, goal) in enumerate(zip(x0, xf)):

        # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
        x = torch.tensor(start)
        goal = torch.tensor(goal)

        start_state = torch.cat( [x[:3], init_rates, start_R.reshape(-1), init_rates], dim=0).to(device).to(torch.float32)
        end_state   = torch.cat( [goal[:3],   init_rates, end_R.reshape(-1), init_rates], dim=0 ).to(device).to(torch.float32)
        
        planner_cfg['start_state'] = start_state
        planner_cfg['end_state'] = end_state

        tnow = time.time()
        torch.cuda.synchronize()

        output = simulate(planner_cfg, nerf.get_density, trial, device)

        torch.cuda.synchronize()
        plan_time = time.time() - tnow
        
        output['plan_time'] = plan_time

        total_data.append(output)
        print(f"Trial {trial} completed")

    # Save trajectory
    data = {
        'scene': scene_name,
        'method': 'nerfnav',
        'radius': radius,
        'mass': mass,
        'inertia': I.tolist(),
        'radius_z': radius_z,
        'radius_config': radius_config,
        'mean_config': mean_config.tolist(),
        'lower_bound': lower_bound.tolist(),
        'upper_bound': upper_bound.tolist(),
        'resolution': resolution,
        'total_data': total_data,
    }

    # create directory if it doesn't exist
    os.makedirs('trajs', exist_ok=True)
    
    # write to the file
    save_path = f'trajs/{scene_name}_nerfnav.json'
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

#%%