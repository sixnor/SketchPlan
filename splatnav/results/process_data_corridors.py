#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from tqdm import tqdm
import json
import polytope
from splat.gsplat_utils import GSplatLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ----------------- Possible Methods ----------------- ###
# method = 'splatplan'
# method = 'sfc-* {* can be 1, 2, 3, 4}'
# method = 'ompl'
### ----------------- Possible Distance Types ----------------- ###

for sparse in [True]:
    for method in ['ompl']:
        for scene_name in ['stonehenge', 'statues', 'flight', 'old_union']:

            # TODO: POPULATE THE UPPER AND LOWER BOUNDS FOR FASTER DISTANCE QUERYING!!!
            if scene_name == 'old_union':

                if sparse:
                    path_to_gsplat = Path('outputs/old_union2/sparse-splat/2024-10-25_113753/config.yml')
                else:
                    path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

            elif scene_name == 'stonehenge':

                if sparse:
                    path_to_gsplat = Path('outputs/stonehenge/sparse-splat/2024-10-25_120323/config.yml')
                else:
                    path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

            elif scene_name == 'statues':
    
                if sparse:
                    path_to_gsplat = Path('outputs/statues/sparse-splat/2024-10-25_114702/config.yml')
                else:
                    path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

            elif scene_name == 'flight':
                radius_z = 0.06
                radius_config = 0.545/2
                mean_config = np.array([0.19, 0.01, -0.02])

                if sparse:
                    path_to_gsplat = Path('outputs/flight/sparse-splat/2024-10-25_115216/config.yml')
                else:
                    path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

            print(f"Running {scene_name} with {method}")

            tnow = time.time()
            gsplat = GSplatLoader(path_to_gsplat, device)
            print('Time to load GSplat:', time.time() - tnow)

            # Load file
            current_path = Path.cwd()  # Get the current working directory as a Path object
            parent_path = current_path.parent  # Get the parent directory

            if sparse:
                save_path = f'trajs/{scene_name}_sparse_{method}.json'
            else:
                save_path = f'trajs/{scene_name}_{method}.json'
            
            with open( os.path.join(str(parent_path), save_path), 'r') as f:
                meta = json.load(f)

            # Load in the data
            total_data = meta['total_data']
            radius = meta['radius']

            total_data_processed = []
            for i, data in enumerate(total_data):
                print(f"Processing trajectory {i}/{len(total_data)}")

                if len(data['traj']) == 0:
                    print(f"Trajectory {i} is empty")
                    data['feasible'] = False
                    total_data_processed.append(data)
                    continue
                elif len(data['traj']) >= 1:
                    if method == 'ompl' or method == 'nerfnav':
                        data['feasible'] = True

                if not data['feasible']:
                    total_data_processed.append(data)
                    continue

                traj = torch.tensor(data['traj'], device=device)[:, :3]

                # Compute the distance to the GSplat
                safety_margin = []
                for pt in traj:
                    h, grad_h, hess_h, info = gsplat.query_distance(pt, radius=radius, distance_type='ball-to-ellipsoid')

                    # NOTE: IMPORTANT!!! h is the squared signed distance minus the radius squared, so we need to undo this, because we want signed distance - radius
                    # record min value of h
                    squared_signed_distance = torch.min(h) + radius**2
                    sign_dist = torch.sign(squared_signed_distance)
                    mag_dist = torch.abs(squared_signed_distance)
                    signed_distance = sign_dist * torch.sqrt(mag_dist) - radius
                    safety_margin.append(signed_distance.item())

                # Compute the total path length
                path_length = torch.sum(torch.norm(traj[1:] - traj[:-1], dim=1)).item()

                data['safety_margin'] = safety_margin
                data['path_length'] = path_length

                if method == 'ompl' or method == 'nerfnav':
                    pass
                else:
                    # Quality of polytopes
                    polytopes = data['polytopes'] #[torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes]

                    polytope_vols = []
                    polytope_radii = []

                    polytope_margin = []
                    for poly in polytopes:

                        poly = np.array(poly)
                        A = poly[:, :-1]
                        b = poly[:, -1]

                        p = polytope.Polytope(A, b)
                        polytope_vols.append(p.volume)
                        polytope_radii.append(np.linalg.norm(p.chebR))

                        vertices = torch.tensor(polytope.extreme(p), device=device, dtype=torch.float32)

                        for vertex in vertices:
                            h, grad_h, hess_h, info = gsplat.query_distance(vertex, radius=radius, distance_type='ball-to-ellipsoid')

                            squared_signed_distance = torch.min(h) + radius**2
                            sign_dist = torch.sign(squared_signed_distance)
                            mag_dist = torch.abs(squared_signed_distance)
                            signed_distance = sign_dist * torch.sqrt(mag_dist) - radius

                            # record min value of h
                            polytope_margin.append(signed_distance.item())

                    data['polytope_vols'] = polytope_vols
                    data['polytope_radii'] = polytope_radii
                    data['polytope_margin'] = polytope_margin

                total_data_processed.append(data)

            meta['total_data'] = total_data_processed

            if sparse:
                write_path = f'trajs/{scene_name}_sparse_{method}_processed.json'
            else:
                write_path = f'trajs/{scene_name}_{method}_processed.json'

            # Save the data
            with open( os.path.join(str(parent_path), write_path), 'w') as f:
                json.dump(meta, f, indent=4)

#%%