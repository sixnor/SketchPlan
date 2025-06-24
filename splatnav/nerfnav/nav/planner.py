import torch
import numpy as np
import json

from splatnav.nerfnav.nav.math_utils import rot_matrix_to_vec, next_rotation, astar
from splatnav.initialization.grid_utils import PointCloudVoxel

class NerfNav:
    def __init__(self, start_state, end_state, cfg, density_fn, device):
        self.nerf = density_fn
        self.device = device

        self.cfg                = cfg
 
        # Gradient / Learning / Optimization parameters
        self.lr                 = cfg['lr']
        self.epochs_init        = cfg['epochs_init']
        self.epochs_update      = cfg['epochs_update']
        self.fade_out_epoch     = cfg['fade_out_epoch']
        self.fade_out_sharpness = cfg['fade_out_sharpness']
        self.penalty            = cfg['penalty']

        # Dynamics / Robot parameters
        self.T_final            = cfg['T_final']
        self.steps              = cfg['steps']

        self.mass               = cfg['mass']
        self.J                  = cfg['I']
        self.radius             = cfg['radius']
        self.g                  = torch.tensor([0., 0., -cfg['g']]).to(device)

        self.resolution        = cfg['resolution']
        self.lower_bound       = cfg['lower_bound']
        self.upper_bound       = cfg['upper_bound']
        self.cutoff            = cfg['cutoff']
        self.robot_pcd         = cfg['robot_pcd']

        # Initial and Goal
        self.start_state = start_state
        self.end_state   = end_state

        # Time discretization
        self.dt = self.T_final / self.steps

        # Straight line trajectory initialization
        slider = torch.linspace(0, 1, self.steps, device=self.device)[1:-1, None]

        states = (1-slider) * self.full_to_reduced_state(start_state) + \
                    slider  * self.full_to_reduced_state(end_state)
        
        self.states = states.clone().detach().requires_grad_(True)
        self.initial_accel = torch.tensor([cfg['g'], cfg['g']]).to(device).requires_grad_(True)

        self.epoch = 0

    def full_to_reduced_state(self, state):
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0., 0. ] , device=self.device)
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle], device=self.device) ], dim = -1).detach()

    def a_star_init(self):

        self.cell_sizes = (self.upper_bound - self.lower_bound) / self.resolution

        # Voxel grid centers
        X, Y, Z = torch.meshgrid(
            torch.linspace(self.lower_bound[0] + self.cell_sizes[0]/2, self.upper_bound[0] - self.cell_sizes[0]/2, self.resolution, device=self.device),
            torch.linspace(self.lower_bound[1] + self.cell_sizes[1]/2, self.upper_bound[1] - self.cell_sizes[1]/2, self.resolution, device=self.device),
            torch.linspace(self.lower_bound[2] + self.cell_sizes[2]/2, self.upper_bound[2] - self.cell_sizes[2]/2, self.resolution, device=self.device)
        )
        self.grid_centers = torch.stack([X, Y, Z], dim=-1)
        
        coords = self.grid_centers.reshape(-1, 3)

        output = self.nerf(coords)

        # 20, 20, 20
        occupied_mask = (output > self.cutoff).squeeze()
        occupied_points = coords[occupied_mask]

        # A* and voxel grid initialization
        self.voxel_grid = PointCloudVoxel(occupied_points, self.lower_bound, self.upper_bound, self.resolution, self.radius, self.device)

        path = self.voxel_grid.create_path(self.start_state[:3], self.end_state[:3])
        path = torch.tensor(path, dtype=torch.float32, device=self.device)
   
        #Diff. flat outputs (x,y,z,yaw)
        states = torch.cat( [path, torch.zeros( (path.shape[0], 1) , device=self.device) ], dim=-1)

        #prevents weird zero derivative issues
        #randomness = torch.normal(mean= 0, std=0.01*torch.ones(states.shape)).to(self.device)
        #states += randomness

        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        states = (prev_smooth + next_smooth + states)/3

        self.states = states.clone().detach().to(self.device).requires_grad_(True)

    def params(self):
        return [self.initial_accel, self.states]

    def calc_everything(self):

        start_pos   = self.start_state[None, 0:3]
        start_v     = self.start_state[None, 3:6]
        start_R     = self.start_state[6:15].reshape((1, 3, 3))
        start_omega = self.start_state[None, 15:]

        end_pos   = self.end_state[None, 0:3]
        end_v     = self.end_state[None, 3:6]
        end_R     = self.end_state[6:15].reshape((1, 3, 3))
        end_omega = self.end_state[None, 15:]

        next_R = next_rotation(start_R, start_omega, self.dt)

        # start, next, decision_states, last, end

        start_accel = start_R @ torch.tensor([0,0,1.0]).to(self.device) * self.initial_accel[0] + self.g
        next_accel = next_R @ torch.tensor([0,0,1.0]).to(self.device) * self.initial_accel[1] + self.g

        next_vel = start_v + start_accel * self.dt
        after_next_vel = next_vel + next_accel * self.dt

        next_pos = start_pos + start_v * self.dt
        after_next_pos = next_pos + next_vel * self.dt
        after2_next_pos = after_next_pos + after_next_vel * self.dt
    
        # position 2 and 3 are unused - but the atached roations are
        current_pos = torch.cat( [start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos], dim=0)

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]

        current_vel = (next_pos - prev_pos)/self.dt
        current_vel = torch.cat( [ current_vel, end_v], dim=0)

        prev_vel = current_vel[:-1, :]
        next_vel = current_vel[1: , :]

        current_accel = (next_vel - prev_vel)/self.dt - self.g

        # duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel = torch.cat( [ current_accel, current_accel[-1,None,:] ], dim=0)

        accel_mag     = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = current_accel/accel_mag

        # remove states with rotations already constrained
        z_axis_body = z_axis_body[2:-1, :]

        z_angle = self.states[:,3]

        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        rot_matrix = torch.cat( [start_R, next_R, rot_matrix, end_R], dim=0)

        current_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        current_omega = torch.cat( [ current_omega, end_omega], dim=0)

        prev_omega = current_omega[:-1, :]
        next_omega = current_omega[1:, :]

        angular_accel = (next_omega - prev_omega)/self.dt
        # duplicate last ang_accceleration - its not actaully used for anything (there is no action at last state)
        angular_accel = torch.cat( [ angular_accel, angular_accel[-1,None,:] ], dim=0)

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]
        actions =  torch.cat([ accel_mag*self.mass, torques ], dim=-1)

        return current_pos, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # if not torch.allclose( actions[:2, 0], self.initial_accel ):
        #     print(actions)
        #     print(self.initial_accel)

        return actions

    def get_next_action(self):
        actions = self.get_actions()
        # fz, tx, ty, tz
        return actions[0, :]

    def body_to_world(self, points):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_state_cost(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0].to(self.device)
        torques = torch.norm(actions[:, 1:], dim=-1).to(self.device)

        # S, B, 3  =  S, _, 3 +      _, B, 3   X    S, _,  3
        # B_body, B_omega = torch.broadcast_tensors(self.robot_body, omega[:,None,:])
        # point_vels = vel[:,None,:] + torch.cross(B_body, B_omega, dim=-1)

        # S, B
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5

        # S, B
        density = self.nerf( self.body_to_world(self.robot_pcd) )
        density = density.squeeze()

        # Ignore densities that are way too high, as they make optimization difficult
        density = torch.clamp(density, max=1e3)

        # multiplied by distance to prevent it from just speed tunnelling
        # S =   S,B * S,_
        collision_prob = torch.mean(density[None,:] * distance[:,None], dim = -1) 

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t)).to(self.device)
            colision_prob = colision_prob * mask

        traj = (pos, vel, accel, rot_matrix, omega, angular_accel, actions)

        diff = pos[1:] - pos[:-1]
        diff = torch.sum(diff**2, dim=-1)

        #PARAM cost function shaping
        return self.penalty*torch.mean(collision_prob) + torch.mean(diff), torch.mean(collision_prob), traj

    def total_cost(self):
        total_cost, collision_loss, traj  = self.get_state_cost()
        return total_cost ,traj, collision_loss

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        for it in range(self.epochs_init):
            opt.zero_grad()
            self.epoch = it
            loss, traj, coll = self.total_cost()
            print(f'Iteration ({it}): {loss.item()}. Collision loss: {coll.item()}')
            loss.backward()
            opt.step()

        pos, vel, accel, rot_matrix, omega, angular_accel, actions = traj

        output = {
            'traj': torch.cat([pos, vel, accel], dim=-1).tolist(),
            'angular': torch.cat([rot_matrix.reshape(-1, 9), omega, angular_accel], dim=-1).tolist(),
            'action': actions.tolist(),
        }

        return output

    def learn_update(self, iteration):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss, traj, coll = self.total_cost()
            print(it, loss)
            loss.backward()
            opt.step()
            # it += 1

            # if (it > self.epochs_update and self.max_residual < 1e-3):
            #     break

            save_step = 50
            if it%save_step == 0:
                if hasattr(self, "basefolder"):
                    self.save_poses(self.basefolder / "replan_poses" / (str(it//save_step)+ f"_time{iteration}.json"))
                    self.save_costs(self.basefolder / "replan_costs" / (str(it//save_step)+ f"_time{iteration}.json"))
                else:
                    print("Warning: data not saved!")

    def update_state(self, measured_state):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)
        self.initial_accel = actions[1:3, 0].detach().requires_grad_(True)
        # print(self.initial_accel.shape)

    def save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}
        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.cpu().detach().numpy()
                pose[:3, 3]  = pos.cpu().detach().numpy()
                pose[3,3] = 1

                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    def save_data(self, data, filename):
        with open(filename,"w+") as f:
            json.dump(data, f, indent=4)

    def save_costs(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        total_cost, colision_loss  = self.get_state_cost()

        output = {"colision_loss": colision_loss.cpu().detach().numpy().tolist(),
                  "pos": positions.cpu().detach().numpy().tolist(),
                  "actions": actions.cpu().detach().numpy().tolist(),
                  "total_cost": total_cost.cpu().detach().numpy().tolist()}

        with open(filename,"w+") as f:
            json.dump( output,  f, indent=4)

    def save_progress(self, filename):
        if hasattr(self.renderer, "config_filename"):
            config_filename = self.renderer.config_filename
        else:
            config_filename = None

        to_save = {"cfg": self.cfg,
                    "start_state": self.start_state,
                    "end_state": self.end_state,
                    "states": self.states,
                    "initial_accel":self.initial_accel,
                    "config_filename": config_filename,
                    }
        torch.save(to_save, filename)