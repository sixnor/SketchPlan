import torch
import torch.nn.functional as F
import math
from drawing.projectUtils import checkTrajCollision, trajBatchAffine, spoofCameraObj
import numpy as np
from pathlib import Path
from splatnav.splat.splat_utils import GSplatLoader
from nerfstudio.data.dataparsers import nerfstudio_dataparser
import pandas as pd
import sys
from copy import deepcopy
import pickle

class Sampler:
    def __init__(self, splatinfo, df, robot_radius=0.15, device="cpu", min_point_dist = 2.0, cameraDict=None, data_folder="drawing/data/", verbose=True):
        self.device = device
        self.splatinfo = splatinfo
        self.df = df[df["2d_human"].notna()]
        self.robot_radius = robot_radius
        self.sketches = torch.tensor(np.stack(self.df["2d_human"]),dtype=torch.float32, device=self.device)
        self.trajs = torch.tensor(np.stack(self.df["3d_gt"]),dtype=torch.float32, device=self.device)
        self.traj_scales = torch.tensor(np.stack(self.df["scale"]),dtype=torch.float32, device=self.device)
        self.bbox_centre = torch.tensor(self.splatinfo["center"],dtype=torch.float32, device=self.device)
        self.bbox_axes = torch.tensor(self.splatinfo["axes"],dtype=torch.float32, device=self.device)
        self.scene_scale = torch.tensor(self.splatinfo["scale"], dtype=torch.float32, device=self.device)
        self.min_point_dist = torch.tensor(min_point_dist, dtype=torch.float32, device=self.device) # in metres
        self.gsplat = GSplatLoader(Path(self.splatinfo["configYML"]), device)
        self.index = torch.tensor(self.df.index, dtype=torch.int, device=self.device)
        self.scene_name = self.splatinfo["name"]
        self.data_folder = data_folder
        self.verbose = verbose

        config = nerfstudio_dataparser.NerfstudioDataParserConfig(data=Path(self.splatinfo["transformsJSON"]), train_split_fraction=1) # No test images
        dataparser = config.setup()
        dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        self.cameraParams = dataparser_outputs.cameras[0] # Note that the camera to world matrix may change, and thus not should not be used.
        if cameraDict:
            self.cameraParams = spoofCameraObj(cameraDict, self.cameraParams)


    def gen(self,n):
        col = torch.ones(n,dtype=torch.bool, device=self.device)
        colfree_c2w = torch.zeros((n,4,4), dtype=torch.float32, device=self.device)
        colfree_inds = torch.zeros(n,dtype=torch.int, device=self.device)
        while torch.any(col):
            x0, xf = self.bboxSample(torch.sum(col))
            pose = torch.cat([x0, xf - x0],dim=-1)
            c2w = self.poseToC2W(pose, convention="opencv")
            seltraj = torch.randint(0,len(self.trajs), (int(torch.sum(col)),), device=self.device)
            candidate_trajs = self.trajs[seltraj]
            colfree_inds[col] = self.index[seltraj]
            colfree_c2w[col] = c2w
            transformed_trajs = trajBatchAffine(candidate_trajs,self.scene_scale, c2w)
            col[col.clone()] = checkTrajCollision(self.gsplat, transformed_trajs, self.robot_radius, self.scene_scale)
            if self.verbose:
                print(f"Trajectories in collision {col.sum()}/{n}")
        return colfree_c2w, colfree_inds
    
    def dumpDepthMaps(self, colfree_c2w, colfree_inds):
        # Takes the first pose of each trajectory and dumps all corresponding renders of depth images.
        run = 0
        rows = []
        #df = pd.DataFrame([self.df.iloc[0]]*len(colfree_c2w))
        for k, (ind, c2w) in enumerate(zip(colfree_inds, colfree_c2w)):
            if not k % 100:
                print(f"TRAJECTORY {k}/{len(colfree_inds)}")

            row = deepcopy(self.df.loc[ind.item()])
            
            gltocv = torch.diag(torch.tensor([1,-1,-1,1],dtype=torch.float32, device=self.device))

            c2w_nerfstudio = c2w @ gltocv

            camera = self.cameraParams
            camera.camera_to_worlds = c2w_nerfstudio[None,:] # Kind of a hacky solution, just switch the matrix in one camera object to create new views
            
            outputs = self.gsplat.splat.pipeline.model.get_outputs_for_camera(camera)
            depths = outputs["depth"].detach().clone().cpu().squeeze()/self.scene_scale.detach().clone().cpu()
            rgb = (outputs["rgb"].detach().clone().cpu().squeeze() * 255).to(torch.uint8).permute(-1,0,1)

            depth_quint8 = torch.quantize_per_tensor_dynamic(depths, torch.quint8, reduce_range=False)

            torch.save(depth_quint8, f"{self.data_folder}depthImages/run_shift_{self.scene_name}_{run}.pt")
            row["depthfile"] = f"run_shift_{self.scene_name}_{run}.pt"

            #torch.save(rgb, f"{self.data_folder}rgb/run_shift_{self.scene_name}_{run}.pt")
            row["rgb"] = f"run_shift_{self.scene_name}_{run}.pt"

            row["env"] = self.scene_name
            row["scale"] = self.scene_scale.item()
            row["c2w"] = c2w.detach().clone().cpu().numpy()
                
            rows.append(pickle.loads(pickle.dumps(row))) # Memory leak if just deepcopy is used. Ugly
            run += 1
        df = pd.DataFrame(rows).reset_index()
        return df


    # pull two points from the bbox, to find initial orientation. Find c2w (should already have the code). 
    # Check if traj collides with gaussian geometry.
    # if not, render image and give back as a sample
    # Bundle a large amount together and call dumpDepthMaps
    # Problem is loading in trajs? 

    def poseToC2W(self, pose, convention):
        """
        Converts a pose or batch of poses [x, y, z, vx, vy, vz] into a camera-to-world matrix using PyTorch.

        Args:
            pose: torch.Tensor of shape (6,) or (N, 6)
            convention: "nerfstudio" or "opencv"

        Returns:
            torch.Tensor of shape (4, 4) or (N, 4, 4)
        """
        pose = torch.as_tensor(pose, dtype=torch.float32)
        batched = pose.ndim == 2

        if not batched:
            pose = pose[None]  # Convert to (1, 6)

        positions = pose[:, 0:3]
        directions = pose[:, 3:6]
        directions = directions / directions.norm(dim=1, keepdim=True)

        if convention == "nerfstudio":
            z_dirs = -directions
            up = torch.tensor([0, 0, 1], dtype=torch.float32, device=pose.device)
        elif convention == "opencv":
            z_dirs = directions
            up = torch.tensor([0, 0, -1], dtype=torch.float32, device=pose.device)
        else:
            raise ValueError("Valid coordinate convention not passed")

        up = up.expand(pose.shape[0], -1)
        x_dirs = torch.cross(up, z_dirs, dim=1)
        x_dirs = x_dirs / x_dirs.norm(dim=1, keepdim=True)
        y_dirs = torch.cross(z_dirs, x_dirs, dim=1)

        R_c2w = torch.stack([x_dirs, y_dirs, z_dirs], dim=2)  # (N, 3, 3)
        T_c2w = torch.eye(4, dtype=torch.float32, device=pose.device).unsqueeze(0).repeat(pose.shape[0], 1, 1)
        T_c2w[:, :3, :3] = R_c2w
        T_c2w[:, :3, 3] = positions

        return T_c2w[0] if not batched else T_c2w
    
    def bboxSample(self, n):
        """
        Samples pairs of points (x0, xf) uniformly from inside a box defined by its center and axes.

        Args:
            centre (Tensor): (3,) tensor representing the center of the box
            axes (Tensor): (3, 3) tensor representing box axes as columns
            n (int): number of samples
            scale (float): scaling factor for the rejection threshold
            rejectDist (float or None): if set, ensures distance between x0 and xf >= rejectDist * scale

        Returns:
            x0, xf: (n, 3) tensors of start and end points
        """
        device = self.bbox_centre.device
        axes = self.bbox_axes.T  # (3, 3) -> (3, 3), assume columns are principal axes

        def sample_points(count):
            rand = torch.rand(3, count, device=device) * 2 - 1  # uniform in [-1, 1]
            return (self.bbox_centre[:, None] + axes @ rand).T  # (count, 3)

        x0 = sample_points(n)
        xf = sample_points(n)

        if self.min_point_dist is not None:
            while True:
                dists = torch.norm(x0 - xf, dim=1)
                tooshort = dists < (self.min_point_dist * self.scene_scale)
                if not torch.any(tooshort):
                    break
                num_resample = tooshort.sum()
                x0[tooshort] = sample_points(num_resample)
                xf[tooshort] = sample_points(num_resample)

        return x0, xf
                

    def chopMix(self, threed, twod, inds):
        """
        threed: (B, 100, 3)
        twod: (B, 100, 2)
        inds: (N, c)
        Returns: s
            new_threed: (N, 100, 3)
            new_twod: (N, 100, 2)
        """
        B, T, D3 = threed.shape
        _, _, D2 = twod.shape
        N, c = inds.shape
        seg_len = T // c  # segment length
        
        new_threed = torch.zeros((N, T, D3), dtype=threed.dtype, device=threed.device)
        new_twod = torch.zeros((N, T, D2), dtype=twod.dtype, device=twod.device)

        for n in range(N):
            for i in range(c):
                idx = inds[n, i]
                start = i * seg_len
                end = (i + 1) * seg_len
                seg3d = threed[idx, start:end].clone()
                seg2d = twod[idx, start:end].clone()

                if i > 0:
                    ### Align 3D ###
                    prev_dir3d = new_threed[n, start-1] - new_threed[n, start-2]
                    cur_dir3d = seg3d[1] - seg3d[0]

                    prev_norm3d = F.normalize(prev_dir3d, dim=0)
                    cur_norm3d = F.normalize(cur_dir3d, dim=0)

                    cross = torch.cross(cur_norm3d, prev_norm3d)
                    dot = torch.dot(cur_norm3d, prev_norm3d).clamp(-1, 1)
                    angle = torch.acos(dot)

                    if torch.norm(cross) > 1e-4:
                        axis = cross / torch.norm(cross)
                        R = self.rotation_matrix(axis, angle)
                        seg3d = (R @ (seg3d - seg3d[0]).T).T + new_threed[n, start - 1]

                    ### Align 2D ###
                    prev_dir2d = new_twod[n, start-1] - new_twod[n, start-2]
                    cur_dir2d = seg2d[1] - seg2d[0]

                    angle2d = self.angle_between_2d(cur_dir2d, prev_dir2d)
                    R2d = self.rotation_matrix_2d(angle2d)

                    seg2d = ((seg2d - seg2d[0]) @ R2d.T) + new_twod[n, start - 1]

                new_threed[n, start:end] = seg3d
                new_twod[n, start:end] = seg2d

        return new_threed, new_twod

    def rotation_matrix(self,axis, theta):
        """
        Rodrigues' rotation formula in 3D
        axis: (3,)
        theta: scalar
        Returns: (3,3) rotation matrix
        """
        axis = F.normalize(axis, dim=0)
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=axis.device)

        I = torch.eye(3, device=axis.device)
        return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    def angle_between_2d(self,v1, v2):
        """Returns signed angle from v1 to v2 in radians"""
        v1 = F.normalize(v1, dim=0)
        v2 = F.normalize(v2, dim=0)
        dot = torch.dot(v1, v2).clamp(-1, 1)
        det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant (z-component of cross in 2D)
        return torch.atan2(det, dot)

    def rotation_matrix_2d(self,theta):
        """Returns 2D rotation matrix for angle theta"""
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=theta.device)