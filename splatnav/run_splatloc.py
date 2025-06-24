# %%
from __future__ import annotations

import json
import os, sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union
 
from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm

import open3d as o3d

# add to the PATH
sys.path.append(f"{Path(__file__).parent.parent}")

from pose_estimator.utils import *


# # # # #
# # # # # Config Path
# # # # #

config_path = 'outputs/old_union2/splatfacto/2024-09-02_151414/config.yml'
scene_name = 'old_union'

dataset_path = 'data/old_union2'

# config path:
config_path = Path(f"{os.path.expanduser(config_path)}")

# name of the scene
scene_name: str = scene_name

# %%
# rescale factor
res_factor = None

# option to visualize environment point cloud
visualize_env_point_cloud = False

# option to visualize downsampled point clouds for efficiency
enable_downsampled_visualization = True

# voxel size for downsampling point cloud
downsampled_voxel_size = 0.01

# option to display computation time statistics
display_compute_stats = True

# feature detector for PnP-RANSAC
# options: POI_Detector.SIFT, POI_Detector.SURF, POI_Detector.ORB
# options = [detector for detector in POI_Detector]
feature_detector = POI_Detector.SIFT

# separator
sep = "-" * 100
lev_2_sep = "*" * 100
sep_space = "\x0c" * 3

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize GSplat
gsplat = GaussianSplat(
    config_path=config_path,
    res_factor=res_factor,
    test_mode="test",  # "inference", "val"
    dataset_mode="val",
    device=device,
)

# camera intrinsics
H, W, K = gsplat.get_camera_intrinsics()
K = K.to(device)

# poses in test dataset
poses = gsplat.get_poses()

# images for evaluation
eval_imgs = gsplat.get_images()

# %%
# # #
# # # Load the Dataset
# # #

# path to the dataset of images and poses for evaluation
dataset_path: Path = Path(dataset_path)

# dataset
gsplat_dataset = load_dataset(data_path=dataset_path, dataset_mode="all")

# images in the dataset
dataset_images = [
    gsplat_dataset.get_image_float32(image_idx)
    for image_idx in range(len(gsplat_dataset._dataparser_outputs.image_filenames))
]
# %%

# # #
# # # Pose Localization
# # #

# option to visualize point clouds
visualize_RGBD_point_clouds = False

# option to print the error stats
print_error_stats = True

# option to visualize registration results
enable_registration_visualization = False

# option to visualize matches in PnP-RANSAC
visualize_PnP_matches = False

# global registration method (RANSAC, FGR)
# If global registration is used, you should provide a point cloud of
# the scene <env_pcd> (from the GSplat) and a point cloud from the camera
# e.g., using a monocular depth estimator to generate a point cloud from
# the camera's pose or RGB observation.
global_registration_method = Global_Registration.RANSAC

# local pose refinement method (ICP, PnP_RANSAC)
local_refinement_method = Local_Refinement.PnP_RANSAC

# local pose refinement method (ICP, COLORED_ICP)
local_registration_method = Local_Registration.COLORED_ICP

# option to disable global registration after the first timestep
disable_global_registration = True

# Global registration is required for local registration.
disable_global_registration = (
    disable_global_registration
    if local_refinement_method != Local_Refinement.ICP
    else False
)

# option to add noise to the initial guess of the pose
perturb_init_guess = True

# option to initialize with the perturbed ground-truth pose
init_with_perturbed_gt_pose: bool = True

# parameters with which to perturb the initial guess=
perturb_init_guess_params = {
    "rotation": np.deg2rad(5),
    "translation": 0.1,
}

# option to save images for debugging
save_image_debug = False

# voxel size for downsampling the point cloud in the pose estimation module
voxel_size = 0.05

# number of trials 'timesteps'
num_trials = len(gsplat_dataset)

# time offset from the zeroth index
t_offset = 0

# estimation frequency
estimation_frequency = 1

# estimation error
pose_error = np.inf * np.ones((num_trials + 1, 2))

# estimated pose
est_pose_over_time = []

for idx, t_step in enumerate(range(t_offset, num_trials, estimation_frequency)):
    # #
    # # Generate/Retrieve the RGB Image
    # #

    # ground-truth RGB image
    cam_rgb = dataset_images[t_step].to(device)

    # ground-truth pose
    gt_pose = torch.eye(4)
    gt_pose[:3] = gsplat_dataset.cameras.camera_to_worlds[t_step].to(device)
    gt_pose = gt_pose.cpu().numpy()

    # initial guess
    if idx == 0 and init_with_perturbed_gt_pose:
        init_guess = gt_pose

    if not disable_global_registration:
        # start time
        t0 = time.perf_counter()

        try:
            cam_pcd
            env_pcd
        except NameError as excp:
            print(
                (
                    "To enable global registration, you must provide "
                    + "a point cloud from the camera and "
                    + "a point cloud from the scene."
                )
            )

            raise excp

        # source and target point clouds
        source_down, target_down, source_fpfh, target_fpfh = preprocess_point_clouds(
            source=cam_pcd, target=env_pcd, voxel_size=voxel_size
        )

        # end time
        t1 = time.perf_counter()

        if display_compute_stats:
            print(sep)
            print(f"Downsampling the Point Cloud took {t1-t0} seconds!")
            print(sep)

    # #
    # # Global Registration
    # #

    if not disable_global_registration and init_guess is not None:  # or idx == 0:

        if global_registration_method == Global_Registration.RANSAC:
            # start time
            t0 = time.perf_counter()

            # execute global registration
            global_est_transformation = execute_global_registration(
                source_down=source_down,
                target_down=target_down,
                source_fpfh=source_fpfh,
                target_fpfh=target_fpfh,
                voxel_size=voxel_size,
            )

            # end time
            t1 = time.perf_counter()

            print(lev_2_sep)
            print(f"RANSAC Global Registration")
            print(lev_2_sep)
        elif global_registration_method == Global_Registration.FGR:

            # # #
            # # # FAST GLOBAL REGISTRATION
            # # #

            # FAST GLOBAL REGISTRATION

            # start time
            t0 = time.perf_counter()

            # execute global registration
            global_est_transformation = execute_fast_global_registration(
                source_down=source_down,
                target_down=target_down,
                source_fpfh=source_fpfh,
                target_fpfh=target_fpfh,
                voxel_size=voxel_size,
            )

            # end time
            t1 = time.perf_counter()

            print(lev_2_sep)
            print(f"Fast Global Registration")
            print(lev_2_sep)

        # estimated pose
        # convert from OPENCV Camera convention to OPENGL convention
        global_est_pose = global_est_transformation.transformation.copy()
        global_est_pose[:, 1] = -global_est_pose[:, 1]
        global_est_pose[:, 2] = -global_est_pose[:, 2]

        # pose error
        error = SE3error(gt_pose, global_est_pose)

        if print_error_stats:
            print(
                f"SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
            )

        if display_compute_stats:
            print(f"Global Registration took {t1-t0} seconds!")

        if enable_registration_visualization:
            visualize_registration_result(
                source=cam_pcd,
                target=env_pcd,
                transformation=global_est_transformation.transformation,
                enable_downsampled_visualization=enable_downsampled_visualization,
                downsampled_voxel_size=downsampled_voxel_size,
            )
    elif (perturb_init_guess or init_with_perturbed_gt_pose) and idx == 0:
        # add noise to the initial guess of the pose

        # generate a random rotation axis
        rand_rot_axis = torch.nn.functional.normalize(
            torch.rand(3, device=device), dim=-1
        )

        # random rotation matrix
        rand_rot = vec_to_rot_matrix(
            perturb_init_guess_params["rotation"] * rand_rot_axis
        )

        # initial guess
        init_guess[:3, :3] = rand_rot.cpu().numpy() @ init_guess[:3, :3]
        init_guess[:3, 3] += (
            perturb_init_guess_params["translation"]
            * torch.nn.functional.normalize(torch.rand(3, device=device), dim=-1)
            .cpu()
            .numpy()
        )

        # estimated pose
        global_est_pose = init_guess

        # pose error
        error = SE3error(gt_pose, global_est_pose)

        if print_error_stats:
            print(
                f"Initial SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
            )

        if enable_registration_visualization:
            visualize_registration_result(
                source=cam_pcd,
                target=env_pcd,
                transformation=global_est_pose,
                enable_downsampled_visualization=enable_downsampled_visualization,
                downsampled_voxel_size=downsampled_voxel_size,
            )

    # #
    # # Pose Refinement
    # #

    try:
        if local_refinement_method == Local_Refinement.ICP:
            # #
            # # Iterative Closest Point (ICP)
            # #

            if local_registration_method == Local_Registration.ICP:
                # ICP

                # start time
                t0 = time.perf_counter()

                # local refinement for registration
                local_est_transformation = ICP_refinement_registration(
                    source=cam_pcd,
                    target=target_down,
                    source_fpfh=source_fpfh,
                    target_fpfh=target_fpfh,
                    voxel_size=voxel_size,
                    global_transformation=global_est_transformation,
                )

                # end time
                t1 = time.perf_counter()

                print(lev_2_sep)
                print(f"Local Refinement: ICP Registration")
                print(lev_2_sep)

            if local_registration_method == Local_Registration.COLORED_ICP:
                # Colored ICP

                # start time
                t0 = time.perf_counter()

                # local refinement for registration
                local_est_transformation = Colored_ICP_refinement_registration(
                    source=cam_pcd,
                    target=target_down,
                    source_fpfh=source_fpfh,
                    target_fpfh=target_fpfh,
                    voxel_size=voxel_size,
                    global_transformation=global_est_transformation,
                )

                # end time
                t1 = time.perf_counter()

                print(lev_2_sep)
                print(f"Local Refinement: Colored ICP Registration")
                print(lev_2_sep)

            # estimated pose
            # convert from OPENCV Camera convention to
            local_est_pose = local_est_transformation.transformation.copy()
            local_est_pose[:, 1] = -local_est_pose[:, 1]
            local_est_pose[:, 2] = -local_est_pose[:, 2]

            # pose error
            error = SE3error(gt_pose, local_est_pose)

            if print_error_stats:
                print(
                    f"SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                )

            if display_compute_stats:
                print(f"Local Registration took {t1-t0} seconds!")

            print(sep)
            print(sep)

            if enable_registration_visualization:
                visualize_registration_result(
                    source=cam_pcd,
                    target=env_pcd,
                    transformation=local_est_transformation.transformation,
                    enable_downsampled_visualization=enable_downsampled_visualization,
                    downsampled_voxel_size=downsampled_voxel_size,
                )
        elif local_refinement_method == Local_Refinement.PnP_RANSAC:
            # #
            # # Perspective-n-Point (PnP)
            # #

            print(sep)
            print(lev_2_sep)
            print(f"Local Refinement: PnP-RANSAC")
            print(lev_2_sep)

            # initial guess
            if idx == 0:
                init_guess = torch.tensor(global_est_pose, device=device).float()
            else:
                init_guess = torch.tensor(local_est_pose, device=device).float()

            # start time
            t0 = time.perf_counter()

            # estimated pose
            # convert from OPENCV Camera convention to
            local_est_pose = execute_PnP_RANSAC(
                gsplat,
                init_guess,
                camera_intrinsics_K=K,
                rgb_input=cam_rgb,
                feature_detector=feature_detector,
                save_image=save_image_debug,
                pnp_figure_filename="figures/pnp_init_guess.png",
                print_stats=True,
                visualize_PnP_matches=visualize_PnP_matches,
                pnp_matches_figure_filename="figures/pnp_matches.png",
            )

            # end time
            t1 = time.perf_counter()

            # pose error
            error = SE3error(gt_pose, local_est_pose)

            if print_error_stats:
                print(
                    f"{sep_space} SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                )

            if display_compute_stats:
                print(f"Local Registration took {t1-t0} seconds!")

            # end time
            t1 = time.perf_counter()

    except:
        print("Error in Pose Refinement")
        break

    # close figures
    plt.close()

    # store the pose error
    pose_error[t_step, :] = error

    # store the estimated pose
    est_pose_over_time.append(local_est_pose)

# %%
# # #
# # # Visualize the Estimation Error
# # #

# figure directory
figure_dir: Path = Path(f"figures/{scene_name}/test_runs")

# make directory, if necessary
figure_dir.mkdir(parents=True, exist_ok=True)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 6), dpi=200)

# for better spacing
# fig.tight_layout()

# pose error
ax1.plot(pose_error[:, 0], label="Rotation Error")
ax2.plot(pose_error[:, 1], label="Translation Error")

# axis label
ax1.set_xlabel("Timestep")
ax2.set_xlabel("Timestep")
ax1.set_ylabel("Rotation Error (degrees)")
ax2.set_ylabel("Translation Error (m)")

# axis options
ax1.grid()
ax2.grid()

plt.show()

# save figure
fig.savefig(f"{figure_dir}/error_stats.png")

plt.close()

# %%
# # #
# # # Save the Estimated Pose
# # #

import json

# path to save the estimated pose
est_pose_filepath: Path = Path(f"results/{scene_name}/test_runs/est_pose.json")

# create parent directory, if necessary
est_pose_filepath.parent.mkdir(exist_ok=True, parents=True)

# data to save
dict_est_pose_over_time = dict(
    (
        (f"frame_{key + t_offset + 1:05d}", value.tolist())
        for key, value in enumerate(est_pose_over_time)
    )
)

# save to JSON
with open(est_pose_filepath, "w") as fh:
    json.dump(dict_est_pose_over_time, fh, indent=4)

# %%
