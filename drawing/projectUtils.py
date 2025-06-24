import cv2
import numpy as np
import json
from splatnav.ellipsoids.covariance_utils import quaternion_to_rotation_matrix as quat2R
from splatnav.ellipsoids.intersection_utils import gs_sphere_intersection_test
import torch
import json
import viser.transforms as tf
from torchvision.transforms import Resize
from scipy.ndimage import distance_transform_edt




def transformTraj(df, ind, pred=False):
    # Takes traj in drone centered view and spits it back out in viser
    c2w = df.loc[ind]["c2w"]
    scale = df.loc[ind]["scale"]
    if pred:
        traj = df.loc[ind]["3d_pred"].T
    else:
        traj = df.loc[ind]["3d_gt"].T
    traj = trajAffine(traj, scale, c2w)
    return traj

def c2wFromQuatpos(quat, pos): # Takes in wxyz quaternion and position in viser convention and returns c2w in nerfstudio format (without scene scaling)
    gltocv = np.diag([1,-1,-1,1])
    xyzw = np.roll(quat, 3) # I really dislike rotations
    crot = tf.SE3.from_rotation_and_translation(tf.SO3.from_quaternion_xyzw(xyzw), pos)
    c2w = crot.as_matrix()
    c2w = c2w @ gltocv
    return c2w

def trajAffine(traj,scale,c2w):
    traj = np.vstack([scale*traj, np.ones(traj.shape[1])])
    traj = (c2w @ traj).T
    traj = traj[:,:-1]
    return traj

def trajBatchAffine(trajs, scales, c2ws):
    # Input Tensor of shape BxTx3
    # Output Tensor of shape BxTx3
    aff = c2ws[:,:-1,:-1] * scales.view((-1,1,1))
    trans = c2ws[:,:-1,-1]
    retrajs = torch.baddbmm(trans.view((-1,1,3)),trajs, aff.mT)
    return retrajs

def reproject2Dto3D(twodpoints, depth, data_folder="drawing/data/"):
    with open(f"{data_folder}cameraParams.json", 'r') as file:
            cJSON = json.load(file)

    cameraMatrix = np.array(cJSON["matrix"])

    dist_params = np.array(cJSON["distortion"])
    fx,fy, cx, cy = cJSON["fx"],cJSON["fy"], cJSON["cx"], cJSON["cy"] 

    undist = cv2.undistortImagePoints(twodpoints, cameraMatrix, dist_params)

    undist = undist.squeeze()
    undist -= np.array([cx, cy])

    depth = depth[:,None]
    pred3d = np.concatenate([depth*undist/np.array([fx,fy]), depth],axis=1)
    return pred3d

def checkTrajCollision(gsplat, trajs, robot_radius, scene_scale, n_eval=100):
    # Trajs should (NxTx3)
    # Trajs should be in nerfstudio convention? 

    trajs = trajs.to(torch.float32)
    device = trajs.device
    eval_points = torch.linspace(0,1,n_eval)[:, None].to(device=device).to(torch.float32)

    means = gsplat.means
    scales = gsplat.scales
    rots = quat2R(gsplat.rots)

    minscale = gsplat.scales.amin(dim=-1)
    maxscale = gsplat.scales.amax(dim=-1)
    
    collisions = torch.zeros(len(trajs), dtype=torch.bool, device=device)
    for i, traj in enumerate(trajs):
        dists = torch.cdist(traj, means)
        over  = dists < maxscale + robot_radius*scene_scale # does the robot intersect the minimum volume enclosing sphere?
        under = dists < minscale + robot_radius*scene_scale # does the robot intersect the maximum volume inscribed sphere?
        if torch.all(~over): # robot never in enclosing sphere
            collisions[i] = False
        elif torch.any(under): # robot in enscribed sphere
            collisions[i] = True
        else: # Edge case, we have to check all points in enclosing sphere
            tinds, ginds = torch.argwhere(over).T
            eval_points = torch.linspace(0, 1, n_eval)[:, None].to(torch.float32).to(trajs.device)
            notcols, _ = gs_sphere_intersection_test(rots[ginds],scales[ginds],robot_radius*scene_scale,means[ginds], traj[tinds], eval_points)
            collisions[i] = torch.any(~notcols)

    return collisions

def loadJSON(jsonfilepath):
    with open(jsonfilepath, "r") as f:
        loaded_data = json.load(f)
    return loaded_data

def bboxSample(centre, axes, n,scale=1.0,rejectDist=None):
    axes = axes.T
    x0 = centre + (axes @ np.random.uniform(-1,1,(3,n))).T
    xf = centre + (axes @ np.random.uniform(-1,1,(3,n))).T
    if rejectDist:
        while np.any(np.linalg.norm(x0-xf,axis=1) < rejectDist*scale): # Rejection sampling for minimum path length go brrr
            tooshort = np.linalg.norm(x0-xf,axis=1) < rejectDist*scale
            x0[tooshort] =  centre + (axes @ np.random.uniform(-1,1,(3,np.sum(tooshort)))).T
            xf[tooshort] =  centre + (axes @ np.random.uniform(-1,1,(3,np.sum(tooshort)))).T
    return x0, xf
      

def pruneDataset(dataset, screenmargin, eps):
    allpos = torch.all(dataset.threedpoints[:,:,-1] > -eps, dim=(1)) # Trajectory should not be behind the camera
    boundx = torch.all((0 + screenmargin < dataset.twodpoints[:,:,0]) & (dataset.twodpoints[:,:,0] < 1280.0 - screenmargin), dim=(1)) # Sketch should not be beyond x limit
    boundy = torch.all((0 + screenmargin < dataset.twodpoints[:,:,1]) & (dataset.twodpoints[:,:,1] < 720.0 - screenmargin), dim=(1)) # Sketch should be beyond y limit
    viable = boundx.cpu() & boundy.cpu() & allpos.cpu()
    print("VIABLE PROPORTION OF DATASET:",(torch.sum(viable)/len(dataset)).item())

    dataset = torch.utils.data.Subset(dataset, torch.argwhere(viable)) # Filter out trajectories which go behind the camera.
    return dataset


def prune_dataframe(df, screenmargin=10, eps=1e-6):
    threed_all = np.stack(df["3d_gt"].values)  # Shape: (B, N, 3)
    twod_all = np.stack(df["2d_projection"].values)      # Shape: (B, N, 2)

    # Condition 1: All 3D points should be in front of the camera
    allpos = np.all(threed_all[:, :, -1] > -eps, axis=1)

    # Condition 2: x-values within screen bounds
    xvals = twod_all[:, :, 0]
    boundx = np.all((screenmargin < xvals) & (xvals < 1280.0 - screenmargin), axis=1)

    # Condition 3: y-values within screen bounds
    yvals = twod_all[:, :, 1]
    boundy = np.all((screenmargin < yvals) & (yvals < 720.0 - screenmargin), axis=1)

    # Combine masks
    viable = allpos & boundx & boundy

    print("VIABLE PROPORTION OF DATAFRAME:", viable.sum() / len(df))

    # Filter dataframe
    return df[viable].reset_index(drop=True)

def move_dict_to_device(data_dict, device):
    """
    Moves all torch.Tensor objects in a dictionary to the specified device.
    
    Args:
        data_dict (dict): Dictionary containing various objects.
        device (str or torch.device): Target device (e.g., "cuda" or "cpu").
        
    Returns:
        dict: A new dictionary with tensors moved to the specified device.
    """
    return {key: val.to(device, non_blocking=True) if isinstance(val, torch.Tensor) else val for key, val in data_dict.items()}

def splitdf(df, fractions, seed=None):
    rng = np.random.default_rng(seed)
    index = np.arange(len(df))
    rng.shuffle(index)

    split_indices = np.cumsum(fractions[:-1]) * len(index)  # Compute split indices
    split_indices = split_indices.astype(int)  # Convert to integer indices

    indarrs = np.split(index, split_indices)

    return [df.iloc[indarr] for indarr in indarrs]
         

def addOverlay(batchimages, batchscaledpoints):
    # In: NxCxHxW, NxTx2
    # Points be normalised between [-1,1]^2
    # Out Nx(C+1)xHxW

    shape = torch.Tensor([*batchimages.shape[2:]])
    thing = (shape*(batchscaledpoints + 1.0)/2.0).to(torch.int) # Undo normalisation
    thing = torch.clamp(torch.min(thing, shape - 1), min=0).to(torch.int64)[:,None] # Clamp to image space
    pads = torch.zeros((len(thing),1,*batchimages.shape[2:])).to(torch.float32) # get indices
    batch_indices = torch.arange(batchscaledpoints.shape[0]).repeat_interleave(batchscaledpoints.shape[1]) # create index 
    pads[batch_indices,0, thing[:,0,:,1].flatten(), thing[:,0,:,0].flatten()] = 1.0 # apply

    return torch.cat([batchimages,pads],dim=1)

"""
def calcMeanStd(df, data_folder="drawing/data/rgb/", key="rgb", resizeshape=None):
    resizer = Resize(resizeshape, antialias=True)
    num_pixels = 0
    sum_ = torch.zeros(3)  # Assuming RGB images
    sum_sq = torch.zeros(3)

    # First pass: Compute mean
    for dfile in df[key]:
        rgb = torch.load(data_folder + dfile).to(torch.float32) / 255.0  # Load tensor
        rgb = resizer(rgb)
        c, h, w = rgb.shape
        num_pixels += h * w
        sum_ += rgb.view(c, -1).sum(dim=1)  # Sum over all pixels

    mean = sum_ / num_pixels

    # Second pass: Compute standard deviation
    sum_sq = torch.zeros(3)

    for dfile in df[key]:
        rgb = torch.load(data_folder + dfile).to(torch.float32) / 255.0
        rgb = resizer(rgb)
        sum_sq += ((rgb - mean.view(3, 1, 1)) ** 2).view(3, -1).sum(dim=1)

    std = torch.sqrt(sum_sq / num_pixels)
    return mean, std
"""
def calcMeanStd(df, data_folder="drawing/data/rgb/", key="rgb", resizeshape=None):
    num_pixels = 0
    num_channels = None
    sum_ = None

    # First pass: compute sum
    for dfile in df[key]:
        img = torch.load(data_folder + dfile).to(torch.float32) / 255.0
        img = Resize(img, resizeshape, antialias=True)
        if img.dim() == 2:
            img = img.unsqueeze(0)  # Convert (H, W) → (1, H, W)
        c, h, w = img.shape

        if sum_ is None:
            num_channels = c
            sum_ = torch.zeros(c)
            sum_sq = torch.zeros(c)

        num_pixels += h * w
        sum_ += img.view(c, -1).sum(dim=1)

    mean = sum_ / num_pixels

    # Second pass: compute squared deviation
    sum_sq = torch.zeros(num_channels)

    for dfile in df[key]:
        img = torch.load(data_folder + dfile).to(torch.float32) / 255.0
        img = Resize(img, resizeshape, antialias=True)
        if img.dim() == 2:
            img = img.unsqueeze(0)
        sum_sq += ((img - mean.view(num_channels, 1, 1)) ** 2).view(num_channels, -1).sum(dim=1)

    std = torch.sqrt(sum_sq / num_pixels)
    return mean, std

def pointsUndistortRedistort(points, sourceCameraDict, targetCameraDict):
    """
    This function undistorts points from the source camera and reprojects them 
    to the target camera's image plane using fisheye calibration parameters.

    :param points: A list of points in the source camera's image [u, v] (2D).
    :param sourceCameraDict: Dictionary containing the source camera's fisheye parameters.
                             Example: 
                             {
                                 "fx": fx1,
                                 "fy": fy1,
                                 "cx": cx1,
                                 "cy": cy1,
                                 "k1": k1, 
                                 "k2": k2,
                                 "k3": k3,
                                 "p1": p1,
                                 "p2": p2
                             }
    :param targetCameraDict: Dictionary containing the target camera's fisheye parameters.
                             Example: 
                             {
                                 "fx": fx2,
                                 "fy": fy2,
                                 "cx": cx2,
                                 "cy": cy2,
                                 "k1": k1,
                                 "k2": k2,
                                 "k3": k3,
                                 "p1": p1,
                                 "p2": p2
                             }
    :return: A list of 2D points in the target camera's image plane.
    """

    # Extract the fisheye parameters for both cameras
    fx1 = sourceCameraDict["fx"]
    fy1 = sourceCameraDict["fy"]
    cx1 = sourceCameraDict["cx"]
    cy1 = sourceCameraDict["cy"]
    k1, k2, k3, p1, p2 = sourceCameraDict["k1"], sourceCameraDict["k2"], sourceCameraDict["k3"], sourceCameraDict["p1"], sourceCameraDict["p2"]

    fx2 = targetCameraDict["fx"]
    fy2 = targetCameraDict["fy"]
    cx2 = targetCameraDict["cx"]
    cy2 = targetCameraDict["cy"]
    k1_t, k2_t, k3_t, p1_t, p2_t = targetCameraDict["k1"], targetCameraDict["k2"], targetCameraDict["k3"], targetCameraDict["p1"], targetCameraDict["p2"]

    # Create camera matrices (3x3)
    K1 = np.array([[fx1, 0, cx1],
                   [0, fy1, cy1],
                   [0, 0, 1]], dtype=np.float32)

    K2 = np.array([[fx2, 0, cx2],
                   [0, fy2, cy2],
                   [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients for both cameras
    dist1 = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    dist2 = np.array([k1_t, k2_t, p1_t, p2_t, k3_t], dtype=np.float32)

    # Convert points to numpy array
    points_np = np.array(points, dtype=np.float32)

    # Undistort the points in the source camera's image
    undistorted_points_source = cv2.undistortPoints(points_np, K1, dist1)

    # Faux place points in 3d space (convert pixel coordinates to homogeneous coordinates)
    homo_points_source = cv2.convertPointsToHomogeneous(undistorted_points_source)

    # Reproject the normalized points to the target camera's image plane
    target_points = cv2.projectPoints(homo_points_source,np.zeros(3), np.zeros(3), K2, dist2)[0]

    # Return the distorted points in the target camera's image plane
    return target_points.reshape(-1, 2)

def spoofCameraObj(cameraDict, cameraObj):
    cameraObj.fx[:] = cameraDict["fx"]
    cameraObj.fy[:] = cameraDict["fy"]
    cameraObj.cx[:] = cameraDict["cx"]
    cameraObj.cy[:] = cameraDict["cy"]
    cameraObj.width[:] = cameraDict["W"]
    cameraObj.height[:] = cameraDict["H"]
    cameraObj.distortion_params[:] = torch.Tensor([cameraDict["k1"], cameraDict["k2"], cameraDict["k3"], 0.0, cameraDict["p1"], cameraDict["p2"]])
    return cameraObj

def arcLength(array):
    # array should BxNx2 or BxNx3
    return torch.sum(torch.norm(array[:,1:] - array[:,:-1], dim=-1), dim=-1)


def poseToC2W(pose, convention):
    # Takes a pose of the form [x,y,z,vx,vy,vz] and converts creates the corresponding camera to world matrix if the camera was looking from the position in the direction of the velocity with the y-axis upwards.
    if convention == "nerfstudio":
        z_dir = -pose[3:6]/np.linalg.norm(pose[3:6])
        provisional_up = np.array([0, 0, 1])  # Default to world Y-axis, 2d up aligns with 3d up
    elif convention == "opencv":
        z_dir = pose[3:6]/np.linalg.norm(pose[3:6])
        provisional_up = -np.array([0, 0, 1])  # Default to world Y-axis, 2d up aligns with 3d down
    else:
        print("Valid coordinate convention not passed")
        
    x_dir = np.cross(provisional_up, z_dir)
    x_dir = x_dir / np.linalg.norm(x_dir)  # Normalize
    y_dir = np.cross(z_dir, x_dir)  # Already normalized if x/z are orthogonal
    R_c2w = np.column_stack([x_dir, y_dir, z_dir])  # Columns = camera axes in world coordinates
    T_c2w = np.identity(4)
    T_c2w[0:3,:] = np.hstack([R_c2w,pose[0:3].reshape(-1,1)])
    return T_c2w

def fill_nans_with_nearest(image):
    """
    image: (H, W) numpy array, grayscale image with possible NaNs.
    Fills NaNs with the nearest non-NaN neighbor value.
    """
    nan_mask = np.isnan(image)

    if not np.any(nan_mask):
        return image  # no NaNs, nothing to do

    # Create a binary mask where non-NaNs are 1, NaNs are 0
    valid_mask = nan_mask

    # Use distance transform to find nearest non-NaN for each pixel
    # indices will be (H,W,2) array giving nearest valid pixel for each point
    distance, indices = distance_transform_edt(valid_mask, return_indices=True)

    # Map the NaN locations to their nearest valid neighbor
    filled_image = image.copy()
    filled_image[nan_mask] = image[indices[0][nan_mask], indices[1][nan_mask]]

    return filled_image

def project_points_torch(points_3d, intrinsics):
    """
    Projects 3D points to image-aligned NDC coordinates in [-1, 1]^2.

    Assumes:
        - Camera frame: x-right, y-down, z-forward
        - Image frame:  x-right, y-down

    Args:
        points_3d: (B, T, 3) tensor of 3D points in camera coordinates.
        intrinsics: Dict with 'fx', 'fy', 'cx', 'cy', 'W', 'H'.

    Returns:
        (B, T, 2) tensor of 2D points in normalized device coordinates.
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    W = intrinsics['W']
    H = intrinsics['H']

    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2].clamp(min=1e-6)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    u_ndc = 2 * (u / W) - 1
    v_ndc = 2 * (v / H) - 1

    return torch.stack((u_ndc, v_ndc), dim=-1)