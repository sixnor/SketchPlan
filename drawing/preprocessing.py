import sys
sys.path.append("..")
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import numpy as np
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import poses
from nerfstudio.data.dataparsers import nerfstudio_dataparser
from pathlib import Path
import torch
from splatnav.splat.splat_utils import GSplatLoader
import json
import cv2
import pandas as pd
import pickle
from scipy.interpolate import splprep, splev, make_interp_spline
from drawing.projectUtils import trajAffine, spoofCameraObj, poseToC2W

class datasetMaker:
    def __init__(self, 
                 splatYML, 
                 transformsJSON, 
                 trajJSON, 
                 data_folder="drawing/data/", 
                 max_trajs=int(10**9), 
                 scene_name=None, 
                 max_len=3, 
                 cameraDict=None, 
                 override_scale=None): # all arguments are string paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        self.splatYML = splatYML
        with open(trajJSON, 'r') as file:
            self.trajsJSON = json.load(file)
        self.transformsJSON = transformsJSON
        self.trajs =  [np.array(run["traj"]) for run in self.trajsJSON["total_data"][:max_trajs]]
        self.data_folder = data_folder

        self.N = len(self.trajs)

        config = nerfstudio_dataparser.NerfstudioDataParserConfig(data=Path(self.transformsJSON), train_split_fraction=1) # No test images
        dataparser = config.setup()
        dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        # We assume that all images (for a given datset) come from the same camera and thus get the instrinsics from the first image.
        self.cameraParams = dataparser_outputs.cameras[0] # Note that the camera to world matrix may change, and thus not should not be used.
        if cameraDict:
            self.cameraParams = spoofCameraObj(cameraDict, self.cameraParams)

        path_to_gsplat = Path(self.splatYML)
        self.gsplat = GSplatLoader(path_to_gsplat, self.device)

        self.m_steps = 100
        self.cut_n_start = 2 # Cut the first something points
        self.max_len = max_len # Maximum arclength for admissable trajectory

        self.n_2d_interp = 100
        self.n_3d_interp = 100

        self.scale = dataparser_outputs.dataparser_scale # Usually inaccurate unless COLMAP data has been scaled.
        if override_scale:
            self.scale = override_scale

        self.scene_name = scene_name

        self.fx, self.fy, self.cx, self.cy = self.cameraParams.fx.item(), self.cameraParams.fy.item(), self.cameraParams.cx.item(), self.cameraParams.cy.item()

        self.cameraMatrix = np.array([[self.fx,0,       self.cx],
                                      [0,      self.fy, self.cy],
                                      [0,      0,       1      ]])
        
        self.df = pd.DataFrame({"env": [self.scene_name]*self.N, 
                                "scale": [self.scale]*self.N,
                                "arclength_gt": [None]*self.N, 
                                "3d_gt": [None]*self.N, 
                                "2d_projection": [None]*self.N, 
                                "2d_human": [None]*self.N, 
                                "arclength_pred": [None]*self.N,
                                "depthfile": [None]*self.N,
                                "3d_pred": [None]*self.N,
                                "c2w": [None]*self.N,
                                "3d_orig": [None]*self.N,
                                "2d_orig": [None]*self.N,
                                "a": [None]*self.N,
                                "b": [None]*self.N}
                                )
        self.dist_params = self.cameraParams.distortion_params.detach().clone().cpu().numpy()[[0,1,4,5]]

    def dumpCameraParamsToJSON(self):
        camera_dict = {"matrix": self.cameraMatrix,
                       "distortion": self.dist_params,
                       "fx": self.fx,
                       "fy": self.fy,
                       "cx": self.cx,
                       "cy": self.cy}
        with open(f"{self.data_folder}cameraParams.json", "w") as file:
            json.dump(camera_dict, file, indent=4, cls=NumpyEncoder)
        return

    def dumpDepthMaps(self, do_rgb=True):
        # Takes the first pose of each trajectory and dumps all corresponding renders of depth images.
        run = 0
        for k, traj in enumerate(self.trajs):
            if not k % 100:
                print(f"TRAJECTORY {k}/{len(self.trajs)}")
            
            pose = traj[0]
            c2w = poseToC2W(pose, convention="nerfstudio")
            camera = self.cameraParams
            camera.camera_to_worlds = torch.Tensor(c2w)[None,:] # Kind of a hacky solution, just switch the matrix in one camera object to create new views
            
            outputs = self.gsplat.splat.pipeline.model.get_outputs_for_camera(camera)
            depths = outputs["depth"].detach().clone().cpu().squeeze()/self.scale
            rgb = (outputs["rgb"].detach().clone().cpu().squeeze() * 255).to(torch.uint8).permute(-1,0,1)

            depth_quint8 = torch.quantize_per_tensor_dynamic(depths, torch.quint8, reduce_range=False)

            torch.save(depth_quint8, f"{self.data_folder}depthImages/run_{self.scene_name}_{run}.pt")
            self.df.at[run, "depthfile"] = f"run_{self.scene_name}_{run}.pt"

            if do_rgb:
                torch.save(rgb, f"{self.data_folder}rgb/run_{self.scene_name}_{run}.pt")
                self.df.at[run, "rgb"] = f"run_{self.scene_name}_{run}.pt"
            
            run += 1
        return
    
    def getPointFeatures(self, write=False):
        inputPoints = []
        outputPoints = []
        run = 0
        for k, traj in enumerate(self.trajs):
            
            fullPath = traj[:,:3].T

            pose = traj[0]
            c2w = poseToC2W(pose, convention="opencv")
            w2c = np.linalg.inv(c2w)
            fullPath= np.vstack([fullPath, np.ones(fullPath.shape[1])]) # add ones for homogeneous transform
            egopath = (w2c @ fullPath)[:-1] # Now in drone coordinate frame

            resampled3d = interpolatePoints(egopath.T/self.scale, max_len=self.max_len, n_interp=self.n_3d_interp).T  # Resampling is done by arclength on spline
            
            rvec = np.zeros(3)
            tvec = np.zeros(3)

            resampled3d2d = interpolatePoints(egopath.T/self.scale, max_len=self.max_len, n_interp=self.n_2d_interp).T  # Resampling is done by arclength on spline
            twodpoints = cv2.projectPoints(resampled3d2d.T[self.cut_n_start:]*self.scale, rvec, tvec, self.cameraMatrix, self.dist_params)[0].squeeze()

            resampled2d = interpolatePoints(twodpoints, n_interp=self.n_2d_interp).T 

            inputPoints.append(resampled2d.T)
            outputPoints.append(resampled3d.T)


            # Call halfspace maker
            # Add halfspaces to df

            #a, b = self.dumpHalfspaces(resampled3d.T, w2c)

            self.df.at[run, "arclength_gt"] = arcLength(resampled3d.T)
            self.df.at[run, "3d_gt"] = resampled3d.T.copy()
            self.df.at[run, "2d_projection"] = resampled2d.T.copy()
            self.df.at[run, "3d_orig"] = traj.copy()
            self.df.at[run, "2d_orig"] = twodpoints.copy()   
            self.df.at[run, "c2w"] = c2w.copy()
            #self.df.at[run, "a"] = a.copy()
            #self.df.at[run, "b"] = b.copy()

            run += 1

        inputPoints = np.stack(inputPoints)
        outputPoints = np.stack(outputPoints)

        if write: torch.save(torch.from_numpy(inputPoints), f"{self.data_folder}twodpoints.pt")
        if write: torch.save(torch.from_numpy(outputPoints), f"{self.data_folder}threedpoints.pt")
        return inputPoints, outputPoints
    
    def dumpHalfspaces(self, traj, w2c):
        # multiply scales by scene_scale
        # transform means into drone centered frame
        scales = self.gsplat.scales.detach().clone()
        scales = scales / self.scale
        w2ct = torch.from_numpy(w2c).to(self.device).to(self.gsplat.scales.dtype)

        means = self.gsplat.means.detach().clone().T
        means = torch.vstack([means, torch.ones(means.shape[1], device=self.device, dtype=self.gsplat.scales.dtype)])
        means = (w2ct @ means).T
        means = means[:,:-1] / self.scale

        maxsplatrad, inds = torch.max(scales,dim=1)

        trajt = torch.from_numpy(traj).to(self.device).to(self.gsplat.scales.dtype)

        # Threshold on cube
        dirs = -means[None,:] + trajt[:,None] # From point on traj to gsplat mean
        dists = torch.norm(dirs, dim=-1)
        mins, mininds = torch.min(dists - maxsplatrad, dim=-1)

        actualdirs = dirs[torch.arange(len(traj)), mininds]
        a = torch.nn.functional.normalize(actualdirs).to(torch.float32)
        margins = maxsplatrad[mininds]

        b = +torch.sum(a * means[mininds],dim=-1) + margins

        #-torch.sum(a * traj, dim=-1) + b > 0 # True means col, false means no col -> neg means no col, pos means col
        a = a.detach().clone().cpu().numpy()
        b = b.detach().clone().cpu().numpy()

        return a, b

class NumpyEncoder(json.JSONEncoder): # From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def createDatasets(listOfDatasetDicts, data_folder="drawing/data/", dfname="drawingdf", cameraDict=None, do_rgb=False):
    twodpoints = []
    threedpoints = []
    dfs = []
    for datasetDict in listOfDatasetDicts:
        print(f"Processing {datasetDict['name']}")
        dataset = datasetMaker(datasetDict["configYML"], 
                               datasetDict["transformsJSON"], 
                               datasetDict["trajsJSON"], 
                               data_folder=data_folder, scene_name=datasetDict['name'], 
                               override_scale=datasetDict["scale"], 
                               cameraDict=cameraDict)
        dataset.dumpDepthMaps(do_rgb=do_rgb)
        twod, threed = dataset.getPointFeatures(write=False)
        dfs.append(dataset.df)
        twodpoints.append(twod)
        threedpoints.append(threed)

    bigdf = pd.concat(dfs)
    bigdf = bigdf.reset_index(drop=True)
    pd.to_pickle(bigdf, f"{data_folder}{dfname}.pkl")

def interpolatePoints(posPoints, n_interp=100, max_len=np.inf, mode="linear"):
        # Interpolates out n equally spaced points on posPoints
        arcLength = np.linalg.norm(posPoints[1:] - posPoints[:-1], axis=1) # Get approximate arc length up to each point
        arcLength = np.insert(arcLength, 0, 0)
        t = np.cumsum(arcLength)
        posPoints = posPoints[t < max_len] # Chop to max len
        t = t[t < max_len]
        t = t/t[-1]
        t,uinds = np.unique(t,return_index=True)
        posPoints = posPoints[uinds]

        evallocs = np.linspace(0, 1, n_interp)

        if mode == "spline":
            spl, u = splprep(posPoints.T, s=1e-5) # takes 2/3xN
            points = np.vstack(splev(evallocs, spl)).T # returns 2/3xN
        elif mode == "linear":
            b = make_interp_spline(t,posPoints,k=3)
            points = b(evallocs)
        else:
            raise NotImplementedError("Interpolation mode does not exist")

        return points

def arcLength(array):
    # array should Nx2 or Nx3
    return np.sum(np.linalg.norm(array[1:] - array[:-1], axis=1))
    