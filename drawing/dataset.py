import torch
import torch.random
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np
from torchvision.transforms import Resize, Normalize
from drawing.projectUtils import addOverlay, calcMeanStd, move_dict_to_device, arcLength

class SketchDataset(Dataset):
    def __init__(self, df, data_folder="drawing/data/", human=False, resizeshape=None, scale=torch.Tensor([1280,720])):
        # Validate input dimensions
        self.data_folder = data_folder
        self.df = df
        if human:
            self.twodpoints = torch.from_numpy(np.stack(self.df["2d_human"])).to(torch.float32)
        else: 
            self.twodpoints = torch.from_numpy(np.stack(self.df["2d_projection"])).to(torch.float32)
        self.threedpoints = torch.from_numpy(np.stack(self.df["3d_gt"])).to(torch.float32)
        self.inds = list(df.index)
        self.a = torch.from_numpy(np.stack(self.df["a"])).to(torch.float32)
        self.b = torch.from_numpy(np.stack(self.df["b"])).to(torch.float32)
        self.c2ws = torch.from_numpy(np.stack(df["c2w"])).to(torch.float32)
        self.scene_scales = torch.from_numpy(np.stack(df["scale"])).to(torch.float32)
        self.envs = self.df["env"].to_list()
        self.scale = scale

        if resizeshape is not None:
            self.resizer = Resize(resizeshape, antialias=True)
        else:
            self.resizer = None

    def __len__(self):
        return len(self.twodpoints)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        dfile = self.df.iloc[idx]["depthfile"]
        depthmap = torch.load(self.data_folder + "depthImages/" + dfile).dequantize().to(torch.float32)
        depthmap = depthmap[None,:,:]
        if self.resizer is not None:
            depthmap = self.resizer(depthmap)

        if self.scale != None:
            twod = 2*(self.twodpoints[idx] / self.scale) - 1
        else: 
            twod = self.twodpoints[idx]
        returndict = {"depthmap": depthmap,
                      "twod": twod,
                      "threed": self.threedpoints[idx],
                      "a": self.a[idx],
                      "b": self.b[idx],
                      "inds": self.inds[idx],
                      "c2ws": self.c2ws[idx],
                      "envs": self.envs[idx],
                      "scene_scales": self.scene_scales[idx]}
        return returndict
    

class SketchPreFetchDataset(Dataset):
    def __init__(self, 
                 df, 
                 data_folder="drawing/data/", 
                 human=False, 
                 resizeshape=None, 
                 overlay=False, 
                 rgb_means=0, 
                 rgb_stds=1, 
                 mode="d", 
                 condnumber=0.8, 
                 device="cpu",
                 postprocessor=None,
                 normParams=None,
                 prediction_type="traj",
                 scale=None):
        # Validate input dimensions
        self.data_folder = data_folder
        self.df = df
        if human:
            self.twodpoints = torch.from_numpy(np.stack(self.df["2d_human"])).to(torch.float32)
            self.projection = torch.from_numpy(np.stack(self.df["2d_projection"])).to(torch.float32)
        else: 
            self.twodpoints = torch.from_numpy(np.stack(self.df["2d_projection"])).to(torch.float32)
            self.projection = self.twodpoints
        self.threedpoints = torch.from_numpy(np.stack(self.df["3d_gt"])).to(torch.float32)
        self.inds = list(df.index)
        #self.a = torch.from_numpy(np.stack(self.df["a"])).to(torch.float32)
        #self.b = torch.from_numpy(np.stack(self.df["b"])).to(torch.float32)
        self.c2ws = torch.from_numpy(np.stack(df["c2w"])).to(torch.float32)
        self.scene_scales = torch.from_numpy(np.stack(df["scale"])).to(torch.float32)
        self.envs = self.df["env"].to_list()
        self.overlay = overlay
        self.rgb_means = rgb_means
        self.rgb_stds = rgb_stds
        self.condnumber = condnumber
        self.device = device
        self.postprocessor = postprocessor
        self.normParams = normParams
        self.projection_type = prediction_type

        self.mode = mode # can be one of "d", "rgb" or "rgbd"

        self.resizeshape = resizeshape
        
        if self.resizeshape is not None:
            self.resizer = Resize(self.resizeshape, antialias=True)
        else:
            self.resizer = None

        all_depthmaps = []

        for idx in range(len(self.df)):  # Loop over all indices of the dataframe
            dfile = self.df.iloc[idx]["depthfile"]  # Get the depthfile for the current index
            depthmap = torch.load(self.data_folder + "depthImages/" + dfile).dequantize().to(torch.float32)  # Load and process the depthmap
            self.scale = torch.tensor(list(depthmap.shape[-2:])[::-1])
            depthmap = depthmap[None, None, :, :]  # Add an extra dimension (for batch)

            if self.resizer is not None:  # Apply resizing if defined
                depthmap = self.resizer(depthmap)    
            all_depthmaps.append(depthmap)  # Append the processed depthmap to the list

        # Stack all depthmaps into a single tensor
        self.depthmaps_tensor = torch.cat(all_depthmaps, dim=0)  # Assuming you want to concatenate along the batch dimension

        
        if self.mode in ["rgb", "rgbd"]:
            all_rgb = []
            for idx in range(len(self.df)):  # Loop over all indices of the dataframe
                dfile = self.df.iloc[idx]["rgb"]  # Get the depthfile for the current index
                rgb = torch.load(self.data_folder + "rgb/" + dfile)  # Load and process the depthmap
                self.scale = torch.tensor(list(depthmap.shape[-2:])[::-1])
                rgb = rgb[None, :, :]  # Add an extra dimension (for batch)
        
                if self.resizer is not None:  # Apply resizing if defined
                    rgb = self.resizer(rgb)    
                all_rgb.append(rgb)  # Append the processed depthmap to the list

            # Stack all depthmaps into a single tensor
            self.rgb_tensor = torch.cat(all_rgb, dim=0).to(torch.float32) / 255.0  # Assuming you want to concatenate along the batch dimension

            self.normalise =  Normalize(self.rgb_means, self.rgb_stds)
            self.rgb_tensor = self.normalise(self.rgb_tensor)

        self.condswitch_tensor = (torch.rand(len(self.twodpoints)) -1)*0.01 + self.condnumber

        if self.scale != None:
            self.scaled_twodpoints = 2*(self.twodpoints / self.scale) - 1
            if human:
                self.projection = 2*(self.projection / self.scale) - 1

        if self.overlay:
            self.depthmaps_tensor = addOverlay(self.depthmaps_tensor, self.scaled_twodpoints)

        if self.projection_type == "sketch":
            if not human:
                raise ValueError("To predict sketches, human must be True")
            tmp = self.scaled_twodpoints.detach().clone()
            self.scaled_twodpoints = torch.cat([self.threedpoints, self.projection],dim=-1)
            self.threedpoints = tmp
            
        if self.mode == "rgbd":
            self.images = torch.cat([self.rgb_tensor, self.depthmaps_tensor], dim=1)
        elif self.mode == "rgb":
            self.images = self.rgb_tensor
        elif self.mode == "d":
            self.images = self.depthmaps_tensor
        else:
            raise ValueError("Mode does not exist!")
        
        if self.normParams:
            self.normalise(*normParams)

        if self.postprocessor:
            self.images = self.postprocessor(self.images)

        if self.device:
            self.images = self.images.to(self.device)
            self.threedpoints = self.threedpoints.to(self.device)
            self.scaled_twodpoints = self.scaled_twodpoints.to(self.device)
            self.condswitch_tensor = self.condswitch_tensor.to(self.device)

        

        #translation = torch.norm(self.threedpoints[:,0] - self.threedpoints[:,-1], dim=-1)
        #arclength = torch.sum(torch.norm(self.threedpoints[:,1:] - self.threedpoints[:,:-1], dim=-1), dim=-1)
        #efficiency = arclength/translation
        self.weights = torch.ones(len(self.twodpoints), dtype=torch.float32, device=self.scaled_twodpoints.device)

        


    def normalise(self, means=None, stds=None, maxval=2.25):
        if (means is not None) and (stds is not None):
            self.images = (self.images - means.to(self.images.device))/stds.to(self.images.device)
            self.images = torch.clamp_max(self.images, maxval)
        else: 
            stds = self.images.std()
            means = self.images.mean()
            self.images = (self.images - means.to(self.images.device))/stds.to(self.images.device)
            self.images = torch.clamp_max(self.images, maxval)
        self.normParams[0], self.normParams[1] = means, stds
        
    def __len__(self):
        return len(self.twodpoints)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        
        if self.scale != None:
            twod = self.scaled_twodpoints[idx]
        else: 
            twod = self.twodpoints[idx]
        
        returndict = {"image": self.images[idx],
                      "twod": twod,
                      "threed": self.threedpoints[idx],
                      #"a": self.a[idx],
                      #"b": self.b[idx],
                      "inds": self.inds[idx],
                      "c2ws": self.c2ws[idx],
                      "envs": self.envs[idx],
                      "scene_scales": self.scene_scales[idx],
                      "condsws": self.condswitch_tensor[idx],
                      "weights": self.weights[idx],
                      "projection": self.projection[idx]}
        
        return returndict
    

    