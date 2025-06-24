import numpy as np
import torch
import scipy as sp
import matplotlib.pyplot as plt
from splatnav.splat.splat_utils import GSplatLoader
from pathlib import Path
import drawing.preprocessing
import plotly.express as px 
from drawing.projectUtils import reproject2Dto3D, checkTrajCollision, loadJSON, pruneDataset, move_dict_to_device, splitdf, calcMeanStd, pointsUndistortRedistort, prune_dataframe
import pandas as pd
import json
from drawing.trainer import Trainer
from drawing.lossFuncs import HalfSpaceLoss, MSELoss, MaximumAxisLoss, ComboLoss, NNLoss
import os
from drawing.dataset import SketchPreFetchDataset

torch.set_float32_matmul_precision("high")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    


### PARAMETERS HERE ###

data_folder = "drawing/data2/"
camera_make = "zed720prect"
dfname = "allflight"

envs = ["berlin", "nyc", "alameda", "london"]

#######################

splatjson = loadJSON(f"{data_folder}splatinfo.json") # Splat, trajectory and scale information
intrinsics = loadJSON(f"{data_folder}intrinsics.json")[camera_make]
drawing.preprocessing.createDatasets([splatjson[env] for env in envs], dfname=dfname, cameraDict=intrinsics, data_folder=data_folder, do_rgb=False)

