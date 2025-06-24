from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers import DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm  
import pandas as pd
from einops import rearrange
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os
import wandb
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline,make_pipeline
from functools import partial

from drawing.encoder import *
from drawing.model import ConvNet
from drawing.convModules import BigConvPointFeatures
from drawing.dataset import SketchPreFetchDataset
from diffusion.unet import ConditionalUnet1D, SequenceChunkDropout
from diffusion.diffusiondataset import DiffusionDataset
from diffusion.diffusiontrainer import DiffusionTrainer
from diffusion.diffusionmodel import DiffusionModel
from drawing.projectUtils import splitdf, loadJSON, calcMeanStd, pruneDataset, move_dict_to_device, prune_dataframe, project_points_torch
from splatnav.splat.splat_utils import GSplatLoader
from drawing.sampler import Sampler
from drawing.adapters import SeqMLP, SketchLinearTransform


torch.set_float32_matmul_precision("high")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    


##### PARAMETERS HERE #####
splatjson = loadJSON("drawing/data/splatinfo.json")
gsplat = GSplatLoader(Path(splatjson["flight"]["configYML"]), device)
intrinsics = loadJSON("drawing/data/intrinsics.json")["zed720prect"]
robot_radius = 0.144

data_folder = "drawing/data/"
cond_switch = 0.8
resizeshape = (224,224) # Set to None to preserve original size
mode = "d" # d or rgb

traindf_name = "train_df.pkl"
testdf_name = "test_df.pkl"

sketchadapterform = "mlp" # Can be "pls", "mlp" or None
###########################


df = pd.read_pickle(f"{data_folder}{traindf_name}")
df = prune_dataframe(df)
train_df, val_df = splitdf(df, [0.9,0.1], seed=0)
train_df_human = train_df[~train_df["2d_human"].isna()]
val_df_human = val_df[~val_df["2d_human"].isna()]

test_df = pd.read_pickle(f"{data_folder}{testdf_name}")
test_df = prune_dataframe(test_df)
test_df_human = test_df[~test_df["2d_human"].isna()]


# Load human dataset for SketchAdapter Training
if mode in ["rgb","rgbd"]: means, stds = calcMeanStd(train_df, resizeshape=resizeshape)
else: means, stds = 0,1

train_dataset = SketchPreFetchDataset(train_df_human, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, device=device, mode=mode, condnumber=cond_switch, normParams=[None, None], human=True)
normParams = train_dataset.normParams
trajMax = train_dataset.threedpoints.amax(dim=(0,1))
trajMin = train_dataset.threedpoints.amin(dim=(0,1))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1500, shuffle=True, num_workers=0)

val_dataset = SketchPreFetchDataset(val_df_human, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, device=device, mode=mode, condnumber=cond_switch, normParams=normParams, human=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=500, shuffle=False, num_workers=0)

test_dataset = SketchPreFetchDataset(test_df_human, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, mode=mode, condnumber=cond_switch, normParams=normParams, human=True)


# human2proj
resshape = np.array([1280,720])
X_train_wide = 2 * np.stack(train_df_human["2d_human"]) / resshape - 1
Y_train_wide = 2*np.stack(train_df_human["2d_projection"]) / resshape -1

X_val_wide = 2*np.stack(val_df_human["2d_human"]) / resshape - 1
Y_val_wide = 2*np.stack(val_df_human["2d_projection"]) / resshape - 1

X_test_wide = 2*np.stack(test_df_human["2d_human"]) / resshape - 1
Y_test_wide = 2*np.stack(test_df_human["2d_projection"]) / resshape - 1


if sketchadapterform == "mlp":
    mlp = SeqMLP(dropout=0.75, hidden=256)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    epochs = 50000
    features = torch.tensor(X_train_wide).to(device).to(torch.float32)
    targets = torch.tensor(Y_train_wide).to(device).to(torch.float32)
    mlp.to(device)
    print("TRAINING MLP SKETCHADAPTER")
    for e in range(epochs):
        mlp.train()
        opt.zero_grad()
        preds = mlp(features)
        l = loss_fn(preds, targets)
        l.backward()
        opt.step()
        print("LOSS:", l.item())
    mlp.eval()
    with torch.no_grad():
        print("MLP SketchAdapter eval loss", torch.mean(torch.norm(mlp(torch.tensor(X_val_wide).to(device).to(torch.float32)) - torch.tensor(Y_val_wide).to(device).to(torch.float32),dim=-1)))
    preprocessor = mlp

elif sketchadapterform == "pls":
    print("FITTING PLS SKETCHADAPTER")
    # Fit PLS
    pls = PLSRegression(n_components=10, scale=True)
    pls.fit(X_train_wide.reshape((-1,200)), Y_train_wide.reshape((-1,200)))

    # Predict
    Y_pred = pls.predict(X_val_wide.reshape((-1,200))).reshape((-1,100,2))
    print("PLS SketchAdapter eval loss", np.mean(np.linalg.norm(Y_pred - Y_val_wide,axis=-1)))

    A = pls.coef_.T
    b = pls.intercept_ - pls._x_mean @ pls.coef_.T

    preprocessor = SketchLinearTransform(torch.tensor(A),torch.tensor(b))
else:
    print("SketchAdapter not utilised")
    preprocessor = None



# Load projection-supervised data


if mode in ["rgb","rgbd"]: means, stds = calcMeanStd(train_df, resizeshape=resizeshape)
else: means, stds = 0,1

train_dataset = SketchPreFetchDataset(train_df, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, device=device, mode=mode, condnumber=cond_switch, normParams=[None, None], human=False)
normParams = train_dataset.normParams
trajMax = train_dataset.threedpoints.amax(dim=(0,1))
trajMin = train_dataset.threedpoints.amin(dim=(0,1))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1500, shuffle=True, num_workers=0)

valf_dataset = SketchPreFetchDataset(train_df.sample(100), resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, device=device, mode=mode, condnumber=cond_switch, normParams=normParams, human=False)
valf_dataloader = torch.utils.data.DataLoader(valf_dataset, batch_size=500, shuffle=False, num_workers=0)

val_dataset = SketchPreFetchDataset(val_df_human, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, device=device, mode=mode, condnumber=cond_switch, normParams=normParams, human=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=500, shuffle=False, num_workers=0)

test_dataset = SketchPreFetchDataset(test_df_human, resizeshape=resizeshape, rgb_means=means, rgb_stds=stds, mode=mode, condnumber=cond_switch, normParams=normParams, human=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=5)

point_dim = 3
embed_dim = 200+384#140

# create network object
backbonemodel = ConditionalUnet1D(
    input_dim = point_dim,
    global_cond_dim = embed_dim + 1,
    dropout=0.0,
    diffusion_step_embed_dim=64,
    down_dims=[64, 128, 256],
    kernel_size=3,                
    n_groups=16, 
)

class InputDropWrapper(nn.Module):
    def __init__(self, drop_module, other_module):
        super().__init__()
        self.drop_module = drop_module
        self.other_module = other_module

    def forward(self, x, y):
        x_dropped = self.drop_module(x)
        x = self.other_module(x_dropped, y)
        return x

encoder = ConvNet(inpoints=100, outpoints=embed_dim, interpolation="raw", dropoutrate=0.4, mode=mode)

# for this demo, we use DDPMScheduler with 200 diffusion iterations # For inference  DPMSolverMultistepScheduler with 15-30 steps and lambda_min_clipped = -1.0 works well
num_diffusion_iters = 200
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

model = DiffusionModel(backbonemodel, encoder, noise_scheduler, trajmax=trajMax, trajmin=trajMin, sketchprocessor=preprocessor)
_ = model.to(device)

# Bulk training

num_epochs = 2000 # typically 2000 is enough
prob_uncond = 0.2
lr = 1e-4

msel = torch.nn.MSELoss(reduction="none")
lossfn = lambda x,y: msel(x,y).mean(dim=tuple(range(1, x.ndim)))

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=lr, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs
)

trainer = DiffusionTrainer(model,optimizer, 
                           lossfn, 
                           noise_scheduler=noise_scheduler, 
                           lr_scheduler=lr_scheduler, 
                           prob_uncond=prob_uncond)

trainer.train(train_dataloader,epochs=num_epochs, eval_loader=test_dataloader,eval2_loader=valf_dataloader,eval_every=20,gsplat=gsplat, robot_radius=robot_radius)


torch.save(trainer.model, f"{data_folder}model.pt")

res = trainer.evaluate(test_dataloader,gsplat=gsplat,robot_radius=robot_radius)
print("RESULTS")
print("PD:", res["MSE"], "COLLISION RATE:", res["COL"])

