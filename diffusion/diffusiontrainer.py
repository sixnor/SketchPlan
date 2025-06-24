import torch
from torch.amp import autocast, GradScaler
from torch import nn
from typing import Callable, Optional, Tuple
import copy
import diffusers
from tqdm.auto import tqdm
from drawing.projectUtils import checkTrajCollision, trajBatchAffine, move_dict_to_device
from splatnav.splat.splat_utils import GSplatLoader
from contextlib import nullcontext
import pandas as pd
from wandb.sdk.wandb_run import Run
from ema_pytorch import PostHocEMA, EMA
from copy import deepcopy

class DiffusionTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        noise_scheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        ema_decay: float = 0.75,
        prob_uncond: float = 0.0,
        mixed_precision: bool = True,
        wandb_log: Run | None = None,
        target  = "threed",
        feature = "twod",
        costgrad_fn=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.ema_decay = ema_decay
        self.device = device
        self.prob_uncond = prob_uncond
        self.mixed_precision = mixed_precision
        self.log = wandb_log
        self.feature = feature
        self.target = target
        self.costgrad_fn = costgrad_fn
        
        # Tracking history
        self.train_losses = []

        self.ema_model = diffusers.EMAModel(parameters=self.model.model.parameters(), power=self.ema_decay)
        self.model.inference_model = deepcopy(self.model.model)

        if self.mixed_precision:
            self.scaler = GradScaler()

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.model.train()
        self.model.encoder.train()
        total_loss = 0.0
        total_items = 0.0
        for sampledict in train_loader:
            self.optimizer.zero_grad()
            sampledict = move_dict_to_device(sampledict, self.device)
            image = sampledict["image"]
            twod = sampledict[self.feature]
            trajs = sampledict[self.target]
            condsws = sampledict["condsws"]
            weights = sampledict["weights"]
            # data normalized in dataset
            # device transfer
            trajs = self.model.normaliseTraj(trajs)
            B = len(trajs)
            # Train both conditional and unconditional model by randomly dropping conditioning
            condsws[torch.rand(B) < self.prob_uncond] = 0.0

            # sample noise to add to actions
            noise = torch.randn(trajs.shape, device=self.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.model.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()
            amp_context = autocast(device_type=self.device) if self.mixed_precision else nullcontext()
            with amp_context:
                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.model.noise_scheduler.add_noise(
                    trajs, noise, timesteps)

                # predict the noise residual
                noise_pred = self.model.pred_noise(
                    noisy_actions, timesteps, image, twod, condsws)
                # L2 loss
                loss = torch.mean(weights * self.loss_fn(noise_pred, noise))

            # optimize
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step() 
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            self.lr_scheduler.step()

            # update Exponential Moving Average of the model weights

            self.ema_model.step(self.model.model.parameters())

            total_loss += loss.item()*len(trajs)
            total_items += len(trajs)

        self.shape = [trajs.shape[1], trajs.shape[2]] # Infer shape from outputs rather than having user specify
        avg_loss = total_loss / total_items
        return avg_loss
        

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        eval_loader: torch.utils.data.DataLoader | None = None,
        eval2_loader: torch.utils.data.DataLoader | None = None,
        eval_every: int = 100,
        gsplat: GSplatLoader | None = None,
        robot_radius: float = 0.15
    ):
        with tqdm(range(epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = self._train_epoch(train_loader)
                logdict = {"loss": epoch_loss}
                if eval_loader and (epoch_idx % eval_every == 0):
                    self.ema_model.copy_to(self.model.inference_model.parameters())
                    res = self.evaluate(eval_loader,guidance_scales=[1.0],gsplat=gsplat, robot_radius=robot_radius)
                    logdict["MSE"] = res["MSE"][0]
                    if gsplat:
                        logdict["COL"] = res["COL"][0]
                    if eval2_loader:
                        res = self.evaluate(eval2_loader,guidance_scales=[1.0],gsplat=gsplat, robot_radius=robot_radius)
                        logdict["MSE_TRAIN"] = res["MSE"][0]
                        if gsplat:
                            logdict["COL_TRAIN"] = res["COL"][0]
                if self.log:
                    self.log.log(logdict)
                tglobal.set_postfix(loss=epoch_loss)
                self.train_losses.append(epoch_loss)
        self.model.inference_model = deepcopy(self.model.model)
        self.ema_model.copy_to(self.model.inference_model.parameters())
        if self.log:
            self.log.finish()

    def evaluate(self, dataloader, guidance_scales = [1.0], gsplat=None, robot_radius=None, origdf = None, costgrad_scale=1.0, costgrad_M=4):
        res = {"GUID": [], 
               "MSE" : [],
               "COL" : []}
        if isinstance(origdf, pd.DataFrame):
            cpdf = origdf.copy()
        for guidance_scale in guidance_scales:
            num_trajs = 0.0
            total_sse = 0.0
            total_cols = 0.0
            
            for sampledict in dataloader:
                sampledict = move_dict_to_device(sampledict, self.device)
                
                image = sampledict["image"]
                twod = sampledict[self.feature]
                trajs = sampledict[self.target]
                scene_scales = sampledict["scene_scales"]
                scene_scale = scene_scales[0].detach().clone().cpu().item()
                c2w = sampledict["c2ws"]
                inds = sampledict["inds"]
                num_trajs += len(trajs)
                if self.costgrad_fn: 
                    pred_trajs = self.model.stochastic_forward(image, twod, guidance_scale=guidance_scale, costgrad=self.costgrad_fn, costgrad_scale=costgrad_scale, M=costgrad_M)
                else:
                    pred_trajs = self.model(image, twod,guidance_scale=guidance_scale)
                total_sse += torch.sum(torch.norm(trajs - pred_trajs,dim=-1)).detach().clone().cpu().item()
                
                if gsplat:
                    transformed_trajs = trajBatchAffine(pred_trajs,scene_scales, c2w)
                    total_cols += torch.sum(checkTrajCollision(gsplat,transformed_trajs, robot_radius,scene_scale)).item()
                if isinstance(origdf, pd.DataFrame):
                    for ind, pred_traj in zip(inds,pred_trajs):
                        cpdf.at[ind.item(), "3d_pred"] = pred_traj.detach().clone().cpu().numpy()
                

            res["GUID"].append(guidance_scale)
            res["MSE"].append(total_sse/num_trajs)
            if gsplat:
                res["COL"].append(total_cols/num_trajs)
        if isinstance(origdf, pd.DataFrame):
            return res, cpdf
        else:
            return res

