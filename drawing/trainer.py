import torch
from typing import Callable, Optional, Tuple
import copy
from drawing.projectUtils import move_dict_to_device, checkTrajCollision, trajBatchAffine
import pandas as pd

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_best: bool = True,
        target  = "threed",
        feature = "twod",
        gsplat = None,
        robot_radius = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.save_best = save_best
        self.feature = feature
        self.target = target
        self.gsplat = gsplat
        self.robot_radius = robot_radius
        
        # Tracking history
        self.train_losses = []
        self.val_losses = []

        # Best model tracking
        self.best_model_weights = None
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0.0
        for sampledict in train_loader:
            sampledict = move_dict_to_device(sampledict, self.device)
            image = sampledict["image"]
            twod = sampledict[self.feature]
            trajs = sampledict[self.target]
            
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(twod, image)
            loss = self.loss_fn(outputs, trajs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update tracking
            total_loss += loss.item()*len(outputs)
            total_items += len(outputs)
            
        avg_loss = total_loss / total_items
        return avg_loss

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        guidance_scales: list[float] = [1.0],         # kept for API symmetry
        gsplat=None,
        robot_radius=None,
        origdf: pd.DataFrame | None = None,
):
        self.model.eval()
        """
        Generic evaluation that works with the “(twod, image) -> trajs” model.
        Returns the same metric dict (and optional dataframe) as the first
        implementation.
        """
        res = {"GUID": [], "MSE": [], "COL": []}

        # If we want the predictions copied back into a dataframe, clone it first
        cpdf = origdf.copy() if isinstance(origdf, pd.DataFrame) else None

        for guidance_scale in guidance_scales:        # loop kept, even if unused
            num_trajs   = 0
            total_sse   = 0.0
            total_cols  = 0.0

            for sampledict in dataloader:
                sampledict = move_dict_to_device(sampledict, self.device)

                image        = sampledict["image"]
                twod         = sampledict[self.feature]
                trajs        = sampledict[self.target]
                scene_scales = sampledict["scene_scales"]
                scene_scale  = scene_scales[0].detach().cpu().item()
                c2w          = sampledict["c2ws"]
                inds         = sampledict["inds"]

                num_trajs += len(trajs)

                # ✧ 2-input model:  (twod, image)  — no guidance_scale argument
                pred_trajs = self.model(twod, image)

                # --- metrics ----------------------------------------------------
                total_sse += torch.sum(torch.norm(trajs - pred_trajs, dim=-1)
                                    ).detach().cpu().item()

                if gsplat is not None:
                    transformed = trajBatchAffine(pred_trajs, scene_scales, c2w)
                    total_cols += torch.sum(
                        checkTrajCollision(gsplat, transformed,
                                        robot_radius, scene_scale)
                    ).item()

                # --- optional dataframe update ----------------------------------
                if cpdf is not None:
                    for ind, pred in zip(inds, pred_trajs):
                        cpdf.at[ind.item(), "3d_pred"] = pred.detach().cpu().numpy()

            # store aggregated numbers for this (possibly dummy) guidance_scale
            res["GUID"].append(guidance_scale)
            res["MSE"].append(total_sse / num_trajs)
            if gsplat is not None:
                res["COL"].append(total_cols / num_trajs)

        # mirror original return signature
        return (res, cpdf) if cpdf is not None else res

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
    ):
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                res = self.evaluate(val_loader,gsplat=self.gsplat,robot_radius=self.robot_radius)
                val_loss = res["MSE"][0]
                val_col = res["COL"][0]
                self.val_losses.append(val_loss)

            # Update best model if validation loss improves
                if self.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1



            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4e}" + (f" | Val MSE: {val_loss:.4e}" if val_loader is not None else "") + (f" | Val COL: {val_col:.4e}" if val_loader is not None else "") + (" 🔖" if epoch + 1 == self.best_epoch else ""))

    def restore_best_model(self):
        """Load the weights of the best performing model"""
        if self.best_model_weights is None:
            raise RuntimeError("No best model weights available. Did you run validation during training?")
        self.model.load_state_dict(self.best_model_weights)

        
    def _dumpFeatures(self, val_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        self.model.eval()
        all_outputs = []
        all_gt = []
        total_loss = 0.0
        total_items = 0.0

        with torch.no_grad():
            for sampledict in val_loader:
                sampledict = move_dict_to_device(sampledict, self.device)
                image = sampledict["image"]
                twod = sampledict["twod"]
                outputs = self.model.getintermediate(twod, image)

                all_outputs.append(outputs.detach().clone().cpu())
                all_gt.append(sampledict["threed"].detach().clone().cpu())

        return torch.cat(all_outputs, dim=0), torch.cat(all_gt, dim=0)