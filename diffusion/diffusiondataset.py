import torch
import numpy as np
import pandas as pd

class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, df, pca_components=20,human=False, normparams=None, cond_switch=1): # For normparams, provide ((emin, emax), (tmin,tmax))
        self.normparams = normparams
        self.cond_switch = cond_switch
        if human is True:
            tdf = df[df["2d_human"].notna()]
            pca = torch.from_numpy(np.stack(tdf["pca_human"])).to(torch.float32)
        else:
            tdf = df
            pca = torch.from_numpy(np.stack(df["pca_projection"])).to(torch.float32)
            
        img_embed = torch.from_numpy(np.stack(tdf["img_embed"])).to(torch.float32)
        self.embeds = torch.concatenate([pca[:,:pca_components],img_embed],dim=1) # Sketch then img
        self.trajs = torch.from_numpy(np.stack(tdf["3d_gt"])).to(torch.float32)
        self.inds = list(tdf.index)
        self.c2w = torch.from_numpy(np.stack(tdf["c2w"])).to(torch.float32)
        self.scene_scale = torch.from_numpy(np.stack(tdf["scale"])).to(torch.float32)
        if self.cond_switch:
            self.embeds = torch.nn.functional.pad(self.embeds, (0,1), mode="constant", value=self.cond_switch) # Conditioning switch
            self.embeds[:,-1] = (torch.rand(len(self.embeds)) - 1.0)*0.01 + cond_switch
        if self.normparams is None:
            emin, emax = self.embeds.amin(dim=0),self.embeds.amax(dim=0)
            tmin, tmax = self.trajs.amin(dim=(0,1)), self.trajs.amax(dim=(0,1))
            self.normparams = ((emin,emax),(tmin,tmax))
        self.embeds, self.trajs = self.normalise((self.embeds,self.trajs), self.normparams)
        self.image_embed_dim = img_embed.shape[1]
        self.sketch_embed_dim = pca_components
        self.embed_dim = self.embeds.shape[1]
        self.seq_len = self.trajs.shape[1]
        self.point_dim = self.trajs.shape[2]
    def normalise(self, tensors, params): # Normalise values to [-1,1]
        scaled_tensors = []
        for tensor, (tmin,tmax) in zip(tensors, params):
            scaled_tensors.append(2*(tensor - tmin)/(tmax - tmin) - 1)
        return scaled_tensors
    def unnormalise(self, tensors, params):  # Convert values back to original range
        original_tensors = []
        for tensor, (tmin, tmax) in zip(tensors, params):
            original_tensors.append((tensor + 1) * (tmax - tmin) / 2 + tmin)
        return original_tensors
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, idx):
        traj = self.trajs[idx]
        embed = self.embeds[idx]
        return traj, embed