import torch
from torch import nn
from torch_kdtree import build_kd_tree

class HalfSpaceLoss(nn.Module):
    def __init__(self, alpha, barrier=torch.nn.functional.elu, margin=0.1):
        super(HalfSpaceLoss, self).__init__()
        self.barrier = barrier
        self.alpha = alpha
        self.MSE = nn.MSELoss(reduction="mean")
        self.margin = margin

    def forward(self, outputs, threed, a, b):
        """
        Computes the custom loss.

        Args:
            predictions (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        mse = self.MSE(outputs, threed)

        #-torch.sum(a * traj, dim=-1) + b # neg means no col, pos means col
        col = self.barrier(-torch.sum(a * outputs, dim=-1) + b + self.margin).mean() # Freakiest function on the books

        loss = mse + self.alpha*col

        return loss    
    
class MaximumAxisLoss(nn.Module):
    def __init__(self, means, scales, margin, barrier=torch.relu):
        super(MaximumAxisLoss, self).__init__()
        self.means = means
        self.scales = scales
        self.margin = margin
        self.maxsplatrad, _ = torch.max(scales,dim=1)
        self.threshold = 0.5 # extra bounds for a padded bbox around the trajectory in metres. Only means within the bbox are considered for collision checking
        self.barrier = barrier

    def forward(self, outputs, datadict):
        c2ws = datadict["c2ws"]
        scene_scales = datadict["scene_scales"]
        aff = c2ws[:,:-1,:-1] * scene_scales.view((-1,1,1))
        trans = c2ws[:,:-1,-1]
        retrajs = torch.baddbmm(trans.view((-1,1,3)),outputs, aff.mT)
        thres = self.threshold*scene_scales[0]
        uvals, _ = torch.max(retrajs, dim=1)
        uvals = uvals + thres
        lvals, _ = torch.min(retrajs, dim=1)
        lvals = lvals - thres
        uvals = uvals.unsqueeze(1)
        lvals = lvals.unsqueeze(1)
        points = self.means.unsqueeze(0)
        inside = torch.all((points >= lvals) & (points <= uvals), dim=2)
        index = torch.argwhere(inside)
        ftrajs = torch.flatten(retrajs, end_dim=-2)
        
        bdim, tdim, _ = retrajs.shape

        index = index.repeat_interleave(tdim,0)
        inc = torch.arange(0,tdim,device=outputs.device).repeat(len(index)//tdim)
        index[:,0] = index[:,0]*tdim + inc

        resdists = torch.norm(ftrajs[index[:,0]] - self.means[index[:,1]],dim=-1) - self.maxsplatrad[index[:,1]] - self.margin
        print("res",resdists.nbytes)
        print("inside",inside.nbytes)
        print("index",index.nbytes)

        collosses = 1e9*torch.ones(bdim*tdim, device=outputs.device,dtype=torch.float32)

        collosses.scatter_reduce_(0,index[:,0], resdists, reduce="amin", include_self=False) 

        total_loss = torch.sum(self.barrier(-collosses)) / (bdim*tdim)

        return total_loss


class NNLoss(nn.Module):
    def __init__(self, means, margin, barrier=torch.relu):
        super(NNLoss, self).__init__()
        self.means = means
        self.margin = margin
        self.barrier = barrier

        self.tree = build_kd_tree(means, device=means.device)

    def forward(self, outputs, datadict):
        c2ws = datadict["c2ws"]
        scene_scales = datadict["scene_scales"]
        aff = c2ws[:,:-1,:-1] * scene_scales.view((-1,1,1))
        trans = c2ws[:,:-1,-1]
        retrajs = torch.baddbmm(trans.view((-1,1,3)),outputs, aff.mT)

        dists, inds = self.tree.query(retrajs.view((-1,3)))

        loss = torch.mean(self.barrier(-(dists - self.margin)))
        return loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.l = torch.nn.MSELoss(reduction="mean")
    
    def forward(self, outputs, datadict):
        return self.l(outputs, datadict["threed"])
    
class ComboLoss(nn.Module):
    def __init__(self, lossFs, weights):
        super(ComboLoss, self).__init__()
        self.lossFs = lossFs
        self.weights = weights
    
    def forward(self, outputs, datadict):
        loss = 0
        for lossF, weight in zip(self.lossFs, self.weights):
            loss += weight*lossF(outputs, datadict)
        return loss
            


