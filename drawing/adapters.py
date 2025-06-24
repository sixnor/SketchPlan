import torch
import torch.nn as nn

class SketchLinearTransform(nn.Module):
    def __init__(self, A, b):
        super().__init__()
        # Initialize buffers (non-trainable parameters)
        self.register_buffer('A', A.to(torch.float32))
        self.register_buffer('b', b.to(torch.float32))
        
    def forward(self, sketches):        
        sketches = (sketches.reshape((-1,200)) @ self.A + self.b).reshape((-1,100,2))
        return sketches

class SeqMLP(nn.Module):
    def __init__(self, hidden=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 200)
        )

    def forward(self, x):                 # x: (B, 100, 2)
        x = x.view((-1, 200))
        x = self.net(x)                   # point-wise
        x = x.view((-1,100,2))
        return x                           # residual


class BlankPointsWrapper(nn.Module): # ugly way to remvoe the sketch
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, points, image):
        blanked_points = torch.zeros_like(points)
        return self.module(blanked_points, image)
