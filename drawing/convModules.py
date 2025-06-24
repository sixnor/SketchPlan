import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze, expand, activation=nn.ReLU):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, 1),
            nn.BatchNorm2d(squeeze),
            activation(inplace=True)
        )
        self.expand_1x1 = nn.Conv2d(squeeze, expand, 1)
        self.expand_3x3 = nn.Conv2d(squeeze, expand, 3, padding=1)
        
    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], dim=1)

class DepthFireNet(nn.Module):
    def __init__(self, embed_dim=256, activation=nn.ReLU):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (1, H, W) [depth image]
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Halve resolution
            nn.BatchNorm2d(64),
            activation(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 1/4 resolution
            
            FireModule(64, 16, 64, activation=activation),   # Out: 64 + 64 = 128
            FireModule(128, 16, 64, activation=activation),  # Out: 64 + 64 = 128
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 1/8
            
            FireModule(128, 32, 128, activation=activation),  # Out: 128 + 128 = 256
            FireModule(256, 32, 128, activation=activation),  # Out: 128 + 128 = 256
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 1/16
            
            FireModule(256, 48, 192, activation=activation),  # Out: 192 + 192 = 384
            FireModule(384, 48, 192, activation=activation),  # Out: 192 + 192 = 384
        )
        
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(384, embed_dim)       # Project to embedding
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        return self.embed(x)
    

class BigConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.convmodule = nn.Sequential()
        # Input: (1, 720, 1280)

        #self.channels = [1,16,32,64,128,256,512]
        #self.kernelsizes = [3,3,3,3,3,3]
        #self.strides = [2,2,2,2,2,2]
        #self.paddings = [1,1,1,1,1,1]

        self.channels = [1,16,32,64,64,64,64]
        self.kernelsizes = [3,3,3,3,3,3]
        self.strides = [2,2,2,2,1,1]
        self.paddings = [1,1,1,1,1,1]

        self.activation = nn.ReLU

        for i in range(len(self.channels) - 1):
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            k = self.kernelsizes[i]
            s = self.strides[i]
            p = self.paddings[i]
            self.convmodule.add_module(f"conv{i}", nn.Conv2d(in_ch, out_ch, k,s,p))
            self.convmodule.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.convmodule.add_module(f"act{i}", self.activation())

        # Adaptive pooling to get fixed spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((100, 1))
    def forward(self, image):
        x = self.convmodule(image)
        # Adaptive pooling
        #x = self.adaptive_pool(x)
        x = torch.flatten(x,start_dim=1)
        return x
    

class BigConvPointFeatures(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.beforepoint = nn.Sequential()
        self.afterpoint = nn.Sequential()
        # Input: (1, 720, 1280)

        #self.channels = [1,16,32,64,128,256,512]
        #self.kernelsizes = [3,3,3,3,3,3]
        #self.strides = [2,2,2,2,2,2]
        #self.paddings = [1,1,1,1,1,1]

        self.bchannels = [in_channels,16,32,64,64,64,64]
        self.bkernelsizes = [3,3,3,3,3,3]
        self.bstrides = [2,2,2,2,1,1]
        self.bpaddings = [1,1,1,1,1,1]

        self.activation = nn.ReLU

        for i in range(len(self.bchannels) - 1):
            in_ch = self.bchannels[i]
            out_ch = self.bchannels[i+1]
            k = self.bkernelsizes[i]
            s = self.bstrides[i]
            p = self.bpaddings[i]
            self.beforepoint.add_module(f"conv{i}", nn.Conv2d(in_ch, out_ch, k,s,p))
            self.beforepoint.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.beforepoint.add_module(f"act{i}", self.activation())


        self.achannels = [64,64]
        self.akernelsizes = [3]
        self.astrides = [2]
        self.apaddings = [1]

        for i in range(len(self.achannels) - 1):
            in_ch = self.achannels[i]
            out_ch = self.achannels[i+1]
            k = self.akernelsizes[i]
            s = self.astrides[i]
            p = self.apaddings[i]
            self.afterpoint.add_module(f"conv{i}", nn.Conv2d(in_ch, out_ch, k,s,p))
            self.afterpoint.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.afterpoint.add_module(f"act{i}", self.activation())

    def forward(self, image, points):
        bx = self.beforepoint(image)

        pointfeatures = torch.nn.functional.grid_sample(bx, points[:,:,None,:], align_corners=False)
        ax = self.afterpoint(bx)
        # Adaptive pooling
        #x = self.adaptive_pool(x)
        ax = torch.flatten(ax,start_dim=1)
        pointfeatures = torch.flatten(pointfeatures, start_dim=1)
        x = torch.cat([ax,pointfeatures], dim=1)
        return x
    
    def forwardextra(self, image, points):
        bx = self.beforepoint(image)

        pointfeatures = torch.nn.functional.grid_sample(bx, points[:,:,None,:], align_corners=False)
        ax = self.afterpoint(bx)
        # Adaptive pooling
        #x = self.adaptive_pool(x)
        ax = torch.flatten(ax,start_dim=1)
        pointfeatures = torch.flatten(pointfeatures, start_dim=1)
        x = torch.cat([ax,pointfeatures], dim=1)
        return x,bx,pointfeatures,ax

class DepthEncoderPointFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.blayers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 1/2 res
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 1/4 res
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # 1/8 res
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1), # 1/16 res
            nn.ReLU()
        )

        self.alayers = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # 1/2 res
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),  # 1/2 res
            nn.ReLU(),
        )
    def forward(self, image, points):
        bx = self.blayers(image)
        pointfeatures = torch.nn.functional.grid_sample(bx, points[:,:,None,:], align_corners=False)
        ax = self.alayers(bx)
        # Adaptive pooling
        #x = self.adaptive_pool(x)
        ax = torch.flatten(ax,start_dim=1)
        pointfeatures = torch.flatten(pointfeatures, start_dim=1)
        x = torch.cat([ax,pointfeatures], dim=1)
        return x