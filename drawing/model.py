import torch
import torch.nn as nn
from drawing.splinehelper import CubicInterpolationSpline, AkimaInterpolationSpline, BSplineEvaluator, DummyInterpolator
import datetime
from drawing.convModules import BigConv, DepthFireNet, BigConvPointFeatures, DepthEncoderPointFeatures

interpolators = {"cubic": CubicInterpolationSpline, 
               "akima": AkimaInterpolationSpline, 
               "bspline": BSplineEvaluator,
               "raw": DummyInterpolator}

modes = {"d"    : 1,
         "rgb"  : 3,
         "rgbd" : 4}

class ConvNet(nn.Module):
    def __init__(self, dropoutrate=0.0, inpoints=100, outpoints=100, controlpoints=15, interpolation="akima", mode="d", fulltraj=False, A=None, b=None):
        super(ConvNet, self).__init__()
        self.interpolation = interpolation # Can be "cubic", "akima", "bspline" or "raw" 
        
        self.activation = nn.ReLU

        self.dropoutrate = dropoutrate
        self.summation = False

        self.channels = modes[mode]

        self.inpoints = inpoints
        self.controlpoints = controlpoints
        self.outpoints = outpoints
        self.fulltraj = fulltraj

        self.spline = interpolators[self.interpolation](self.controlpoints, self.outpoints)
        if interpolation != "raw":
            outsize = self.controlpoints - 1
        else:
            outsize = self.outpoints
        #self.linsizes = [7*7*64 + 64*self.inpoints + 2*self.inpoints, 1000,500,500,500, 3*outsize] # Before I had out as in, 3000,2000,2000,2000, fin # 512*100
        
        #self.linsizes = [7*7*64 + 64*self.inpoints + 2*self.inpoints, 1000,500,500,500,32,3*outsize] # Before I had out as in, 3000,2000,2000,2000, fin # 512*100
        if self.fulltraj:
            self.linsizes = [7*7*64 + 64*self.inpoints + 2*self.inpoints, 1000,200,3*outsize] # Before I had out as in, 3000,2000,2000,2000, fin # 512*100
        else:
            self.linsizes = [7*7*64 + 64*self.inpoints + 2*self.inpoints, 1000,200,outsize] # Before I had out as in, 3000,2000,2000,2000, fin # 512*100
        self.name = f"model_{interpolation}_in_{self.inpoints}_out_{self.outpoints}_control_{self.controlpoints}_{datetime.datetime.now().strftime('%d%m%y_%H%M')}"
        
        self.convmodule = BigConvPointFeatures(self.channels)
        
    
        self.linmodule = nn.Sequential()

        for i in range(len(self.linsizes) - 2):
            in_size = self.linsizes[i]
            out_size = self.linsizes[i+1]
            self.linmodule.add_module(f"lin{i}", nn.Linear(in_size, out_size))
            self.linmodule.add_module(f"bn{i}", nn.BatchNorm1d(out_size))
            self.linmodule.add_module(f"drop{i}", nn.Dropout(self.dropoutrate))
            self.linmodule.add_module(f"act{i}", self.activation())
        self.linmodule.add_module(f"lin{i+1}", nn.Linear(self.linsizes[-2], self.linsizes[-1]))

        if (A is not None) and (b is not None):
            self.register_buffer("A", A.to(torch.float32))
            self.register_buffer("b", b.to(torch.float32))

    def forward(self, points, image):
        if not self.training and hasattr(self, 'A') and hasattr(self, 'b'):
            points = (points.reshape((-1, 200)) @ self.A + self.b).reshape((-1, 100, 2))

        # Feature extraction
        x = self.convmodule(image, points)
        #x = self.convmodule(image)
        x = torch.cat([x, torch.flatten(points,start_dim=1)],dim=1)
        x = self.linmodule(x)
        if self.interpolation != "raw":
            x = torch.reshape(x, (-1,self.controlpoints-1,3))
            zeros = torch.zeros((x.shape[0],1,3),device=x.device) # add zeros so that the first control point of any trajectory is always zero
            x = torch.cat([zeros, x],dim=1)
            x = self.spline(x)
        elif self.fulltraj:
            x = torch.reshape(x, (-1,self.outpoints,3))
        else:
            #x = torch.reshape(x, (-1,self.outpoints,3))
            pass
        if self.summation:
            x = torch.cumsum(x,dim=1)
        return x
    
    def getintermediate(self, points, image):
        x = self.convmodule(image, points)
        #x = self.convmodule(image)
        x = torch.cat([x, torch.flatten(points,start_dim=1)],dim=1)
        return x
    
    def freezeconv(self):
        for param in self.convmodule.parameters():
            param.requires_grad = False


    def unfreezeconv(self):
        for param in self.convmodule.parameters():
            param.requires_grad = True


    def setDropoutrate(self, rate):
        for idx, m in enumerate(self.named_modules()): 
            component = m[1]
            if isinstance(component, torch.nn.Dropout):
                component.p = rate

# Test the network
if __name__ == "__main__":
    # Create random input tensor (batch_size=1, channels=1, height=720, width=1280)
    x = torch.randn(4, 1, 720, 1280)
    points = torch.randn(4,100,2)
    model = ConvNet()
    output = model((points,x))
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")