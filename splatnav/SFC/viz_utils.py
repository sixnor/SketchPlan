import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse  
import torch


def plot_ellipsoid(R, S, d, ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}) if S.size == 2 else plt.subplots(subplot_kw={'projection': '3d'})
 
    # plot in 2D
    if S.size()[0] == 2:  
        print('Plotting 2D')
        fix, ax = plt.subplots()
        angle = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)
        ell = Ellipse(xy=d[:2], width=2*S[0], height=2*S[1], angle=angle, edgecolor='r', facecolor='none')
        ax.add_patch(ell)
        ax.set_xlim(d[0] - S[0]*2, d[0] + S[0]*2)
        ax.set_ylim(d[1] - S[1]*2, d[1] + S[1]*2)
        ax.scatter(*d[:2], color='red', s=100, label='d')

    # plot in 3D
    else:
        print('Plotting 3D')
        # init boundaries 
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)

        # scale
        x = S[0] * np.outer(np.cos(u), np.sin(v))
        y = S[1] * np.outer(np.sin(u), np.sin(v))
        z = S[2] * np.outer(np.ones(np.size(u)), np.cos(v))

        # convert to torch
        x = torch.tensor(x, dtype=R.dtype, device=R.device)
        y = torch.tensor(y, dtype=R.dtype, device=R.device)
        z = torch.tensor(z, dtype=R.dtype, device=R.device)
        
        # rotate * translate
        for i in range(len(x)):
            for j in range(len(x[0])):
                xyz = torch.tensor([x[i, j], y[i, j], z[i, j]], dtype=R.dtype, device=R.device)
                xyz = R @ xyz + d
                x[i, j], y[i, j], z[i, j] = xyz[0], xyz[1], xyz[2]

            ax.plot_surface(x, y, z, color='b', edgecolor='none', alpha=0.2)


    plt.show()

