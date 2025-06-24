#%%
import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

scene_names = ['stonehenge', 'statues', 'flight', 'old_union']
methods = ['splatplan', 'sfc-1', 'sfc-2', 'sfc-3', 'sfc-4']

fig, ax = plt.subplots(3, 2, figsize=(10, 10), dpi=200)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

ax[0,0].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[0,0].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

ax[1,0].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[1,0].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

ax[1,1].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[1,1].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

ax[0,1].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[0,1].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

ax[2,1].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[2,1].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

ax[2,0].axvspan(0., 1.1, facecolor='gray', alpha=0.7)
ax[2,0].axvspan(2.2, 3.3, facecolor='gray', alpha=0.7)

for l, sparse in enumerate([False, True]):
    for k, scene_name in enumerate(scene_names):

        for j, method in enumerate(methods):

            if sparse:
                read_path = f'trajs/{scene_name}_sparse_{method}_processed.json'
            else:
                read_path = f'trajs/{scene_name}_{method}_processed.json'
            save_fp = os.path.join(str(Path(os.getcwd()).parent.absolute()), read_path)

            with open(save_fp, 'r') as f:
                meta = json.load(f)

            success = []
            safety = []
            polytope_safety = []
            times = []
            polytope_vols = []
            polytope_radii = []
            path_length = []
            eccentricity = []

            num_facets = []

            datas = meta['total_data']

            if method == 'sfc-1':
                col = '#34A853'
                linewidth= 3

            elif method == 'sfc-2':

                col = '#FBBc05'
                linewidth=3

            elif method == 'sfc-3':
                col = '#EA4335'
                linewidth=3

            elif method == 'sfc-4':
                col = '#ff46df'
                linewidth=3
    
            elif method == 'splatplan':
                col = '#4285F4'
                linewidth=3

            print(f'{scene_name}_{method}')

            # Per trajectory
            for i, data in enumerate(datas):

                # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
                if not data['feasible']:
                    success.append(False)
                    continue

                else:
                    success.append(True)

                num_polytopes = data['num_polytopes']

                # record the times
                traj_time = np.array([data['times_astar'], data['times_collision_set'],
                                    data['times_polytope'], data['times_opt']])
                times.append(traj_time)
                
                # record the min safety margin
                safety.append(np.array(data['safety_margin']).min())
                polytope_safety.append(np.array(data['polytope_margin']).min())
                path_length.append(data['path_length'])

                # record the polytope stats (min/max/mean/std)
                polytope_vols_entry = np.array(data['polytope_vols'])
                polytope_radii_entry = np.array(data['polytope_radii'])

                polytope_eccentricity = (4/3 * np.pi * (polytope_radii_entry**3)) / polytope_vols_entry

                polytope_vols.append([polytope_vols_entry.min(), polytope_vols_entry.max(), polytope_vols_entry.mean(), polytope_vols_entry.std()])
                polytope_radii.append([polytope_radii_entry.min(), polytope_radii_entry.max(), polytope_radii_entry.mean(), polytope_radii_entry.std()])
                eccentricity.append([polytope_eccentricity.min(), polytope_eccentricity.max(), polytope_eccentricity.mean(), polytope_eccentricity.std()])

                polytope_length = np.array([len(np.array(poly)) for poly in data['polytopes']]).mean()

                num_facets.append(polytope_length)

            success = np.array(success)
            safety = np.array(safety)
            polytope_safety = np.array(polytope_safety)
            times = np.array(times)
            polytope_vols = np.array(polytope_vols)
            polytope_radii = np.array(polytope_radii)
            eccentricity = np.array(eccentricity)
            path_length = np.array(path_length)

            print(f'{scene_name}_{method}', np.array(num_facets).mean())
            print('success rate', success.sum()/len(success))
            # Computation Time

            # TODO: This plots the individual times of each component of the algorithm
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, qp_solve_time.mean(), bottom = 0, width=0.15, color= adjust_lightness(col, 0.5), linewidth=3, ec='k', label='qp')
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, cbf_solve_time.mean(), bottom=qp_solve_time.mean(), width=0.15, color=adjust_lightness(col, 1.0), linewidth=3, ec='k', label='cbf')
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean(), bottom=cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3, hatch='-', ec='k', label='prune')
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean() + cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3,  ec='k', label='prune')

           # Computation Time
            width = 0.06
            offset = l/10
            linewidth = 2
            try:
                plt00 = ax[1, 0].bar(1.1*k + j/len(methods) + 0.25/2 + l/10, times.sum(axis=1).mean(), width=width, color=col, capsize=10, edgecolor=adjust_lightness(col, 0.5), linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)
            except:
                pass

            # Safety Margin
            # For trajectory points
            width = 0.075
            offset = l/10
            try:
                violinplot = ax[0, 1].violinplot(1e2*safety, positions=[1.1*k + j/len(methods) + 0.25/2 + l/10], widths=width, showmeans=False, showextrema=False, showmedians=False)
                ax[0, 1].scatter(1.1*k + j/len(methods) + 0.25/2 + l/10, 1e2*safety.mean(), s=100, color='k', alpha=0.5, marker='4')

                for pc in violinplot['bodies']:
                    # pc.set_facecolor(col)
                    # pc.set_edgecolor('black')
                    # pc.set_alpha(1)
                    pc.set_color(col)
                    pc.set_alpha(0.8)
            except:
                pass

            # For polytope vertices
            width = 0.075
            offset = l/10
            try:
                violinplot = ax[1, 1].violinplot(1e2*polytope_safety, positions=[1.1*k + j/len(methods) + 0.25/2 + offset], widths=width, showmeans=False, showextrema=False, showmedians=False)
                ax[1, 1].scatter(1.1*k + j/len(methods) + 0.25/2 + offset, 1e2*polytope_safety.mean(), s=100, color='k', alpha=0.5, marker='4')

                for pc in violinplot['bodies']:
                    # pc.set_facecolor(col)
                    # pc.set_edgecolor('black')
                    # pc.set_alpha(1)
                    pc.set_color(col)
                    pc.set_alpha(0.8)
            except:
                pass


            # # Polytope Volume
            width = 0.075
            offset = l/10
            try:
                violinplot = ax[2, 1].violinplot(polytope_vols[:, 2]*1e3, positions=[1.1*k + j/len(methods) + 0.25/2 + offset], widths=width, showmeans=False, showextrema=False, showmedians=False)
                ax[2, 1].scatter(1.1*k + j/len(methods) + 0.25/2 + offset, 1e3*polytope_vols[:, 2].mean(), s=100, color='k', alpha=0.5, marker='4')

                for pc in violinplot['bodies']:
                    # pc.set_facecolor(col)
                    # pc.set_edgecolor('black')
                    # pc.set_alpha(1)
                    pc.set_color(col)
                    pc.set_alpha(0.8)
            except:
                pass

            # # Polytope Radii
            # errors = np.abs(polytope_radii[:, 2].mean().reshape(-1, 1) - np.array([polytope_radii[:, 0].min(), polytope_radii[:, 1].max()]).reshape(-1, 1))

            # ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_radii[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
            # ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_radii[:, 2])), polytope_radii[:, 2], s=250, color=col, alpha=0.04)
            # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_radii[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

            # Eccentricity
            # errors = np.abs(eccentricity[:, 2].mean().reshape(-1, 1) - np.array([eccentricity[:, 0].min(), eccentricity[:, 1].max()]).reshape(-1, 1))
            
            # ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, eccentricity[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
            # ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(eccentricity[:, 2])), eccentricity[:, 2], s=250, color=col, alpha=0.04)
            # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, eccentricity[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

            # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 + l/10, eccentricity[:, 2].mean(), s=200, color='k', alpha=1, marker='-')
            # violinplot = ax[1, 1].violinplot(eccentricity[:, 2], positions=[k + 0.75*j/len(methods) + 0.25/2 + l/10], widths=0.1, showmeans=False, showextrema=False, showmedians=False)

            # for pc in violinplot['bodies']:
            #     # pc.set_facecolor(col)
            #     # pc.set_edgecolor('black')
            #     # pc.set_alpha(1)
            #     pc.set_color(col)
            #     pc.set_alpha(0.8)

            # Path Length
            width = 0.075
            offset = l/10
            try:
                violinplot = ax[2, 0].violinplot(path_length, positions=[1.1*k + j/len(methods) + 0.25/2 + offset], widths=width, showmeans=False, showextrema=False, showmedians=False)
                ax[2, 0].scatter(1.1*k + j/len(methods) + 0.25/2 + offset, path_length.mean(), s=100, color='k', alpha=0.5, marker='4')

                for pc in violinplot['bodies']:
                    # pc.set_facecolor(col)
                    #pc.set_edgecolor('black')
                    # pc.set_alpha(1)
                    pc.set_color(col)
                    pc.set_alpha(0.8)
            except:
                pass

            # Success Rate
            width = 0.06
            offset = l/10
            linewidth = 2
            try:
                plt21 = ax[0, 0].bar(1.1*k + j/len(methods) + 0.25/2 + offset, int((1 - success.sum()/len(success))*100), width=width, color=col, capsize=10, edgecolor=adjust_lightness(col, 0.5), linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)
            except:
                pass

# COMPUTATION TIME
ax[1, 0].set_title(r'Computation Time (s) $\downarrow$' , fontsize=20, fontweight='bold')
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].grid(which='both', axis='y', linewidth=1, color='k', linestyle=':', alpha=0.25, zorder=0)
ax[1, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 0].spines[location].set_linewidth(4)
ax[1, 0].tick_params(which='major', direction="out", length=12, width=2)
ax[1, 0].tick_params(which='minor', direction="out", length=6, width=1.5)
ax[1, 0].set_yscale('log')
ax[1, 0].set_xlim(0., 4.45)

# SAFETY MARGIN
ax[0, 1].set_title(r'Min. Distance (Traj., $10^{-2}$)', fontsize=15, fontweight='bold')
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].axhline(y = 0., color = 'k', linestyle = '-', linewidth=3, alpha=0.7, zorder=0) 
ax[0, 1].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[0, 1].set_axisbelow(True)
ax[0, 1].yaxis.tick_right()
ax[0, 1].yaxis.tick_right()
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 1].spines[location].set_linewidth(4)
ax[0, 1].set_xlim(0., 4.45)

# PATH LENGTH
ax[2, 0].set_title(r'Path Length $\downarrow$', fontsize=20, fontweight='bold')
ax[2, 0].get_xaxis().set_visible(False)
ax[2, 0].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[2, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[2, 0].spines[location].set_linewidth(4)
ax[2, 0].set_xlim(0., 4.45)

# SUCCESS RATE
ax[0, 0].set_title(r'Failure Rate (%) $\downarrow$', fontsize=20, fontweight='bold')
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[0, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 0].spines[location].set_linewidth(4)
ax[0, 0].set_xlim(0., 4.45)

# # POLYTOPE VOLUME
# ax[1, 0].set_title(r'Polytope Volume $\uparrow$', fontsize=25, fontweight='bold')
# ax[1, 0].get_xaxis().set_visible(False)
# ax[1, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
# ax[1, 0].set_axisbelow(True)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[1, 0].spines[location].set_linewidth(4)

# # POLYTOPE RADII
# ax[1, 1].set_title(r'Polytope Radius $\uparrow$', fontsize=25, fontweight='bold')
# ax[1, 1].get_xaxis().set_visible(False)
# ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
# ax[1, 1].set_axisbelow(True)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[1, 1].spines[location].set_linewidth(4)

# POLYTOPE Distances
ax[1, 1].set_title(r'Min. Distance (Vertices, $10^{-2}$)', fontsize=15, fontweight='bold')
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[1, 1].set_axisbelow(True)
ax[1, 1].yaxis.tick_right()
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 1].spines[location].set_linewidth(4)
ax[1, 1].set_xlim(0., 4.45)

# POLYTOPE Volume
ax[2, 1].set_title(r'Polytope Volume $(10^{-3})$ $\uparrow$', fontsize=20, fontweight='bold')
ax[2, 1].get_xaxis().set_visible(False)
ax[2, 1].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[2, 1].set_axisbelow(True)
ax[2, 1].yaxis.tick_right()
for location in ['left', 'right', 'top', 'bottom']:
    ax[2, 1].spines[location].set_linewidth(4)
ax[2, 1].set_xlim(0., 4.45)

plt.tight_layout()
plt.savefig(f'sfc_ablations.pdf', dpi=2000)

#%%