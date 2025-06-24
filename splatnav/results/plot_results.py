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
methods = ['splatplan', 'sfc-2', 'ompl', 'nerfnav']

fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

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

for l, sparse in enumerate([False, True]):
    for k, scene_name in enumerate(scene_names):

        for j, method in enumerate(methods):

            if sparse:
                if method == 'nerfnav':
                    continue
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

            if method == 'sfc-2':
                col = '#34A853'
                linewidth= 3
    
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
                    traj_time = np.array([data['times_astar'], data['times_collision_set'], data['times_ellipsoid'],
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

            elif method == 'splatplan':
                col = '#4285F4'
                linewidth=3

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

            elif method == 'ompl':

                col = '#FBBc05'
                linewidth=3

                # Per trajectory
                for i, data in enumerate(datas):

                    # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
                    if not data['feasible']:
                        success.append(False)
                        continue
                    else:
                        success.append(True)

                    # record the times
                    traj_time = np.array(data['times_rrt'])
                    times.append(traj_time)
                    
                    # record the min safety margin
                    safety.append(np.array(data['safety_margin']).min())
                    path_length.append(data['path_length'])

                success = np.array(success)
                safety = np.array(safety)
                times = np.array(times)
                path_length = np.array(path_length)

                print(f'{scene_name}_{method}')
                print('success rate', success.sum()/len(success))

            elif method == 'nerfnav':
                col = '#EA4335'
                linewidth=3

                # Per trajectory
                for i, data in enumerate(datas):

                    # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
                    if not data['feasible']:
                        success.append(False)
                        continue
                    else:
                        success.append(True)

                    # record the times
                    traj_time = np.array(data['plan_time'])
                    times.append(traj_time)
                    
                    # record the min safety margin
                    safety.append(np.array(data['safety_margin']).min())
                    path_length.append(data['path_length'])

                success = np.array(success)
                safety = np.array(safety)
                times = np.array(times)
                path_length = np.array(path_length)

                print(f'{scene_name}_{method}')
                print('success rate', success.sum()/len(success))

            print(f'{scene_name}_{method}')

            # Computation Time
            if method == 'ompl' or method == 'nerfnav':
                if method == 'nerfnav':
                    # No sparse version for nerfnav
                    width = 0.15
                    offset = 0.04
                else:
                    width = 0.075
                    offset = l/10

                plt00 = ax[1, 0].bar(1.1*k + j/len(methods) + 0.25/2 + offset, times.mean(), width=width, color=col, capsize=10, edgecolor=adjust_lightness(col, 0.5), linewidth=linewidth, 
                            linestyle='-', joinstyle='round', rasterized=True)
                
            else:
                plt00 = ax[1, 0].bar(1.1*k + j/len(methods) + 0.25/2 + l/10, times.sum(axis=1).mean(), width=0.075, color=col, capsize=10, edgecolor=adjust_lightness(col, 0.5), linewidth=linewidth, 
                            linestyle='-', joinstyle='round', rasterized=True)

            # Safety Margin
            # For trajectory points
            if method == 'nerfnav':
                # No sparse version for nerfnav
                width = 0.15
                offset = 0.04
            else:
                width = 0.075
                offset = l/10

            violinplot = ax[0, 1].violinplot(safety, positions=[1.1*k + j/len(methods) + 0.25/2 + offset], widths=width, showmeans=False, showextrema=False, showmedians=False)

            for pc in violinplot['bodies']:
                # pc.set_facecolor(col)
                # pc.set_edgecolor('black')
                # pc.set_alpha(1)
                pc.set_color(col)
                pc.set_alpha(0.8)

            ax[0, 1].scatter(1.1*k + j/len(methods) + 0.25/2 + offset, safety.mean(), s=100, color='k', alpha=0.5, marker='4')

            # Path Length
            if method == 'nerfnav':
                # No sparse version for nerfnav
                width = 0.15
                offset = 0.04
            else:
                width = 0.075
                offset = l/10

            violinplot = ax[1, 1].violinplot(path_length, positions=[1.1*k + j/len(methods) + 0.25/2 + offset], widths=width, showmeans=False, showextrema=False, showmedians=False)

            for pc in violinplot['bodies']:
                # pc.set_facecolor(col)
                #pc.set_edgecolor('black')
                # pc.set_alpha(1)
                pc.set_color(col)
                pc.set_alpha(0.8)

            ax[1, 1].scatter(1.1*k + j/len(methods) + 0.25/2 + offset, path_length.mean(), s=100, color='k', alpha=0.5, marker='4')

            # Success Rate
            if method == 'nerfnav':
                # No sparse version for nerfnav
                width = 0.15
                offset = 0.04
            else:
                width = 0.075
                offset = l/10
            plt21 = ax[0, 0].bar(1.1*k + j/len(methods) + 0.25/2 + offset, int((1 - success.sum()/len(success))*100), width=0.075, color=col, capsize=10, edgecolor=adjust_lightness(col, 0.5), linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)


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
ax[1, 0].set_xlim(0., 4.4)

# SAFETY MARGIN
ax[0, 1].set_title('Min. Distance (Traj)', fontsize=20, fontweight='bold')
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].axhline(y = 0., color = 'k', linestyle = '-', linewidth=3, alpha=0.7, zorder=0) 
ax[0, 1].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[0, 1].set_axisbelow(True)
ax[0, 1].yaxis.tick_right()
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 1].spines[location].set_linewidth(4)
ax[0, 1].set_xlim(0., 4.4)

# PATH LENGTH
ax[1, 1].set_title(r'Path Length $\downarrow$', fontsize=20, fontweight='bold')
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[1, 1].set_axisbelow(True)
ax[1, 1].yaxis.tick_right()
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 1].spines[location].set_linewidth(4)
ax[1, 1].set_xlim(0., 4.4)

# SUCCESS RATE
ax[0, 0].set_title(r'Failure Rate (%) $\downarrow$', fontsize=20, fontweight='bold')
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].grid(axis='y', linewidth=2, color='k', linestyle='-', alpha=0.25, zorder=0)
ax[0, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 0].spines[location].set_linewidth(4)
ax[0, 0].set_xlim(0., 4.4)
plt.tight_layout()
plt.savefig(f'simulation_stats.pdf', dpi=2000)

#%%