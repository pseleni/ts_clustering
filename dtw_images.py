import numpy as np
import matplotlib
from dtaidistance import dtw, ed
from dtaidistance import dtw_visualisation as dtwvis

s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
s2 = np.array([0., 1, 3, 6, 2, 0, 0, 0, 2, 1, 0, 0, 0])
dtw_path = dtw.warping_path(s1, s2)
# dtwvis.plot_warping_single_ax(s1, s2, dtw_path, filename="dtw.pdf")
dtwvis.plot_warping(s1, s2, dtw_path, warping_line_options={
                    'linewidth': 4, 'color': 'pink', 'alpha': 0.8}, filename="dtw.pdf")

distance = ed.distance(s1, s2)
ed_path = [(i, i) for i in range(len(s1))]
# print(distance)
dtwvis.plot_warping(s1, s2, ed_path, warping_line_options={
                    'linewidth': 4, 'color': 'orange', 'alpha': 0.8},filename="ed.pdf")

# paths = np.ones(shape=(14,14))
# dtwvis.plot_warpingpaths(s1, s2, paths, best_path,  filename="edpath.pdf")


d, paths = dtw.warping_paths(s1, s2, psi=2)
dtw_path = dtw.best_path(paths)
fig, ax = dtwvis.plot_warpingpaths(s1, s2, paths, showlegend=True)

py, px = zip(*ed_path)
ax3 = ax[3]
ax3.plot(px, py, ".-", linewidth= 5, color="orange", markersize=15, label = 'Euclidean')

py, px = zip(*dtw_path)
ax3 = ax[3]
ax3.plot(px, py, ".-", linewidth= 5, color="pink", markersize=15, label='DTW')

props = dict(boxstyle='round', facecolor='white', alpha=1)
ax0 = ax[0]
# ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0] + 1, p[-1][1] + 1]))
ax0.text(0, 0.1, rf'\phantom{{here}}                         \n                   ',
                verticalalignment='top',
                transform=ax0.transAxes, bbox=props)

fig.legend()    
fig.show()
fig.savefig("paths.pdf")

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
# import random
# import numpy as np
# x = np.arange(0, 20, .5)
# s1 = np.sin(x)
# s2 = np.sin(x - 1)
# random.seed(1)
# for idx in range(len(s2)):
#     if random.random() < 0.05:
#         s2[idx] += (random.random() - 0.5) / 2
# d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
# best_path = dtw.best_path(paths)
# dtwvis.plot_warpingpaths(s1, s2, paths, best_path)
# print(paths[0])
