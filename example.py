import numpy as np
import matplotlib.pyplot as plt
from cvi import *
from fcm import *
from dataset import make_spiral_clusters

# generate some data with known number of clusters and some noise
m = 2.0
c_true = 7
x, v_true = make_spiral_clusters(c_true, 2000, 500)

results = []

# cluster data for different number of clusters
cs = np.arange(2, 10)
for c in cs:
    v = fcm(x, c)

    # calculate cluster validity indices
    results.append([])
    for method in methods:
        u = fcm_get_u(x, v, m)

        result = method(x, u, v, m)

        results[-1].append(result)

results = np.array(results)

ny = 4
nx = 2

# plot cluster validity indices
for i, method in enumerate(methods):
    plt.subplot(ny, nx, 1 + i)
    column = results[:, i]
    plt.plot(cs, column)

    # find best cluster size for cluster validity index
    if targets[i] == "min":
        c = cs[np.argmin(column)]
    else:
        c = cs[np.argmax(column)]
    
    plt.title("%s, %s is at %d"%(method.__name__, targets[i], c))
    
    plt.plot([c, c], [np.min(column), np.max(column)])

plt.tight_layout()
plt.savefig("cvi.png", dpi=300)

# plot data
plt.clf()
v = fcm(x, c_true)
plt.plot(x[:, 0], x[:, 1], 'bo', markersize=3, markeredgewidth=0.5, markeredgecolor='black')
plt.plot(v[:, 0], v[:, 1], 'ro', markersize=3, markeredgewidth=0.5, markeredgecolor='black')

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
