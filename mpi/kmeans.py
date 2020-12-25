from operator import itemgetter
from mpi4py import MPI
import numpy as np
from numpy.core.fromnumeric import var
from scipy.cluster.hierarchy import centroid
from scipy.cluster.vq import kmeans, whiten
from sklearn import datasets, mixture
import matplotlib.pyplot as plt

# https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

np.random.seed(seed=rank)

# generate data
varied_x, varied_y, center = datasets.make_blobs(
    n_samples=1500,
    cluster_std=[1.0, 2.5, 0.4, 1.5, 2.0],
    centers=5,
    random_state=100,
    return_centers=True,
)

k = 5
n_all = 10
n = n_all // size

centroid, distortion = kmeans(varied_x, k, n)
results = comm.gather((centroid, distortion), root=0)
if rank == 0:
    results.sort(key=itemgetter(1))
    result = results[0]
    print("Best distortion for %d tries: %f" % (n_all, result[1]))
    print("Best centroids: %s" % result[0])
    print("Groud truth: %s" % center)

    fig, ax = plt.subplots()
    colors = np.array(["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628"])
    ax.scatter(varied_x[:, 0], varied_x[:, 1], s=10, color=colors[varied_y.astype(int)])
    ax.plot(center[:, 0], center[:, 1], "*k", markersize=8, label="groud_truth")
    ax.plot(result[0][:, 0], result[0][:, 1], "xr", markersize=8, label="output")
    ax.legend()
    plt.show()
