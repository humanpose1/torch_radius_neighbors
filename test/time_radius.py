# Goal: compare different implementation to see which one is faster


import torch
import matplotlib.pyplot as plt
import numpy as np
import neighbors
import time
from scipy.spatial import cKDTree

if __name__ == "__main__":

    print("measure simply the time of execution radius search and compare with scipy")
    list_time_scipy = []
    list_time_nanoflann = []
    list_size = np.linspace(10000, 200000, 30)

    for i, size in enumerate(list_size):
        radius = 0.1
        a = torch.randn(int(size), 3)

        t0 = time.time()
        res = neighbors.radius_search(a, a, radius)
        list_time_nanoflann.append(time.time()-t0)
        t0 = time.time()
        tree = cKDTree(a.detach().numpy())
        col = tree.query_ball_point(a.detach().numpy(), radius)
        list_time_scipy.append(time.time()-t0)

    plt.plot(list_size, list_time_nanoflann, 'bo', label='with nanoflann')
    plt.plot(list_size, list_time_scipy, 'ro', label='with scipy')
    plt.title("time of execution for simple radius neighbors")
    plt.xlabel("size of the point cloud")
    plt.ylabel("time of execution")
    plt.legend()
    plt.show()
