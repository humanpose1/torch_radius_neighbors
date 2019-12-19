# Goal: compare different implementation to see which one is faster


import torch
import matplotlib.pyplot as plt
import numpy as np
import neighbors
import time
from scipy.spatial import cKDTree


def compute_batch_radius_with_scipy(x, y, batch_x, batch_y, radius):
    x_ = torch.cat([x, 2 * radius * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y_ = torch.cat([y, 2 * radius * batch_y.view(-1, 1).to(y.dtype)], dim=-1)
    tree = cKDTree(x_.detach().numpy())
    col = tree.query_ball_point(y_.detach().numpy(), radius)
    return col


if __name__ == "__main__":

    print("measure simply the time of execution radius search and compare with scipy")
    list_time_scipy = []
    list_time_nanoflann = []
    list_size = np.linspace(10000, 200000, 30)

    for i, size in enumerate(list_size):
        radius = 0.1
        a = torch.randn(int(size), 3)
        # generate a random batch
        b = torch.randint(0, 16, (int(size),))
        b = b.sort()[0]

        t0 = time.time()
        res = neighbors.batch_radius_search(a, a, b, b, radius, -1, 0)
        list_time_nanoflann.append(time.time()-t0)
        t0 = time.time()

        res = compute_batch_radius_with_scipy(a, a, b, b, radius)
        list_time_scipy.append(time.time()-t0)

    plt.plot(list_size, list_time_nanoflann, 'bo', label='with nanoflann')
    plt.plot(list_size, list_time_scipy, 'ro', label='with scipy')
    plt.xlabel("size of the point cloud")
    plt.ylabel("time of execution")
    plt.legend()
    plt.show()
