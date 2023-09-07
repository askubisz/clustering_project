import numpy as np
from sklearn.datasets import make_blobs

# Set seed
np.random.seed(0)

sizes=[5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000]

for size in sizes:
    data, labels = make_blobs(n_samples=5000, n_features=10, centers=size, cluster_std=1)
    np.savetxt("k_"+str(size)+".csv", data, delimiter=',')