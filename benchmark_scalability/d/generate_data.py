import numpy as np
from sklearn.datasets import make_blobs

# Set seed
np.random.seed(0)

sizes=[2, 5, 10, 25, 50, 75, 100, 150, 200, 350, 500, 750, 1000]

for size in sizes:
    data, labels = make_blobs(n_samples=5000, n_features=size, centers=10, cluster_std=1)
    np.savetxt("d_"+str(size)+".csv", data, delimiter=',')