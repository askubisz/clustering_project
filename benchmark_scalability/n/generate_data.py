import numpy as np
from sklearn.datasets import make_blobs

# Set seed
np.random.seed(0)

sizes=[100, 500, 1000, 3000, 8000, 16000, 32000, 50000, 75000, 100000, 200000, 350000, 600000, 1000000]

for size in sizes:
    data, labels = make_blobs(n_samples=size, n_features=10, centers=10, cluster_std=1)
    np.savetxt("n_"+str(size)+".csv", data, delimiter=',')