from sklearn.datasets import make_blobs
from numpy import savetxt


X_varied, y_varied = make_blobs(n_samples=1500,
                                centers=3,
                                cluster_std=[2.0, 1, 4.0],
                                random_state=10)

savetxt("diff_variance_overlap_data.csv", X_varied, delimiter=',')
savetxt("diff_variance_overlap_labels.csv", y_varied, delimiter=',')

X_varied, y_varied = make_blobs(n_samples=300,
                                centers=3,
                                cluster_std=1,
                                random_state=1)

savetxt("wrong_k_data.csv", X_varied, delimiter=',')
savetxt("wrong_k_labels.csv", y_varied, delimiter=',')