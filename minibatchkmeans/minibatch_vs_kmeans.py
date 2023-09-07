import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
import clustbench
from itertools import cycle, islice
import time

np.random.seed(14)

centers = [[1, 1], [-0.5, -0.5], [1, -1]]
X1, labels_true = make_blobs(n_samples=30000, centers=centers, cluster_std=0.4)

X2=clustbench.load_dataset('sipu', 's3')
X2=X2.data

start=time.time()
k_means1 = KMeans(init="k-means++", n_clusters=3, n_init=1).fit(X1)
t1=(time.time()-start)

start=time.time()
k_means2 = KMeans(init="k-means++", n_clusters=15, n_init=1).fit(X2)
t2=(time.time()-start)

start=time.time()
mbk1 = MiniBatchKMeans(
    init="k-means++",
    n_clusters=3,
    batch_size=256,
    n_init=1,
    max_no_improvement=10,
    verbose=0,
).fit(X1)
t3=(time.time()-start)

start=time.time()
mbk2 = MiniBatchKMeans(
    init="k-means++",
    n_clusters=15,
    batch_size=100,
    n_init=1,
    max_no_improvement=10,
    verbose=0,
).fit(X2)
t4=(time.time()-start)

k_means_cluster_centers = k_means1.cluster_centers_
order = pairwise_distances_argmin(k_means1.cluster_centers_, mbk1.cluster_centers_)
mbk_means_cluster_centers = mbk1.cluster_centers_[order]

k_means_labels1 = pairwise_distances_argmin(X1, k_means_cluster_centers)
mbk_means_labels1 = pairwise_distances_argmin(X1, mbk_means_cluster_centers)

k_means_cluster_centers = k_means2.cluster_centers_
order = pairwise_distances_argmin(k_means2.cluster_centers_, mbk2.cluster_centers_)
mbk_means_cluster_centers = mbk2.cluster_centers_[order]

k_means_labels2 = pairwise_distances_argmin(X2, k_means_cluster_centers)
mbk_means_labels2 = pairwise_distances_argmin(X2, mbk_means_cluster_centers)


# PLOT RESULTS 

plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.07)
plt.rcParams.update({
    "text.usetex": True
})

plt.subplot(2, 3, 1)
colors = np.array(["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.scatter(X1[:, 0], X1[:, 1], s=10, color=colors[k_means_labels1])
plt.title(r'$k$-means', size=30)
plt.xticks(())
plt.yticks(())
plt.text(0.01, 0.01, ("%.3f s" % t1), transform=plt.gca().transAxes, size=25, horizontalalignment="left")
plt.text(0.99, 0.01, "Inertia: %f" % (k_means1.inertia_), transform=plt.gca().transAxes, size=25, horizontalalignment="right")

plt.subplot(2, 3, 2)
plt.scatter(X1[:, 0], X1[:, 1], s=10, color=colors[mbk_means_labels1])
plt.title(r'Mini batch $k$-means', size=30)
plt.xticks(())
plt.yticks(())
plt.text(0.01, 0.01, ("%.3f s" % t3), transform=plt.gca().transAxes, size=25, horizontalalignment="left")
plt.text(0.99, 0.01, "Inertia: %f" % (mbk1.inertia_), transform=plt.gca().transAxes, size=25, horizontalalignment="right")

different = mbk_means_labels1 == 4
for k in range(3):
    different += (k_means_labels1 == k) != (mbk_means_labels1 == k)
identical = np.logical_not(different)

plt.subplot(2, 3, 3)
plt.scatter(X1[identical, 0], X1[identical, 1], s=10, color="lightgrey")
plt.scatter(X1[different, 0], X1[different, 1], s=10, color="red")
plt.title("Difference", size=30)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 3, 4)
max_pred=15
colors = np.array(list(islice(cycle([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd",
    "#ffbb78", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd",
    "#ffbb78", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd",
    "#ffbb78", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd",
    "#ffbb78", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd"
    ]), max_pred,)))
plt.scatter(X2[:, 0], X2[:, 1], s=10, color=colors[k_means_labels2])
plt.scatter(k_means2.cluster_centers_[:,0], k_means2.cluster_centers_[:,1], marker="+", c="k", s=25)
plt.xticks(())
plt.yticks(())
plt.text(0.01, 0.01, ("%.3f s" % t2), transform=plt.gca().transAxes, size=25, horizontalalignment="left")
plt.text(0.99, 0.01, "Inertia: %f" % (k_means2.inertia_), transform=plt.gca().transAxes, size=25, horizontalalignment="right")

plt.subplot(2, 3, 5)
plt.scatter(X2[:, 0], X2[:, 1], s=10, color=colors[mbk_means_labels2])
plt.scatter(mbk2.cluster_centers_[:,0], mbk2.cluster_centers_[:,1], marker="+", c="k", s=25)
plt.xticks(())
plt.yticks(())
plt.text(0.01, 0.01, ("%.3f s" % t4), transform=plt.gca().transAxes, size=25, horizontalalignment="left")
plt.text(0.99, 0.01, "Inertia: %f" % (mbk2.inertia_), transform=plt.gca().transAxes, size=25, horizontalalignment="right")

different = mbk_means_labels2 == 101
for k in range(15):
    different += (k_means_labels2 == k) != (mbk_means_labels2 == k)
identical = np.logical_not(different)

plt.subplot(2, 3, 6)
plt.scatter(X2[identical, 0], X2[identical, 1], s=10, color="lightgrey")
plt.scatter(X2[different, 0], X2[different, 1], s=10, color="red")
plt.xticks(())
plt.yticks(())

plt.savefig('difference_minibatch_kmeans.pdf')
