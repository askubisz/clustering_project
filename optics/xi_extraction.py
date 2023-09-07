import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_xi
import matplotlib.pyplot as plt
from itertools import cycle, islice

np.random.seed(14)

colours_distinct=[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2",  "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
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
    ]

C1 = np.column_stack((np.random.uniform(-5,5,1000), np.random.uniform(-5,5,1000))) # Noise
C2 = [0, 0.5] + np.random.randn(500, 2)
C3 = [2.5, -3] + 0.6 * np.random.randn(500, 2)
C4 = [-3, 3] + 0.7 * np.random.randn(300, 2)
C5 = [-3, -3] + 0.1 * np.random.randn(50, 2)
X = np.vstack((C1, C2, C3, C4, C5))

# Fit model
cham_model=OPTICS(min_samples=15, xi=0.01, min_cluster_size=0.05, metric='euclidean').fit(X)
cham_model2=OPTICS(min_samples=15, xi=0.04, min_cluster_size=0.05, metric='euclidean').fit(X)

space_cham = np.arange(len(X))
reachability_cham = cham_model.reachability_[cham_model.ordering_]
reachability_cham2 = cham_model2.reachability_[cham_model2.ordering_]
labels = cham_model.labels_[cham_model.ordering_]
labels2 = cham_model2.labels_[cham_model2.ordering_]

plt.rcParams.update({
    "text.usetex": True
})

plt.figure(figsize=(15, 7))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.1)

plt.subplot(2,2,1)
colors = np.array(list(islice(cycle(colours_distinct), len(cham_model.labels_))))
for klass, color in zip(range(0, len(cham_model.labels_)), colors):
    Xk = X[cham_model.labels_ == klass]
    plt.scatter(Xk[:, 0], Xk[:, 1], s=20, color=color)
plt.scatter(X[cham_model.labels_ == -1, 0], X[cham_model.labels_ == -1, 1], s=20, color='black')
plt.title(r'Cluster allocation when $\xi=0.01$', size=18)
plt.xticks(())
plt.yticks(())


plt.subplot(2,2,2)
for klass, color in zip(range(0, len(cham_model.labels_)), colors):
    Xk = space_cham[labels == klass]
    Rk = reachability_cham[labels == klass]
    plt.scatter(Xk, Rk, color=color, s=7)
plt.scatter(space_cham[labels == -1], reachability_cham[labels == -1], s=7, color='black')
plt.title(r'Reachability plot for $\xi=0.01$', size=18)
plt.xlabel("Points ordered by OPTICS")
#plt.xticks(())
#plt.yticks(())

plt.subplot(2,2,3)
colors = np.array(list(islice(cycle(colours_distinct), len(cham_model2.labels_))))
for klass, color in zip(range(0, len(cham_model2.labels_)), colors):
    Xk = X[cham_model2.labels_ == klass]
    plt.scatter(Xk[:, 0], Xk[:, 1], s=20, color=color)
plt.scatter(X[cham_model2.labels_ == -1, 0], X[cham_model2.labels_ == -1, 1], s=20, color='black')
plt.title(r'Cluster allocation when $\xi=0.04$', size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2,2,4)
for klass, color in zip(range(0, len(cham_model2.labels_)), colors):
    Xk = space_cham[labels2 == klass]
    Rk = reachability_cham2[labels2 == klass]
    plt.scatter(Xk, Rk, color=color, s=7)
plt.scatter(space_cham[labels2 == -1], reachability_cham2[labels2 == -1], s=7, color='black')
plt.title(r'Reachability plot for $\xi=0.04$', size=18)
plt.xlabel("Points ordered by OPTICS")
#plt.xticks(())
#plt.yticks(())

plt.tight_layout()
plt.savefig("xi_extraction.pdf")