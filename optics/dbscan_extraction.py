import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt
from itertools import cycle, islice

np.random.seed(14)

plt.rcParams.update({
    "text.usetex": True
})

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
model=OPTICS(min_samples=15, metric='euclidean').fit(X)

space = np.arange(len(X))
reachability = model.reachability_[model.ordering_]

labels_026 = cluster_optics_dbscan(
    reachability=model.reachability_,
    core_distances=model.core_distances_,
    ordering=model.ordering_,
    eps=0.26,
)
labels_055 = cluster_optics_dbscan(
    reachability=model.reachability_,
    core_distances=model.core_distances_,
    ordering=model.ordering_,
    eps=0.55,
)

plt.figure(figsize=(15, 7))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.1)

plt.subplot(1,3,1)
colors = np.array(list(islice(cycle(colours_distinct), len(labels_055))))
colors = np.append(colors, ["#000000"])
plt.scatter(X[:, 0], X[:, 1], s=20, color=colors[labels_055])
plt.title(r'Cluster allocation when $\varepsilon=0.55$', size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(1,3,2)
colors = np.array(list(islice(cycle(colours_distinct), len(labels_026))))
colors = np.append(colors, ["#000000"])
plt.scatter(X[:, 0], X[:, 1], s=20, color=colors[labels_026])
plt.title(r'Cluster allocation when $\varepsilon=0.26$', size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(1,3,3)
plt.plot(space, reachability, 'black')
plt.plot(space, np.full_like(space, 0.55, dtype=float), "k--")
plt.plot(space, np.full_like(space, 0.26, dtype=float), "k-.")
plt.title(r'Reachability plot, $MinPts=15$', size=18)
plt.xlabel("Points ordered by OPTICS")
#plt.xticks(())
#plt.yticks(())

plt.tight_layout()
plt.savefig("dbscan_extraction.pdf")