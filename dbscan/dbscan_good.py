import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import clustbench
from itertools import cycle, islice

np.random.seed(14)

# Define colours
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

# Load data
compound=clustbench.load_dataset('sipu', 'compound')
compound=compound.data
aggregation=clustbench.load_dataset('sipu', 'aggregation')
aggregation=aggregation.data
spiral=clustbench.load_dataset('sipu', 'spiral')
spiral=spiral.data

# Fit models
compound1=DBSCAN(eps=0.13, min_samples=5).fit(compound)
aggregation1=DBSCAN(eps=0.13, min_samples=6).fit(aggregation)
spiral1=DBSCAN(eps=0.3, min_samples=5).fit(spiral)

# PLOT
plt.figure(figsize=(15, 6))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.07)
plt.rcParams.update({
    "text.usetex": True
})

### COMPOUND
plt.subplot(1, 3, 1)
colors = np.array(list(islice(cycle(colours_distinct), int(max(compound1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(compound[:, 0], compound[:, 1], s=40, color=colors[compound1.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.13$', size=18)
plt.xticks(())
plt.yticks(())

### AGGREGATION
plt.subplot(1, 3, 2)
colors = np.array(list(islice(cycle(colours_distinct), int(max(aggregation1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(aggregation[:, 0], aggregation[:, 1], s=40, color=colors[aggregation1.labels_])
plt.title(r'$MinPts$: $6$, $\varepsilon$: $0.13$', size=18)
plt.xticks(())
plt.yticks(())

### SPIRAL
plt.subplot(1, 3, 3)
colors = np.array(list(islice(cycle(colours_distinct), int(max(spiral1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(spiral[:, 0], spiral[:, 1], s=40, color=colors[spiral1.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.3$', size=18)
plt.xticks(())
plt.yticks(())


#plt.suptitle("DBSCAN", size=30)
plt.savefig("dbscan_good.pdf")