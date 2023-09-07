from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
import time
from itertools import islice, cycle
import numpy as np
import clustbench

np.random.seed(0)

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
jain=clustbench.load_dataset('sipu', 'jain')
jain=jain.data
compound=clustbench.load_dataset('sipu', 'compound')
compound=compound.data
aggregation=clustbench.load_dataset('sipu', 'aggregation')
aggregation=aggregation.data
engytime=clustbench.load_dataset('fcps', 'engytime')
engytime=engytime.data

# FIT MODELS
jain_model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans').fit(jain)
compound_model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=15, assign_labels='kmeans').fit(compound)
aggregation_model = SpectralClustering(n_clusters=7, affinity='nearest_neighbors', n_neighbors=20, assign_labels='kmeans').fit(aggregation)
engytime_model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=8, assign_labels='kmeans').fit(engytime)

# PLOT
plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.07)

plt.subplot(2, 2, 1)
colors = np.array(list(islice(cycle(colours_distinct), int(max(jain_model.labels_) + 1))))
plt.scatter(jain[:, 0], jain[:, 1], s=40, color=colors[jain_model.labels_])
plt.title("jain dataset", size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 2)
colors = np.array(list(islice(cycle(colours_distinct), int(max(compound_model.labels_) + 1))))
plt.scatter(compound[:, 0], compound[:, 1], s=40, color=colors[compound_model.labels_])
plt.title("compound dataset", size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 3)
colors = np.array(list(islice(cycle(colours_distinct), int(max(aggregation_model.labels_) + 1))))
plt.scatter(aggregation[:, 0], aggregation[:, 1], s=40, color=colors[aggregation_model.labels_])
plt.title("aggregation dataset", size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 4)
colors = np.array(list(islice(cycle(colours_distinct), int(max(engytime_model.labels_) + 1))))
plt.scatter(engytime[:, 0], engytime[:, 1], s=40, color=colors[engytime_model.labels_])
plt.title("engytime dataset", size=18)
plt.xticks(())
plt.yticks(())

plt.savefig('spectral.pdf')