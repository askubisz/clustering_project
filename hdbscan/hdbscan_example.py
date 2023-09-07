import numpy as np
import hdbscan
#from sklearn.cluster import HDBSCAN
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
jain=clustbench.load_dataset('sipu', 'jain')
jain=jain.data

# Fit models
jain_model=hdbscan.HDBSCAN(min_cluster_size=9, min_samples=3, gen_min_span_tree=True).fit(jain)

plt.figure(figsize=(20, 8))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.3, hspace=0.1)
ax1 = plt.subplot(1,2,1)
ax1.set_title("Minimum spanning tree", size=20)
ax2 = plt.subplot(1,2,2)
ax2.set_title("Resulting dendrogram", size=20)

jain_model.minimum_spanning_tree_.plot(axis=ax1, edge_cmap='viridis')
jain_model.single_linkage_tree_.plot(axis=ax2, cmap='viridis', truncate_mode='lastp', p=150, colorbar=True)

plt.savefig("hdbscan_tree_and_hierarchy.pdf")

plt.close()

plt.figure(figsize=(20, 8))
plt.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.1)
ax1 = plt.subplot(1,2,1)
ax1.set_title("Condensed dendrogram", size=20)
ax2 = plt.subplot(1,2,2)
ax2.set_title("Resulting clustering", size=20)

jain_model.condensed_tree_.plot(axis=ax1, select_clusters=True)

colors = np.array(list(islice(cycle(colours_distinct), int(max(jain_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
ax2.scatter(jain[:, 0], jain[:, 1], s=40, color=colors[jain_model.labels_])
ax2.set_xticks(())
ax2.set_yticks(())
plt.savefig("hdbscan_dendrogram_and_clusters.pdf")