import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import clustbench
from itertools import islice, cycle

from sklearn.cluster import AgglomerativeClustering

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


# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Load data
aggregation=clustbench.load_dataset('sipu', 'aggregation')
aggregation=aggregation.data

# setting distance_threshold=0 ensures we compute the full tree.
single_model = AgglomerativeClustering(distance_threshold=0.2, n_clusters=None, linkage="single").fit(aggregation)
complete_model = AgglomerativeClustering(distance_threshold=2.5, n_clusters=None, linkage="complete").fit(aggregation)
average_model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, linkage="average").fit(aggregation)

space_aggregation = np.arange(len(aggregation))

plt.figure(figsize=(20, 20))

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.3)

### PLOT
plt.subplot(3, 2, 1)
plot_dendrogram(single_model, truncate_mode="level", p=3)
plt.plot(space_aggregation, np.full_like(space_aggregation, 0.2, dtype=float), "k--")
plt.title("Dendrogram for aggreagation (single linkage)", size=22)


plt.subplot(3, 2, 2)
colors = np.array(list(islice(cycle(colours_distinct), int(max(single_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(aggregation[:, 0], aggregation[:, 1], s=40, color=colors[single_model.labels_])
plt.title("Cluster results at 0.2 level (single linkage)", size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 2, 3)
plot_dendrogram(complete_model, truncate_mode="level", p=3)
plt.plot(space_aggregation, np.full_like(space_aggregation, 2.5, dtype=float), "k--")
plt.title("Dendrogram for aggregation (complete linkage)", size=22)


plt.subplot(3, 2, 4)
colors = np.array(list(islice(cycle(colours_distinct), int(max(complete_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(aggregation[:, 0], aggregation[:, 1], s=40, color=colors[complete_model.labels_])
plt.title("Cluster results at 2.5 level (complete linkage)", size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 2, 5)
plot_dendrogram(average_model, truncate_mode="level", p=3)
plt.plot(space_aggregation, np.full_like(space_aggregation, 1, dtype=float), "k--")
plt.title("Dendrogram for aggregation (average linkage)", size=22)


plt.subplot(3, 2, 6)
colors = np.array(list(islice(cycle(colours_distinct), int(max(average_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(aggregation[:, 0], aggregation[:, 1], s=40, color=colors[average_model.labels_])
plt.title("Cluster results at 1 level (average linkage)", size=22)
plt.xticks(())
plt.yticks(())

plt.savefig('all_linkage.pdf')
