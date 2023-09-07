import clustbench
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import islice, cycle

np.random.seed(0)

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

wrong_k=pd.read_csv('wrong_k_data.csv', header=None)
non_spherical=clustbench.load_dataset('sipu', 'jain').data
unbalanced=clustbench.load_dataset('sipu', 'unbalance').data
diff_var=pd.read_csv('diff_variance_overlap_data.csv', header=None)

wrong_k_model=KMeans(n_clusters=2, init='random').fit(wrong_k)
non_spherical_model=KMeans(n_clusters=2, init='random').fit(non_spherical)
unbalanced_model=KMeans(n_clusters=8, init='random').fit(unbalanced)
diff_var_model=KMeans(n_clusters=3, init='random').fit(diff_var)

# PLOT
plt.figure(figsize=(15, 8))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.1)
plt.rcParams.update({
    "text.usetex": True
})


plt.subplot(2, 2, 1)
colors = np.array(list(islice(cycle(colours_distinct), int(max(wrong_k_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(wrong_k.iloc[:, 0], wrong_k.iloc[:, 1], s=40, color=colors[wrong_k_model.labels_])
plt.title(r'Wrong $k$', size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 2)
colors = np.array(list(islice(cycle(colours_distinct), int(max(non_spherical_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(non_spherical[:, 0], non_spherical[:, 1], s=40, color=colors[non_spherical_model.labels_])
plt.title("Non-spherical shape", size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 3)
colors = np.array(list(islice(cycle(colours_distinct), int(max(unbalanced_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(unbalanced[:, 0], unbalanced[:, 1], s=40, color=colors[unbalanced_model.labels_])
plt.title("Unbalanced cluster sizes", size=18)
plt.xticks(())
plt.yticks(())

plt.subplot(2, 2, 4)
colors = np.array(list(islice(cycle(colours_distinct), int(max(diff_var_model.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(diff_var.iloc[:, 0], diff_var.iloc[:, 1], s=40, color=colors[diff_var_model.labels_])
plt.title("Different variance in clusters", size=18)
plt.xticks(())
plt.yticks(())

plt.savefig('downfalls_kmeans.pdf')