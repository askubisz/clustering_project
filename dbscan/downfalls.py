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
jain=clustbench.load_dataset('sipu', 'jain')
jain=jain.data
wingnut=clustbench.load_dataset('fcps', 'wingnut')
wingnut=wingnut.data
diamonds=clustbench.load_dataset('fcps', 'twodiamonds')
diamonds=diamonds.data

# Fit models
jain1=DBSCAN(eps=0.18, min_samples=5).fit(jain)
jain2=DBSCAN(eps=0.23, min_samples=5).fit(jain)
jain3=DBSCAN(eps=0.25, min_samples=5).fit(jain)

wingnut1=DBSCAN(eps=0.15, min_samples=5).fit(wingnut)
wingnut2=DBSCAN(eps=0.22, min_samples=5).fit(wingnut)
wingnut3=DBSCAN(eps=0.25, min_samples=5).fit(wingnut)

diamonds1=DBSCAN(eps=0.1, min_samples=5).fit(diamonds)
diamonds2=DBSCAN(eps=0.1, min_samples=6).fit(diamonds)
diamonds3=DBSCAN(eps=0.1, min_samples=7).fit(diamonds)

#### PLOT K-NEIGHBOURS PLOT
plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.33, hspace=0.01)


plt.subplot(1,3,1)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(jain)
distances, indices = neighbors_fit.kneighbors(jain)
distances = np.sort(distances, axis=0)[::-1]
distances = distances[:,4]
plt.title("jain")
plt.ylabel("5-NN distance")
plt.xlabel("Points sorted by the distance")
plt.plot(distances)

plt.subplot(1,3,2)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(wingnut)
distances, indices = neighbors_fit.kneighbors(wingnut)
distances = np.sort(distances, axis=0)[::-1]
distances = distances[:,4]
plt.title("wingnut")
plt.ylabel("6-NN distance")
plt.xlabel("Points sorted by the distance")
plt.plot(distances)

plt.subplot(1,3,3)
neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(diamonds)
distances, indices = neighbors_fit.kneighbors(diamonds)
distances = np.sort(distances, axis=0)[::-1]
distances = distances[:,5]
plt.title("two diamonds")
plt.ylabel("6-NN distance")
plt.xlabel("Points sorted by the distance")
plt.plot(distances)

plt.savefig("kneighbours.pdf")


##### PLOT CLUSTERS 
plt.figure(figsize=(9 * 2 + 3, 13))
plt.rcParams.update({
    "text.usetex": True
})
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.1)

### JAIN
plt.subplot(3, 3, 1)
colors = np.array(list(islice(cycle(colours_distinct), int(max(jain1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(jain[:, 0], jain[:, 1], s=40, color=colors[jain1.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.18$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 2)
colors = np.array(list(islice(cycle(colours_distinct), int(max(jain2.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(jain[:, 0], jain[:, 1], s=40, color=colors[jain2.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.23$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 3)
colors = np.array(list(islice(cycle(colours_distinct), int(max(jain3.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(jain[:, 0], jain[:, 1], s=40, color=colors[jain3.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.25$', size=22)
plt.xticks(())
plt.yticks(())

### WINGNUT

plt.subplot(3, 3, 4)
colors = np.array(list(islice(cycle(colours_distinct), int(max(wingnut1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(wingnut[:, 0], wingnut[:, 1], s=40, color=colors[wingnut1.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.15$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 5)
colors = np.array(list(islice(cycle(colours_distinct), int(max(wingnut2.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(wingnut[:, 0], wingnut[:, 1], s=40, color=colors[wingnut2.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.22$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 6)
colors = np.array(list(islice(cycle(colours_distinct), int(max(wingnut3.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(wingnut[:, 0], wingnut[:, 1], s=40, color=colors[wingnut3.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.25$', size=22)
plt.xticks(())
plt.yticks(())

### DIAMONDS

plt.subplot(3, 3, 7)
colors = np.array(list(islice(cycle(colours_distinct), int(max(diamonds1.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(diamonds[:, 0], diamonds[:, 1], s=40, color=colors[diamonds1.labels_])
plt.title(r'$MinPts$: $5$, $\varepsilon$: $0.1$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 8)
colors = np.array(list(islice(cycle(colours_distinct), int(max(diamonds2.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(diamonds[:, 0], diamonds[:, 1], s=40, color=colors[diamonds2.labels_])
plt.title(r'$MinPts$: $6$, $\varepsilon$: $0.1$', size=22)
plt.xticks(())
plt.yticks(())

plt.subplot(3, 3, 9)
colors = np.array(list(islice(cycle(colours_distinct), int(max(diamonds3.labels_) + 1))))
colors = np.append(colors, ["#000000"])
plt.scatter(diamonds[:, 0], diamonds[:, 1], s=40, color=colors[diamonds3.labels_])
plt.title(r'$MinPts$: $7$, $\varepsilon$: $0.1$', size=22)
plt.xticks(())
plt.yticks(())

#plt.suptitle("DBSCAN", size=30)
plt.savefig("dbscan_downfalls.pdf")