import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering

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

p1=[0,1]
p2=[0,0.5]
p3=[-1,-1]
p4=[1,1.5]
p5=[-0.5,-0.25]

X = np.vstack((p1,p2,p3,p4,p5))

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)

plt.figure(figsize=(20, 8))

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.05, hspace=0.1)

### PLOT
plt.subplot(1, 2, 1)
plot_dendrogram(model, truncate_mode="level", p=3)
plt.title("Dendrogram", size=22)
# plt.xticks(())
# plt.yticks(())

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], s=40)
plt.title("Dataset", size=22)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.savefig('dendrogram.png')