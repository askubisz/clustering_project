import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, estimate_bandwidth
from denmune import DenMune
from classix import CLASSIX
from sklearn_extra.cluster import KMedoids
from finch import FINCH
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import cycle, islice

# Set seed
np.random.seed(0)

# Function to fix labeling of denmune clustering so it yields indices of labels in standard range starting from -1
def fix_denmune_labels(labels):
    transformed_labels=[]
    seen_labels={}
    counter=0
    for value in labels:
        if value < 0:
            transformed_labels.append(-1)
        elif str(value) in seen_labels:
            transformed_labels.append(seen_labels[str(value)])
        elif str(value) not in seen_labels:
            seen_labels[str(value)]=counter
            transformed_labels.append(seen_labels[str(value)])
            counter+=1
    return transformed_labels

# Define 100 distinct colours for labels
colours_distinct=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', 
                  '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', 
                  '#9edae5', '#8c6d31', '#393b79', '#ad494a', '#3182bd', '#8d8ac9', 
                  '#e6c260', '#4bf2f9', '#e70e18', '#0739e1', '#2e92d9', '#941d9b', 
                  '#41b273', '#e716b7', '#dd73de', '#064fd5', '#873514', '#bb8c6d', 
                  '#2eaf47', '#5655f1', '#d01de6', '#6e47b1', '#903e36', '#a4968d', 
                  '#2230c1', '#3c3fd8', '#4635be', '#08388e', '#f359b0', '#a680d1', 
                  '#01b4f4', '#0ed77b', '#a24016', '#871754', '#7fcfe8', '#ff0868', 
                  '#7ff978', '#e75c0f', '#25a8f6', '#a802d0', '#28c108', '#30af20', 
                  '#786615', '#5aa448', '#470d04', '#4d7ee4', '#39991d', '#395406', 
                  '#18f706', '#27b094', '#831ef5', '#add0e3', '#2e3c84', '#5fc713', 
                  '#d7dafc', '#635c7b', '#5a1fd1', '#396f43', '#b57590', '#26a0c0', 
                  '#1c3356', '#f28373', '#20ee6a', '#3689cc', '#2752d1', '#234f64', 
                  '#8f2564', '#2765bc', '#ff0109', '#481993', '#ec8b01', '#7d83ad', 
                  '#98f4f2', '#8f9af4', '#7af4c4', '#dfe28b', '#d5e2b6', '#a41009', 
                  '#7dd27d', '#27ef48', '#409347', '#fb6124']

# Load data
blobs=pd.read_csv('diff_variance_overlap_data.csv', header=None)
blobs_labels=pd.read_csv('diff_variance_overlap_labels.csv', header=None)
blobs_labels=blobs_labels.iloc[:,0].values.astype('int')

# Standardise data
scaled_blobs=StandardScaler().fit_transform(blobs)

# Define parameters

n_clusters=3
min_samples=10
min_cluster_size=20
eps=0.19
n_neighbors=16
classix_radius=0.13
bandwidth = estimate_bandwidth(scaled_blobs, quantile=0.3)

# Fit models
kmeans=KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(scaled_blobs)
minibatch_kmeans=MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(scaled_blobs)
pam=KMedoids(n_clusters=n_clusters, init='k-medoids++').fit(scaled_blobs)
dbscan=DBSCAN(min_samples=min_samples, eps=eps).fit(scaled_blobs)
optics=OPTICS(min_samples=min_samples, metric='euclidean', cluster_method='dbscan', eps=eps).fit(scaled_blobs)
hdbscan=HDBSCAN(min_cluster_size=min_cluster_size).fit(scaled_blobs)
mean_shift=MeanShift(bandwidth=bandwidth, cluster_all=False).fit(scaled_blobs)
single=AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit(scaled_blobs)
complete=AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit(scaled_blobs)
ward=AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(scaled_blobs)
spectral=SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, assign_labels='kmeans').fit(scaled_blobs)
classix=CLASSIX(sorting='pca', group_merging='density', radius=classix_radius, minPts=min_cluster_size, verbose=0).fit(scaled_blobs)
denmune, _=DenMune(train_data=scaled_blobs, k_nearest=n_neighbors).fit_predict(show_plots=False, show_analyzer=False)
c, num_clust, req_c=FINCH(data=scaled_blobs, req_clust=n_clusters, distance='euclidean', verbose=False)

denmune_labels=fix_denmune_labels(denmune['train'])

# Plot results
plt.figure(figsize=(23, 12))
G = gridspec.GridSpec(4, 8)
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.96, wspace=0.05, hspace=0.15)
plt.rcParams.update({
    "text.usetex": True
})
ax1 = plt.subplot(G[0, 0:2])
ax2 = plt.subplot(G[0, 2:4])
ax3 = plt.subplot(G[0, 4:6])
ax4 = plt.subplot(G[0, 6:8])
ax5 = plt.subplot(G[1, 0:2])
ax6 = plt.subplot(G[1, 2:4])
ax7 = plt.subplot(G[1, 4:6])
ax8 = plt.subplot(G[1, 6:8])
ax9 = plt.subplot(G[2, 0:2])
ax10 = plt.subplot(G[2, 2:4])
ax11 = plt.subplot(G[2, 4:6])
ax12 = plt.subplot(G[2, 6:8])
ax13 = plt.subplot(G[3, 1:3])
ax14 = plt.subplot(G[3, 3:5])
ax15 = plt.subplot(G[3, 5:7])

# kmeans plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(kmeans.labels_)))))
colors = np.append(colors, ["#000000"])
ax1.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[kmeans.labels_], s=20)
ax1.set_title(r'$k$-means', size=20)
ax1.set_xticks(())
ax1.set_yticks(())

# mini batch kmeans plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(minibatch_kmeans.labels_)))))
colors = np.append(colors, ["#000000"])
ax2.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[minibatch_kmeans.labels_], s=20)
ax2.set_title(r'Mini batch $k$-means', size=20)
ax2.set_xticks(())
ax2.set_yticks(())

# pam plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(pam.labels_)))))
colors = np.append(colors, ["#000000"])
ax3.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[pam.labels_], s=20)
ax3.set_title("PAM", size=20)
ax3.set_xticks(())
ax3.set_yticks(())

# dbscan plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(dbscan.labels_)))))
colors = np.append(colors, ["#000000"])
ax4.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[dbscan.labels_], s=20)
ax4.set_title("DBSCAN", size=20)
ax4.set_xticks(())
ax4.set_yticks(())

# optics plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(optics.labels_)))))
colors = np.append(colors, ["#000000"])
ax5.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[optics.labels_], s=20)
ax5.set_title("OPTICS", size=20)
ax5.set_xticks(())
ax5.set_yticks(())

# hdbscan plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(hdbscan.labels_)))))
colors = np.append(colors, ["#000000"])
ax6.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[hdbscan.labels_], s=20)
ax6.set_title("HDBSCAN", size=20)
ax6.set_xticks(())
ax6.set_yticks(())

# mean shift plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(mean_shift.labels_)))))
colors = np.append(colors, ["#000000"])
ax7.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[mean_shift.labels_], s=20)
ax7.set_title("Mean Shift", size=20)
ax7.set_xticks(())
ax7.set_yticks(())

# single linkage plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(single.labels_)))))
colors = np.append(colors, ["#000000"])
ax8.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[single.labels_], s=20)
ax8.set_title("Single linkage", size=20)
ax8.set_xticks(())
ax8.set_yticks(())

# complete linkage plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(complete.labels_)))))
colors = np.append(colors, ["#000000"])
ax9.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[complete.labels_], s=20)
ax9.set_title("Complete linkage", size=20)
ax9.set_xticks(())
ax9.set_yticks(())

# ward linkage plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(ward.labels_)))))
colors = np.append(colors, ["#000000"])
ax10.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[ward.labels_], s=20)
ax10.set_title("Ward linkage", size=20)
ax10.set_xticks(())
ax10.set_yticks(())

# spectral clustering plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(spectral.labels_)))))
colors = np.append(colors, ["#000000"])
ax11.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[spectral.labels_], s=20)
ax11.set_title("Spectral clustering", size=20)
ax11.set_xticks(())
ax11.set_yticks(())

# classix clustering plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(classix.labels_)))))
colors = np.append(colors, ["#000000"])
ax12.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[classix.labels_], s=20)
ax12.set_title("CLASSIX", size=20)
ax12.set_xticks(())
ax12.set_yticks(())

# denmune clustering plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(denmune_labels)))))
colors = np.append(colors, ["#000000"])
ax13.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[denmune_labels], s=20)
ax13.set_title("DenMune", size=20)
ax13.set_xticks(())
ax13.set_yticks(())

# finch clustering plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(req_c)))))
colors = np.append(colors, ["#000000"])
ax14.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[req_c], s=20)
ax14.set_title("FINCH", size=20)
ax14.set_xticks(())
ax14.set_yticks(())

# true labels plot
colors = np.array(list(islice(cycle(colours_distinct), len(np.unique(blobs_labels))+1)))
ax15.scatter(scaled_blobs[:, 0], scaled_blobs[:, 1], color=colors[blobs_labels], s=20)
ax15.set_title("True labels", size=20)
ax15.set_xticks(())
ax15.set_yticks(())


plt.savefig("diff_variance.pdf")