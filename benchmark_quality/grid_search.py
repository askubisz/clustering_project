import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, estimate_bandwidth
from denmune import DenMune
from classix import CLASSIX
from sklearn_extra.cluster import KMedoids
from finch import FINCH
from sklearn.preprocessing import StandardScaler
import clustbench
from sklearn import metrics
from tqdm import tqdm

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

def grid_search(data, labels, parameters, name, n_clusters=None, denmune=False):
    if denmune==True:
        best_ar=float('-inf')
        for param in parameters:
            preds, _=DenMune(train_data=data, k_nearest=param).fit_predict(show_plots=False, show_analyzer=False)
            preds=fix_denmune_labels(preds['train'])
            ar=metrics.adjusted_rand_score(labels, preds)
            if ar>best_ar:
                best_ar=ar
                best_parameters=param
        print("denmune: Best parameters for "+name+" dataset: "+str(best_parameters)+", achievied score "+str(best_ar))
    else:
        best_ar=float('-inf')
        for param in parameters:
            model=HDBSCAN(min_samples=param[0], min_cluster_size=param[1]).fit(data)
            ar=metrics.adjusted_rand_score(labels, model.labels_)
            if ar>best_ar:
                best_ar=ar
                best_parameters=param
                if best_ar==1:
                    break
        print("Best parameters for "+name+" dataset: "+str(best_parameters)+", achievied score "+str(best_ar))


fcps_datasets=['atom', 'chainlink', 'engytime', 'hepta', 'lsun', 'target', 'tetra', 'twodiamonds', 'wingnut']

sipu_datasets=['a1', 'a2', 'a3', 'aggregation', 'compound', 'd31', 'r15', 'flame', 'jain', 'pathbased', 'spiral', 's1', 's2', 's3', 's4', 'unbalance']

graves_datasets=['dense', 'fuzzyx', 'line', 'parabolic', 'ring_noisy', 'ring_outliers', 'ring', 'zigzag_noisy', 'zigzag_outliers', 'zigzag']

uci_datasets=['ecoli', 'glass', 'ionosphere', 'sonar', 'statlog', 'wdbc', 'wine', 'yeast']

# DBSCAN parameters
# eps=np.linspace(0.01, 0.5, 50)
# minpts=[5]
# parameters = []
# for i in eps:
#     for j in minpts:
#         parameters.append((round(i, 4), j))

# HDBSCAN parameters
min_cluster_size=range(3,30)
min_samples=range(3,30)
parameters = []
for i in min_samples:
    for j in min_cluster_size:
        parameters.append((round(i, 4), j))

# Meanshift parameters
# quantile=np.linspace(0.1, 0.5, 20)
# parameters=list(quantile)

# Spectral parameters
# n_neighbours=range(2,35)
# parameters=list(n_neighbours)

# CLASSIX parameters
# radius=np.linspace(0.01, 0.5, 50)
# minpts=range(3,25)
# parameters = []
# for i in radius:
#     for j in minpts:
#         parameters.append((round(i, 4), j))

# DenMune parameters
# n_neighbours=range(3,40)
# parameters=list(n_neighbours)


for name in sipu_datasets:
    data=clustbench.load_dataset('sipu', name)
    scaled_data=StandardScaler().fit_transform(data.data)
    grid_search(scaled_data, data.labels[0], parameters, name, data.n_clusters[0], False)