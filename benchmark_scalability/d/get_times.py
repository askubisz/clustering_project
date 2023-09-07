import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, estimate_bandwidth
from denmune import DenMune
from classix import CLASSIX
from sklearn_extra.cluster import KMedoids
from finch import FINCH
import os
import glob
import time
import statistics

# Make a dictionary with sorted .csv files in increasing order
cwd_path = os.getcwd()
csv_files = glob.glob(os.path.join(cwd_path, "d_*.csv"))
ordered_paths=[]

for f in csv_files:
    size=int(f.split("_")[-1].split(".")[0])
    ordered_paths.append({"path":f, "size": size})

# Auxiliary function to sort
def sortFn(dict):
    return dict['size']

ordered_paths.sort(key=sortFn)


# Define auxilary function to fit the model
def aux_fit_model(algo_name, data, n_clusters):
    if algo_name=='KM':
        start_time=time.time()
        KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='MBKM':
        start_time=time.time()
        MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1, batch_size=1024).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='PAM':
        start_time=time.time()
        KMedoids(n_clusters=n_clusters, init='k-medoids++').fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='DBSCAN':
        start_time=time.time()
        DBSCAN(min_samples=5, eps=0.5).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='OPTICS':
        start_time=time.time()
        OPTICS(min_samples=5, metric='euclidean', cluster_method='dbscan', eps=0.5).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='HDBSCAN':
        start_time=time.time()
        HDBSCAN(min_samples=5, min_cluster_size=5).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='MS':
        start_time=time.time()
        MeanShift(bandwidth=22, cluster_all=True).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='Single':
        start_time=time.time()
        AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='Comp':
        start_time=time.time()
        AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='Ward':
        start_time=time.time()
        AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='Spectral':
        start_time=time.time()
        SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=8, assign_labels='kmeans').fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='CLASSIX':
        start_time=time.time()
        CLASSIX(sorting='pca', group_merging='density', radius=0.2, minPts=5, verbose=0).fit(data)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='DenMune':
        start_time=time.time()
        DenMune(train_data=data, k_nearest=20).fit_predict(show_plots=False, show_analyzer=False, validate=False)
        end_time=time.time()
        return(end_time-start_time)
    if algo_name=='FINCH':
        start_time=time.time()
        FINCH(data=data, req_clust=n_clusters, distance='euclidean', verbose=False)
        end_time=time.time()
        return(end_time-start_time)

# Make an empty dataframe for measurements
algo_names=['KM', 'MBKM', 'PAM', 'DBSCAN', 'OPTICS', 'HDBSCAN', 'MS', 'Single', 'Comp', 'Ward', 'Spectral', 'CLASSIX', 'DenMune', 'FINCH']
sizes=[item['size'] for item in ordered_paths]
df_times=pd.DataFrame(columns=algo_names, index=sizes)

# Iterate through every file and fit on each model
for item in ordered_paths:
    size=item['size']
    path=item['path']
    data=pd.read_csv(path)
    # Go through each algorithm
    for algo_name in algo_names:
        # # Discard algorithms that run out of memory or take too long
        # Time issues
        if size>=50 and (algo_name=="OPTICS" or algo_name=='MS' or algo_name=='DenMune' or algo_name=='CLASSIX'):
            continue
        times=[]
        print("Fitting "+algo_name+" on data with "+str(size)+" number of dimensions")
        # Repeat each fit 5 times
        for i in range(5):
            result_time=aux_fit_model(algo_name, data, 10)
            times.append(result_time)
            print(times)
            if result_time>10:
                break
        median_time=statistics.median(times)
        df_times.at[size, algo_name]=median_time


df_times.to_csv(os.path.join(cwd_path, "times.csv"))