import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, estimate_bandwidth
from denmune import DenMune
from sklearn.datasets import make_blobs
from classix import CLASSIX
from sklearn_extra.cluster import KMedoids
from finch import FINCH
import os
import glob
import time
import statistics

np.random.seed(0)

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
sizes=[5,10,25,50,75,100,150,200,250,500,750,1000]
df_times=pd.DataFrame(columns=algo_names, index=sizes)

# Iterate through every file and fit on each model
for size in sizes:
    times={}
    for algo_name in algo_names:
        times[algo_name]=[]
    for i in range(5):
        data, labels=make_blobs(n_samples=5000, n_features=10, centers=size, cluster_std=1)
        # Go through each algorithm
        for algo_name in algo_names:
            print("Fitting "+algo_name+" on data of "+str(size)+" number of clusters "+str(i))
            result_time=aux_fit_model(algo_name, data, size)
            times[algo_name]=times[algo_name]+[result_time]
            print(times)
    for algo_name in algo_names:
        median_time=statistics.median(times[algo_name])
        df_times.at[size, algo_name]=median_time


df_times.to_csv("times_diff.csv")