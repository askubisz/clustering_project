import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, estimate_bandwidth
from denmune import DenMune
from classix import CLASSIX
from hdbscan import validity_index
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

datasets = [('a1', {"n_clusters": 20, "min_samples": 13, "min_cluster_size": 18, "min_samples_hdbscan": 3, "eps": 0.1, "n_neighbors": 15, "denmune_neigh": 34, "classix_minpts": 17, "classix_radius": 0.05, "quantile": 0.036}),
            ('a2', {"n_clusters": 35, "min_samples": 10, "min_cluster_size": 27, "min_samples_hdbscan": 3, "eps": 0.07, "n_neighbors": 13, "denmune_neigh": 51, "classix_minpts": 27, "classix_radius": 0.03, "quantile": 0.018}),
            ('a3', {"n_clusters": 50, "min_samples": 13, "min_cluster_size": 19, "min_samples_hdbscan": 4, "eps": 0.07, "n_neighbors": 19, "denmune_neigh": 57, "classix_minpts": 29, "classix_radius": 0.02, "quantile": 0.014}),
            ('aggregation', {"n_clusters": 7, "min_samples": 13, "min_cluster_size": 3, "min_samples_hdbscan": 10, "eps": 0.22, "n_neighbors": 18, "denmune_neigh": 8, "classix_minpts": 6, "classix_radius": 0.09, "quantile": 0.122}),
            ('compound', {"n_clusters": 6, "min_samples": 4, "min_cluster_size": 3, "min_samples_hdbscan": 4, "eps": 0.2, "n_neighbors": 19, "denmune_neigh": 8, "classix_minpts": 5, "classix_radius": 0.13, "quantile": 0.194}),
            ('d31', {"n_clusters": 31, "min_samples": 13, "min_cluster_size": 23, "min_samples_hdbscan": 3, "eps": 0.09, "n_neighbors": 18, "denmune_neigh": 37, "classix_minpts": 23, "classix_radius": 0.03, "quantile": 0.023}),
            ('r15', {"n_clusters": 15, "min_samples": 13, "min_cluster_size": 5, "min_samples_hdbscan": 3, "eps": 0.17, "n_neighbors": 16, "denmune_neigh": 20, "classix_minpts": 10, "classix_radius": 0.09, "quantile": 0.05}),
            ('flame', {"n_clusters": 2, "min_samples": 11, "min_cluster_size": 3, "min_samples_hdbscan": 4, "eps": 0.45, "n_neighbors": 5, "denmune_neigh": 8, "classix_minpts": 16, "classix_radius": 0.18, "quantile": 0.339}),
            ('jain', {"n_clusters": 2, "min_samples": 14, "min_cluster_size": 19, "min_samples_hdbscan": 3, "eps": 0.23, "n_neighbors": 8, "denmune_neigh": 20, "classix_minpts": 3, "classix_radius": 0.35, "quantile": 0.267}),
            ('pathbased', {"n_clusters": 3, "min_samples": 13, "min_cluster_size": 27, "min_samples_hdbscan": 10, "eps": 0.33, "n_neighbors": 5, "denmune_neigh": 8, "classix_minpts": 6, "classix_radius": 0.23, "quantile": 0.122}),
            ('spiral', {"n_clusters": 3, "min_samples": 3, "min_cluster_size": 3, "min_samples_hdbscan": 3, "eps": 0.17, "n_neighbors": 3, "denmune_neigh": 4, "classix_minpts": 3, "classix_radius": 0.25, "quantile": 0.045}),
            ('s1', {"n_clusters": 15, "min_samples": 14, "min_cluster_size": 28, "min_samples_hdbscan": 4, "eps": 0.13, "n_neighbors": 9, "denmune_neigh": 40, "classix_minpts": 21, "classix_radius": 0.05, "quantile": 0.045}),
            ('s2', {"n_clusters": 15, "min_samples": 12, "min_cluster_size": 27, "min_samples_hdbscan": 4, "eps": 0.11, "n_neighbors": 19, "denmune_neigh": 54, "classix_minpts": 15, "classix_radius": 0.01, "quantile": 0.041}),
            ('s3', {"n_clusters": 15, "min_samples": 3, "min_cluster_size": 29, "min_samples_hdbscan": 4, "eps": 0.06, "n_neighbors": 19, "denmune_neigh": 39, "classix_minpts": 22, "classix_radius": 0.02, "quantile": 0.045}),
            ('s4', {"n_clusters": 15, "min_samples": 4, "min_cluster_size": 29, "min_samples_hdbscan": 3, "eps": 0.06, "n_neighbors": 7, "denmune_neigh": 10, "classix_minpts": 22, "classix_radius": 0.02, "quantile": 0.05}),
            ('unbalance', {"n_clusters": 8, "min_samples": 3, "min_cluster_size": 8, "min_samples_hdbscan": 4, "eps": 0.36, "n_neighbors": 19, "denmune_neigh": 48, "classix_minpts": 7, "classix_radius": 0.3, "quantile": 0.194}),
            ]

dataset_names=[var[0] for var in datasets]
algo_names=['KM', 'MBKM', 'PAM', 'DBSCAN', 'OPTICS', 'HDBSCAN', 'MS', 'Single', 'Comp', 'Ward', 'Spectral', 'CLASSIX', 'DenMune', 'FINCH']

df_extrinsic=pd.DataFrame(columns=algo_names, index=dataset_names)
df_intrinsic=pd.DataFrame(columns=algo_names, index=dataset_names)

for i_dataset, (dataset, params) in tqdm(enumerate(datasets)):
    data=clustbench.load_dataset('sipu', dataset)
    scaled_data=StandardScaler().fit_transform(data.data)

    bandwidth=estimate_bandwidth(scaled_data, quantile=params['quantile'])

    models={}
    ar=[]
    ami=[]
    chi=[]
    dbi=[]

    models['KM']=KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10).fit(scaled_data)
    models['MBKM']=MiniBatchKMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10, batch_size=256).fit(scaled_data)
    models['PAM']=KMedoids(n_clusters=params['n_clusters'], init='k-medoids++').fit(scaled_data)
    models['DBSCAN']=DBSCAN(min_samples=params['min_samples'], eps=params['eps']).fit(scaled_data)
    models['OPTICS']=OPTICS(min_samples=params['min_samples'], metric='euclidean', cluster_method='dbscan', eps=params['eps']).fit(scaled_data)
    models['HDBSCAN']=HDBSCAN(min_samples=params['min_samples_hdbscan'], min_cluster_size=params['min_cluster_size']).fit(scaled_data)
    models['MS']=MeanShift(bandwidth=bandwidth, cluster_all=True).fit(scaled_data)
    models['Single']=AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='single').fit(scaled_data)
    models['Comp']=AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='complete').fit(scaled_data)
    models['Ward']=AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward').fit(scaled_data)
    models['Spectral']=SpectralClustering(n_clusters=params['n_clusters'], affinity='nearest_neighbors', n_neighbors=params['n_neighbors'], assign_labels='kmeans').fit(scaled_data)
    models['CLASSIX']=CLASSIX(sorting='pca', group_merging='density', radius=params['classix_radius'], minPts=params['classix_minpts'], verbose=0).fit(scaled_data)
    models['DenMune'], _=DenMune(train_data=scaled_data, k_nearest=params['denmune_neigh']).fit_predict(show_plots=False, show_analyzer=False)
    c, num_clust, models['FINCH']=FINCH(data=scaled_data, req_clust=params['n_clusters'], distance='euclidean', verbose=False)

    for model in list(models.keys()):
        if model=='DenMune':
            preds=fix_denmune_labels(models[model]['train'])
        if model=='FINCH':
            preds=models[model]
        elif model!='DenMune':
            preds=models[model].labels_

        # Convert preds to be a numpy array
        preds=np.array(preds)

        # Eliminate singletons and clusters of size 2 for possibility of calculating DBCV index
        for label in np.unique(preds):
            if sum(preds==label)==1 or sum(preds==label)==2:
                print("Singleton or cluster of size 2 detected for "+str(model)+" and "+str(dataset)+" dataset")
                index_of_singleton=np.where(preds==label)[0]
                preds[index_of_singleton]=-1

        print("dataset: "+str(dataset)+" model: "+str(model))
        ar=round(metrics.adjusted_rand_score(data.labels[0], preds), 3)
        ami=round(metrics.adjusted_mutual_info_score(data.labels[0], preds), 3)
        chi=round(metrics.calinski_harabasz_score(scaled_data, preds), 3)
        silhouette=round(metrics.silhouette_score(scaled_data, preds), 3)
        dbcv=round(validity_index(scaled_data, preds), 3)
        df_extrinsic.at[dataset, model]=str(ar)+"/"+str(ami)
        df_intrinsic.at[dataset, model]=str(chi)+"/"+str(silhouette)+"/"+str(dbcv)

df_extrinsic.to_csv('sipu_extrinsic.csv')
df_intrinsic.to_csv('sipu_intrinsic.csv')