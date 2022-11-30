from typing import List, Dict, Union, Tuple, Optional 
from math import floor 
from copy import deepcopy

import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity



def spectral_clustering(X: np.array, n_clusters: int)->np.array:
    '''
    wrapper for spectral clustering  
    '''
    clf = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='rbf')
    # sim = cosine_similarity(X, X) 
    label_pred = clf.fit_predict(X)
    return label_pred 

def dbscan(X: np.ndarray, n_clusters: int, eps: Optional[float] = None)-> np.ndarray:
    '''
    wrapper for dbscan clustering algorithm.
    eps: the max distance for the points to be considered a neighbor.

    '''
    distances = euclidean_distances(X) # (n, n)
    eps2cluster = {} 

    if eps == None: 
        for eps in range(5, 20):
            clf = DBSCAN(eps=eps, min_samples=5, metric='euclidean', n_jobs=4)
            label_pred = clf.fit_predict(X) 
            found_clusters = np.max(label_pred)
            eps2cluster[eps] = found_clusters
        
        min_diff = 1e6 
        best_eps = None 
        for eps in eps2cluster: 
            if eps2cluster[eps] != -1:
                diff = abs(eps2cluster[eps] - n_clusters)
                if diff < min_diff or not best_eps:
                    best_eps = eps 
                    min_diff = diff 


        eps = best_eps 
    
    clf = DBSCAN(eps=eps, min_samples=5, metric='euclidean', n_jobs=4)
    label_pred = clf.fit_predict(X) 
        
    # searched_eps = set([eps, ]) 
    # while found_clusters == 0 or found_clusters == -1:
    #     if found_clusters == 0: 
    #         # eps is too large 
    #         eps -=1 
    #     else:
    #         eps +=1 
    #     if eps in searched_eps: 
    #         break 
    #     clf = DBSCAN(eps=eps, min_samples=5, metric='euclidean', n_jobs=4)
    #     label_pred = clf.fit_predict(X) 
    #     found_clusters = np.max(label_pred)
    #     searched_eps.add(eps)
    

    
    idx2cluster = { i: cluster for i, cluster in enumerate(label_pred) if (cluster!=-1 or cluster>=n_clusters)} 
    # assign -1 instances to their nearest neighbors 
    neighbors = np.argsort(distances, axis=1) 
    new_label_pred = deepcopy(label_pred)
    for i, cluster in enumerate(label_pred):
        if cluster == -1:
            for n in neighbors[i,1:]:# exclude self 
                if n in idx2cluster:
                    new_label_pred[i] = idx2cluster[n]
                    break 


    return new_label_pred, eps 

def agglomerative_clustering(X:np.array, n_clusters: int)-> np.array:
    clf = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
    label_pred = clf.fit_predict(X)
    return label_pred 

def agglomerative_ward(X: np.array, n_clusters:int)-> np.array:
    clf = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    label_pred = clf.fit_predict(X)
    return label_pred 

