import numpy as np

from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import cdist

from src import sinkhorn
from utils.helpers import *

def aggregate_points(X, weights, n_clusters):
    '''
    aggregate points and weights with the use of AgglomerativeClustering
    
    X: ndarray (num_points, dim_X)
        the initial cloud of points
    weights: ndarray (num_points, )
        the weights of the histogram
    n_clusters: int
        number of clusters, in which the data is splitted
    '''
    
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clustering.fit(X)
    
    X_agg = np.zeros(shape = (n_clusters, X.shape[1]))
    weights_agg = np.zeros(shape = n_clusters)
    #labels are needed to reassign back the points to clusters
    labels = agg_clustering.labels_
    
    for label, x, w in zip(labels, X, weights):
        X_agg[label] += x
        weights_agg[label] += w
        
    # turn the sum of coordinates into mean 
    X_agg = X_agg / np.bincount(labels)[:, None]
        
    return X_agg, weights_agg, labels

def refine_transport_plan(C, reg, a, b, G_agg, a_agg, b_agg, inverted_labels_i, inverted_labels_f, use_n_max = 1):
    '''
    given the coarsened transprot plan for the clusters G_agg, refine it
    
    C: ndarray (dim_a, dim_b)
        the cost matrix
    reg: float
        the regularization term
    a: ndarray, shape (dim_a,)
        samples weights in the source domain
    b: ndarray, shape (dim_b,)
        samples weights in the target domain
    G_agg: ndarray (dim_a_agg, dim_b_agg)
        transportation plan for the coarsened domain
    a_agg: ndarray, shape (dim_a_agg,)
        samples weights in the source domain
    b_agg: ndarray, shape (dim_b_agg,)
        samples weights in the target domain
    inverted_labels_i: dict
        for each cluster from the source key of the dict points to the points with the corresponding cluster
    inverted_labels_i: dict
        for each cluster from the destination key of the dict points to the points with the corresponding cluster
    use_n_max: int
        for each cluster of the source take use_n_max most relevant sinks
    '''
    
    G = np.zeros((len(a), len(b)))
    
    for i in range(a_agg.size):
        j_max = np.argpartition(-G_agg[i], use_n_max)[:use_n_max]
        
        src_indices = inverted_labels_i[i]
        dst_indices = concatenate_lists([inverted_labels_f[j] for j in j_max])
        
        #selected rows and columns
        row_col = np.ix_(src_indices, dst_indices)
        
        a_norm = a[src_indices] / sum(a[src_indices])
        b_norm = b[dst_indices] / sum(b[dst_indices])
        
        G_partial = sinkhorn.sinkhorn_knopp(C[row_col], reg, a = a_norm, b = b_norm) 
        
        G[row_col] += G_partial 
    
    #normalize
    return G / a_agg.size

def sinkhorn_multigrid(C, x_i, x_f, reg_coarse, reg_fine, a, b, n_clusters_a, n_clusters_b, use_n_max = 1):
    '''
    performs the sinkhorn iterations on multigrid domain : 
    1) Agglomerative clustering on the points
    2) Sinkhorn algorithm on the coarsened matrix
    3) For each of the clusters perform refinement

    C: ndarray (dim_a, dim_b)
        the cost matrix
    x_i: (dim_a, space_dim)
        the sources locations
    x_f: (dim_b, space_dim)
        the sinks locations
    reg_coarse: float
        the regularization term
    reg_fine: float
        the regularization term
    a: ndarray, shape (dim_a,)
        samples weights in the source domain
    b: ndarray, shape (dim_b,)
        samples weights in the target domain
    n_clusters_a: int
        number of clusters, in which the sources are splitted
    n_clusters_b: int
        number of clusters, in which the sinks are splitted
    use_n_max: int
        for each cluster of the source take use_n_max most relevant sinks
    '''
    
    x_i_agg, a_agg, labels_i = aggregate_points(x_i, weights=a, n_clusters=n_clusters_a)
    x_f_agg, b_agg, labels_f = aggregate_points(x_f, weights=b, n_clusters=n_clusters_b)

    #these dicts are needed to choose for each cluster points belonging to it
    inverted_labels_i = get_inverted_labels(labels_i)
    inverted_labels_f = get_inverted_labels(labels_f)
    
    C_agg = cdist(x_i_agg, x_f_agg)
    
    G_coarse = sinkhorn.sinkhorn_knopp(C_agg, reg = reg_coarse, a = a_agg, b = b_agg)
    G_refine = refine_transport_plan(C, reg = reg_fine, a = a, b = b, 
                                     G_agg=G_coarse, a_agg=a_agg, b_agg=b_agg, 
                                     inverted_labels_i=inverted_labels_i, 
                                     inverted_labels_f=inverted_labels_f, 
                                     use_n_max = use_n_max)
    
    return G_refine