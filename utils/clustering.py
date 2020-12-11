from sklearn.cluster import KMeans


def KMeans_clustering(data, n_clust):
    clt = KMeans(n_clusters=n_clust)
    clt.fit(data)
    
    return clt.labels_, clt.cluster_centers_