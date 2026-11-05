def agglomerative(data, k=3, linkage="single"):
    # Start with each point as its own cluster
    clusters = [[i] for i in range(len(data))]
    
    def cluster_distance(c1, c2):
        dists = [euclidean(data[i], data[j]) for i in c1 for j in c2]
        if linkage == "single":
            return min(dists)
        elif linkage == "complete":
            return max(dists)
        else:  # average
            return sum(dists) / len(dists)
    
    while len(clusters) > k:
        # Find closest pair of clusters
        min_dist = float('inf')
        to_merge = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    to_merge = (i, j)
        
        # Merge them
        i, j = to_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    # Assign labels
    labels = [0] * len(data)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    
    return labels

# labels = agglomerative(data, k=3, linkage="single")
# labels = agglomerative(data, k=3, linkage="complete")