import random
import math

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def kmeans(data, k, max_iter=100):
    # Random initial centroids
    centroids = random.sample(data, k)
    
    for _ in range(max_iter):
        # Assignment step
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean(point, c) for c in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
        
        # Update step
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_c = [sum(dim) / len(cluster) for dim in zip(*cluster)]
                new_centroids.append(new_c)
            else:
                new_centroids.append(random.choice(data))
        
        # Check convergence
        if all(euclidean(old, new) < 1e-4 for old, new in zip(centroids, new_centroids)):
            break
        centroids = new_centroids
    
    # Final labels
    labels = []
    for point in data:
        distances = [euclidean(point, c) for c in centroids]
        labels.append(distances.index(min(distances)))
    
    return labels, centroids

# Example usage (replace with your data)
# data = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], ...]
# labels, centroids = kmeans(data, k=3)