import random
import math

def fuzzy_c_means(data, c=3, m=2.0, eps=1e-5, max_iter=100):
    n = len(data)
    d = len(data[0])
    
    # Initialize membership matrix U (c x n)
    U = [[random.random() for _ in range(n)] for _ in range(c)]
    for j in range(n):
        s = sum(U[i][j] for i in range(c))
        for i in range(c):
            U[i][j] /= s
    
    for it in range(max_iter):
        # Update centroids
        centroids = []
        for i in range(c):
            num = [0] * d
            den = 0
            for j in range(n):
                um = U[i][j] ** m
                den += um
                for dim in range(d):
                    num[dim] += um * data[j][dim]
            centroids.append([num[dim] / den if den else 0 for dim in range(d)])
        
        # Update membership
        U_new = [[0] * n for _ in range(c)]
        for j in range(n):
            for i in range(c):
                dist_i = euclidean(data[j], centroids[i])
                if dist_i == 0:
                    U_new[i][j] = 1
                    continue
                s = 0
                for k in range(c):
                    dist_k = euclidean(data[j], centroids[k])
                    if dist_k == 0:
                        s = float('inf')
                        break
                    s += (dist_i / dist_k) ** (2 / (m - 1))
                U_new[i][j] = 1 / s if s != float('inf') else 0
        
        # Check convergence
        max_diff = max(abs(U_new[i][j] - U[i][j]) for i in range(c) for j in range(n))
        U = U_new
        if max_diff < eps:
            break
    
    # Hard labels (highest membership)
    labels = []
    for j in range(n):
        memberships = [U[i][j] for i in range(c)]
        labels.append(memberships.index(max(memberships)))
    
    return labels, centroids, U

# labels, centroids, U = fuzzy_c_means(data, c=3, m=2.0)