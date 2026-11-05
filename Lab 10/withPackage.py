import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, confusion_matrix)
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# ==================== Shared Setup ====================
SEED = 42
COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

iris = load_iris()
X = pd.DataFrame(iris.data, columns=COLS)
y = iris.target
Xs = StandardScaler().fit_transform(X)

def match_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    mapping = {cl: k for k, cl in zip(r, c)}
    return np.array([mapping[p] for p in y_pred])

def evaluate(Xd, y_true, y_pred):
    mapped = match_labels(y_true, y_pred)
    return {
        "silhouette": silhouette_score(Xd, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "accuracy": np.mean(mapped == y_true)
    }, mapped

# ==================== Question 1: K-Means ====================
print("1. K-MEANS")
print("Choice of k (Elbow + Silhouette):")
print(f"{'k':>4} {'SSE':>12} {'Silhouette':>12}")
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED).fit(Xs)
    sil = silhouette_score(Xs, km.labels_) if k > 1 else np.nan
    print(f"{k:4d} {km.inertia_:12.4f} {sil:12.4f}")

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=SEED).fit(Xs)
km_scores, km_mapped = evaluate(Xs, y, kmeans.labels_)
print(f"\nFinal K-Means (k=3): iterations={kmeans.n_iter_}, SSE={kmeans.inertia_:.4f}")
print(km_scores)

# ==================== Question 2: Fuzzy C-Means ====================
def fuzzy_c_means(Xd, c=3, m=2.0, eps=1e-5, max_iter=300, seed=SEED):
    rng = np.random.default_rng(seed)
    n = Xd.shape[0]
    U = rng.random((c, n))
    U /= U.sum(axis=0)
    history = []
    for it in range(1, max_iter + 1):
        Um = U ** m
        V = (Um @ Xd) / Um.sum(axis=1, keepdims=True)
        D = np.linalg.norm(Xd[None, :, :] - V[:, None, :], axis=2)
        D = np.fmax(D, 1e-12)
        J = np.sum(Um * D ** 2)
        U_new = 1.0 / np.sum((D[:, None, :] / D[None, :, :]) ** (2 / (m - 1)), axis=1)
        delta = np.abs(U_new - U).max()
        history.append((it, J, delta))
        U = U_new
        if delta < eps:
            break
    return V, U, history

print("\n2. FUZZY C-MEANS")
V, U, hist = fuzzy_c_means(Xs)
fcm_labels = U.argmax(axis=0)
fcm_scores, fcm_mapped = evaluate(Xs, y, fcm_labels)
print(f"Converged in {len(hist)} iterations")
print(f"FPC = {np.sum(U**2)/U.shape[1]:.4f}")
print(fcm_scores)

# ==================== Question 3: Agglomerative ====================
print("\n3. AGGLOMERATIVE CLUSTERING")
dist = pdist(Xs, metric="euclidean")
print(f"{'Linkage':<12} {'Cophenetic':>12} {'Silhouette':>12} {'ARI':>8} {'NMI':>8} {'Acc':>8}")
agg_results = {}
for method in ["single", "complete", "average", "ward"]:
    Z = linkage(Xs, method=method)
    coph, _ = cophenet(Z, dist)
    lab = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(Xs)
    sc, mapped = evaluate(Xs, y, lab)
    agg_results[method] = (Z, lab, mapped, sc, coph)
    print(f"{method:<12} {coph:12.4f} {sc['silhouette']:12.4f} {sc['ARI']:8.4f} {sc['NMI']:8.4f} {sc['accuracy']:8.4f}")

# ==================== Comparison ====================
print("\nCOMPARISON")
print(f"{'Algorithm':<25} {'Silhouette':>12} {'ARI':>8} {'NMI':>8} {'Acc':>8}")
for name, sc in [("K-Means (k=3)", km_scores),
                 ("Fuzzy C-Means (m=2)", fcm_scores),
                 ("Agglomerative (Ward)", agg_results["ward"][3])]:
    print(f"{name:<25} {sc['silhouette']:12.4f} {sc['ARI']:8.4f} {sc['NMI']:8.4f} {sc['accuracy']:8.4f}")

# ==================== Optional Plots ====================
pca = PCA(n_components=2).fit(Xs)
X2 = pca.transform(Xs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="viridis", s=20)
plt.title("True Species")

plt.subplot(1, 3, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=km_mapped, cmap="viridis", s=20)
plt.title("K-Means")

plt.subplot(1, 3, 3)
plt.scatter(X2[:, 0], X2[:, 1], c=agg_results["ward"][2], cmap="viridis", s=20)
plt.title("Ward Agglomerative")
plt.tight_layout()
plt.show()