# Pure Python PCA (no NumPy) — full working example

X = [
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0]
]

n = len(X)
d = len(X[0])

# ---------- Step 1: Center the data ----------
mu = [0.0] * d
for i in range(n):
    for j in range(d):
        mu[j] += X[i][j]
for j in range(d):
    mu[j] /= n

Xc = [[X[i][j] - mu[j] for j in range(d)] for i in range(n)]

print("Mean vector μ:", mu)
print("Centered data Xc:")
for row in Xc:
    print(row)

# ---------- Step 2: Covariance matrix ----------
C = [[0.0] * d for _ in range(d)]
for i in range(d):
    for j in range(d):
        s = 0.0
        for k in range(n):
            s += Xc[k][i] * Xc[k][j]
        C[i][j] = s / (n - 1)

print("\nCovariance matrix C:")
for row in C:
    print(row)

# ---------- Step 3: Eigen-decomposition (2×2 analytic) ----------
a, b, c = C[0][0], C[0][1], C[1][1]
trace = a + c
det   = a * c - b * b
disc  = (trace * trace - 4 * det)  0.5

lambda1 = (trace + disc) / 2
lambda2 = (trace - disc) / 2

def eigenvector(lam):
    # Solve (a - λ)x + b y = 0
    if abs(b) > 1e-12:
        x = 1.0
        y = -(a - lam) / b
    else:
        x, y = 1.0, 0.0
    norm = (x * x + y * y)  0.5
    return [x / norm, y / norm]

w1 = eigenvector(lambda1)
w2 = eigenvector(lambda2)

print("\nEigenvalues: λ1 =", round(lambda1, 4), ", λ2 =", round(lambda2, 4))
print("Principal component 1 (w1):", [round(v, 4) for v in w1])
print("Principal component 2 (w2):", [round(v, 4) for v in w2])

# ---------- Step 4: Select top-k ----------
k = 1
W = [w1]  # keep only first principal component

explained_ratio = lambda1 / (lambda1 + lambda2)
print("\nExplained variance ratio (PC1):", round(explained_ratio, 4))

# ---------- Step 5: Project ----------
Z = []
for i in range(n):
    z = 0.0
    for j in range(d):
        z += Xc[i][j] * W[0][j]
    Z.append(z)

print("\nProjected data Z (1D):")
for z in Z:
    print(round(z, 4))

# ---------- Optional: Reconstruction ----------
X_recon = []
for i in range(n):
    row = []
    for j in range(d):
        row.append(Z[i] * W[0][j] + mu[j])
    X_recon.append(row)

print("\nReconstructed X:")
for row in X_recon:
    print([round(v, 4) for v in row])