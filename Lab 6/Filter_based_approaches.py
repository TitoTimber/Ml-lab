import math
from collections import Counter

# Iris data (first 10 samples for demo — replace with full data if needed)
X = [
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1]
]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # all setosa for demo
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# ---------- Variance ----------
def variance(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n

print("=== VARIANCE ===")
for i, f in enumerate(features):
    col = [row[i] for row in X]
    print(f"{f}: {variance(col):.4f}")

# ---------- Correlation (Pearson) ----------
def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))
    return num / (den_x * den_y) if den_x * den_y != 0 else 0

print("\n=== CORRELATION ===")
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        col_i = [row[i] for row in X]
        col_j = [row[j] for row in X]
        print(f"{features[i]} & {features[j]}: {correlation(col_i, col_j):.4f}")

# ---------- Chi-Square ----------
def chi_square(feature, labels, bins=3):
    min_val = min(feature)
    max_val = max(feature)
    bin_width = (max_val - min_val) / bins
    bin_counts = [Counter() for _ in range(bins)]
    class_counts = Counter(labels)
    
    for val, lab in zip(feature, labels):
        b = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[b][lab] += 1
    
    chi = 0
    total = len(labels)
    for b in range(bins):
        bin_total = sum(bin_counts[b].values())
        for c in class_counts:
            O = bin_counts[b][c]
            E = (bin_total * class_counts[c]) / total
            if E > 0:
                chi += (O - E) ** 2 / E
    return chi

print("\n=== CHI-SQUARE ===")
for i, f in enumerate(features):
    col = [row[i] for row in X]
    print(f"{f}: {chi_square(col, y):.4f}")

# ---------- Information Gain ----------
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

def information_gain(feature, labels, bins=3):
    H_y = entropy(labels)
    min_val = min(feature)
    max_val = max(feature)
    bin_width = (max_val - min_val) / bins
    bin_labels = [[] for _ in range(bins)]
    
    for val, lab in zip(feature, labels):
        b = min(int((val - min_val) / bin_width), bins - 1)
        bin_labels[b].append(lab)
    
    H_cond = 0
    total = len(labels)
    for b in bin_labels:
        if b:
            H_cond += (len(b) / total) * entropy(b)
    return H_y - H_cond

print("\n=== INFORMATION GAIN ===")
for i, f in enumerate(features):
    col = [row[i] for row in X]
    print(f"{f}: {information_gain(col, y):.4f}")