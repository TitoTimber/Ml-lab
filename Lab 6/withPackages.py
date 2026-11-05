import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# ==================== Load Data ====================
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names

print("Features:", features)
print("X shape:", X.shape)

# ==================== A.1.1 VARIANCE ====================
print("\n=== A.1.1 VARIANCE ===")
variances = np.var(X, axis=0)
for f, v in zip(features, variances):
    print(f"{f}: {v:.4f}")

# ==================== A.1.2 CORRELATION ====================
print("\n=== A.1.2 CORRELATION ===")
corr = np.corrcoef(X.T)
corr_df = pd.DataFrame(corr, index=features, columns=features)
print(corr_df.round(4))

# ==================== A.1.3 CHI-SQUARE ====================
print("\n=== A.1.3 CHI-SQUARE ===")
# Chi2 needs non-negative features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_scores, _ = chi2(X_scaled, y)
for f, s in zip(features, chi_scores):
    print(f"{f}: {s:.4f}")

# ==================== A.1.4 INFORMATION GAIN ====================
print("\n=== A.1.4 INFORMATION GAIN (Mutual Information) ===")
ig_scores = mutual_info_classif(X, y, random_state=42)
for f, s in zip(features, ig_scores):
    print(f"{f}: {s:.4f}")

# ==================== Summary Table ====================
print("\n=== SUMMARY ===")
results = pd.DataFrame({
    "Feature": features,
    "Variance": variances,
    "Chi-Square": chi_scores,
    "Information Gain": ig_scores
})
print(results.round(4))

# ==================== A.2 GA + SVM ====================
print("\n=== A.2 GA BASED FEATURE SELECTION + SVM ===")

np.random.seed(42)
n_features = X.shape[1]
pop_size = 20
max_gen = 50
mutation_rate = 0.1

def fitness(chromosome):
    selected = chromosome[:n_features].astype(bool)
    if not selected.any():
        return 0.0
    scales = chromosome[n_features:]
    X_new = X[:, selected] * scales[selected]
    return cross_val_score(SVC(kernel='rbf'), X_new, y, cv=5).mean()

# Initialize population (binary features + continuous scales)
population = []
for _ in range(pop_size):
    binary = np.random.randint(0, 2, n_features)
    scales = np.random.uniform(0.5, 2.0, n_features)
    population.append(np.concatenate([binary, scales]))

avg_fitness_history = []

for gen in range(max_gen):
    fitnesses = np.array([fitness(ind) for ind in population])
    avg_fit = fitnesses.mean()
    avg_fitness_history.append(avg_fit)
    
    # Early stopping
    if gen > 5 and abs(avg_fitness_history[-1] - avg_fitness_history[-6]) < 0.001:
        print(f"Stopped at generation {gen}")
        break
    
    # Tournament selection + crossover + mutation
    new_pop = []
    for _ in range(pop_size):
        # Tournament
        idx1 = np.random.choice(pop_size, 3)
        idx2 = np.random.choice(pop_size, 3)
        p1 = population[idx1[np.argmax(fitnesses[idx1])]]
        p2 = population[idx2[np.argmax(fitnesses[idx2])]]
        
        # Crossover
        point = np.random.randint(1, len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        
        # Mutation
        for i in range(n_features):
            if np.random.rand() < mutation_rate:
                child[i] = 1 - child[i]
        for i in range(n_features, len(child)):
            if np.random.rand() < mutation_rate:
                child[i] = np.random.uniform(0.5, 2.0)
        
        new_pop.append(child)
    population = new_pop

# Best result
final_fitnesses = [fitness(ind) for ind in population]
best_idx = np.argmax(final_fitnesses)
best = population[best_idx]
selected = best[:n_features].astype(bool)

print(f"Best Accuracy: {final_fitnesses[best_idx]:.4f}")
print("Selected features:", [features[i] for i in range(n_features) if selected[i]])

# Plot average fitness
plt.figure(figsize=(8, 4))
plt.plot(avg_fitness_history, marker='o')
plt.xlabel("Generation")
plt.ylabel("Average Fitness (SVM Accuracy)")
plt.title("GA Feature Selection - Average Fitness")
plt.grid(True)
plt.show()