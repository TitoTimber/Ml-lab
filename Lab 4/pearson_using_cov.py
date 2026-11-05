from sklearn.datasets import load_iris
import math

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()
feature_names = iris.feature_names

# Calculate mean
def calculate_mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)

# Calculate covariance between two variables
def calculate_covariance(X, Y):
    mean_x = calculate_mean(X)
    mean_y = calculate_mean(Y)
    
    cov_sum = 0.0
    for i in range(len(X)):
        dx = X[i] - mean_x
        dy = Y[i] - mean_y
        cov_sum += dx * dy
        
    return cov_sum / (len(X) - 1)  # Sample covariance (dividing by n - 1)

# Calculate Pearson correlation coefficient using covariance
def pearson_correlation(X, Y):
    cov_xy = calculate_covariance(X, Y)
    
    # Standard deviation is the square root of variance, and Var(X) = Cov(X, X)
    std_x = math.sqrt(calculate_covariance(X, X))
    std_y = math.sqrt(calculate_covariance(Y, Y))
    
    denominator = std_x * std_y
    if denominator == 0:
        return 0.0
        
    return cov_xy / denominator

# Calculate correlation matrix manually
def correlation_matrix(data):
    features = len(data[0])
    matrix = []
    for i in range(features):
        row = []
        for j in range(features):
            x_values = []
            y_values = []
            for sample in data:
                x_values.append(sample[i])
                y_values.append(sample[j])

            correlation = pearson_correlation(x_values, y_values)
            row.append(correlation)
        matrix.append(row)

    return matrix

# Main program
matrix = correlation_matrix(X)
print("Pearson Correlation Matrix")

for row in matrix:
    print("[", end=" ")
    for value in row:
        print(round(value, 3), end=" ")
    print("]")

print("\nFeature Names:")
for name in feature_names:
    print(name)