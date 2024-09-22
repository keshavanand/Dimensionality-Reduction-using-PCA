# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:02:41 2024

@author: arshd
"""
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


# Generate Swiss roll dataset
X, t = make_swiss_roll(n_samples=1000, noise=0.1)

# Plot the generated Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis')
ax.set_title("Swiss Roll Dataset")
plt.show()

# Function to apply Kernel PCA and plot the results
def plot_kpca(X, kernel, gamma=None):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma)
    X_kpca = kpca.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='viridis')
    plt.title(f"Kernel PCA with {kernel} kernel")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

# Apply and plot Kernel PCA with different kernels
plot_kpca(X, 'linear')
plot_kpca(X, 'rbf', gamma=0.04)  # gamma can be adjusted for RBF kernel
plot_kpca(X, 'sigmoid', gamma=0.004)

# Discretize the target variable t into 10 bins 
binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
t_binned = binner.fit_transform(t.reshape(-1, 1)).astype(int).ravel()  # Discrete labels for classification

# Now split the data with the binned target variable
X_train, X_test, t_train, t_test = train_test_split(X, t_binned, test_size=0.2, random_state=42)

# Create a pipeline with Kernel PCA and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression(max_iter=5000))
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {
        'kpca__kernel': ['rbf', 'sigmoid', 'linear'],
        'kpca__gamma': np.linspace(0.01, 0.1, 10),
        'log_reg__C': np.logspace(-3, 3, 7)
    }
]

# Apply GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, t_train)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Best Kernel PCA model based on GridSearchCV results
best_kpca = grid_search.best_estimator_.named_steps['kpca']
X_best_kpca = best_kpca.transform(X_train)

# Plot the transformed test set using the best parameters
plt.figure(figsize=(6, 6))
plt.scatter(X_best_kpca[:, 0], X_best_kpca[:, 1], c=t_train, cmap='viridis')
plt.title("Best kPCA results from GridSearchCV")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



