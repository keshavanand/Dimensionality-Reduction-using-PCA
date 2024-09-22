# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import the necessory libraries
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# load mnist dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# retrive features and target labels
X , Y = mnist['data'], mnist['target']

# convert y to integers
Y = Y.astype(int)


# get index of all unique digits for ploting
unique_digits = np.unique(Y)
digit_indices = [np.where(Y== digit)[0][0] for digit in unique_digits]

# Function to display digit
def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    
# Display all digits
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plot_digit(X[digit_indices[i]])
plt.show()

# Use PCA to retrieve the first and second principal component 
# Apply PCA to reduce the dataset to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance ratio of the first two components
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance of PC1: {explained_variance[0]:.4f}")
print(f"Explained variance of PC2: {explained_variance[1]:.4f}")

# Plot projections onto 1D hyperplane (PC1 vs PC2)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap="viridis", alpha=0.6)
plt.colorbar(label='Digit')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projection of MNIST onto 2D PCA space')
plt.show()

# Reduce MNIST to 154 dimensions using Incremental PCA
n_components = 154
ipca = IncrementalPCA(n_components=n_components, batch_size=200)
X_ipca = ipca.fit_transform(X)

# Output the cumulative explained variance
cumulative_variance = sum(ipca.explained_variance_ratio_)
print(f"Cumulative explained variance with {n_components} components: {cumulative_variance:.4f}")

# Project the reduced data back to original space
X_reconstructed = ipca.inverse_transform(X_ipca)

# Display original and reconstructed digits side by side
plt.figure(figsize=(10, 8))
for i in range(10):
    # Original
    plt.subplot(4, 10, i+1)
    plot_digit(X[digit_indices[i]])
    plt.title("Original")

    # Reconstructed
    plt.subplot(4, 10, i+11)
    plot_digit(X_reconstructed[digit_indices[i]])
    plt.title("Recon")
plt.tight_layout() 
plt.show()