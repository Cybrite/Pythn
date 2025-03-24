question 1

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target  # Actual labels (for comparison)
# feature_names = iris.feature_names
# target_names = iris.target_names

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Apply K-Means clustering (K=3 since Iris has 3 species)
# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# cluster_labels = kmeans.fit_predict(X_scaled)

# # For visualization, we'll use PCA to reduce to 2 dimensions
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Create a scatter plot
# plt.figure(figsize=(10, 8))

# # Plot the clusters
# colors = ['blue', 'green', 'red']
# markers = ['o', 's', '^']

# for i in range(3):
#     # Plot points in each cluster
#     plt.scatter(
#         X_pca[cluster_labels == i, 0], 
#         X_pca[cluster_labels == i, 1],
#         s=50, c=colors[i], marker=markers[i], alpha=0.7,
#         label=f'Cluster {i+1}'
#     )

# # Plot cluster centers
# centers_pca = pca.transform(kmeans.cluster_centers_)
# plt.scatter(
#     centers_pca[:, 0], centers_pca[:, 1],
#     s=200, c='yellow', marker='.', edgecolors='black',
#     label='Centroids'
# )

# # Add labels and title
# plt.title('K-Means Clustering of Iris Dataset (PCA-reduced)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add a text showing the explained variance ratio
# explained_variance = pca.explained_variance_ratio_
# plt.annotate(
#     f'Explained variance: {sum(explained_variance):.2f}',
#     xy=(0.05, 0.95), xycoords='axes fraction'
# )

# # Show the plot
# plt.tight_layout()
# plt.show()