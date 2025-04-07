# #question 1
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score, adjusted_rand_score

# # Load and preprocess data
# iris = load_iris()
# X_scaled = StandardScaler().fit_transform(iris.data)

# # Reduce dimensions for visualization with a reusable PCA object
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Apply clustering algorithms
# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# kmeans_labels = kmeans.fit_predict(X_scaled)
# # Use the fitted PCA object to transform cluster centers
# kmeans_centers_pca = pca.transform(kmeans.cluster_centers_)

# dbscan = DBSCAN(eps=0.6, min_samples=5)
# dbscan_labels = dbscan.fit_predict(X_scaled)

# # Visualization
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# colors = ['blue', 'green', 'red']
# markers = ['o', 's', '^']

# # K-Means plot
# for i in range(3):
#     mask = kmeans_labels == i
#     ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, c=colors[i], 
#                 marker=markers[i], alpha=0.7, label=f'Cluster {i+1}')

# ax1.scatter(kmeans_centers_pca[:, 0], kmeans_centers_pca[:, 1], s=200, 
#             c='yellow', marker='.', edgecolors='black', label='Centroids')
# ax1.set(title='K-Means Clustering', xlabel='PC1', ylabel='PC2')
# ax1.grid(True, linestyle='--', alpha=0.7)
# ax1.legend()

# # DBSCAN plot
# unique_labels = np.unique(dbscan_labels)
# colors_dbscan = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# for i, label in enumerate(unique_labels):
#     mask = dbscan_labels == label
#     color = 'black' if label == -1 else colors_dbscan[i]
#     marker = 'x' if label == -1 else 'o'
#     alpha = 0.5 if label == -1 else 0.7
#     label_text = 'Noise' if label == -1 else f'Cluster {i+1}'
    
#     ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50, c=color, 
#                 marker=marker, alpha=alpha, label=label_text)

# ax2.set(title='DBSCAN Clustering', xlabel='PC1', ylabel='PC2')
# ax2.grid(True, linestyle='--', alpha=0.7)
# ax2.legend()

# plt.tight_layout()
# plt.show()

# # Performance evaluation
# n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
# n_noise = np.sum(dbscan_labels == -1)
# print(f"DBSCAN found {n_dbscan_clusters} clusters with {n_noise} noise points")

# kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
# print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

# # DBSCAN silhouette (excluding noise)
# non_noise_mask = dbscan_labels != -1
# if np.sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
#     dbscan_silhouette = silhouette_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
#     print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.3f}")

# # Comparison with ground truth
# print(f"K-Means Adjusted Rand Index: {adjusted_rand_score(iris.target, kmeans_labels):.3f}")
# # Fixed the double colon typo in the format string
# print(f"DBSCAN Adjusted Rand Index: {adjusted_rand_score(iris.target, dbscan_labels):.3f}")

# # Parameter exploration - condensed output
# print("\nDBSCAN Parameter Exploration:")
# results = []
# for eps in [0.3, 0.5, 0.7, 0.9]:
#     for min_samples in [3, 5, 10]:
#         labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
#         n_clusters = len(np.unique(labels[labels != -1]))
#         n_noise = np.sum(labels == -1)
#         results.append((eps, min_samples, n_clusters, n_noise))

# # Format as a table
# print(f"{'eps':<6}{'min_samples':<12}{'clusters':<10}{'noise points':<12}")
# print("-" * 40)
# for eps, min_samples, n_clusters, n_noise in results:
#     print(f"{eps:<6}{min_samples:<12}{n_clusters:<10}{n_noise:<12}")

#question 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Read the data
df = pd.read_csv('Online Retail.csv')

# Basic preprocessing
df_clean = df.dropna()
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
max_date = df_clean['InvoiceDate'].max()
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.contains('C')]
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Create RFM features
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Standardize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply DBSCAN with fixed parameters
dbscan = DBSCAN(eps=1.0, min_samples=10)
clusters = dbscan.fit_predict(rfm_scaled)

# Add cluster information to the RFM dataframe
rfm['Cluster'] = clusters

# Visualize clusters with PCA for dimensionality reduction
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# Create a dataframe with PCA results
pca_df = pd.DataFrame(data=rfm_pca, columns=['Component 1', 'Component 2'])
pca_df['Cluster'] = clusters

# Plot PCA results colored by cluster
plt.figure(figsize=(12, 10))

# Define colors for clusters
unique_clusters = sorted(set(clusters))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

# Plot each cluster
for i, cluster in enumerate(unique_clusters):
    cluster_data = pca_df[pca_df['Cluster'] == cluster]
    if cluster == -1:
        plt.scatter(
            cluster_data['Component 1'],
            cluster_data['Component 2'],
            s=80,
            c='black',
            alpha=0.4,
            label='Noise'
        )
    else:
        plt.scatter(
            cluster_data['Component 1'],
            cluster_data['Component 2'],
            s=80,
            c=[colors[i]],
            alpha=0.8,
            label=f'Cluster {cluster}'
        )

plt.title('Customer Segments using DBSCAN', fontsize=18)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.show()