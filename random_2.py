import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Read the data
df = pd.read_csv('Online Retail.csv')

# Basic preprocessing
df_clean = df.dropna().copy()
df_clean.loc[:, 'InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
max_date = df_clean['InvoiceDate'].max()
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.contains('C')].copy()
df_clean.loc[:, 'TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

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

# Try different DBSCAN parameters
print("Exploring DBSCAN parameters:")
best_eps = 0.5
best_min_samples = 5
best_n_clusters = 0

# Parameter exploration
for eps in [0.3, 0.5, 0.7, 0.9, 1.1]:
    for min_samples in [3, 5, 8, 10, 15]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(rfm_scaled)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        print(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points")
        
        if n_clusters > best_n_clusters:
            best_n_clusters = n_clusters
            best_eps = eps
            best_min_samples = min_samples

print(f"\nBest parameters: eps={best_eps}, min_samples={best_min_samples} with {best_n_clusters} clusters")

# Apply DBSCAN with best parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
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
            label=f'Noise ({len(cluster_data)} points)'
        )
    else:
        plt.scatter(
            cluster_data['Component 1'],
            cluster_data['Component 2'],
            s=80,
            c=[colors[i]],
            alpha=0.8,
            label=f'Cluster {cluster} ({len(cluster_data)} points)'
        )

plt.title('Customer Segments using DBSCAN', fontsize=18)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze the clusters
if best_n_clusters > 0:
    print("\nCluster Statistics:")
    for cluster in sorted(set(clusters)):
        if cluster != -1:
            cluster_data = rfm[rfm['Cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_data)} customers):")
            print(cluster_data.describe().round(2))