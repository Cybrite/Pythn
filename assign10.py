#question 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Reduce dimensions for visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)
centers_pca = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)

# Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

for i in range(3):
    plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1],
                s=50, c=colors[i], marker=markers[i], alpha=0.7, label=f'Cluster {i+1}')

plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='yellow', 
            marker='.', edgecolors='black', label='Centroids')

plt.title('K-Means Clustering of Iris Dataset (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#question 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess data
data = pd.read_csv('Online Retail.csv')
data = data.dropna(subset=['CustomerID'])
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
latest_date = pd.to_datetime(data['InvoiceDate']).max()

# Create RFM features
customer_data = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - pd.to_datetime(x.max())).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Cluster customers
X = StandardScaler().fit_transform(customer_data)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X)

# Visualize with PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
centers = pca.transform(kmeans.cluster_centers_)

# Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

for i in range(3):
    mask = customer_data['Cluster'] == i
    plt.scatter(coords[mask, 0], coords[mask, 1], 
                s=50, c=colors[i], marker=markers[i], alpha=0.7, label=f'Cluster {i}')

plt.scatter(centers[:, 0], centers[:, 1], s=200, c='yellow', 
            marker='.', edgecolors='black', label='Centroids')

plt.title('Customer Segmentation (PCA-reduced)')
plt.xlabel('PC1'), plt.ylabel('PC2')
plt.legend(), plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
