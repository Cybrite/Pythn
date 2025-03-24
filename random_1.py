import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Online Retail.csv')
print(f"Initial dataset shape: {data.shape}")

# Data preprocessing
data.dropna(subset=['CustomerID'], inplace=True)
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Filter out any potential outliers or errors (negative quantities or prices)
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
print(f"Dataset shape after cleaning: {data.shape}")

latest_date = pd.to_datetime(data['InvoiceDate']).max()


customer_data = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - pd.to_datetime(x.max())).days,  # Recency
    'InvoiceNo': 'nunique',      # Frequency (number of orders)
    'Quantity': 'sum',           # Total items purchased
    'TotalPrice': 'sum'          # Monetary value
}).reset_index()

customer_data.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalQuantity', 'TotalSpend']
print(f"Number of unique customers: {customer_data.shape[0]}")

# Descriptive statistics of features
print("\nFeature Statistics:")
print(customer_data.describe())

# Feature scaling
scaler = StandardScaler()
features = ['Recency', 'Frequency', 'TotalQuantity', 'TotalSpend']
scaled_data = scaler.fit_transform(customer_data[features])

# Elbow method for optimal K
wcss = []
silhouette_scores = []
range_k = range(2, 10)

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(scaled_data, labels))


# Apply K-means with optimal K (from visual inspection, let's say k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Get cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers = pd.DataFrame(centers, columns=features)
cluster_centers['Cluster'] = range(optimal_k)

# Customer count by cluster
cluster_counts = customer_data['Cluster'].value_counts().sort_index()
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}: {count} customers ({count/customer_data.shape[0]:.1%})")


cluster_means = customer_data.groupby('Cluster')[features].mean()


# Visualize clusters with PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
customer_data['PCA1'] = pca_result[:, 0]
customer_data['PCA2'] = pca_result[:, 1]

# Plot clusters using PCA
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'v']

for i in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == i]
    plt.scatter(
        cluster_data['PCA1'], 
        cluster_data['PCA2'],
        s=50, c=colors[i % len(colors)], marker=markers[i % len(markers)], alpha=0.7,
        label=f'Cluster {i}'
    )

# Plot PCA-transformed cluster centers
pca_centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    pca_centers[:, 0], pca_centers[:, 1],
    s=200, c='yellow', marker='.', edgecolors='black',
    label='Centroids'
)

plt.title('Customer Segmentation (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate(
    f'Explained variance: {sum(pca.explained_variance_ratio_):.2f}',
    xy=(0.05, 0.95), xycoords='axes fraction'
)
plt.tight_layout()
plt.show()

