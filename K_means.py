import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.random.rand(400, 2)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], s=20, cmap='summer')
plt.show()

kmeans = KMeans(n_clusters=3,max_iter=100)

kmeans.fit(X)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], s=20, c=kmeans.labels_, cmap='summer')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=50, c="r",alpha=0.9, marker='x')
plt.show()