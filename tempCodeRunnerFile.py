n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
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