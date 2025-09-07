import pandas as pd
import numpy as np

data = {
    "Closed Meetings": [4, 1, 5, 0, 3, 2, 6, 1, 3, 2],
    "Closed Calls": [7, 2, 10, 1, 5, 3, 12, 0, 4, 6],
    "Email Received": [15, 5, 20, 3, 12, 8, 22, 2, 14, 10],
    "Email Sent": [10, 8, 18, 4, 9, 7, 19, 3, 12, 8]
}
df = pd.DataFrame(data)
X = df.to_numpy()  

def silhouette_score(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)

    # Compute full pairwise distance matrix
    dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        label_i = labels[i]

        # Intra-cluster distances (a(i))
        mask_same = (labels == label_i)
        mask_same[i] = False  # exclude self
        a_i = np.mean(dist_matrix[i, mask_same]) if np.any(mask_same) else 0

        # Inter-cluster distances (b(i))
        b_i = np.inf
        for other_label in unique_labels:
            if other_label != label_i:
                mask_other = (labels == other_label)
                if np.any(mask_other):
                    b_i = min(b_i, np.mean(dist_matrix[i, mask_other]))

        denom = max(a_i, b_i)
        silhouette_vals[i] = 0 if denom == 0 else (b_i - a_i) / denom

    return np.mean(silhouette_vals)

def kmeans_init(X, k, random_seed=0):
    np.random.seed(random_seed)
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    idx = np.random.choice(n_samples)
    centroids[0] = X[idx]
    distances = np.full(n_samples, np.inf)
    for i in range(1, k):
        for j in range(n_samples):
            dist = np.linalg.norm(X[j] - centroids[i-1])
            if dist < distances[j]:
                distances[j] = dist
        probabilities = distances ** 2
        probabilities /= probabilities.sum()
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[next_idx]
    return centroids

def run_kmeans(X, k, max_iterations=100, random_seed=0):
    centroids = kmeans_init(X, k, random_seed)

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        assignments = []
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in centroids]
            assigned_cluster = np.argmin(distances)
            assignments.append(assigned_cluster)
        assignments = np.array(assignments)

        new_centroids = []
        for cluster_id in range(k):
            points_in_cluster = X[assignments == cluster_id]
            if len(points_in_cluster) > 0:
                new_centroid = points_in_cluster.mean(axis=0)
            else:
                new_centroid = X[np.random.choice(X.shape[0])]
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            print("Convergence reached.")
            break

        centroids = new_centroids

    return assignments, centroids

best_k = None
best_score = -1
best_assignments = None
best_centroids = None

for k in range(2, 7):  
    assignments, centroids = run_kmeans(X, k)
    score = silhouette_score(X, assignments)
    print(f"k={k} silhouette score={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
        best_assignments = assignments
        best_centroids = centroids

print(f"\nBest k value by silhouette score: {best_k} with score {best_score:.4f}")

df['Cluster'] = best_assignments

feature_cols = ["Closed Meetings", "Closed Calls", "Email Received", "Email Sent"]
cluster_means = df.groupby('Cluster')[feature_cols].mean()
print(cluster_means)

normalized_means = cluster_means.copy()
for col in cluster_means.columns:
    col_mean = cluster_means[col].mean()
    col_std = cluster_means[col].std()
    normalized_means[col] = (cluster_means[col] - col_mean) / col_std

print(normalized_means)

personas = []

for i, row in normalized_means.iterrows():
    if all(row > 0.5):  
        personas.append("Engaged Decision-Maker")
    elif all(row < -0.5):  
        personas.append("Passive Observer")
    elif row['Closed Meetings'] > 0.5 and row['Closed Calls'] > 0.5:
        personas.append("Warm Lead")
    elif row['Email Received'] > 0.5 and row['Email Sent'] > 0.5:
        personas.append("Email Engager")
    else:
        personas.append("Mixed Engagement")

persona_mapping = {cluster_id: label for cluster_id, label in zip(normalized_means.index, personas)}
df['Persona'] = df['Cluster'].map(persona_mapping)

print(df)