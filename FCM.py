from transform import translation_non_negatives
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

# def custom_distance(a, b):
#    return np.sum((a - b) * np.power(np.log(a) - np.log(b), 2))

def custom_distance(a, b):
    epsilon = 1e-10  # Small constant to avoid zero or negative values
    return np.sum((a - b) * np.power(np.log(a + epsilon) - np.log(b + epsilon), 2))

def fuzzy_membership(data, cluster_centers, m):
    distances = np.array([[custom_distance(data[i], cluster_centers[j]) for j in range(cluster_centers.shape[0])] for i in range(data.shape[0])])
    distances = np.fmax(distances, np.finfo(np.float64).eps)
    memberships = 1 / np.power(distances, 2 / (m - 1))
    memberships /= np.sum(memberships, axis=1, keepdims=True)
    return memberships

def fuzzy_c_means_custom_distance(data, num_clusters, m, max_iters=100, tol=1e-4):
    n_samples, n_features = data.shape
    U = np.random.rand(n_samples, num_clusters)
    U /= np.sum(U, axis=1, keepdims=True)

    for iteration in range(max_iters):
        # Update cluster centroids
        V = np.dot(U.T, data) / np.sum(U, axis=0, keepdims=True).T

        # Update fuzzy memberships
        memberships = fuzzy_membership(data, V, m)

        # Check convergence
        if np.linalg.norm(U - memberships) < tol:
            break

        U = memberships

    return np.argmax(U, axis=1), V

def main():
    # Example usage
    # Suppose 'translated_data' is your translated data
    your_data, true_labels = make_blobs(n_samples=3, centers=3, random_state=42)
    #your_data = np.array([[1, 2, 3], [-2, 5, 1], [0, 1, -4]])
    translated_data,_ = translation_non_negatives(your_data)
    num_clusters = 3
    m = 2  # Fuzziness parameter

    # Apply Fuzzy C-Means clustering with custom distance
    cluster_assignments, cluster_centers = fuzzy_c_means_custom_distance(translated_data, num_clusters, m)

    # Display the results
    print("Cluster Assignments:\n", cluster_assignments)
    print("\nCluster Centers:\n", cluster_centers)


if __name__=="__main__":
    main()
