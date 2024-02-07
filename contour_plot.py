from FCM import fuzzy_c_means_custom_distance, fuzzy_membership
from transform import translation_non_negatives
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin_min


def main():
    # Generate sample data with three clusters
    data, true_labels = make_blobs(n_samples=3, centers=3, random_state=42)
    data,_ = translation_non_negatives(data)
    # Apply Fuzzy C-Means clustering with custom distance
    num_clusters =3 
    m = 2
    cluster_assignments, cluster_centers = fuzzy_c_means_custom_distance(data, num_clusters, m)

    # Create a meshgrid for contour plot
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

    # Compute membership values for each point in the meshgrid
    meshgrid_data = np.c_[xx.ravel(), yy.ravel()]
    membership_values = fuzzy_membership(meshgrid_data, cluster_centers, m)

    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, membership_values[:, 0].reshape(xx.shape), cmap="viridis", alpha=0.5)
    plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap="viridis", edgecolors="k", marker="o", s=100)
    plt.title("Contour Plot of FCM Memberships (Cluster 1) with Custom Distance")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(contour, label="Membership Value")
    plt.show()

if __name__=="__main__":
    main()
