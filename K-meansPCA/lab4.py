import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename)
    return data

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def assign_to_centroids(data, centroids):
    num_data_points = data.shape[0]
    num_centroids = centroids.shape[0]
    assignments = np.zeros(num_data_points, dtype=int)      

    for i in range(num_data_points):
        min_distance = float('inf')
        closest_centroid = -1   

        for j in range(num_centroids):
            distance = np.linalg.norm(data[i] - centroids[j])

            if distance < min_distance:
                min_distance = distance
                closest_centroid = j    

        assignments[i] = closest_centroid
    return assignments

def update_centroids(data, assignments, k):
    num_features = data.shape[1]
    centroids = np.zeros((k, num_features))

    for i in range(k):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)

    return centroids

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        assignments = assign_to_centroids(data, centroids)
        new_centroids = update_centroids(data, assignments, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, assignments

def plot_results(data, centroids, assignments):
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Load data
data = load_data('./data_kmeans.txt')

# Cluster data using K-means
k = 3
centroids, assignments = k_means(data, k)

# Plot training results
plot_results(data, centroids, assignments)

# Test data
test_data = np.array([[1.5, 4.5],
                      [5.5, 4.5],
                      [6.5, 3.5]])

# Assign test data to centroids
test_assignments = assign_to_centroids(test_data, centroids)

# Plot test results
plot_results(test_data, centroids, test_assignments)