import numpy as np
import random
import matplotlib.pyplot as plt
import time

initial_centroids = np.array([(2, 3), (7, 8), (3, 9)])  
num_points = 300  

def generate_points(centroid, num_points, std_dev):
    cx, cy = centroid
    points = []
    for _ in range(num_points):
        x = random.gauss(cx, std_dev)  
        y = random.gauss(cy, std_dev)  
        points.append((x, y))
    return np.array(points)

all_points = []
for centroid in initial_centroids:
    all_points.extend(generate_points(centroid, 300, std_dev=1.0))
all_points = np.array(all_points) 

def k_means(points, centroids, max_iterations=500):
    for _ in range(max_iterations):
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2) 
        closest_centroids = np.argmin(distances, axis=1) 

        new_centroids = np.array([points[closest_centroids == i].mean(axis=0) for i in range(len(centroids))])

        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return centroids, closest_centroids

start_time = time.time()

final_centroids, final_clusters = k_means(all_points, initial_centroids)

end_time = time.time()
print(f"K-means completed in {end_time - start_time:.4f} seconds")

plt.figure(figsize=(8, 6))

for i in range(len(final_centroids)):
    cluster_points = all_points[final_clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}", s=10, alpha=0.7)

plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', s=100, label="Centroid", marker='X')

plt.title("K-means Clusters and Centroids")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.savefig("k_means_clusters_vectorized.png") 
plt.show()