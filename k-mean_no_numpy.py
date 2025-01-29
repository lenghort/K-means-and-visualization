import numpy as np
import matplotlib.pyplot as plt
import random
import time

initial_centroids = [(2, 3), (7, 8), (3, 9)] 
num_points = 300

def generate_points(centroid, num_points, std_dev):
    cx, cy = centroid
    points = []
    for _ in range(num_points):
        x = random.gauss(cx, std_dev)  
        y = random.gauss(cy, std_dev)
        points.append((x, y))
    return points

all_points = []
for centroid in initial_centroids:
    all_points.extend(generate_points(centroid, 300, std_dev=1.0))  

plt.figure(figsize=(8, 6))
for centroid in initial_centroids:
    plt.scatter(*centroid, color='red', s=100, label="Centroid") 

def k_means(points, centroids, max_iterations = 500):
    for i in range(max_iterations):
        clusters = [[] for i in range(len(centroids))]
        
        for point in points:
            distances = []
            for centroid in centroids:
                dist = ((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2) ** 0.5 
                distances.append(dist)
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:  
                new_x = sum([p[0] for p in cluster]) / len(cluster)
                new_y = sum([p[1] for p in cluster]) / len(cluster)
                new_centroids.append((new_x, new_y))
            else:
                new_centroids.append(centroids[clusters.index(cluster)])

        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids, clusters

start_time = time.time()
final_centroids, final_clusters = k_means(all_points, initial_centroids)
end_time = time.time()

print(f"K-means completed in {end_time - start_time:.4f} seconds")

plt.figure(figsize=(8, 6))

for i, cluster in enumerate(final_clusters):
    cluster_x = [p[0] for p in cluster]
    cluster_y = [p[1] for p in cluster]
    plt.scatter(cluster_x, cluster_y, label=f"Cluster {i+1}", s=10, alpha=0.7)

for centroid in final_centroids:
    plt.scatter(*centroid, color='red', s=100, label="Centroid", marker='X')

plt.title("Random Points Around Guessed Centroids")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.savefig("guessed_centroids_plain.png") 
plt.show()

    