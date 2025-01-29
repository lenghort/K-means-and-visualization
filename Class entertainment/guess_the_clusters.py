import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_path = 'points.csv'
points= []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        points.append([float(row[0]), float(row[1])])

points = np.array(points)

num_clusters = 3

kmeans = KMeans(num_clusters, random_state=0)
kmeans.fit(points)
centroids_kmeans = kmeans.cluster_centers_
labels = kmeans.labels_

x_min, y_min = points.min(axis=0)
x_max, y_max = points.max(axis=0)

plt.figure(figsize=(8, 6))

for i in range(num_clusters):
    plt.scatter(points[labels == i][:, 0], points[labels == i][:, 1], label=f'Cluster {i+1}')

plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], color='red', marker='x', s=100, label='K-means Centroids')

plt.axvline(x=x_min, color='green', linestyle='--', label='X Bounds')
plt.axvline(x=x_max, color='green', linestyle='--')
plt.axhline(y=y_min, color='purple', linestyle='--', label='Y Bounds')
plt.axhline(y=y_max, color='purple', linestyle='--')

plt.title('Class entertainment - Guess the clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.savefig("class entertain.png")

plt.show()