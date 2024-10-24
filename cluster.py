import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#read in the data
data = pd.read_csv('dataset 2.csv')
print(data.head())
# clean data
data = data.dropna(subset=['danceability', 'energy', 'valence', 'tempo'])
data = data.drop_duplicates(subset='track_id')

#what the clustering is based off of 
features = data[['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'speechiness']]

#store sum of squared errors
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)

#use the plot to determine a good k based on the elbow method
plt.plot(range(1, 11), sse)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()

# Assuming 'x' is your feature matrix and 'df' is your DataFrame containing the original data
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize KMeans with 3 clusters
kmeans.fit(features) # Fit the model to your feature data

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the DataFrame
data['cluster'] = labels

# Separate the DataFrame into individual clusters
cluster_0 = data[data['cluster'] == 0]
cluster_1 = data[data['cluster'] == 1]
cluster_2 = data[data['cluster'] == 2]

# Print information about each cluster
print("Cluster 0:")
print(cluster_0.head())  # Print the first few rows of Cluster 0

print("\nCluster 1:")
print(cluster_1.head())  # Print the first few rows of Cluster 1

print("\nCluster 2:")
print(cluster_2.head())  # Print the first few rows of Cluster 2

cluster_summary = data.groupby('cluster')[['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'speechiness']].mean()
print(f"cluster summary: {cluster_summary}")
