import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Download the Spotify dataset from Kaggle
import kagglehub
try:
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    print('Data downloaded to:', path)
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit()

# Load the dataset into a pandas DataFrame
try:
    df = pd.read_csv(path + '/SpotifyFeatures.csv')
    print('Data loaded')
    print(df.head())
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Define the musical features we want to analyze
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Check if any of our desired features are missing from the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print('Missing features:', missing_features)
    exit()

# Extract the feature values and remove any rows with missing data
X = df[features].dropna().values

# Standardize the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Data scaled')

# Perform K-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Perform PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print('PCA completed')

# Create the scatter plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Cluster ID')  # Add a label to the colorbar
plt.title('PCA of Spotify Songs with K-means Clustering')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('spotify_clusters.png')  # Save the plot as a PNG file
plt.close() # Close the plot to free memory

df_clustered = df.dropna(subset=features.copy())
df_clustered['Cluster'] = clusters
df_clustered['pca1'] = X_pca[:, 0]
df_clustered['pca2'] = X_pca[:, 1]

# Handle NaN values in the 'track_name' column before performing the string search
song = df_clustered[df_clustered['track_name'].fillna('').str.contains('Enter Sandman', case=False)]
if not song.empty:
    print('Song found:', song[['track_name', 'Cluster', 'pca1', 'pca2']])
else:
    print('Song not found in the dataset.')
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.scatter(song.iloc[0]['pca1'], song.iloc[0]['pca2'], color='red', s=100, label=song.iloc[0]['track_name'], edgecolor='black')
plt.title('PCA of Spotify Songs with K-means Clustering')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.savefig('spotify_clusters_with_song.png')  # Save the plot as a PNG file
plt.close() # Close the plot to free memory