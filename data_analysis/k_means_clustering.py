import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_embeddings(embedding_dir: str):
    """Load all embeddings from the directory"""
    embeddings = []
    filenames = []
    
    print("Loading embeddings...")
    for file in tqdm(os.listdir(embedding_dir)):
        if file.endswith('.npy'):
            embedding_path = os.path.join(embedding_dir, file)
            embedding = np.load(embedding_path)
            embeddings.append(embedding.flatten())  # Flatten in case of 2D arrays
            filenames.append(file)
    
    return np.array(embeddings), filenames

def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10):
    """Perform k-means clustering"""
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, output_dir: str):
    """Visualize clusters using PCA"""
    print("Reducing dimensionality for visualization...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6)
    plt.title('Job Description Clusters (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'clusters_visualization.png'))
    plt.close()

def analyze_clusters(labels: np.ndarray, filenames: list, kmeans: KMeans, output_dir: str):
    """Analyze cluster distributions and save results"""
    # Count documents per cluster
    cluster_counts = defaultdict(int)
    cluster_files = defaultdict(list)
    
    for label, filename in zip(labels, filenames):
        cluster_counts[label] += 1
        cluster_files[label].append(filename)
    
    # Save analysis results
    with open(os.path.join(output_dir, 'cluster_analysis.txt'), 'w') as f:
        f.write("Cluster Analysis Results\n")
        f.write("=======================\n\n")
        
        for cluster in sorted(cluster_counts.keys()):
            f.write(f"Cluster {cluster}:\n")
            f.write(f"Number of documents: {cluster_counts[cluster]}\n")
            f.write(f"Sample files: {', '.join(cluster_files[cluster][:5])}\n\n")

def main():
    # Configure directories
    embedding_dir = "data/job_descriptions/format_embedding"
    output_dir = "data/job_descriptions/clustering_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    embeddings, filenames = load_embeddings(embedding_dir)
    
    # Perform clustering
    labels, kmeans = perform_clustering(embeddings)
    
    # Visualize results
    visualize_clusters(embeddings, labels, output_dir)
    
    # Analyze and save cluster information
    analyze_clusters(labels, filenames, kmeans, output_dir)
    
    print(f"Clustering analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()