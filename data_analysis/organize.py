import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_files_by_cluster():
    """
    Read cluster analysis results and organize txt files into cluster-specific folders
    """
    # Configure paths
    cluster_analysis_file = "data/job_descriptions/clustering_results/cluster_analysis.txt"
    source_txt_dir = "data/job_descriptions/format_txt"
    output_base_dir = "data/job_descriptions/clustered_txt"
    
    # Create base output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store cluster assignments
    cluster_files = {}
    
    # Read cluster analysis file and extract file assignments
    print("Reading cluster analysis file...")
    current_cluster = None
    with open(cluster_analysis_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Cluster"):
                current_cluster = line.split()[1].rstrip(':')
                cluster_files[current_cluster] = []
            elif line.startswith("Sample files:"):
                files = line.replace("Sample files:", "").strip().split(', ')
                cluster_files[current_cluster].extend([f.replace('.npy', '.txt') for f in files])
    
    # Create cluster directories and copy files
    print("Organizing files into clusters...")
    for cluster, files in tqdm(cluster_files.items()):
        # Create cluster directory
        cluster_dir = os.path.join(output_base_dir, f"cluster_{cluster}")
        Path(cluster_dir).mkdir(exist_ok=True)
        
        # Copy files to cluster directory
        for filename in files:
            source_path = os.path.join(source_txt_dir, filename)
            dest_path = os.path.join(cluster_dir, filename)
            
            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
                continue

    print(f"Files organized successfully in {output_base_dir}")

if __name__ == "__main__":
    organize_files_by_cluster()