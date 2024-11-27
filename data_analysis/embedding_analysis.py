import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import faiss
import logging
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embedding_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingAnalyzer:
    def __init__(self):
        self.field_embeddings = {}
        self.job_embeddings = {}
        self.job_ids = []  # To maintain mapping between index and job ID
        self.resume_embeddings = defaultdict(dict)
        self.index = None
        self.dimension = None
    
    def load_embeddings(self, resume_base_dir: str, job_dir: str):
        """Load all embeddings and build FAISS index"""
        logger.info("Loading embeddings...")
        
        # Validate directories
        if not os.path.exists(resume_base_dir) or not os.path.exists(job_dir):
            raise ValueError("Directory does not exist")
        
        # Load job embeddings first to determine dimension
        job_files = [f for f in os.listdir(job_dir) if f.endswith('.npy')]
        if not job_files:
            raise ValueError(f"No embedding files found in job directory: {job_dir}")
        
        # Load all job embeddings first to ensure consistent dimensions
        job_embeddings_list = []
        for file in tqdm(job_files, desc="Loading job embeddings"):
            embedding_path = os.path.join(job_dir, file)
            try:
                embedding = np.load(embedding_path)
                # Ensure 1D array and normalize
                embedding = embedding.flatten()  # Convert to 1D if needed
                embedding = embedding / np.linalg.norm(embedding)
                job_embeddings_list.append(embedding)
                self.job_ids.append(file)
            except Exception as e:
                logger.error(f"Error loading job embedding {file}: {str(e)}")
                continue
        
        if not job_embeddings_list:
            raise ValueError("No valid job embeddings loaded")
        
        # Stack all embeddings and initialize FAISS index
        job_embeddings_array = np.vstack(job_embeddings_list)
        self.dimension = job_embeddings_array.shape[1]
        logger.info(f"Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index with correct dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(job_embeddings_array)
        
        # Load resume embeddings by field
        fields = [d for d in os.listdir(resume_base_dir) 
                 if os.path.isdir(os.path.join(resume_base_dir, d))]
        
        for field in fields:
            field_path = os.path.join(resume_base_dir, field)
            if os.path.isdir(field_path):
                logger.info(f"Loading embeddings for field: {field}")
                field_files = [f for f in os.listdir(field_path) if f.endswith('.npy')]
                
                field_vectors = []
                for file in tqdm(field_files, desc=f"Loading {field} embeddings"):
                    embedding_path = os.path.join(field_path, file)
                    try:
                        embedding = np.load(embedding_path)
                        # Normalize for cosine similarity
                        embedding = embedding / np.linalg.norm(embedding)
                        self.resume_embeddings[field][file] = embedding
                        field_vectors.append(embedding)
                    except Exception as e:
                        logger.error(f"Error loading resume embedding {file}: {str(e)}")
                        continue
                
                # Calculate field embedding (average of all resumes in the field)
                if field_vectors:
                    field_embedding = np.mean(field_vectors, axis=0)
                    # Normalize the mean vector
                    self.field_embeddings[field] = field_embedding / np.linalg.norm(field_embedding)
                else:
                    logger.warning(f"No valid embeddings found for field: {field}")
        
        logger.info(f"Loaded {len(self.job_ids)} job embeddings")
        logger.info(f"Loaded resume embeddings for {len(self.field_embeddings)} fields")
    
    def find_matching_jobs(self, field: str, top_n: int = 10, include_negative: bool = True) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Find top N matching and bottom N non-matching jobs for a given field using FAISS"""
        field_embedding = self.field_embeddings[field].reshape(1, -1)
        
        # Get all similarities to find both top and bottom matches
        k = len(self.job_ids) if include_negative else top_n
        similarities, indices = self.index.search(field_embedding, k)
        
        # Convert to list of (job_id, similarity) tuples
        all_matches = [
            (self.job_ids[idx], float(sim))
            for sim, idx in zip(similarities[0], indices[0])
        ]
        
        # Split into positive and negative matches
        positive_matches = all_matches[:top_n]
        negative_matches = all_matches[-top_n:] if include_negative else []
        
        return positive_matches, negative_matches
    
    def analyze_field_job_matches(self, output_dir: str, top_n: int = 10):
        """Analyze matches between fields and jobs, including both positive and negative matches"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        for field in self.field_embeddings.keys():
            logger.info(f"Analyzing matches for field: {field}")
            
            # Get top matching and least matching jobs
            matching_jobs, negative_matches = self.find_matching_jobs(field, top_n)
            
            # Store results
            results[field] = {
                'top_matches': [
                    {
                        'job_id': job_id.replace('.npy', ''),
                        'similarity_score': float(score)
                    }
                    for job_id, score in matching_jobs
                ],
                'least_matches': [
                    {
                        'job_id': job_id.replace('.npy', ''),
                        'similarity_score': float(score)
                    }
                    for job_id, score in negative_matches
                ]
            }
        
        # Save results
        import json
        output_path = os.path.join(output_dir, 'field_job_matches.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to: {output_path}")
        return results

def main():
    # Configure directories
    base_dir = "data/analysis_results/model_embedding"
    models = ["b1ade", "sfr"]
    
    for model in models:
        logger.info(f"Analyzing embeddings for {model} model")
        
        resume_base_dir = f"data/resume/{model}_embedding"
        job_dir = f"data/job_descriptions/{model}_embedding"
        output_dir = os.path.join(base_dir, model)
        
        # Initialize analyzer
        analyzer = EmbeddingAnalyzer()
        
        # Load embeddings
        analyzer.load_embeddings(resume_base_dir, job_dir)
        
        # Perform analysis
        results = analyzer.analyze_field_job_matches(output_dir, top_n=10000)
        
        logger.info(f"Analysis completed for {model} model!")

if __name__ == "__main__":
    main()