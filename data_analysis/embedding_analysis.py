import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import logging

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
        self.resume_embeddings = defaultdict(dict)
        
    def load_embeddings(self, resume_base_dir: str, job_dir: str):
        """Load all embeddings from the directories"""
        logger.info("Loading embeddings...")
        
        # Load job embeddings
        for file in tqdm(os.listdir(job_dir), desc="Loading job embeddings"):
            if file.endswith('.npy'):
                embedding_path = os.path.join(job_dir, file)
                self.job_embeddings[file] = np.load(embedding_path)
        
        # Load resume embeddings by field
        for field in os.listdir(resume_base_dir):
            field_path = os.path.join(resume_base_dir, field)
            if os.path.isdir(field_path):
                logger.info(f"Loading embeddings for field: {field}")
                for file in tqdm(os.listdir(field_path), desc=f"Loading {field} embeddings"):
                    if file.endswith('.npy'):
                        embedding_path = os.path.join(field_path, file)
                        self.resume_embeddings[field][file] = np.load(embedding_path)
                
                # Calculate field embedding (average of all resumes in the field)
                field_vectors = list(self.resume_embeddings[field].values())
                self.field_embeddings[field] = np.mean(field_vectors, axis=0)
    
    def find_matching_jobs(self, field: str, top_n: int = 10):
        """Find top N matching jobs for a given field"""
        field_embedding = self.field_embeddings[field]
        
        # Calculate similarities
        similarities = {}
        for job_id, job_embedding in self.job_embeddings.items():
            similarity = cosine_similarity(
                field_embedding.reshape(1, -1),
                job_embedding.reshape(1, -1)
            )[0][0]
            similarities[job_id] = similarity
        
        # Sort and get top N
        sorted_jobs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_jobs[:top_n]
    
    def analyze_field_job_matches(self, output_dir: str, top_n: int = 10):
        """Analyze matches between fields and jobs"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        for field in self.field_embeddings.keys():
            logger.info(f"Analyzing matches for field: {field}")
            
            # Get top matching jobs
            matching_jobs = self.find_matching_jobs(field, top_n)
            
            # Store results
            results[field] = {
                'top_matches': [
                    {
                        'job_id': job_id.replace('.npy', ''),
                        'similarity_score': float(score)
                    }
                    for job_id, score in matching_jobs
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
    resume_base_dir = "data/resume/format_embedding"
    job_dir = "data/job_descriptions/b1ade_embedding"
    output_dir = "data/analysis_results"
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer()
    
    # Load embeddings
    analyzer.load_embeddings(resume_base_dir, job_dir)
    
    # Perform analysis
    results = analyzer.analyze_field_job_matches(output_dir,top_n=1000)
    
    logger.info("Analysis completed!")

if __name__ == "__main__":
    main()