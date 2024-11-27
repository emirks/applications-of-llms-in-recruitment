import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('matcher_dataset.log')
    ]
)
logger = logging.getLogger(__name__)

class MatcherDatasetCreator:
    def __init__(self, top_n: int = 1000):
        self.top_n = top_n
        self.matches_data = defaultdict(list)
        
    def load_field_matches(self, results_file: str) -> None:
        """Load matching results from the analysis output"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for field, field_data in data.items():
                # Get top positive matches
                positive_matches = sorted(
                    field_data['top_matches'],
                    key=lambda x: x['similarity_score'],
                    reverse=True
                )[:self.top_n]
                
                # Get top negative matches
                negative_matches = sorted(
                    field_data['least_matches'],
                    key=lambda x: x['similarity_score']
                )[:self.top_n]
                
                # Store matches with their type and score
                for match in positive_matches:
                    self.matches_data[field].append({
                        'job_id': match['job_id'],
                        'similarity_score': match['similarity_score'],
                        'match_type': 'positive'
                    })
                
                for match in negative_matches:
                    self.matches_data[field].append({
                        'job_id': match['job_id'],
                        'similarity_score': match['similarity_score'],
                        'match_type': 'negative'
                    })
                
            logger.info(f"Loaded matches for {len(self.matches_data)} fields")
            
        except Exception as e:
            logger.error(f"Error loading matches file: {e}")
            raise

    def create_dataset(self) -> pd.DataFrame:
        """Create a unified dataset from the loaded matches"""
        dataset_rows = []
        
        for field, matches in tqdm(self.matches_data.items(), desc="Creating dataset"):
            for match in matches:
                dataset_rows.append({
                    'field': field,
                    'job_id': match['job_id'],
                    'similarity_score': match['similarity_score'],
                    'match_type': match['match_type'],
                    'is_match': 1 if match['match_type'] == 'positive' else 0
                })
        
        df = pd.DataFrame(dataset_rows)
        
        # Add some basic statistics
        total_pairs = len(df)
        positive_pairs = len(df[df['is_match'] == 1])
        negative_pairs = len(df[df['is_match'] == 0])
        
        logger.info(f"Dataset created with {total_pairs} total pairs")
        logger.info(f"Positive pairs: {positive_pairs}")
        logger.info(f"Negative pairs: {negative_pairs}")
        
        return df

    def save_dataset(self, df: pd.DataFrame, output_dir: str, model_name: str) -> None:
        """Save the dataset to CSV and provide statistics"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        output_path = os.path.join(output_dir, f"{model_name}_matcher_dataset.csv")
        df.to_csv(output_path, index=False)
        
        # Convert groupby results to a more JSON-friendly format
        field_match_counts = {}
        for (field, match_type), count in df.groupby(['field', 'match_type']).size().items():
            if field not in field_match_counts:
                field_match_counts[field] = {}
            field_match_counts[field][match_type] = int(count)
        
        # Create and save dataset statistics
        stats = {
            'total_pairs': len(df),
            'unique_jobs': df['job_id'].nunique(),
            'unique_fields': df['field'].nunique(),
            'positive_pairs': len(df[df['is_match'] == 1]),
            'negative_pairs': len(df[df['is_match'] == 0]),
            'avg_similarity_positive': float(df[df['is_match'] == 1]['similarity_score'].mean()),
            'avg_similarity_negative': float(df[df['is_match'] == 0]['similarity_score'].mean()),
            'fields_distribution': df['field'].value_counts().to_dict(),
            'match_type_by_field': field_match_counts  # Using the converted format
        }
        
        stats_path = os.path.join(output_dir, f"{model_name}_dataset_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Statistics saved to: {stats_path}")

    def copy_job_texts(self, df: pd.DataFrame, model_name: str, output_dir: str) -> None:
        """Copy job description text files for the jobs in the dataset"""
        source_dir = os.path.join("data/job_descriptions/format_txt")
        target_dir = os.path.join(output_dir, model_name, "format_txt")
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Get unique job IDs from the dataset
        unique_jobs = df['job_id'].unique()
        
        copied_count = 0
        missing_count = 0
        
        logger.info(f"Copying job description texts for {len(unique_jobs)} unique jobs")
        for job_id in tqdm(unique_jobs, desc="Copying job texts"):
            source_file = os.path.join(source_dir, f"{job_id}.txt")
            target_file = os.path.join(target_dir, f"{job_id}.txt")
            
            try:
                if os.path.exists(source_file):
                    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(source_file, target_file)
                    copied_count += 1
                else:
                    logger.warning(f"Source file not found: {source_file}")
                    missing_count += 1
            except Exception as e:
                logger.error(f"Error copying file {job_id}: {str(e)}")
                missing_count += 1
        
        logger.info(f"Copied {copied_count} job description files")
        if missing_count > 0:
            logger.warning(f"Failed to copy {missing_count} job description files")

def main():
    # Configure paths
    base_dir = "data/analysis_results/model_embedding"
    output_dir = "data/matcher_dataset"
    models = ["b1ade", "sfr"]
    
    for model in models:
        logger.info(f"Creating dataset for {model} model")
        
        # Initialize dataset creator
        creator = MatcherDatasetCreator(top_n=1000)
        
        # Load matches from analysis results
        results_file = os.path.join(base_dir, model, "field_job_matches.json")
        creator.load_field_matches(results_file)
        
        # Create dataset
        dataset_df = creator.create_dataset()
        
        # Save dataset and statistics
        creator.save_dataset(dataset_df, output_dir, model)
        
        # Copy job description text files
        creator.copy_job_texts(dataset_df, model, output_dir)
        
        logger.info(f"Completed dataset creation for {model} model")

if __name__ == "__main__":
    main() 