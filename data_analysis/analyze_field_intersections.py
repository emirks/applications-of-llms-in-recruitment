import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('field_intersections.log')
    ]
)
logger = logging.getLogger(__name__)

class FieldIntersectionAnalyzer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.raw_matches = defaultdict(list)
        self.field_jobs: Dict[str, Set[str]] = defaultdict(set)
        self.intersection_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.job_similarity_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def load_analysis_results(self, results_file: str):
        """Load the analysis results from the JSON file"""
        logger.info(f"Loading analysis results from {results_file}")
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        # Store raw matches first
        for field, field_data in data.items():
            self.raw_matches[field] = sorted(
                field_data['top_matches'],
                key=lambda x: x['similarity_score'],
                reverse=True
            )
        
        logger.info(f"Loaded raw data for {len(self.raw_matches)} fields")
    
    def filter_top_n_matches(self, n: int):
        """Filter to only include top N matches for each field"""
        self.field_jobs.clear()
        self.job_similarity_scores.clear()
        
        for field, matches in self.raw_matches.items():
            for match in matches[:n]:
                job_id = match['job_id']
                similarity = match['similarity_score']
                self.field_jobs[field].add(job_id)
                self.job_similarity_scores[job_id][field] = similarity
    
    def analyze_intersections_for_n(self, n: int):
        """Analyze intersections for top N matches"""
        self.filter_top_n_matches(n)
        return self.analyze_intersections()
    
    def analyze_multiple_thresholds(self, start_n: int, step: int, min_n: int):
        """Analyze intersections across different top N thresholds"""
        results = {}
        for n in range(start_n, min_n - 1, -step):
            logger.info(f"Analyzing top {n} matches")
            intersection_matrix = self.analyze_intersections_for_n(n)
            cross_domain_jobs = self.find_cross_domain_jobs(min_fields=2)
            
            results[n] = {
                'intersection_matrix': intersection_matrix,
                'cross_domain_count': len(cross_domain_jobs),
                'total_unique_jobs': len(set.union(*self.field_jobs.values()) if self.field_jobs else set())
            }
        
        return results
    
    def visualize_threshold_results(self, results: Dict[int, dict], output_dir: str):
        """Create visualizations for threshold analysis"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot trends
        thresholds = sorted(results.keys())
        cross_domain_counts = [results[n]['cross_domain_count'] for n in thresholds]
        unique_job_counts = [results[n]['total_unique_jobs'] for n in thresholds]
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, cross_domain_counts, 'b-', label='Cross-domain Jobs')
        plt.plot(thresholds, unique_job_counts, 'r-', label='Total Unique Jobs')
        plt.xlabel('Top N Matches')
        plt.ylabel('Count')
        plt.title('Job Counts vs. Match Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
        plt.close()
        
        # Create heatmaps for selected thresholds
        for n in thresholds:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                results[n]['intersection_matrix'],
                annot=True,
                fmt='d',
                cmap='YlOrRd',
                square=True
            )
            plt.title(f'Field Intersections (Top {n} Matches)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'intersections_top_{n}.png'))
            plt.close()
    
    def generate_threshold_report(self, results: Dict[int, dict], output_dir: str):
        """Generate a report comparing different thresholds"""
        report_path = os.path.join(output_dir, 'threshold_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("Threshold Analysis Report\n")
            f.write("=======================\n\n")
            
            for n in sorted(results.keys()):
                data = results[n]
                f.write(f"Top {n} Matches Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total unique jobs: {data['total_unique_jobs']}\n")
                f.write(f"Cross-domain jobs: {data['cross_domain_count']}\n")
                f.write(f"Cross-domain percentage: {(data['cross_domain_count'] / data['total_unique_jobs'] * 100):.2f}%\n\n")
                
                # Add intersection statistics
                matrix = data['intersection_matrix']
                # Convert pandas DataFrame to numpy array for calculations
                matrix_values = matrix.values
                non_zero_mask = matrix_values > 0
                if non_zero_mask.any():
                    avg_intersection = float(matrix_values[non_zero_mask].mean())
                    max_intersection = float(matrix_values[non_zero_mask].max())
                else:
                    avg_intersection = 0.0
                    max_intersection = 0.0
                
                f.write(f"Average intersection size: {avg_intersection:.2f}\n")
                f.write(f"Maximum intersection size: {max_intersection:.0f}\n\n")
    
    def analyze_intersections(self):
        """Analyze intersections between different fields"""
        fields = list(self.field_jobs.keys())
        intersection_matrix = pd.DataFrame(0, index=fields, columns=fields)
        
        # Calculate intersections
        for i, field1 in enumerate(fields):
            for field2 in fields[i:]:
                intersection = self.field_jobs[field1] & self.field_jobs[field2]
                self.intersection_counts[(field1, field2)] = len(intersection)
                intersection_matrix.loc[field1, field2] = len(intersection)
                intersection_matrix.loc[field2, field1] = len(intersection)
        
        return intersection_matrix
    
    def find_cross_domain_jobs(self, min_fields: int = 2) -> Dict[str, List[str]]:
        """Find jobs that appear in multiple fields"""
        job_fields: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Collect all fields for each job
        for field, jobs in self.field_jobs.items():
            for job_id in jobs:
                similarity = self.job_similarity_scores[job_id][field]
                job_fields[job_id].append((field, similarity))
        
        # Filter for jobs that appear in multiple fields
        cross_domain_jobs = {
            job_id: sorted(fields, key=lambda x: x[1], reverse=True)
            for job_id, fields in job_fields.items()
            if len(fields) >= min_fields
        }
        
        return cross_domain_jobs
    
    def analyze_embedding_statistics(self):
        """Analyze statistical properties of the embeddings"""
        stats = {}
        
        # Analyze field embeddings
        field_dims = {field: emb.shape for field, emb in self.field_embeddings.items()}
        field_norms = {field: np.linalg.norm(emb) for field, emb in self.field_embeddings.items()}
        
        # Analyze job embeddings
        job_dims = {job_id: emb.shape for job_id, emb in self.job_embeddings.items()}
        job_norms = {job_id: np.linalg.norm(emb) for job_id, emb in self.job_embeddings.items()}
        
        # Calculate basic statistics
        stats['embedding_dimensions'] = {
            'fields': list(set(d[0] for d in field_dims.values())),
            'jobs': list(set(d[0] for d in job_dims.values()))
        }
        
        stats['norm_statistics'] = {
            'fields': {
                'mean': np.mean(list(field_norms.values())),
                'std': np.std(list(field_norms.values())),
                'min': np.min(list(field_norms.values())),
                'max': np.max(list(field_norms.values()))
            },
            'jobs': {
                'mean': np.mean(list(job_norms.values())),
                'std': np.std(list(job_norms.values())),
                'min': np.min(list(job_norms.values())),
                'max': np.max(list(job_norms.values()))
            }
        }
        
        return stats
    
    def generate_embedding_report(self, output_dir: str):
        """Generate a report on embedding statistics"""
        stats = self.analyze_embedding_statistics()
        report_path = os.path.join(output_dir, 'embedding_statistics_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Embedding Statistics Report\n")
            f.write("=========================\n\n")
            
            f.write("Dimensionality:\n")
            f.write(f"Field embeddings: {stats['embedding_dimensions']['fields']}\n")
            f.write(f"Job embeddings: {stats['embedding_dimensions']['jobs']}\n\n")
            
            f.write("Field Embedding Norms:\n")
            for metric, value in stats['norm_statistics']['fields'].items():
                f.write(f"- {metric}: {value:.4f}\n")
            
            f.write("\nJob Embedding Norms:\n")
            for metric, value in stats['norm_statistics']['jobs'].items():
                f.write(f"- {metric}: {value:.4f}\n")
    
    def save_model_specific_results(self, results: Dict[int, dict], output_dir: str):
        """Save results in model-specific directory"""
        model_dir = os.path.join(output_dir, self.model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save visualizations
        self.visualize_threshold_results(results, model_dir)
        self.generate_threshold_report(results, model_dir)
        
        # Save raw results for later comparison
        results_file = os.path.join(model_dir, 'intersection_analysis_results.json')
        serializable_results = {
            str(k): {
                'cross_domain_count': v['cross_domain_count'],
                'total_unique_jobs': v['total_unique_jobs'],
                'intersection_matrix': v['intersection_matrix'].to_dict()
            }
            for k, v in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

def compare_model_results(output_dir: str, models: List[str]):
    """Compare results across different embedding models"""
    comparison_data = {}
    
    for model in models:
        model_dir = os.path.join(output_dir, model)
        results_file = os.path.join(model_dir, 'intersection_analysis_results.json')
        
        with open(results_file, 'r') as f:
            results = json.load(f)
            comparison_data[model] = results
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 8))
    for model in models:
        thresholds = sorted([int(k) for k in comparison_data[model].keys()])
        cross_domain_percentages = [
            float(comparison_data[model][str(t)]['cross_domain_count']) / 
            float(comparison_data[model][str(t)]['total_unique_jobs']) * 100
            for t in thresholds
        ]
        plt.plot(thresholds, cross_domain_percentages, label=f'{model} Model')
    
    plt.xlabel('Top N Matches')
    plt.ylabel('Cross-domain Jobs (%)')
    plt.title('Cross-domain Job Percentage by Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def main():
    # Configure paths
    base_dir = "data/analysis_results/model_embedding"
    
    
    # Models to analyze
    models = ["b1ade", "sfr"]
    
    for model in models:
        logger.info(f"Analyzing results for {model} model")
        results_file = os.path.join(base_dir, f"{model}/field_job_matches.json")
        output_dir = os.path.join(base_dir, f"{model}/intersections")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer for this model
        analyzer = FieldIntersectionAnalyzer(model)
        analyzer.load_analysis_results(results_file)
        
        # Analyze multiple thresholds
        threshold_results = analyzer.analyze_multiple_thresholds(
            start_n=1000,
            step=100,
            min_n=100
        )
        
        # Save model-specific results
        analyzer.save_model_specific_results(threshold_results, output_dir)
        
        logger.info(f"Completed analysis for {model} model")
    
    # Compare results across models
    compare_model_results(output_dir, models)
    
    logger.info("Analysis completed! Check the output directory for results.")

if __name__ == "__main__":
    main()