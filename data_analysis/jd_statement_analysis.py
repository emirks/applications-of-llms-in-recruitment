import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Added for progress indication

def analyze_job_statements(base_path: str) -> Dict:
    """
    Analyze job description statements from JSON files.
    
    Args:
        base_path: Path to the directory containing job description JSON files
        
    Returns:
        Dictionary containing analysis results
    """
    # Initialize containers for analysis
    stats = defaultdict(list)
    
    # Fields to analyze
    fields = [
        'must_have_requirements',
        'nice_to_have_requirements',
        'responsibilities',
        'required_skills',
        'experience_required',
        'educational_requirements'
    ]
    
    # Get total file count for progress bar
    json_files = [f for f in os.listdir(base_path) if f.endswith('.json')]
    
    # Process each file with progress bar
    file_count = 0
    for filename in tqdm(json_files, desc="Processing job descriptions", unit="file"):
        file_count += 1
        with open(os.path.join(base_path, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Add statements to respective categories
                for field in fields:
                    if field in data:
                        stats[field].extend(data[field])
            except json.JSONDecodeError:
                print(f"Error reading file: {filename}")
                continue
    
    print("\nCalculating statistics...")
    
    # Calculate statistics
    analysis = {
        'total_files_processed': file_count,
        'categories': {}
    }
    
    for field in fields:
        statements = stats[field]
        if not statements:
            continue
            
        # Calculate lengths of statements
        lengths = [len(str(stmt).split()) for stmt in statements]
        
        # Get unique statements
        unique_statements = set(str(stmt) for stmt in statements)
        
        category_stats = {
            'total_statements': len(statements),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'unique_statements': len(unique_statements),
            'duplicate_rate': 1 - (len(unique_statements) / len(statements)),
            'sample_statements': list(statements[:3]) if statements else []
        }
        
        analysis['categories'][field] = category_stats
    
    # Calculate overall metrics
    all_statements = []
    for field in fields:
        if field in stats:
            all_statements.extend(stats[field])
    
    if all_statements:
        lengths = [len(str(stmt).split()) for stmt in all_statements]
        unique_statements = set(str(stmt) for stmt in all_statements)
        
        analysis['overall_metrics'] = {
            'total_statements': len(all_statements),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'unique_statements': len(unique_statements),
            'duplicate_rate': 1 - (len(unique_statements) / len(all_statements)),
            'statements_per_job': len(all_statements) / file_count if file_count > 0 else 0
        }
    
    return analysis

def print_analysis(analysis: Dict):
    """Print analysis results in a readable format"""
    print("\nJob Description Statement Analysis")
    print("=================================")
    print(f"Total Files Processed: {analysis['total_files_processed']}")
    
    # Print overall metrics first
    if 'overall_metrics' in analysis:
        print("\nOverall Metrics")
        print("--------------")
        metrics = analysis['overall_metrics']
        print(f"Total Statements: {metrics['total_statements']}")
        print(f"Unique Statements: {metrics['unique_statements']}")
        print(f"Duplicate Rate: {metrics['duplicate_rate']:.1%}")
        print(f"Statements per Job: {metrics['statements_per_job']:.1f}")
        print(f"Statement Lengths:")
        print(f"  Mean: {metrics['mean_length']:.1f} words")
        print(f"  Median: {metrics['median_length']:.1f} words")
        print(f"  Std Dev: {metrics['std_length']:.1f} words")
        print(f"  Range: {metrics['min_length']} - {metrics['max_length']} words")
    
    for category, stats in analysis['categories'].items():
        print(f"\n{category.replace('_', ' ').title()}")
        print("-" * len(category))
        print(f"Total Statements: {stats['total_statements']}")
        print(f"Unique Statements: {stats['unique_statements']}")
        print(f"Duplicate Rate: {stats['duplicate_rate']:.1%}")
        print(f"Statement Lengths:")
        print(f"  Mean: {stats['mean_length']:.1f} words")
        print(f"  Median: {stats['median_length']:.1f} words")
        print(f"  Std Dev: {stats['std_length']:.1f} words")
        print(f"  Range: {stats['min_length']} - {stats['max_length']} words")
        print("\nSample Statements:")
        for stmt in stats['sample_statements']:
            print(f"  - {stmt}")

def main():
    # Update this path to your job descriptions directory
    base_path = "matcher_dataset/job_descriptions/statements/format_json"
    
    # Run analysis
    analysis = analyze_job_statements(base_path)
    
    # Print results
    print_analysis(analysis)
    
    # Optional: Save results to JSON
    print("\nSaving results to JSON...")
    with open('job_description_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Analysis complete!")

if __name__ == "__main__":
    main()