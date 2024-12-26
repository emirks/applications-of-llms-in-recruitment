import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

def analyze_resume_statements(base_path: str) -> Dict:
    """
    Analyze resume statements from JSON files.
    
    Args:
        base_path: Path to the directory containing resume JSON files
        
    Returns:
        Dictionary containing analysis results
    """
    # Initialize containers for analysis
    stats = defaultdict(list)
    skill_stats = defaultdict(list)
    
    # Fields to analyze
    fields = [
        'education',
        'certifications',
        'personality_traits'
    ]
    
    # Get all JSON files recursively (including subdirectories)
    json_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Process each file with progress bar
    file_count = 0
    for filepath in tqdm(json_files, desc="Processing resumes", unit="file"):
        file_count += 1
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process regular fields
                for field in fields:
                    if field in data:
                        stats[field].extend(data[field])
                
                # Process skills separately due to nested structure
                if 'skills' in data:
                    for skill in data['skills']:
                        # Add skill name to list of skills
                        skill_stats['names'].append(skill['name'])
                        skill_stats['years'].append(skill['years'])
                        skill_stats['levels'].append(skill['level'])
                        skill_stats['descriptions'].append(skill['description'])
                        # Add each evidence statement separately
                        skill_stats['evidence'].extend(skill['evidence'])
                        
        except json.JSONDecodeError:
            print(f"Error reading file: {filepath}")
            continue
    
    print("\nCalculating statistics...")
    
    # Calculate statistics
    analysis = {
        'total_files_processed': file_count,
        'categories': {},
        'skills_analysis': {}
    }
    
    # Process regular fields
    for field in fields:
        statements = stats[field]
        if not statements:
            continue
            
        lengths = [len(str(stmt).split()) for stmt in statements]
        unique_statements = set(str(stmt) for stmt in statements)
        
        category_stats = {
            'total_statements': len(statements),
            'statements_per_resume': len(statements) / file_count if file_count > 0 else 0,
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
    
    # Process skills statistics
    if skill_stats['names']:
        unique_skills = set(skill_stats['names'])
        skill_years = np.array(skill_stats['years'])
        
        # Analyze skill evidence statements
        evidence_lengths = [len(str(stmt).split()) for stmt in skill_stats['evidence']]
        unique_evidence = set(str(stmt) for stmt in skill_stats['evidence'])
        
        analysis['skills_analysis'] = {
            'unique_skills': len(unique_skills),
            'total_skill_mentions': len(skill_stats['names']),
            'skills_per_resume': len(skill_stats['names']) / file_count if file_count > 0 else 0,
            'years_of_experience': {
                'mean': np.mean(skill_years),
                'median': np.median(skill_years),
                'std': np.std(skill_years),
                'min': np.min(skill_years),
                'max': np.max(skill_years)
            },
            'level_distribution': {
                level: skill_stats['levels'].count(level) / len(skill_stats['levels'])
                for level in set(skill_stats['levels'])
            },
            'evidence_statements': {
                'total': len(skill_stats['evidence']),
                'unique': len(unique_evidence),
                'mean_length': np.mean(evidence_lengths),
                'median_length': np.median(evidence_lengths),
                'std_length': np.std(evidence_lengths),
                'min_length': min(evidence_lengths),
                'max_length': max(evidence_lengths)
            },
            'top_skills': sorted([(skill, skill_stats['names'].count(skill)) 
                                for skill in unique_skills], 
                               key=lambda x: x[1], 
                               reverse=True)[:10]
        }
    
    return analysis

def print_analysis(analysis: Dict):
    """Print analysis results in a readable format"""
    print("\nResume Statement Analysis")
    print("========================")
    print(f"Total Files Processed: {analysis['total_files_processed']}")
    
    # Print regular categories
    for category, stats in analysis['categories'].items():
        print(f"\n{category.replace('_', ' ').title()}")
        print("-" * len(category))
        print(f"Total Statements: {stats['total_statements']}")
        print(f"Statements per Resume: {stats['statements_per_resume']:.1f}")
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
    
    # Print skills analysis
    if 'skills_analysis' in analysis:
        print("\nSkills Analysis")
        print("--------------")
        skills = analysis['skills_analysis']
        print(f"Unique Skills: {skills['unique_skills']}")
        print(f"Total Skill Mentions: {skills['total_skill_mentions']}")
        print(f"Skills per Resume: {skills['skills_per_resume']:.1f}")
        
        print("\nYears of Experience:")
        yoe = skills['years_of_experience']
        print(f"  Mean: {yoe['mean']:.1f} years")
        print(f"  Median: {yoe['median']:.1f} years")
        print(f"  Std Dev: {yoe['std']:.1f} years")
        print(f"  Range: {yoe['min']:.1f} - {yoe['max']:.1f} years")
        
        print("\nSkill Level Distribution:")
        for level, percentage in skills['level_distribution'].items():
            print(f"  {level}: {percentage:.1%}")
        
        print("\nEvidence Statements:")
        evidence = skills['evidence_statements']
        print(f"  Total: {evidence['total']}")
        print(f"  Unique: {evidence['unique']}")
        print(f"  Mean Length: {evidence['mean_length']:.1f} words")
        print(f"  Median Length: {evidence['median_length']:.1f} words")
        
        print("\nTop 10 Skills:")
        for skill, count in skills['top_skills']:
            print(f"  - {skill}: {count} mentions")

def main():
    # Update this path to your resumes directory
    base_path = "matcher_dataset/resume/statements/format_json"
    
    # Run analysis
    analysis = analyze_resume_statements(base_path)
    
    # Print results
    print_analysis(analysis)
    
    # Save results to JSON
    print("\nSaving results to JSON...")
    with open('resume_statement_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Analysis complete!")

if __name__ == "__main__":
    main() 