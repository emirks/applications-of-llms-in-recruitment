import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

def load_job_statements(jd_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Load statements from specified job descriptions
    
    Returns:
        Tuple of (all statements, count by category)
    """
    # Fields to analyze from job descriptions
    jd_fields = [
        'must_have_requirements',
        'nice_to_have_requirements',
        'responsibilities',
        'required_skills',
        'experience_required',
        'educational_requirements'
    ]
    
    # Specific JD files to analyze
    target_files = [
        '63cb0c1cc008b86fd11e1b42.json',  # software
        '6346bdfd582a3c1beb0dcf44.json',  # civil engineer
        '6372c536756e3538cd74c3ea.json'   # hr
    ]
    
    all_statements = []
    category_counts = defaultdict(int)
    
    print(f"\nProcessing {len(target_files)} job descriptions...")
    for filename in tqdm(target_files, desc="Loading JD statements"):
        filepath = os.path.join(jd_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filename}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for field in jd_fields:
                    if field in data:
                        statements = data[field]
                        all_statements.extend(statements)
                        category_counts[field] += len(statements)
            except json.JSONDecodeError:
                print(f"Error reading file: {filename}")
                continue
    
    return all_statements, dict(category_counts)

def load_resume_statements(resume_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Load skill descriptions from resumes
    
    Returns:
        Tuple of (skill descriptions, count statistics)
    """
    skill_descriptions = []
    stats = defaultdict(int)
    
    # Get all JSON files recursively
    json_files = []
    for root, _, files in os.walk(resume_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"\nProcessing {len(json_files)} resumes...")
    for filepath in tqdm(json_files, desc="Loading resume statements"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process skills
                if 'skills' in data:
                    for skill in data['skills']:
                        if 'description' in skill:
                            skill_descriptions.append(skill['description'])
                            stats['skill_descriptions'] += 1
                        # Track evidence count separately
                        if 'evidence' in skill:
                            stats['total_skill_evidences'] += len(skill['evidence'])
                            
                # Track other fields count for reference
                for field in ['education', 'certifications', 'personality_traits']:
                    if field in data:
                        stats[field] += len(data[field])
        
        except json.JSONDecodeError:
            print(f"Error reading file: {filepath}")
            continue
    
    return skill_descriptions, dict(stats)

def analyze_potential_pairs(jd_path: str, resume_path: str, num_jd_samples: int) -> Dict:
    """Analyze potential statement pairs between N random JDs and resume skill descriptions"""
    
    # Load statements
    jd_statements, jd_counts = load_job_statements(jd_path)
    skill_descriptions, resume_counts = load_resume_statements(resume_path)
    
    # Calculate statistics
    total_pairs = len(jd_statements) * len(skill_descriptions)
    
    analysis = {
        'job_descriptions': {
            'sampled_count': num_jd_samples,
            'total_statements': len(jd_statements),
            'statements_per_category': jd_counts,
            'avg_statements_per_jd': len(jd_statements) / num_jd_samples if num_jd_samples > 0 else 0
        },
        'resumes': {
            'total_skill_descriptions': len(skill_descriptions),
            'avg_skill_descriptions_per_resume': len(skill_descriptions) / (resume_counts.get('skill_descriptions', 1) or 1),
            'other_statement_counts': {k: v for k, v in resume_counts.items() if k != 'skill_descriptions'},
            'total_skill_evidences': resume_counts.get('total_skill_evidences', 0)
        },
        'pairs': {
            'total_possible_pairs': total_pairs,
            'pairs_per_jd': total_pairs / num_jd_samples if num_jd_samples > 0 else 0
        }
    }
    
    return analysis

def print_analysis(analysis: Dict):
    """Print analysis results in a readable format"""
    print("\nStatement Pair Analysis")
    print("=====================")
    
    # Job Description Stats
    print("\nJob Description Statistics:")
    print(f"Number of JDs sampled: {analysis['job_descriptions']['sampled_count']}")
    print(f"Total statements: {analysis['job_descriptions']['total_statements']}")
    print(f"Average statements per JD: {analysis['job_descriptions']['avg_statements_per_jd']:.1f}")
    print("\nStatements per category:")
    for category, count in analysis['job_descriptions']['statements_per_category'].items():
        print(f"  {category.replace('_', ' ').title()}: {count}")
    
    # Resume Stats
    print("\nResume Skill Statistics:")
    print(f"Total skill descriptions: {analysis['resumes']['total_skill_descriptions']}")
    print(f"Average skill descriptions per resume: {analysis['resumes']['avg_skill_descriptions_per_resume']:.1f}")
    print(f"Total skill evidences: {analysis['resumes']['total_skill_evidences']}")
    
    print("\nOther resume statement counts (for reference):")
    for category, count in analysis['resumes']['other_statement_counts'].items():
        if category != 'total_skill_evidences':
            print(f"  {category.replace('_', ' ').title()}: {count}")
    
    # Pair Stats
    print("\nPotential Pairs (JD statements × Skill descriptions):")
    print(f"Total possible pairs: {analysis['pairs']['total_possible_pairs']:,}")
    print(f"Average pairs per JD: {analysis['pairs']['pairs_per_jd']:,.1f}")

def main():
    # Update these paths to match your directory structure
    jd_path = "matcher_dataset/job_descriptions/statements/format_json"
    resume_path = "matcher_dataset/resume/statements/format_json"
    
    # Run analysis
    analysis = analyze_potential_pairs(jd_path, resume_path, 3)  # Fixed to 3 JDs
    
    # Print results
    print_analysis(analysis)
    
    # Save results to JSON
    print("\nSaving results to JSON...")
    with open('statement_pair_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("Analysis complete!")

if __name__ == "__main__":
    main()