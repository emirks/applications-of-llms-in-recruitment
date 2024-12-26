import os
import json
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def load_all_statements(jd_path: str, resume_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all statements from job descriptions and resumes efficiently
    
    Returns:
        Tuple of (jd_statements, skill_statements) where each statement includes full metadata
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
    
    # Load JD statements
    jd_statements = []
    print("\nLoading job description statements...")
    for filename in tqdm([f for f in os.listdir(jd_path) if f.endswith('.json')]):
        with open(os.path.join(jd_path, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                jd_id = filename.split('.')[0]
                
                # Get job type from job info
                job_type = 'unknown'
                if 'job_info' in data:
                    for info in data['job_info']:
                        if info.startswith('Job Title:'):
                            job_type = info.split(':')[1].strip().lower().replace(' ', '_')
                
                # Collect all statements with metadata
                for field in jd_fields:
                    if field in data:
                        for stmt in data[field]:
                            jd_statements.append({
                                'statement': stmt,
                                'jd_id': jd_id,
                                'job_type': job_type,
                                'category': field
                            })
            except json.JSONDecodeError:
                print(f"Error reading file: {filename}")
                continue
    
    # Load resume skill statements
    skill_statements = []
    print("\nLoading resume skill statements...")
    for root, _, files in os.walk(resume_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        resume_id = Path(filepath).stem
                        resume_type = Path(filepath).parent.name
                        
                        if 'skills' in data:
                            for skill in data['skills']:
                                if 'description' in skill:
                                    skill_statements.append({
                                        'statement': skill['description'],
                                        'resume_id': resume_id,
                                        'resume_type': resume_type,
                                        'skill_name': skill['name'],
                                        'skill_level': skill.get('level', 'Not specified'),
                                        'skill_years': skill.get('years', 0),
                                        'evidence_count': len(skill.get('evidence', []))
                                    })
                except json.JSONDecodeError:
                    print(f"Error reading file: {filepath}")
                    continue
    
    return jd_statements, skill_statements

def create_random_pairs(jd_statements: List[Dict], 
                       skill_statements: List[Dict], 
                       num_pairs: int) -> List[Dict]:
    """
    Create completely random statement pairs
    
    Args:
        jd_statements: List of all JD statements with metadata
        skill_statements: List of all skill statements with metadata
        num_pairs: Number of random pairs to create
    """
    pairs = []
    
    print(f"\nCreating {num_pairs:,} random statement pairs...")
    for _ in tqdm(range(num_pairs), desc="Creating pairs"):
        # Randomly select one JD statement and one skill statement
        jd_stmt = random.choice(jd_statements)
        skill_stmt = random.choice(skill_statements)
        
        pair = {
            'jd_statement': jd_stmt['statement'],
            'skill_statement': skill_stmt['statement'],
            'metadata': {
                'jd': {
                    'id': jd_stmt['jd_id'],
                    'job_type': jd_stmt['job_type'],
                    'category': jd_stmt['category']
                },
                'resume': {
                    'id': skill_stmt['resume_id'],
                    'type': skill_stmt['resume_type'],
                    'skill_name': skill_stmt['skill_name'],
                    'skill_level': skill_stmt['skill_level'],
                    'skill_years': skill_stmt['skill_years'],
                    'evidence_count': skill_stmt['evidence_count']
                }
            }
        }
        
        pairs.append(pair)
    
    return pairs

def main():
    # Update these paths to match your directory structure
    jd_path = "matcher_dataset/job_descriptions/statements/format_json"
    resume_path = "matcher_dataset/resume/statements/format_json"
    output_path = "matcher_dataset/statement_pairs"
    
    # Number of random pairs to create
    num_pairs = 100000  # Adjust this number as needed
    
    # Load all statements
    jd_statements, skill_statements = load_all_statements(jd_path, resume_path)
    
    print(f"\nLoaded {len(jd_statements):,} JD statements and {len(skill_statements):,} skill statements")
    
    # Create random pairs
    pairs = create_random_pairs(jd_statements, skill_statements, num_pairs)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save all pairs to a single file
    output_file = os.path.join(output_path, 'statement_pairs_random.json')
    print("\nSaving pairs to file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_pairs': len(pairs),
            'sampling_params': {
                'total_jd_statements': len(jd_statements),
                'total_skill_statements': len(skill_statements),
                'requested_pairs': num_pairs
            },
            'pairs': pairs
        }, f, indent=2)
    print(f"Saved {len(pairs):,} pairs to {output_file}")
    
    print("\nRandom pair creation complete!")

if __name__ == "__main__":
    main() 