import os
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def load_job_statements(jd_path: str) -> List[Dict]:
    """
    Load statements from specified job descriptions with metadata
    
    Returns:
        List of dictionaries containing statements and their metadata
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
    
    # Specific JD files to analyze with their job types
    target_files = [
        ('63cb0c1cc008b86fd11e1b42.json', 'software'),
        ('6346bdfd582a3c1beb0dcf44.json', 'civil_engineer'),
        ('63404c34470c7f10c915357c.json', 'hr')
    ]
    
    all_statements = []
    
    print(f"\nProcessing {len(target_files)} job descriptions...")
    for filename, job_type in tqdm(target_files, desc="Loading JD statements"):
        filepath = os.path.join(jd_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filename}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                jd_id = filename.split('.')[0]
                
                for field in jd_fields:
                    if field in data:
                        statements = data[field]
                        for stmt in statements:
                            all_statements.append({
                                'statement': stmt,
                                'jd_id': jd_id,
                                'job_type': job_type,
                                'category': field
                            })
            except json.JSONDecodeError:
                print(f"Error reading file: {filename}")
                continue
    
    return all_statements

def load_resume_statements(resume_path: str) -> List[Dict]:
    """
    Load skill descriptions from resumes with metadata
    
    Returns:
        List of dictionaries containing skill descriptions and their metadata
    """
    skill_statements = []
    
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
    
    return skill_statements

def create_statement_pairs(jd_path: str, resume_path: str, output_path: str):
    """Create all possible pairs between JD statements and resume skill descriptions"""
    
    # Load statements with metadata
    jd_statements = load_job_statements(jd_path)
    resume_statements = load_resume_statements(resume_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create pairs and save by job type
    print("\nCreating statement pairs...")
    pairs_by_job = defaultdict(list)
    
    for jd_stmt in tqdm(jd_statements, desc="Creating pairs"):
        job_type = jd_stmt['job_type']
        
        for resume_stmt in resume_statements:
            pair = {
                'jd_statement': jd_stmt['statement'],
                'skill_statement': resume_stmt['statement'],
                'metadata': {
                    'jd': {
                        'id': jd_stmt['jd_id'],
                        'job_type': jd_stmt['job_type'],
                        'category': jd_stmt['category']
                    },
                    'resume': {
                        'id': resume_stmt['resume_id'],
                        'type': resume_stmt['resume_type'],
                        'skill_name': resume_stmt['skill_name'],
                        'skill_level': resume_stmt['skill_level'],
                        'skill_years': resume_stmt['skill_years'],
                        'evidence_count': resume_stmt['evidence_count']
                    }
                }
            }
            pairs_by_job[job_type].append(pair)
    
    # Save pairs to separate files by job type
    print("\nSaving pairs to files...")
    for job_type, pairs in pairs_by_job.items():
        output_file = os.path.join(output_path, f'statement_pairs_{job_type}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'job_type': job_type,
                'total_pairs': len(pairs),
                'pairs': pairs
            }, f, indent=2)
        print(f"Saved {len(pairs):,} pairs for {job_type} to {output_file}")

def main():
    # Update these paths to match your directory structure
    jd_path = "matcher_dataset/job_descriptions/statements/format_json"
    resume_path = "matcher_dataset/resume/statements/format_json"
    output_path = "matcher_dataset/statement_pairs"
    
    # Create pairs
    create_statement_pairs(jd_path, resume_path, output_path)
    print("\nPair creation complete!")

if __name__ == "__main__":
    main() 