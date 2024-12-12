import os
import json
from typing import List, Dict
from ..models.data_models import ResumeStatements, Skill

def load_resume_statements(base_path: str, categories: List[str]) -> List[Dict]:
    """Load resume statements from the directory structure"""
    resumes = []
    json_base = os.path.join(base_path, "statements", "format_json")
    
    for category in categories:
        category_path = os.path.join(json_base, category)
        if not os.path.exists(category_path):
            continue
            
        for filename in os.listdir(category_path):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(category_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert raw data to ResumeStatements model
            skills = [
                Skill(
                    name=skill['name'],
                    years=skill['years'],
                    level=skill['level'],
                    description=skill['description'],
                    evidence=skill['evidence']
                )
                for skill in data['skills']
            ]
            
            statements = ResumeStatements(
                personal_info=data['personal_info'],
                education=data['education'],
                certifications=data['certifications'],
                personality_traits=data['personality_traits'],
                skills=skills
            )
            
            resumes.append({
                'id': f"{category}/{filename}",
                'category': category,
                'statements': statements
            })
    
    return resumes 