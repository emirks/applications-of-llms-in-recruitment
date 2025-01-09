import os
import json
from datetime import datetime
from typing import List, Dict
import logging
from ..models.data_models import JobRequirement, MatchResult
import numpy as np

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, JobRequirement):
            return {
                'text': obj.text,
                'type': obj.type,
                'category': obj.category
            }
        elif isinstance(obj, MatchResult):
            return {
                'requirement': obj.requirement,
                'matched_statements': obj.matched_statements,
                'score': float(obj.score)
            }
        return super().default(obj)

def save_matching_results(matches: List[Dict], job_description: Dict, output_dir: str = "matcher_results"):
    """Save matching results in both JSON and readable text formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON format
    json_path = os.path.join(output_dir, f"matches_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'job_description': job_description,
            'matches': matches
        }, f, indent=2, cls=CustomJSONEncoder)
    
    # Save human-readable format
    txt_path = os.path.join(output_dir, f"matches_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        # Write job requirements
        f.write("JOB REQUIREMENTS\n")
        f.write("===============\n\n")
        
        f.write("Must-Have Requirements:\n")
        for req in job_description['must_have_requirements']:
            f.write(f"- {req.text}\n")
        
        f.write("\nNice-to-Have Requirements:\n")
        for req in job_description['nice_to_have_requirements']:
            f.write(f"- {req.text}\n")
        
        f.write("\n\nMATCHING RESULTS\n")
        f.write("================\n")
        
        # Write matches
        for match in matches:
            f.write(f"\nResume: {match['id']}\n")
            f.write(f"Category: {match['category']}\n")
            f.write(f"Overall Match Score: {match['score']:.2%}\n")
            
            f.write("\nMust-Have Requirements Matches:\n")
            for req_match in match['requirement_matches']['must_have']:
                f.write(f"\nRequirement: {req_match['requirement'].text}\n")
                for stmt in req_match['matches'].matched_statements[:3]:
                    f.write(f"* {stmt['text']} (score: {stmt['score']:.2f})\n")
            
            f.write("\nNice-to-Have Requirements Matches:\n")
            for req_match in match['requirement_matches']['nice_to_have']:
                f.write(f"\nRequirement: {req_match['requirement'].text}\n")
                for stmt in req_match['matches'].matched_statements[:3]:
                    f.write(f"* {stmt['text']} (score: {stmt['score']:.2f})\n")
            
            f.write("\n" + "="*50 + "\n")
    
    logger.info(f"Results saved to:\n- JSON: {json_path}\n- Text: {txt_path}")
    return json_path, txt_path 