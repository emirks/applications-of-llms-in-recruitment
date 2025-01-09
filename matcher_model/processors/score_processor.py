from typing import Dict, List
import numpy as np
from ..models.data_models import JobRequirement, MatchResult

class ScoreProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.weights = config['weights']
        
    def aggregate_resume_scores(self, job_description: Dict, resume_matches: Dict) -> Dict[str, float]:
        scores = {}
        
        for resume_id, matches in resume_matches.items():
            # Scores are already normalized, just use them directly
            must_have_scores = [m['matches'].score for m in matches['must_have']]
            nice_to_have_scores = [m['matches'].score for m in matches['nice_to_have']]
            
            # Calculate coverage penalties
            must_have_coverage = len(must_have_scores) / len(job_description['must_have_requirements'])
            nice_to_have_coverage = len(nice_to_have_scores) / max(len(job_description['nice_to_have_requirements']), 1)
            
            # Calculate weighted scores with coverage
            if must_have_scores:
                must_have_score = np.mean(must_have_scores) * must_have_coverage * self.weights['must_have']
            else:
                must_have_score = 0
                
            if nice_to_have_scores:
                nice_to_have_score = np.mean(nice_to_have_scores) * nice_to_have_coverage * self.weights['nice_to_have']
            else:
                nice_to_have_score = 0
                
            scores[resume_id] = must_have_score + nice_to_have_score
            
        return scores 