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
            must_have_scores = [m['matches'].score for m in matches['must_have']]
            nice_to_have_scores = [m['matches'].score for m in matches['nice_to_have']]
            
            # Calculate weighted average
            if must_have_scores:
                must_have_score = np.mean(must_have_scores) * self.weights['must_have']
            else:
                must_have_score = 0
                
            if nice_to_have_scores:
                nice_to_have_score = np.mean(nice_to_have_scores) * self.weights['nice_to_have']
            else:
                nice_to_have_score = 0
                
            scores[resume_id] = must_have_score + nice_to_have_score
            
        return scores 