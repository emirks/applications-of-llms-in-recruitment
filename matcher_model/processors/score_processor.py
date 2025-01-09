from typing import Dict, List
import numpy as np
from ..models.data_models import JobRequirement, MatchResult
from ..models.data_models import ResumeStatements

class ScoreProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.weights = config['weights']
        
    def calculate_statement_coverage_multiplier(self, matched_statements: int, total_statements: int) -> float:
        """Calculate multiplier based on the ratio of matched statements to total statements"""
        if total_statements == 0:
            return 0
        
        ratio = matched_statements / total_statements
        # Use a simple multiplier that:
        # - Gives 1.0 for perfect coverage (all statements matched)
        # - Penalizes proportionally as the ratio of unmatched statements increases
        # - Has a minimum multiplier of 0.5 to avoid over-penalization
        return max(0.2, ratio)
        
    def get_total_statements(self, resume_statements: ResumeStatements) -> int:
        """Calculate total number of statements in a resume that are used in matching"""
        total = 0
        
        # Count skill statements and their evidence
        for skill in resume_statements.skills:
            # Count main skill statement
            total += 1
            # Count each evidence statement
            total += len(skill.evidence)
        
        # Count other relevant statement types used in matching
        total += len(resume_statements.education)
        total += len(resume_statements.certifications)
        total += len(resume_statements.personality_traits)
        
        return total
        
    def calculate_requirement_coverage_multiplier(self, matched_requirements: int, total_requirements: int) -> float:
        """Calculate coverage multiplier based on matched vs total requirements"""
        if total_requirements == 0:
            return 0
        return matched_requirements / total_requirements
        
    def aggregate_resume_scores(self, job_description: Dict, resume_matches: Dict, resumes: List[Dict]) -> Dict[str, Dict]:
        scores = {}
        
        # Create a lookup for total statements per resume
        resume_statements_count = {r['id']: self.get_total_statements(r['statements']) for r in resumes}
        
        for resume_id, matches in resume_matches.items():
            # Get total statements for this resume
            total_statements = resume_statements_count[resume_id]
            
            # Scores are already normalized, just use them directly
            must_have_scores = [m['matches'].score for m in matches['must_have']]
            nice_to_have_scores = [m['matches'].score for m in matches['nice_to_have']]
            
            # Only consider positive scores as valid matches
            must_have_coverage_list = [m['matches'].score for m in matches['must_have'] if m['matches'].score > 0.5]
            nice_to_have_coverage_list = [m['matches'].score for m in matches['nice_to_have'] if m['matches'].score > 0.5]

            # Calculate requirement coverage using the new method
            must_have_coverage = self.calculate_requirement_coverage_multiplier(
                len(must_have_coverage_list),
                len(job_description['must_have_requirements'])
            )
            nice_to_have_coverage = self.calculate_requirement_coverage_multiplier(
                len(nice_to_have_coverage_list),
                max(len(job_description['nice_to_have_requirements']), 1)
            )
            
            # Calculate statement coverage multiplier
            total_matches = len(must_have_coverage_list) + len(nice_to_have_coverage_list)
            statement_coverage_multiplier = self.calculate_statement_coverage_multiplier(total_matches, total_statements)
            
            # Add minimum coverage threshold
            if must_have_coverage < 0.3:  # Require at least 30% of must-have requirements
                scores[resume_id] = {
                    'score': 0,
                    'must_have_coverage': 0,
                    'nice_to_have_coverage': 0,
                    'statement_coverage': 0
                }
                continue
                
            # Calculate weighted scores with coverage
            if must_have_scores:
                must_have_score = np.mean(must_have_scores) * must_have_coverage * self.weights['must_have']
            else:
                must_have_score = 0
                
            if nice_to_have_scores:
                nice_to_have_score = np.mean(nice_to_have_scores) * nice_to_have_coverage * self.weights['nice_to_have']
            else:
                nice_to_have_score = 0
                
            # Apply statement coverage multiplier to final score
            scores[resume_id] = {
                'score': (must_have_score + nice_to_have_score) * statement_coverage_multiplier,
                'must_have_coverage': must_have_coverage,
                'nice_to_have_coverage': nice_to_have_coverage,
                'statement_coverage': statement_coverage_multiplier
            }
            
        return scores