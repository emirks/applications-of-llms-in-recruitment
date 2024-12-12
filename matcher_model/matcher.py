from typing import Dict, List
from tqdm.auto import tqdm
import logging
from .processors import StatementProcessor, ScoreProcessor

logger = logging.getLogger(__name__)

class Matcher:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.statement_processor = StatementProcessor(self.config)
        self.score_processor = ScoreProcessor(self.config)
    
    def match(self, job_description: Dict, resumes: List[Dict]) -> List[Dict]:
        """Main matching process"""
        # Process each requirement type separately
        resume_matches = {}
        
        # Create progress bar for resumes
        with tqdm(total=len(resumes), desc="Processing resumes", unit="resume") as pbar:
            for resume in resumes:
                matches = {
                    'must_have': [],
                    'nice_to_have': []
                }
                
                # Match must-have requirements
                for req in tqdm(job_description['must_have_requirements'], 
                              desc=f"Must-have reqs for {resume['id']}", 
                              leave=False):
                    matched_statements = self.statement_processor.find_matching_statements(
                        req,
                        resume['statements']
                    )
                    matches['must_have'].append({
                        'requirement': req,
                        'matches': matched_statements
                    })
                    
                # Match nice-to-have requirements
                for req in tqdm(job_description['nice_to_have_requirements'], 
                              desc=f"Nice-to-have reqs for {resume['id']}", 
                              leave=False):
                    matched_statements = self.statement_processor.find_matching_statements(
                        req,
                        resume['statements']
                    )
                    matches['nice_to_have'].append({
                        'requirement': req,
                        'matches': matched_statements
                    })
                    
                resume_matches[resume['id']] = matches
                pbar.update(1)
        
        # Calculate scores and rank resumes
        logger.info("Calculating final scores")
        resume_scores = self.score_processor.aggregate_resume_scores(
            job_description,
            resume_matches
        )
        
        # Prepare final results
        ranked_resumes = []
        for resume_id, score in sorted(resume_scores.items(), key=lambda x: x[1], reverse=True):
            resume_data = next(r for r in resumes if r['id'] == resume_id)
            ranked_resumes.append({
                'id': resume_id,
                'score': score,
                'category': resume_data['category'],
                'requirement_matches': resume_matches[resume_id]
            })
            
        return ranked_resumes[:self.config['search']['top_n_resumes']]

    def _get_default_config(self) -> Dict:
        return {
            'models': {
                'bi_encoder': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
                },
                'cross_encoder': {
                    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                }
            },
            'search': {
                'top_k_statements': 5,
                'top_n_resumes': 10
            },
            'weights': {
                'must_have': 0.7,
                'nice_to_have': 0.3
            }
        }