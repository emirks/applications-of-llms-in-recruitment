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
        resume_matches = {}
        
        # Get all requirements
        all_requirements = (
            [(req, 'must_have') for req in job_description['must_have_requirements']] +
            [(req, 'nice_to_have') for req in job_description['nice_to_have_requirements']]
        )
        
        # Create progress bar for resumes
        with tqdm(total=len(resumes), desc="Processing resumes", unit="resume") as pbar:
            for resume in resumes:
                # Batch process all requirements for this resume
                batch_results = self.statement_processor.batch_find_matching_statements(
                    [req for req, _ in all_requirements],
                    current_resume_id=resume['id']
                )
                
                # Organize results
                matches = {
                    'must_have': [],
                    'nice_to_have': []
                }
                
                for req, req_type in all_requirements:
                    matches[req_type].append({
                        'requirement': req,
                        'matches': batch_results[req.text]
                    })
                
                resume_matches[resume['id']] = matches
                pbar.update(1)
        
        # Calculate scores and rank resumes
        logger.info("Calculating final scores")
        resume_scores = self.score_processor.aggregate_resume_scores(
            job_description,
            resume_matches,
            resumes
        )
        
        return self._prepare_ranked_results(resume_scores, resume_matches, resumes)
    
    def match_new(self, job_description: Dict, resumes: List[Dict]) -> List[Dict]:
        """Main matching process"""
        # Get all requirements
        all_requirements = (
            [(req, 'must_have') for req in job_description['must_have_requirements']] +
            [(req, 'nice_to_have') for req in job_description['nice_to_have_requirements']]
        )
        
        # Get all matches in one pass
        all_matches = self.statement_processor.batch_find_all_matching_statements(
            [req for req, _ in all_requirements]
        )
        
        # Organize results by resume
        resume_matches = {}
        for resume in resumes:
            matches = {
                'must_have': [],
                'nice_to_have': []
            }
            
            for req, req_type in all_requirements:
                if resume['id'] in all_matches[req.text]:
                    matches[req_type].append({
                        'requirement': req,
                        'matches': all_matches[req.text][resume['id']]
                    })
            
            resume_matches[resume['id']] = matches
        
        # Calculate scores and rank
        resume_scores = self.score_processor.aggregate_resume_scores(
            job_description,
            resume_matches,
            resumes
        )
        
        return self._prepare_ranked_results(resume_scores, resume_matches, resumes)

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
                'top_k_statements': 150,
                'top_n_resumes': 5
            },
            'weights': {
                'must_have': 0.7,
                'nice_to_have': 0.3
            },
            'normalization': {
                'factor': 10.0  # Adjust this based on your typical score ranges
            }
        }

    def _prepare_ranked_results(self, resume_scores: Dict[str, Dict], resume_matches: Dict, resumes: List[Dict]) -> List[Dict]:
        """Prepare final ranked results with detailed scoring information"""
        # Convert scores to list and sort
        scored_resumes = [
            {
                'id': resume_id,
                'score': score_info['score'],
                'must_have_coverage': score_info['must_have_coverage'],
                'nice_to_have_coverage': score_info['nice_to_have_coverage'],
                'statement_coverage': score_info['statement_coverage'],
                'requirement_matches': {
                    'must_have': matches['must_have'],
                    'nice_to_have': matches['nice_to_have']
                },
                'category': next((r['category'] for r in resumes if r['id'] == resume_id), 'unknown')
            }
            for resume_id, (score_info, matches) in ((rid, (score, resume_matches[rid])) 
                for rid, score in resume_scores.items())
            if score_info['score'] > 0
        ]
        
        scored_resumes.sort(key=lambda x: x['score'], reverse=True)
        return scored_resumes