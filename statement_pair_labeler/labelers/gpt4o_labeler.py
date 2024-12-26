from . import BaseLabeler
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import json

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

class LabelingError(Exception):
    """Custom exception for labeling errors"""
    pass

class GPT4OLabeler(BaseLabeler):
    def __init__(self, batch_size: int = 12):
        super().__init__()
        self.client = client
        self.batch_size = batch_size
        
    def label_pairs_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Label a batch of statement pairs using GPT-4o-mini"""
        
        prompt = """You are an expert at matching job requirements with candidate skills.
For each pair below, evaluate how well the candidate's skill matches the job requirement.

Classify each match into one of these categories and provide a score:
- NotRelevant (0.0-0.1): The statements contain no relevant information to the requirement
- NotYetQualified (0.1-0.5): Contains relevant skills but proficiency level is insufficient
- NearlyQualified (0.5-0.8): Shows skills with proficiency level close to requirements
- Qualified (0.8-1.0): Demonstrates skills that fully meet the requirements

For each pair, provide:
1. A score between 0.0 and 1.0
2. The qualification category
3. A brief explanation of the match

Format your response as a list of JSONs of the following format:

{
    "score": 0.85,
    "category": "Qualified",
    "explanation": "Candidate has X years of experience..."
}

Pairs to evaluate:
"""
        # Add each pair to the prompt
        for i, pair in enumerate(pairs, 1):
            prompt += f"\n{i}. Job Requirement: \"{pair['jd_statement']}\""
            prompt += f"\n   Candidate Skill: \"{pair['skill_statement']}\""
            prompt += f"\n   Context:"
            prompt += f"\n   - Job Type: {pair['metadata']['jd']['job_type']}"
            prompt += f"\n   - Category: {pair['metadata']['jd']['category']}"
            prompt += f"\n   - Skill: {pair['metadata']['resume']['skill_name']} (Level: {pair['metadata']['resume']['skill_level']}, Years: {pair['metadata']['resume']['skill_years']})"
            prompt += "\n"
        
        try:
            response = self.client.complete(prompt)
            response_text = response.text.strip()
            
            # Remove any markdown code block markers
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            results = json.loads(response_text)
            
            # Validate results
            if len(results) != len(pairs):
                raise LabelingError(
                    f"Mismatch in number of results. Expected {len(pairs)}, got {len(results)}"
                )
            
            # Validate each result
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    raise LabelingError(f"Invalid result format at position {i}")
                
                if 'score' not in result or not (0.0 <= result['score'] <= 1.0):
                    raise LabelingError(f"Invalid score in result at position {i}")
                
                if 'category' not in result or result['category'] not in [
                    "NotRelevant", "NotYetQualified", "NearlyQualified", "Qualified"
                ]:
                    raise LabelingError(f"Invalid category in result at position {i}")
                
                if 'explanation' not in result or not result['explanation']:
                    raise LabelingError(f"Missing explanation in result at position {i}")
            
            return results
            
        except Exception as e:
            error_msg = (
                f"Error in GPT4O batch labeling: {str(e)}\n"
                f"Raw response: {response.text}\n"
                f"First pair in batch: {pairs[0]['jd_statement'][:100]}..."
            )
            raise LabelingError(error_msg) from e
    
    def label_pair(self, jd_statement: str, skill_statement: str, metadata: Dict[str, Any]) -> float:
        """Label a single statement pair (implemented for compatibility)"""
        pair = {
            'jd_statement': jd_statement,
            'skill_statement': skill_statement,
            'metadata': metadata
        }
        result = self.label_pairs_batch([pair])[0]
        return result['score'] 