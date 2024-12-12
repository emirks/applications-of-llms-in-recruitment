from typing import List, Dict, Any
import logging
import torch
import numpy as np
from ..models.data_models import ResumeStatements, JobRequirement, MatchResult
from ..models.encoders import BiEncoder, CrossEncoder

logger = logging.getLogger(__name__)

class StatementProcessor:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Initializing encoders")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.bi_encoder = BiEncoder(config['models']['bi_encoder'])
        self.cross_encoder = CrossEncoder(config['models']['cross_encoder'])
        self.batch_size = config.get('batch_size', 32)
    
    def prepare_statements(self, resume_statements: ResumeStatements) -> List[Dict[str, Any]]:
        """Convert resume statements into searchable format"""
        statements = []
        
        # Add skill statements with evidence
        logger.debug("Processing skills and evidence")
        for skill in resume_statements.skills:
            statements.append({
                'text': f"{skill.name}: {skill.description}",
                'type': 'skill',
                'skill_data': skill.dict()
            })
            for evidence in skill.evidence:
                statements.append({
                    'text': evidence,
                    'type': 'skill_evidence',
                    'skill_name': skill.name
                })
        
        # Add other statement types
        logger.debug("Processing other statement types")
        for stmt_type in ['education', 'certifications', 'personality_traits']:
            for stmt in getattr(resume_statements, stmt_type):
                statements.append({
                    'text': stmt,
                    'type': stmt_type
                })
        
        logger.debug(f"Prepared {len(statements)} statements")
        return statements

    def find_matching_statements(self, requirement: JobRequirement, resume_statements: ResumeStatements) -> MatchResult:
        """Find statements matching a job requirement"""
        logger.debug(f"Finding matches for requirement: {requirement.text}")
        
        statements = self.prepare_statements(resume_statements)
        statement_texts = [s['text'] for s in statements]
        
        # Batch process embeddings
        req_embedding = self.bi_encoder.encode(requirement.text)
        
        # Process statement embeddings in batches
        all_embeddings = []
        for i in range(0, len(statement_texts), self.batch_size):
            batch = statement_texts[i:i + self.batch_size]
            embeddings = self.bi_encoder.encode(batch)
            all_embeddings.append(embeddings)
        
        statement_embeddings = np.vstack(all_embeddings)
        
        # Get top-k matches
        similarities = statement_embeddings @ req_embedding.T
        top_k = min(self.config['search']['top_k_statements'], len(statements))
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Rerank using cross-encoder in batches
        pairs = [[requirement.text, statement_texts[idx]] for idx in top_indices]
        scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self.cross_encoder.predict(batch)
            scores.extend(batch_scores if isinstance(batch_scores, list) else [batch_scores])
        
        scored_statements = [
            {**statements[idx], 'score': float(score)}
            for idx, score in zip(top_indices, scores)
        ]
        
        scored_statements.sort(key=lambda x: x['score'], reverse=True)
        
        return MatchResult(
            requirement=requirement,
            matched_statements=scored_statements,
            score=max([s['score'] for s in scored_statements], default=0.0)
        )