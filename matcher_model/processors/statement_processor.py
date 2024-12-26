from typing import List, Dict, Any
import logging
import torch
import numpy as np
from ..models.data_models import ResumeStatements, JobRequirement, MatchResult
from ..models.encoders import BiEncoder, CrossEncoder
from ..models.vector_store.vector_db import VectorDB
import os
from collections import defaultdict

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
        
        # Cache for resume statement embeddings
        self.statement_cache = {}
        self.vector_store_path = config.get('vector_store', {}).get('path', 'data/vector_store')
        self.vector_store_type = config.get('vector_store', {}).get('index_type', 'IP')
        self.vector_db = None

    def initialize_vector_store(self, statements_data: List[Dict]):
        """Initialize or load vector store"""
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        
        if os.path.exists(f"{self.vector_store_path}.index"):
            logger.info("Loading existing vector store")
            self.vector_db = VectorDB.load(self.vector_store_path)
            logger.info(f"Loaded vector store with {len(self.vector_db.metadata)} entries")
            return
        
        logger.info("Creating new vector store")
        # Get dimension from bi-encoder
        sample_embedding = self.bi_encoder.encode("sample text")
        self.vector_db = VectorDB(
            dimension=len(sample_embedding),
            index_type=self.vector_store_type
        )
        
        # Process all statements in batches
        logger.info(f"Processing {len(statements_data)} statements in batches of {self.batch_size}")
        all_embeddings = []
        all_metadata = []
        
        for i in range(0, len(statements_data), self.batch_size):
            batch = statements_data[i:i + self.batch_size]
            texts = [s['text'] for s in batch]
            embeddings = self.bi_encoder.encode(texts)
            all_embeddings.append(embeddings)
            all_metadata.extend(batch)
            
            if (i + 1) % (self.batch_size * 10) == 0:
                logger.info(f"Processed {i + 1}/{len(statements_data)} statements")
        
        # Combine all embeddings and add to vector store
        combined_embeddings = np.vstack(all_embeddings)
        self.vector_db.add(combined_embeddings, all_metadata)
        
        logger.info(f"Added {len(all_metadata)} entries to vector store")
        
        # Save vector store
        logger.info("Saving vector store")
        self.vector_db.save(self.vector_store_path)

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

    def batch_find_matching_statements(self, requirements: List[JobRequirement], resume_statements: List[Dict]) -> Dict[str, List[MatchResult]]:
        """Batch process multiple requirements against multiple resumes"""
        # Encode all requirements at once
        req_texts = [req.text for req in requirements]
        req_embeddings = self.bi_encoder.encode(req_texts)
        
        # Use vector store for initial retrieval for all requirements
        all_distances = []
        all_indices = []
        for req_emb in req_embeddings:
            distances, indices = self.vector_db.search(
                req_emb.reshape(1, -1),
                k=self.config['search']['top_k_statements']
            )
            all_distances.append(distances[0])
            all_indices.append(indices[0])
        
        # Prepare cross-encoder pairs for all requirements
        all_pairs = []
        pair_mapping = defaultdict(list)  # Maps pair index to (req_idx, stmt_idx)
        current_pair_idx = 0
        
        for req_idx, (req, indices) in enumerate(zip(requirements, all_indices)):
            candidate_statements = self.vector_db.get_metadata(indices)
            for stmt_idx, stmt in enumerate(candidate_statements):
                all_pairs.append([req.text, stmt['text']])
                pair_mapping[current_pair_idx] = (req_idx, stmt_idx)
                current_pair_idx += 1
        
        # Batch process with cross-encoder
        all_scores = []
        for i in range(0, len(all_pairs), self.batch_size):
            batch = all_pairs[i:i + self.batch_size]
            if batch:
                batch_scores = self.cross_encoder.predict(batch)
                all_scores.extend(batch_scores if isinstance(batch_scores, list) else [batch_scores])
        
        # Organize results
        results = defaultdict(list)
        for pair_idx, score in enumerate(all_scores):
            req_idx, stmt_idx = pair_mapping[pair_idx]
            req = requirements[req_idx]
            stmt = self.vector_db.get_metadata(all_indices[req_idx])[stmt_idx]
            
            if req.text not in results:
                results[req.text] = []
            results[req.text].append({**stmt, 'score': float(score)})
        
        # Create MatchResults
        final_results = {}
        for req in requirements:
            matched_statements = results.get(req.text, [])
            matched_statements.sort(key=lambda x: x['score'], reverse=True)
            final_results[req.text] = MatchResult(
                requirement=req,
                matched_statements=matched_statements,
                score=max([s['score'] for s in matched_statements], default=0.0)
            )
        
        return final_results