from typing import List, Dict, Any, Union
import logging
import torch
import numpy as np
from ..models.data_models import ResumeStatements, JobRequirement, MatchResult
from ..models.encoders import BiEncoder, CrossEncoder
from ..models.vector_store.vector_db import VectorDB
import os
from collections import defaultdict
import json
from tqdm import tqdm

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

    def normalize_score(self, score: float) -> float:
        """Normalize a score to [-1, 1] range using sigmoid-like normalization"""
        return 2.0 / (1.0 + np.exp(-score/2)) - 1.0

    def normalize_scores(self, scores: Union[List[float], float]) -> Union[List[float], float]:
        """Normalize a single score or list of scores"""
        if isinstance(scores, list):
            return [self.normalize_score(score) for score in scores]
        return self.normalize_score(scores)

    def initialize_vector_store(self, statements: List[Dict], force_recreate: bool = False) -> None:
        """Initialize or load vector store with statements"""
        vector_store_path = os.path.join("data", "vector_store", "resume_statements")
        
        if os.path.exists(f"{vector_store_path}.index") and not force_recreate:
            logger.info("Loading existing vector store")
            self.vector_db = VectorDB.load(vector_store_path)
            logger.info(f"Loaded vector store with {len(self.vector_db.metadata)} entries")
            return
            
        logger.info("Creating new vector store")
        # Get embeddings for all statements
        texts = [stmt['text'] for stmt in statements]
        embeddings = self.bi_encoder.encode(texts)
        
        # Initialize vector store
        self.vector_db = VectorDB(dimension=embeddings.shape[1])
        self.vector_db.add(embeddings, statements)
        
        # Save vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        self.vector_db.save(vector_store_path)
        logger.info(f"Created and saved vector store with {len(statements)} entries")

    def prepare_statements(self, resume_statements: ResumeStatements, resume_id: str) -> List[Dict]:
        """Convert resume statements into searchable format with resume ID"""
        statements = []
        
        # Add skill statements with evidence
        for skill in resume_statements.skills:
            # Add main skill statement with more context
            skill_text = f"{skill.name}: {skill.description}"
            if skill.years > 0:
                skill_text = f"{skill_text} ({skill.years} years experience)"
            statements.append({
                'text': skill_text,
                'type': 'skill',
                'skill_data': skill.dict(),
                'resume_id': resume_id
            })
            
            # Add evidence with context
            for evidence in skill.evidence:
                statements.append({
                    'text': f"{skill.name} - {evidence}",
                    'type': 'skill_evidence',
                    'skill_name': skill.name,
                    'resume_id': resume_id
                })
        
        # Add other statement types
        for stmt_type in ['education', 'certifications', 'personality_traits']:
            for stmt in getattr(resume_statements, stmt_type):
                statements.append({
                    'text': stmt,
                    'type': stmt_type,
                    'resume_id': resume_id
                })
        
        return statements

    def batch_find_matching_statements(self, requirements: List[JobRequirement], current_resume_id: str) -> Dict[str, List[MatchResult]]:
        """Find matching statements for requirements, filtered by resume ID"""
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
            
            # Only include statements that belong to the current resume
            if stmt['resume_id'] == current_resume_id:
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

    def batch_find_all_matching_statements(self, requirements: List[JobRequirement]) -> Dict[str, Dict[str, MatchResult]]:
        """Find matching statements for all requirements and all resumes in one pass"""
        logger.info("\n=== Starting Matching Process ===")
        
        # Get embeddings for all requirements
        req_texts = [req.text for req in requirements]
        req_embeddings = self.bi_encoder.encode(req_texts)
        
        # Use vector store for initial retrieval
        all_distances = []
        all_indices = []
        
        # Debug: Track matches through pipeline
        debug_matches = defaultdict(list)
        
        for req_idx, req_emb in enumerate(req_embeddings):
            req_text = req_texts[req_idx]
            logger.info(f"\nProcessing requirement: {req_text[:100]}...")
            
            # Initial bi-encoder search
            distances, indices = self.vector_db.search(
                req_emb.reshape(1, -1),
                k=self.config['search']['top_k_statements']
            )
            
            # Log initial matches
            initial_matches = self.vector_db.get_metadata(indices[0])
            engineering_matches = [m for m in initial_matches if 'engineering' in m['resume_id']]
            logger.info(f"Bi-encoder found {len(initial_matches)} total matches, {len(engineering_matches)} from engineering")
            
            # Log top engineering matches
            logger.info("\nTop 5 engineering matches from bi-encoder:")
            for idx, match in enumerate(engineering_matches[:5]):
                logger.info(f"{idx+1}. Score: {distances[0][idx]:.3f}, Text: {match['text'][:100]}...")
                debug_matches[req_text].append({
                    'stage': 'bi-encoder',
                    'score': float(distances[0][idx]),
                    'text': match['text'],
                    'resume_id': match['resume_id']
                })
            
            all_distances.append(distances[0])
            all_indices.append(indices[0])
        
        # Prepare cross-encoder pairs and track metadata
        all_pairs = []
        pair_mapping = []
        
        for req_idx, indices in enumerate(all_indices):
            req_text = req_texts[req_idx]
            candidate_statements = self.vector_db.get_metadata(indices)
            
            for stmt_idx, stmt in enumerate(candidate_statements):
                all_pairs.append([req_text, stmt['text']])
                pair_mapping.append((req_idx, stmt_idx, stmt['resume_id']))
        
        logger.info(f"\nPrepared {len(all_pairs)} pairs for cross-encoder")
        
        # Batch process with cross-encoder and normalize immediately
        all_scores = []
        for i in tqdm(range(0, len(all_pairs), self.batch_size)):
            batch = all_pairs[i:i + self.batch_size]
            if batch:
                batch_scores = self.cross_encoder.predict(batch)
                # Use the normalize_scores method
                normalized_scores = self.normalize_scores(batch_scores)
                all_scores.extend(normalized_scores if isinstance(normalized_scores, list) else [normalized_scores])
        
        # Organize results and log cross-encoder matches
        results = {}
        for req_idx, req in enumerate(requirements):
            req_text = req.text
            results[req_text] = {}
            
            # Get all scores for this requirement
            req_scores = [
                (score, mapping[2], self.vector_db.get_metadata(all_indices[mapping[0]])[mapping[1]])
                for score, mapping in zip(all_scores, pair_mapping)
                if mapping[0] == req_idx
            ]
            
            # Log cross-encoder scores for engineering resumes
            eng_scores = [(s, r, stmt) for s, r, stmt in req_scores if 'engineering' in r]
            eng_scores.sort(key=lambda x: x[0], reverse=True)
            
            logger.info(f"\nTop 5 engineering matches from cross-encoder for: {req_text[:100]}...")
            for score, resume_id, stmt in eng_scores[:5]:
                logger.info(f"Score: {score:.3f}, Resume: {resume_id}")
                logger.info(f"Text: {stmt['text'][:100]}...")
                debug_matches[req_text].append({
                    'stage': 'cross-encoder',
                    'score': float(score),
                    'text': stmt['text'],
                    'resume_id': resume_id
                })
            
            # Group by resume_id
            resume_groups = {}
            for score, resume_id, stmt in req_scores:
                if resume_id not in resume_groups:
                    resume_groups[resume_id] = []
                resume_groups[resume_id].append((score, stmt))
            
            # Create MatchResults for each resume
            for resume_id, matches in resume_groups.items():
                matches.sort(key=lambda x: x[0], reverse=True)
                results[req_text][resume_id] = MatchResult(
                    requirement=req,
                    matched_statements=[{**stmt, 'score': float(score)} for score, stmt in matches],
                    score=max(score for score, _ in matches)
                )
        
        # Save debug matches for analysis
        with open('debug_matches.json', 'w') as f:
            json.dump(debug_matches, f, indent=2)
        
        return results