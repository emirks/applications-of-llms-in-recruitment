import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, dimension: int, index_type: str = "IP"):
        """Initialize FAISS index
        Args:
            dimension: Size of the vectors to be stored
            index_type: Type of FAISS index ('L2' or 'IP' for inner product)
        """
        if index_type.upper() == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type.upper() == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.metadata = []
        logger.info(f"Initialized {index_type} index with dimension {dimension}")
    
    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors and their metadata to the index"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors and metadata entries must match")
        
        # Normalize vectors for IP similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(vectors)
            
        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(vectors)} vectors to index")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        return self.index.search(query_vector, k)
    
    def get_metadata(self, indices: List[int]) -> List[Dict]:
        """Get metadata for given indices"""
        return [self.metadata[i] for i in indices]
    
    def save(self, path: str):
        """Save the index and metadata to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorDB':
        """Load index and metadata from disk"""
        # Load FAISS index
        index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance and set attributes
        instance = cls(index.d)
        instance.index = index
        instance.metadata = metadata
        
        return instance
