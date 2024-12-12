from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import torch

class BiEncoder:
    def __init__(self, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(config.get('model_name')).to(self.device)
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into vector embeddings"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device
        )
        return embeddings