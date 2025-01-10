from sentence_transformers import CrossEncoder as SentenceCrossEncoder
from typing import List, Dict, Union
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CrossEncoder:
    def __init__(self, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if we should use trained model
        if config.get('use_trained_model', False):
            model_path = config.get('trained_model_path', 'matcher_model/trained_models/final')
            self.model = SentenceCrossEncoder(
                model_path,
                device=self.device
            )
        else:
            # Use default pretrained model
            self.model = SentenceCrossEncoder(
                config.get('model_name'),
                device=self.device
            )
        
    def predict(self, texts: Union[List[List[str]], List[str]]) -> Union[List[float], float]:
        """Score the similarity between texts"""
        scores = self.model.predict(
            texts,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Handle both single predictions and batches
        if isinstance(scores, (float, np.float32, np.float64)):
            return float(scores)
        return [float(score) for score in scores]