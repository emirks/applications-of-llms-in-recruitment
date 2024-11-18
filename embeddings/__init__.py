import numpy as np
from ai_systems.utils.decorators import cache_embedding_result

class BaseEmbedding:
    @cache_embedding_result()
    def get_embedding(self, text: str) -> np.ndarray:
        raise NotImplementedError("BaseEmbedding subclasses should implement 'calculate' function")
    
__all__ = ['BaseEmbedding', 'BertBaseUncased', 'Gpt2']