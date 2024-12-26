from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLabeler(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def label_pair(self, jd_statement: str, skill_statement: str, metadata: Dict[str, Any]) -> float:
        """
        Label a single statement pair
        Returns: float between 0 and 1
        """
        pass 