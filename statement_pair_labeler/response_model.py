from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Literal
from enum import Enum

class QualificationCategory(str, Enum):
    NOT_RELEVANT = "NotRelevant"
    NOT_YET_QUALIFIED = "NotYetQualified"
    NEARLY_QUALIFIED = "NearlyQualified"
    QUALIFIED = "Qualified"

class LabelDetails(BaseModel):
    score: float = Field(..., ge=0, le=1)
    category: QualificationCategory
    explanation: str

class LabeledPair(BaseModel):
    jd_statement: str
    skill_statement: str
    label_details: LabelDetails
    metadata: Dict[str, Any]
    labeling_metadata: Dict[str, Any] = {}

    @property
    def label(self) -> float:
        """For backwards compatibility"""
        return self.label_details.score

class LabeledPairsData(BaseModel):
    total_pairs: int
    sampling_params: Dict[str, Any]
    labeled_pairs: list[LabeledPair]
    labeling_info: Dict[str, Any] = {}  # Information about the labeling process 