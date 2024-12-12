from typing import List, Dict, Optional
from pydantic import BaseModel

class SkillEvidence(BaseModel):
    text: str
    type: Optional[str] = None
    description: Optional[str] = None

class Skill(BaseModel):
    name: str
    years: float
    level: str
    description: str
    evidence: List[str]

class ResumeStatements(BaseModel):
    personal_info: List[str]
    education: List[str]
    certifications: List[str]
    personality_traits: List[str]
    skills: List[Skill]

class JobRequirement(BaseModel):
    text: str
    type: str  # 'must_have' or 'nice_to_have'
    category: Optional[str] = None  # 'skill', 'education', 'experience', etc.

class MatchResult(BaseModel):
    requirement: JobRequirement
    matched_statements: List[Dict]
    score: float

class JobDescription(BaseModel):
    title: str
    company: str
    location: Optional[str]
    job_type: Optional[str]
    must_have_requirements: List[JobRequirement]
    nice_to_have_requirements: List[JobRequirement]
    responsibilities: List[str]
    required_skills: List[str]
    experience_required: List[str]
    educational_requirements: List[str]
    additional_info: List[str]