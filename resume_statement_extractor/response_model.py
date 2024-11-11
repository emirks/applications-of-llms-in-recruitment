from pydantic import BaseModel
from typing import List

class StatementData(BaseModel):
    personal_info: List[str] = []
    education: List[str] = []
    certifications: List[str] = []
    personality_traits: List[str] = []
    skills: List[str] = []

    @classmethod
    def to_prompt(cls) -> str:
        return """
{
    "personal_info": list<str>,
    "education": list<str>,
    "certifications": list<str>,
    "personality_traits": list<str>,
    "skills": list<str>
}
"""
