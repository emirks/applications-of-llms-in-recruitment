from pydantic import BaseModel
from typing import List

class StatementData(BaseModel):
    job_info: List[str] = []
    must_have_requirements: List[str] = []
    nice_to_have_requirements: List[str] = []
    responsibilities: List[str] = []
    required_skills: List[str] = []
    experience_required: List[str] = []
    educational_requirements: List[str] = []
    additional_info: List[str] = []

    @classmethod
    def to_prompt(cls) -> str:
        return """
{
    "job_info": list<str>,
    "must_have_requirements": list<str>,
    "nice_to_have_requirements": list<str>,
    "responsibilities": list<str>,
    "required_skills": list<str>,
    "experience_required": list<str>,
    "educational_requirements": list<str>,
    "additional_info": list<str>
}
"""
