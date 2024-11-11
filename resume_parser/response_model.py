from typing import List, Optional, Dict, Union
import json


class Link:
    url: str

    def __init__(self, url: str, title: str):
        self.url = url

    @staticmethod
    def to_prompt() -> str:
        return json.dumps(
            {
                "url": "str",
            }
        )

    def to_json(self) -> Dict[str, str]:
        return {
            "url": self.url,
        }


class BasicInfo:
    full_name: str
    email: str
    about: str
    phone: str
    location: Optional[str]
    links: List[Link]
    role: Optional[str]

    def __init__(
        self,
        full_name: str,
        email: str,
        about: str,
        phone: str,
        location: Optional[str] = None,
        links: Optional[List[Link]] = None,
        role: Optional[str] = None,
    ):
        self.full_name = full_name
        self.email = email
        self.about = about
        self.phone = phone
        self.location = location
        self.links = links if links else []
        self.role = role

    @staticmethod
    def to_prompt() -> str:
        return {
            "full_name": {
                "type": "string",
                "description": "The full name of the individual.",
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                "description": "The email address of the individual in the standard email format.",
            },
            "phone": {
                "type": "string",
                "pattern": "^\\+?[1-9]\\d{1,14}$",
                "description": "The phone number of the individual in E.164 format, e.g., +123456789.",
            },
            "location": {
                "type": "string",
                "description": "The geographical location of the individual, typically including city and country.",
            },
            "links": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^https?://[\\w.-]+(?:\\.[\\w\\.-]+)+[/\\w\\.-]*$",
                    "description": "A list of URLs related to the individual, such as LinkedIn or portfolio links.",
                },
            },
            "about": {
                "type": "string",
                "description": "A description of the individual, typically including career goals and aspirations.",
            },
            "role": {
                "type": "string",
                "description": "The role or job title of the individual.",
            },
        }

    def to_json(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "links": [link.to_json() for link in self.links],
        }


class Skills:
    skill_names: list[str]

    def __init__(self, name: list[str]):
        self.skill_names = name

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "string",
                "description": "This step is very critical! Include all the skills that are both included and implied in resume, e.g., 'Python', 'Project Management'.",
            },
        }

    def to_json(self) -> Dict[str, Union[str, str]]:
        return {
            "name": self.skill_names,
        }


class Education:
    degree_type: str
    field_of_study: List[str]
    start_year: int
    graduation_year: int
    institution: str
    education_level: str

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {
                        "type": "string",
                        "description": "The degree obtained, e.g., 'Bachelor of Science'.",
                    },
                    "field_of_study": {
                        "type": "string",
                        "description": "The field of study, e.g., 'Computer Science'.",
                    },
                    "institution": {
                        "type": "string",
                        "description": "The name of the institution where the degree was obtained.",
                    },
                    "education_level": {
                        "type": "string",
                        "enum": ["high school", "bachelors", "masters", "doctoral"],
                        "description": "The education level, one of the following: 'high school', 'bachelors', 'masters', 'doctoral'.",
                    },
                    "start_year": {
                        "type": "integer",
                        "description": "The year of starting the degree in the format YYYY, e.g., '2019'.",
                    },
                    "graduation_year": {
                        "type": "integer",
                        "description": "The year of graduation in the format YYYY, e.g., '2023'.",
                    },
                },
            },
        }

    def to_json(self) -> Dict[str, Union[str, int, List[str]]]:
        return {
            "degree": self.degree_type,
            "field_of_study": self.field_of_study,
            "start_year": self.start_year,
            "graduation_year": self.graduation_year,
            "institution": self.institution,
            "education_level": self.education_level,
        }


class WorkExperience:
    job_title: str
    company: str
    description: Optional[str]
    start_year: Optional[int]
    start_month: Optional[int]
    end_year: Optional[int]
    end_month: Optional[int]
    duration: Optional[str]

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_title": {
                        "type": "string",
                        "description": "The job title of the individual in a specific role.",
                    },
                    "company": {
                        "type": "string",
                        "description": "The name of the company where the individual worked.",
                    },
                    "description": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A description of responsibilities or achievements in the job.",
                        },
                    },
                    "start_year": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{4})$",
                        "description": "The start year of the job or null if not applicable.",
                    },
                    "start_month": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{2})$",
                        "description": "The start month of the job in the format 'MM' or null if not applicable.",
                    },
                    "end_year": {
                        "type": ["integer", "string", "null"],
                        "pattern": "^(\\d{4}|present)$",
                        "description": "The end year of the job or 'present' if currently working there or null if not applicable.",
                    },
                    "end_month": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{2})$",
                        "description": "The end month of the job in the format 'MM' or null if not applicable.",
                    },
                    "duration": {
                        "type": ["string", "null"],
                        "pattern": "^(\\d+[MY])$",
                        "description": "Duration should be a number followed by 'M' for months or 'Y' for years, e.g., '8M' for 8 months or '4Y' for 4 years, or null if not applicable.",
                    },
                },
            },
        }

    def to_json(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "job_title": self.job_title,
            "company": self.company,
            "description": self.description,
            "start_year": self.start_year,
            "start_month": self.start_month,
            "end_year": self.end_year,
            "end_month": self.end_month,
            "duration": self.duration,
        }


class Languages:
    language_names: list[tuple[str, str]]

    def __init__(self, name: list[str]):
        self.language_names = name

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The language spoken by the individual, e.g., 'English'.",
                    },
                    "proficiency": {
                        "type": "string",
                        "enum": [
                            "Beginner",
                            "Intermediate",
                            "Advanced",
                            "Fluent",
                            "Native",
                            "Bilingual",
                        ],
                        "description": "The proficiency level in the language, one of the following: 'Beginner', 'Intermediate', 'Advanced', 'Fluent', 'Native', 'Bilingual'.",
                    },
                },
            },
        }

    def to_json(self) -> Dict[str, Union[str, str]]:
        return {
            "language": self.language_names,
        }


class ExtraActivity:
    title: str
    organization: Optional[str]
    description: str
    start_year: Optional[int]
    start_month: Optional[int]
    end_year: Optional[int]
    end_month: Optional[int]
    duration: Optional[str]

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the activity, e.g., 'Volunteer Data Analyst'.",
                    },
                    "organization": {
                        "type": "string",
                        "description": "The organization where the activity took place, e.g., 'Nonprofit XYZ'.",
                    },
                    "description": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A description of the activities or responsibilities undertaken.",
                        },
                    },
                    "start_year": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{4})$",
                        "description": "The start year of the activity or null if not applicable.",
                    },
                    "start_month": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{2})$",
                        "description": "The start month of the activity in the format 'MM' or null if not applicable.",
                    },
                    "end_year": {
                        "type": ["integer", "string", "null"],
                        "pattern": "^(\\d{4}|present)$",
                        "description": "The end year of the activity or 'present' if currently active or null if not applicable.",
                    },
                    "end_month": {
                        "type": ["integer", "null"],
                        "pattern": "^(\\d{2})$",
                        "description": "The end month of the activity in the format 'MM' or null if not applicable.",
                    },
                    "duration": {
                        "type": ["string", "null"],
                        "pattern": "^(\\d+[MY])$",
                        "description": "Duration should be a number followed by 'M' for months or 'Y' for years, e.g., '12M' for 12 months, or null if not applicable.",
                    },
                },
            },
        }

    def to_json(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "title": self.title,
            "team": self.team,
            "description": self.description,
        }


class Certification:
    name: str
    institution: str
    year_obtained: int

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the certification, e.g., 'AWS Certified Solutions Architect'.",
                    },
                    "institution": {
                        "type": "string",
                        "description": "The institution that issued the certification, e.g., 'Amazon Web Services'.",
                    },
                    "year_obtained": {
                        "type": "int",
                        "pattern": "^(19|20)\\d{2}$",
                        "description": "The year when the certification was obtained.",
                    },
                },
            },
        }

    def to_json(self) -> Dict[str, Union[str, int]]:
        return {
            "name": self.name,
            "institution": self.institution,
            "year_obtained": self.year_obtained,
        }


class Seniority:
    name: str

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def to_prompt() -> str:
        return {
            "type": "string",
            "enum": [
                "Entry-level",
                "Junior",
                "Mid-level",
                "Senior",
                "Lead",
                "Manager",
                "Director",
                "Vice President",
                "C-level",
                "Founder/Owner",
            ],
            "description": "The seniority level of the individual, one of the specified options.",
        }


class CVData:
    basic_info: BasicInfo
    skills: List[str]
    educations: List[Education]
    languages: Languages
    certifications: List[Certification]
    seniority: Seniority
    work_experiences: List[WorkExperience]
    extra_activities: List[ExtraActivity]

    remote: Optional[str]

    # THINGS THAT CAN BE EXTRACTED LATER:
    # current_company: Optional[str]
    # graduated_from_school: Optional[str]
    # graduated_from_field: Optional[str]
    # education_level: Optional[str]
    # current_job_title: Optional[str]
    # current_employment_status: Optional[str]

    # desired_locations: List[str]
    # desired_position: Optional[str]
    # industries: List[str]
    # desired_job_type: Optional[str]
    # job_search_progress: Optional[str]
    # is_freelancer: Optional[str]
    # graduate_year: Optional[int]
    # graduate_degree_type: Optional[str]
    # work_experience: Optional[str]
    # relevant_work_experience: Optional[str]
    # number_of_management: Optional[str]
    # profession: List[str]
    # rn_work_experience: Optional[str]

    @staticmethod
    def to_prompt() -> str:
        return json.dumps(
            {
                "type": "object",
                "properties": {
                    "basic_info": BasicInfo.to_prompt(),
                    "skills": Skills.to_prompt(),
                    "educations": Education.to_prompt(),
                    "languages": Languages.to_prompt(),
                    "experiences": WorkExperience.to_prompt(),
                    "seniority_level": Seniority.to_prompt(),
                    "extra_activities": ExtraActivity.to_prompt(),
                    "certifications": Certification.to_prompt(),
                    "remote": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Whether the individual is open to remote work or not.",
                    },
                },
            }
        )

    @staticmethod
    def check_json(json_data) -> bool:
        if not isinstance(json_data, dict):
            return False

        for key in CVData.__dict__.keys():
            if key not in json_data:
                return False

        return True

    @staticmethod
    def from_json(json_data) -> None:
        if not CVData.check_json(json_data):
            return None

        instance = CVData()
        for key in instance.__dict__.keys():
            setattr(instance, key, json_data[key])

        instance.work_experiences = [
            WorkExperience.from_json(work_experience)
            for work_experience in json_data["work_experiences"]
        ]
        instance.educations = [
            Education.from_json(education) for education in json_data["educations"]
        ]
        instance.extra_activities = [
            ExtraActivity.from_json(extra_activity)
            for extra_activity in json_data["extra-activities"]
        ]

        return instance

    def to_json(self) -> Dict[str, Union[str, List[Dict[str, Union[str, int, bool]]]]]:
        return {
            "name": self.name,
            "email": self.email,
            "about": self.about,
            "country": self.country,
            "city": self.city,
            "current_company": self.current_company,
            "graduated_from_school": self.graduated_from_school,
            "graduated_from_field": self.graduated_from_field,
            "education_level": self.education_level,
            "remote": self.remote,
            "languages": self.languages,
            "location": self.location,
            "desired_locations": self.desired_locations,
            "current_job_title": self.current_job_title,
            "desired_position": self.desired_position,
            "work_experiences": [
                work_experience.to_json() for work_experience in self.work_experiences
            ],
            "educations": [education.to_json() for education in self.educations],
            "skills": self.skills,
            "role": self.role,
            "current_employment_status": self.current_employment_status,
        }


if __name__ == "__main__":
    cv_data_dict = CVData.to_prompt()
    print(json.loads(CVData.to_prompt()))
