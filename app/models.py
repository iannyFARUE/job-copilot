from pydantic import BaseModel, Field


class Project(BaseModel):
    id: str
    name: str
    description: str
    technologies: list[str]
    evidence: list[str] = Field(min_length=1)


class CandidateProfile(BaseModel):
    name: str
    target_roles: list[str]
    skills: list[str]
    projects: list[Project]