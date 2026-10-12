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

from typing import Literal


class JobExtractionRequest(BaseModel):
    description: str = Field(
        min_length=30,
        max_length=20000,
        description="The full job description to analyze.",
    )


class JobRequirements(BaseModel):
    title: str | None
    company: str | None
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_years_experience: float | None
    work_arrangement: Literal["remote", "hybrid", "onsite"] | None
    location: str | None