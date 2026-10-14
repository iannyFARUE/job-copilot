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

class SkillMatch(BaseModel):
    skill: str
    status: Literal["project_evidence", "listed_only", "not_found"]
    project_ids: list[str]
    evidence: list[str]


class JobMatchResult(BaseModel):
    title: str | None
    required_matches: list[SkillMatch]
    preferred_matches: list[SkillMatch]
    needs_review: list[str]


class JobAnalysisResult(BaseModel):
    requirements: JobRequirements
    match: JobMatchResult

class AgentRequest(BaseModel):
    question: str = Field(min_length=5, max_length=3000)


class AgentResponse(BaseModel):
    answer: str
    lookups: list[SkillMatch]