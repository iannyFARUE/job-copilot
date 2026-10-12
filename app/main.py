from fastapi import FastAPI

from app.models import CandidateProfile
from app.candidate_data import candidate_profile


app = FastAPI(
    title="Job Application Copilot",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/profile", response_model=CandidateProfile)
def get_profile() -> CandidateProfile:
    return candidate_profile