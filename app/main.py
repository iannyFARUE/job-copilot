from fastapi import FastAPI

from app.models import CandidateProfile, JobExtractionRequest, JobRequirements
from app.candidate_data import candidate_profile
import logging

from fastapi import HTTPException
from openai import (
    APIError,
    APITimeoutError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
)
from pydantic import ValidationError

from app.llm_service import ExtractionError, extract_job_requirements


logger = logging.getLogger(__name__)

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

@app.post("/jobs/extract", response_model=JobRequirements)
def extract_job(request: JobExtractionRequest) -> JobRequirements:
    try:
        return extract_job_requirements(request.description)

    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The model request timed out. Please try again.",
        ) from exc

    except APIError as exc:
        logger.warning(
            "LLM request failed: %s, status=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        raise HTTPException(
            status_code=502,
            detail="The model request failed. Check server logs and API configuration.",
        ) from exc

    except (
        ExtractionError,
        ValidationError,
        LengthFinishReasonError,
        ContentFilterFinishReasonError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail="The model could not produce a valid extraction.",
        ) from exc