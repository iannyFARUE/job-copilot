from fastapi import FastAPI

from app.models import AgentResumeRequest, CandidateProfile, JobExtractionRequest, JobRequirements,JobMatchResult, JobAnalysisResult, AgentRequest, AgentResponse
from app.agent_service import AgentError, ask_agent, build_agent_graph,resume_agent
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
from app.matching_service import match_job
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Request
from langgraph.checkpoint.sqlite import SqliteSave

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    database_path = data_dir / "checkpoints.sqlite"

    with SqliteSave.from_conn_string(str(database_path)) as saver:
        saver.setup()
        app.state.agent_graph = build_agent_graph(saver)
        yield


app = FastAPI(
    title="Job Application Copilot",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/profile", response_model=CandidateProfile)
def get_profile() -> CandidateProfile:
    return candidate_profile

@app.post("/jobs/extract", response_model=JobRequirements)
def extract_job(request: JobExtractionRequest) -> JobRequirements:
    return extract_job_or_raise(request.description)

def extract_job_or_raise(description: str) -> JobRequirements:
    try:
        return extract_job_requirements(description)

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

@app.post("/jobs/match", response_model=JobMatchResult)
def match_job_endpoint(job: JobRequirements) -> JobMatchResult:
    return match_job(job, candidate_profile)


@app.post("/jobs/analyze", response_model=JobAnalysisResult)
def analyze_job(request: JobExtractionRequest) -> JobAnalysisResult:
    requirements = extract_job_or_raise(request.description)

    match_result = match_job(
        job=requirements,
        profile=candidate_profile,
    )

    return JobAnalysisResult(
        requirements=requirements,
        match=match_result,
    )

@app.post("/agent/ask", response_model=AgentResponse)
def agent_ask(request: AgentRequest,
    http_request: Request) -> AgentResponse:
    try:
        return ask_agent(
            question=request.question,
            thread_id=request.thread_id,
            graph=http_request.app.state.agent_graph,
        )

    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The model request timed out.",
        ) from exc

    except APIError as exc:
        logger.warning(
            "Agent request failed: %s, status=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        raise HTTPException(
            status_code=502,
            detail="The model request failed. Check server logs.",
        ) from exc

    except AgentError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The model supplied invalid tool arguments.",
        ) from exc


@app.post("/agent/resume", response_model=AgentResponse)
def agent_resume(
    request: AgentResumeRequest,
    http_request: Request,
) -> AgentResponse:
    try:
        return resume_agent(
            thread_id=request.thread_id,
            graph=http_request.app.state.agent_graph,
        )

    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The model request timed out.",
        ) from exc

    except APIError as exc:
        logger.warning(
            "Resume failed: %s, status=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        raise HTTPException(
            status_code=502,
            detail="The model request failed. Check server logs.",
        ) from exc

    except AgentError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The model supplied invalid tool arguments.",
        ) from exc