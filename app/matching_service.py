from app.models import (
    CandidateProfile,
    JobMatchResult,
    JobRequirements,
    SkillMatch,
)


def normalize(skill: str) -> str:
    return skill.strip().casefold()


def match_skill(
    skill: str,
    profile: CandidateProfile,
) -> SkillMatch:
    normalized_skill = normalize(skill)

    matching_projects = [
        project
        for project in profile.projects
        if normalized_skill
        in {normalize(tech) for tech in project.technologies}
    ]

    if matching_projects:
        return SkillMatch(
            skill=skill,
            status="project_evidence",
            project_ids=[project.id for project in matching_projects],
            evidence=[
                f"{project.name}: {statement}"
                for project in matching_projects
                for statement in project.evidence
            ],
        )

    listed_skills = {normalize(item) for item in profile.skills}

    if normalized_skill in listed_skills:
        return SkillMatch(
            skill=skill,
            status="listed_only",
            project_ids=[],
            evidence=[],
        )

    return SkillMatch(
        skill=skill,
        status="not_found",
        project_ids=[],
        evidence=[],
    )


def match_job(
    job: JobRequirements,
    profile: CandidateProfile,
) -> JobMatchResult:
    needs_review = []

    if job.minimum_years_experience is not None:
        needs_review.append(
            f"The role requires at least "
            f"{job.minimum_years_experience:g} years of experience. "
            "Your profile does not yet include employment dates."
        )

    if job.work_arrangement is not None or job.location is not None:
        needs_review.append(
            "Confirm the job's location and work arrangement suit you. "
            "Your profile does not yet include these preferences."
        )

    return JobMatchResult(
        title=job.title,
        required_matches=[
            match_skill(skill, profile)
            for skill in job.required_skills
        ],
        preferred_matches=[
            match_skill(skill, profile)
            for skill in job.preferred_skills
        ],
        needs_review=needs_review,
    )