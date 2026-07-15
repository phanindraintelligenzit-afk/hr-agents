"""Candidate scoring rubric for the HR Recruitment ATS Agent.

Scoring Dimensions (per design doc):
| Dimension          | Weight |
|--------------------|--------|
| Skills match       | 35%    |
| Experience level   | 25%    |
| Education          | 15%    |
| Certifications     | 10%    |
| Location/Remote    | 10%    |
| Communication      |  5%    |
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hr_agents.shared.llm import LLMClient, get_llm

logger = logging.getLogger("hr_agents.recruitment.scoring")

# Scoring rubric weights (must sum to 1.0)
SCORING_WEIGHTS: dict[str, float] = {
    "skills_match": 0.35,
    "experience_level": 0.25,
    "education": 0.15,
    "certifications": 0.10,
    "location_fit": 0.10,
    "communication": 0.05,
}


def score_candidate(
    candidate: dict[str, Any],
    jd_parsed: dict[str, Any],
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Score a single candidate against the job description.

    Uses LLM for semantic dimensions and direct comparison for structured dimensions.

    Returns a dict with per-dimension scores and an overall weighted score (0-1).
    """
    if llm is None:
        llm = get_llm()

    # ── Skills match (35%) ──
    skills_required = jd_parsed.get("skills_required", [])
    skills_nice = jd_parsed.get("skills_nice_to_have", [])
    candidate_skills = [s.lower() for s in candidate.get("skills", [])]

    if skills_required:
        required_matches = sum(1 for s in skills_required if s.lower() in candidate_skills)
        skills_score = required_matches / len(skills_required)
    else:
        skills_score = 0.5

    # Bonus for nice-to-have skills
    if skills_nice:
        nice_matches = sum(1 for s in skills_nice if s.lower() in candidate_skills)
        skills_score = min(1.0, skills_score + nice_matches * 0.05)

    # ── Experience level (25%) ──
    min_exp = jd_parsed.get("min_experience_years", 0)
    candidate_exp = candidate.get("experience_years", 0)

    if min_exp > 0 and candidate_exp > 0:
        if candidate_exp >= min_exp * 1.5:
            exp_score = 1.0
        elif candidate_exp >= min_exp:
            exp_score = 0.7 + 0.3 * (candidate_exp - min_exp) / (min_exp * 0.5) if min_exp > 0 else 0.7
        else:
            exp_score = max(0, candidate_exp / min_exp)
    elif candidate_exp > 0:
        exp_score = 0.7  # No minimum specified but candidate has experience
    else:
        exp_score = 0.5

    exp_score = min(1.0, max(0.0, exp_score))

    # ── Education (15%) ──
    education = candidate.get("education", "").lower()
    edu_req = jd_parsed.get("education_required", "").lower()

    if not edu_req:
        edu_score = 0.8  # No specific requirement
    elif "phd" in education or "ph.d" in education:
        edu_score = 1.0
    elif "master" in education or "mba" in education or "msc" in education:
        edu_score = 0.9 if "bachelor" in edu_req else 0.7
    elif "bachelor" in education or "bsc" in education or "b.tech" in education:
        edu_score = 0.8 if "bachelor" in edu_req or "master" in edu_req else 0.5
    else:
        edu_score = 0.3

    # ── Certifications (10%) ──
    candidate_certs = [c.lower() for c in candidate.get("certifications", [])]
    preferred_certs = [c.lower() for c in jd_parsed.get("certifications_preferred", [])]

    if preferred_certs:
        cert_matches = sum(1 for c in preferred_certs if any(pc in c for pc in candidate_certs) or any(cc in c for cc in candidate_certs))
        # Also check the other direction
        cert_matches = max(
            cert_matches,
            sum(1 for cc in candidate_certs if any(pc in cc for pc in preferred_certs))
        )
        cert_score = min(1.0, cert_matches / len(preferred_certs))
    elif candidate_certs:
        cert_score = 0.7  # Has certs even if not specifically requested
    else:
        cert_score = 0.3

    # ── Location fit (10%) ──
    jd_location = jd_parsed.get("location", "").lower()
    candidate_location = candidate.get("location", "").lower()

    if not jd_location or jd_location == "remote":
        loc_score = 1.0  # Remote role → everyone fits
    elif jd_location in candidate_location or candidate_location in jd_location:
        loc_score = 1.0
    else:
        loc_score = 0.5  # Might need relocation

    # ── Communication (5%) ──
    cover = candidate.get("cover_letter", "")
    if cover and len(cover) > 50:
        comm_score = 0.8
    elif cover:
        comm_score = 0.5
    else:
        comm_score = 0.5  # Default: no cover letter

    # ── Use LLM for nuanced scoring ──
    try:
        llm_scores = _llm_scoring(candidate, jd_parsed, llm)
        # Blend LLM scores with rule-based scores (50/50 blend)
        skills_score = 0.5 * skills_score + 0.5 * llm_scores.get("skills_score", skills_score)
        exp_score = 0.5 * exp_score + 0.5 * llm_scores.get("experience_score", exp_score)
        edu_score = 0.5 * edu_score + 0.5 * llm_scores.get("education_score", edu_score)
        cert_score = 0.5 * cert_score + 0.5 * llm_scores.get("certification_score", cert_score)
        comm_score = 0.5 * comm_score + 0.5 * llm_scores.get("communication_score", comm_score)
    except Exception:
        logger.warning("LLM scoring fell back to rule-based only")

    # ── Compute weighted overall ──
    breakdown = {
        "skills_match": round(skills_score, 4),
        "experience_level": round(exp_score, 4),
        "education": round(edu_score, 4),
        "certifications": round(cert_score, 4),
        "location_fit": round(loc_score, 4),
        "communication": round(comm_score, 4),
    }

    overall = sum(breakdown[k] * SCORING_WEIGHTS[k] for k in SCORING_WEIGHTS)
    overall = round(min(1.0, max(0.0, overall)), 4)

    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "name": candidate.get("name", "Unknown"),
        "overall_score": overall,
        "score_breakdown": breakdown,
    }


def _llm_scoring(
    candidate: dict[str, Any],
    jd_parsed: dict[str, Any],
    llm: LLMClient,
) -> dict[str, float]:
    """Use LLM to get nuanced scores for each dimension."""
    prompt = (
        "You are an expert HR recruiter. Score this candidate against the job description.\n\n"
        f"## Job Description\n{json.dumps(jd_parsed, indent=2)}\n\n"
        f"## Candidate\n{json.dumps(candidate, indent=2)}\n\n"
        "Output a JSON object with scores (0.0 to 1.0) for these dimensions:\n"
        "- skills_score\n- experience_score\n- education_score\n"
        "- certification_score\n- location_score\n- communication_score\n\n"
        "Respond ONLY with JSON."
    )

    response = llm.invoke(prompt, system_prompt="You are an expert recruiter. Output JSON only.")
    try:
        parsed = json.loads(response)
        return {
            "skills_score": float(parsed.get("skills_score", 0.5)),
            "experience_score": float(parsed.get("experience_score", 0.5)),
            "education_score": float(parsed.get("education_score", 0.5)),
            "certification_score": float(parsed.get("certification_score", 0.5)),
            "location_score": float(parsed.get("location_score", 0.5)),
            "communication_score": float(parsed.get("communication_score", 0.5)),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def rank_candidates(
    scores: list[dict[str, Any]],
    max_shortlist: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank candidates by overall score and return shortlist.

    Args:
        scores: List of score dicts from score_candidate()
        max_shortlist: Maximum number of candidates to shortlist
        min_score: Minimum overall score to be shortlisted

    Returns:
        Shortlisted candidates (sorted by score descending).
    """
    ranked = sorted(scores, key=lambda s: s.get("overall_score", 0), reverse=True)
    shortlisted = [s for s in ranked if s.get("overall_score", 0) >= min_score]
    return shortlisted[:max_shortlist]