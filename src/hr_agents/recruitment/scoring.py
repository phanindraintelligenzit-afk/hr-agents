"""Candidate scoring rubric for the HR Recruitment ATS Agent — LLM-driven.

When a real LLM (OpenRouter/OpenAI) is connected, DeepSeek analyzes the FULL
candidate profile against the JD and returns genuine scores with reasoning.
When mock is used, falls back to rule-based keyword matching.

Scoring Dimensions:
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

    When a real LLM is configured, sends full candidate profile + JD to the LLM
    for genuine semantic analysis. When mock, uses rule-based comparison.

    Returns a dict with per-dimension scores, overall weighted score (0-1),
    and LLM reasoning (when available).
    """
    if llm is None:
        llm = get_llm()

    # If using a real LLM (not mock), do full LLM-driven scoring
    if llm.provider != "mock":
        return _llm_full_scoring(candidate, jd_parsed, llm)

    # Otherwise, use rule-based scoring
    return _rule_based_scoring(candidate, jd_parsed)


def _llm_full_scoring(
    candidate: dict[str, Any],
    jd_parsed: dict[str, Any],
    llm: LLMClient,
) -> dict[str, Any]:
    """Use LLM to deeply analyze candidate vs JD and return scores + reasoning."""
    prompt = f"""You are an expert technical recruiter. Analyze this candidate against the job description and provide a detailed scored evaluation.

## JOB DESCRIPTION
Title: {jd_parsed.get('title', 'Unknown')}
Department: {jd_parsed.get('department', 'N/A')}
Location: {jd_parsed.get('location', 'Remote')}
Min Experience: {jd_parsed.get('min_experience_years', 0)} years
Required Skills: {', '.join(jd_parsed.get('skills_required', []))}
Nice-to-Have Skills: {', '.join(jd_parsed.get('skills_nice_to_have', []))}
Education Required: {jd_parsed.get('education_required', 'Not specified')}
Preferred Certifications: {', '.join(jd_parsed.get('certifications_preferred', []))}
Responsibilities:
{chr(10).join('- ' + r for r in jd_parsed.get('responsibilities', []))}

## CANDIDATE PROFILE
Name: {candidate.get('name', 'Unknown')}
Skills: {', '.join(candidate.get('skills', []))}
Years of Experience: {candidate.get('experience_years', 0)}
Education: {candidate.get('education', 'Not specified')}
Certifications: {', '.join(candidate.get('certifications', []))}
Location: {candidate.get('location', 'Not specified')}
Resume Summary: {candidate.get('raw_text', 'N/A')[:1000]}

## INSTRUCTIONS
Evaluate the candidate across these 6 dimensions. For each, provide:
1. A score from 0.0 to 1.0 with justification
2. Specific evidence from the candidate's profile

Dimensions:
1. skills_match (weight 35%) — How well do the candidate's skills match required + nice-to-have?
2. experience_level (weight 25%) — Does the candidate have enough relevant experience? Consider quality too.
3. education (weight 15%) — Does education match or exceed requirements?
4. certifications (weight 10%) — Relevant certifications?
5. location_fit (weight 10%) — Good location fit (remote-friendly)?
6. communication (weight 5%) — Based on clarity of resume/cover letter

Respond ONLY with valid JSON:
{{
  "skills_score": 0.0-1.0,
  "experience_score": 0.0-1.0,
  "education_score": 0.0-1.0,
  "certification_score": 0.0-1.0,
  "location_score": 0.0-1.0,
  "communication_score": 0.0-1.0,
  "reasoning": {{
    "skills_match": "brief justification",
    "experience_level": "brief justification",
    "education": "brief justification",
    "certifications": "brief justification",
    "location_fit": "brief justification",
    "communication": "brief justification",
    "summary": "1-2 sentence overall assessment"
  }}
}}"""

    response = llm.invoke(
        prompt,
        system_prompt="You are an expert technical recruiter. Output JSON only. Be thorough and fair.",
        temperature=0.2,
        max_tokens=4096,
    )
    logger.info("LLM scoring response: %s", response[:200])

    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("LLM scoring JSON parse failed: %s — response: %s", e, response[:300])
        return _rule_based_scoring(candidate, jd_parsed)

    # Extract scores
    skills_score = max(0.0, min(1.0, float(parsed.get("skills_score", 0.5))))
    exp_score = max(0.0, min(1.0, float(parsed.get("experience_score", 0.5))))
    edu_score = max(0.0, min(1.0, float(parsed.get("education_score", 0.5))))
    cert_score = max(0.0, min(1.0, float(parsed.get("certification_score", 0.5))))
    loc_score = max(0.0, min(1.0, float(parsed.get("location_score", 0.5))))
    comm_score = max(0.0, min(1.0, float(parsed.get("communication_score", 0.5))))

    reasoning = parsed.get("reasoning", {})

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
        "llm_reasoning": reasoning,
    }


def _rule_based_scoring(
    candidate: dict[str, Any],
    jd_parsed: dict[str, Any],
) -> dict[str, Any]:
    """Fallback: deterministic keyword-based scoring when no real LLM is available."""

    # ── Skills match (35%) ──
    skills_required = jd_parsed.get("skills_required", [])
    skills_nice = jd_parsed.get("skills_nice_to_have", [])
    candidate_skills = [s.lower() for s in candidate.get("skills", [])]

    if skills_required:
        required_matches = sum(1 for s in skills_required if s.lower() in candidate_skills)
        skills_score = required_matches / len(skills_required)
    else:
        skills_score = 0.5

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
        exp_score = 0.7
    else:
        exp_score = 0.5
    exp_score = min(1.0, max(0.0, exp_score))

    # ── Education (15%) ──
    education = candidate.get("education", "").lower()
    edu_req = jd_parsed.get("education_required", "").lower()

    if not edu_req:
        edu_score = 0.8
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
        cert_matches = max(cert_matches, sum(1 for cc in candidate_certs if any(pc in cc for pc in preferred_certs)))
        cert_score = min(1.0, cert_matches / len(preferred_certs))
    elif candidate_certs:
        cert_score = 0.7
    else:
        cert_score = 0.3

    # ── Location fit (10%) ──
    jd_location = jd_parsed.get("location", "").lower()
    candidate_location = candidate.get("location", "").lower()

    if not jd_location or jd_location == "remote":
        loc_score = 1.0
    elif jd_location in candidate_location or candidate_location in jd_location:
        loc_score = 1.0
    else:
        loc_score = 0.5

    # ── Communication (5%) ──
    cover = candidate.get("cover_letter", "")
    comm_score = 0.8 if cover and len(cover) > 50 else (0.5 if cover else 0.5)

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
        "llm_reasoning": {"summary": "Rule-based scoring (no LLM connected). Switch to OpenRouter for AI analysis."},
    }


def rank_candidates(
    scores: list[dict[str, Any]],
    max_shortlist: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank candidates by overall score and return shortlist."""
    ranked = sorted(scores, key=lambda s: s.get("overall_score", 0), reverse=True)
    shortlisted = [s for s in ranked if s.get("overall_score", 0) >= min_score]
    return shortlisted[:max_shortlist]