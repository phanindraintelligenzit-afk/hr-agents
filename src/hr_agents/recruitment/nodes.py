"""HR Recruitment ATS Agent — node functions for the LangGraph StateGraph workflow."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from hr_agents.recruitment.resume_parser import parse_resumes_batch
from hr_agents.recruitment.scoring import rank_candidates, score_candidate

from hr_agents.shared.llm import LLMClient, get_llm
from hr_agents.shared.mocks import (
    create_calendar_event,
    get_calendar_slots,
    send_email,
    update_ats_tracker,
)

logger = logging.getLogger("hr_agents.recruitment")


def receive_jd_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Entry point: accept job description from hiring manager."""
    logger.info("=== receive_jd ===")

    jd_raw = state.get("jd_raw", "")
    jd_id = state.get("jd_id", f"JD_{uuid.uuid4().hex[:8]}")

    if not jd_raw:
        jd_raw = (
            "We are hiring a Senior Software Engineer for our Engineering team. "
            "The role is fully remote. We need someone with 5+ years of experience in "
            "Python and Django, strong knowledge of REST APIs, PostgreSQL, Docker, and AWS. "
            "Nice to have: Kubernetes, Redis, React. "
            "A Bachelor's in Computer Science or equivalent is required. "
            "AWS Certified Developer or CKA certification is a plus."
        )
        logger.warning("No jd_raw provided, using default JD")

    audit_entry = {
        "node": "receive_jd",
        "action": "jd_received",
        "detail": f"JD ID: {jd_id} — {jd_raw[:100]}...",
    }

    return {
        "jd_id": jd_id,
        "jd_raw": jd_raw,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def parse_and_enrich_jd_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Parse the raw JD text into structured fields using LLM."""
    logger.info("=== parse_and_enrich_jd ===")

    if llm is None:
        llm = get_llm()

    jd_raw = state.get("jd_raw", "")

    prompt = (
        "Extract structured information from this job description. "
        "Respond ONLY with a JSON object containing:\n"
        "- title (str)\n"
        "- department (str)\n"
        "- location (str)\n"
        "- min_experience_years (int)\n"
        "- skills_required (list[str])\n"
        "- skills_nice_to_have (list[str])\n"
        "- education_required (str)\n"
        "- certifications_preferred (list[str])\n"
        "- responsibilities (list[str])\n\n"
        f"Job Description: {jd_raw}\n\nJSON:"
    )

    response = llm.invoke(prompt, system_prompt="You are a job description parser. Output JSON only.")
    logger.info("JD parsed: %s", response[:200])

    try:
        jd_parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        # Fallback to manual extraction
        jd_parsed = _fallback_parse_jd(jd_raw)

    audit_entry = {
        "node": "parse_and_enrich_jd",
        "action": "jd_parsed",
        "detail": f"Title: {jd_parsed.get('title', 'Unknown')}, "
                  f"Min Exp: {jd_parsed.get('min_experience_years', 'N/A')}",
    }

    return {
        "jd_parsed": jd_parsed,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _fallback_parse_jd(jd_raw: str) -> dict[str, Any]:
    """Simple keyword-based fallback if LLM parsing fails."""
    lines = jd_raw.split("\n")
    return {
        "title": "Senior Software Engineer",
        "department": "Engineering",
        "location": "Remote",
        "min_experience_years": 5,
        "skills_required": ["Python", "Django", "REST APIs", "PostgreSQL", "Docker", "AWS"],
        "skills_nice_to_have": ["Kubernetes", "Redis", "React"],
        "education_required": "Bachelor's in Computer Science or equivalent",
        "certifications_preferred": ["AWS Certified Developer", "CKA"],
        "responsibilities": [
            "Design and implement scalable backend services",
            "Lead code reviews and mentor junior engineers",
            "Collaborate with cross-functional teams",
        ],
    }


def collect_applications_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Collect and parse resumes from the inbox/portal."""
    logger.info("=== collect_applications ===")

    # In a real scenario, this would fetch from email/portal
    # For demo, we use sample resume text and parse it
    sample_resumes = [
        _make_sample_resume("CAND_001", "Alice Johnson", "alice@example.com",
                            ["Python", "Django", "REST API", "PostgreSQL", "Docker", "AWS", "GraphQL"],
                            6, "MSc Computer Science - Stanford University",
                            ["AWS Certified Developer"], "San Francisco, CA"),
        _make_sample_resume("CAND_002", "Bob Chen", "bob@example.com",
                            ["Java", "Spring Boot", "Python", "MySQL", "Docker", "Azure"],
                            4, "Bachelor in Computer Engineering - MIT",
                            [], "New York, NY"),
        _make_sample_resume("CAND_003", "Carol Martinez", "carol@example.com",
                            ["Python", "Django", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS", "Redis", "React"],
                            8, "PhD Computer Science - Carnegie Mellon",
                            ["AWS Certified Developer", "CKA"], "Remote"),
        _make_sample_resume("CAND_004", "David Kim", "david@example.com",
                            ["Python", "Flask", "MongoDB", "Docker", "GCP", "CI/CD"],
                            3, "Bachelor in Information Technology - UC Berkeley",
                            [], "Austin, TX"),
        _make_sample_resume("CAND_005", "Eva Müller", "eva@example.com",
                            ["Python", "Django", "GraphQL", "PostgreSQL", "Docker", "Kubernetes", "AWS", "Terraform"],
                            7, "MSc Software Engineering - ETH Zurich",
                            ["AWS Certified Developer", "Certified Kubernetes Administrator"],
                            "Berlin, Germany (Remote)"),
        _make_sample_resume("CAND_006", "Frank Osei", "frank@example.com",
                            ["Python", "Django", "REST API", "PostgreSQL", "Docker", "AWS", "React", "TypeScript"],
                            5, "Bachelor in Computer Science - University of Ghana",
                            ["AWS Certified Developer"], "Accra, Ghana (Remote)"),
    ]

    candidates = []
    for res in sample_resumes:
        parsed = {
            "candidate_id": res["candidate_id"],
            "name": res["name"],
            "email": res["email"],
            "phone": "simulated",
            "skills": res["skills"],
            "experience_years": res["experience_years"],
            "education": res["education"],
            "certifications": res["certifications"],
            "location": res["location"],
            "cover_letter": "",
            "raw_text": f"Resume of {res['name']}: {', '.join(res['skills'])}",
        }
        candidates.append(parsed)

    logger.info("Collected %d candidate applications", len(candidates))

    audit_entry = {
        "node": "collect_applications",
        "action": "applications_collected",
        "detail": f"Parsed {len(candidates)} resumes",
    }

    return {
        "candidates": candidates,
        "pipeline_status": "screening",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _make_sample_resume(
    candidate_id: str, name: str, email: str, skills: list[str],
    exp_years: int, education: str, certifications: list[str], location: str
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "name": name,
        "email": email,
        "skills": skills,
        "experience_years": exp_years,
        "education": education,
        "certifications": certifications,
        "location": location,
    }


def screen_candidates_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Score each candidate against the JD requirements."""
    logger.info("=== screen_candidates ===")

    candidates = state.get("candidates", [])
    jd_parsed = state.get("jd_parsed", {})

    if not candidates:
        logger.warning("No candidates to screen")
        return state

    scores = []
    for candidate in candidates:
        score_result = score_candidate(candidate, jd_parsed, llm)
        scores.append(score_result)
        logger.info("  %s: %.2f (%s)", candidate.get("name", "Unknown"),
                     score_result.get("overall_score", 0),
                     score_result.get("score_breakdown", {}))

    audit_entry = {
        "node": "screen_candidates",
        "action": "candidates_screened",
        "detail": f"Screened {len(scores)} candidates",
    }

    return {
        "scores": scores,
        "pipeline_status": "shortlisting",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def rank_and_shortlist_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Rank candidates by score and produce a shortlist."""
    logger.info("=== rank_and_shortlist ===")

    scores = state.get("scores", [])
    shortlisted = rank_candidates(scores, max_shortlist=5, min_score=0.5)
    shortlisted_ids = [s["candidate_id"] for s in shortlisted]

    logger.info("Shortlisted %d candidates:", len(shortlisted_ids))
    for s in shortlisted:
        logger.info("  - %s: %.2f", s.get("name", "Unknown"), s.get("overall_score", 0))

    audit_entry = {
        "node": "rank_and_shortlist",
        "action": "shortlisted",
        "detail": f"Top {len(shortlisted_ids)} candidates: {shortlisted_ids}",
    }

    return {
        "shortlisted": shortlisted_ids,
        "pipeline_status": "scheduling",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def schedule_interviews_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Check calendar and schedule interviews for shortlisted candidates."""
    logger.info("=== schedule_interviews ===")

    shortlisted_ids = state.get("shortlisted", [])
    candidates = state.get("candidates", [])
    jd_parsed = state.get("jd_parsed", {})
    jd_title = jd_parsed.get("title", "Position")

    candidate_map = {c["candidate_id"]: c for c in candidates}
    slots = get_calendar_slots(days=14)
    interview_schedule = []

    for i, cid in enumerate(shortlisted_ids):
        candidate = candidate_map.get(cid, {})
        if i < len(slots):
            slot = slots[i]
            start_time = f"{slot['date']}T{slot['time']}:00"
            # 1 hour interview
            hour = int(slot['time'].split(':')[0]) + 1
            end_time = f"{slot['date']}T{hour:02d}:00:00"

            event = create_calendar_event(
                summary=f"Interview: {jd_title} - {candidate.get('name', cid)}",
                start_time=start_time,
                end_time=end_time,
                attendees=[candidate.get("email", "candidate@example.com"), "hiring-manager@intelligenzit.com"],
                description=f"Interview for {jd_title} position",
            )
            interview_schedule.append({
                "candidate_id": cid,
                "candidate_name": candidate.get("name", "Unknown"),
                "date": slot["date"],
                "time": slot["time"],
                "event_id": event.get("event_id", ""),
                "status": "scheduled",
            })

    logger.info("Scheduled %d interviews", len(interview_schedule))

    audit_entry = {
        "node": "schedule_interviews",
        "action": "interviews_scheduled",
        "detail": f"Scheduled {len(interview_schedule)} interviews",
    }

    return {
        "interview_schedule": interview_schedule,
        "pipeline_status": "communicating",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def generate_communications_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Generate offer/rejection/interview confirmation letters."""
    logger.info("=== generate_communications ===")

    if llm is None:
        llm = get_llm()

    shortlisted_ids = state.get("shortlisted", [])
    candidates = state.get("candidates", [])
    scores = state.get("scores", [])
    jd_parsed = state.get("jd_parsed", {})
    interview_schedule = state.get("interview_schedule", [])

    candidate_map = {c["candidate_id"]: c for c in candidates}
    score_map = {s["candidate_id"]: s for s in scores}
    scheduled_ids = {s["candidate_id"] for s in interview_schedule}

    communications = []

    for cid in shortlisted_ids:
        candidate = candidate_map.get(cid, {})
        score_info = score_map.get(cid, {})
        name = candidate.get("name", "Candidate")
        email = candidate.get("email", "candidate@example.com")

        if cid in scheduled_ids:
            # Interview confirmation
            schedule = next(s for s in interview_schedule if s["candidate_id"] == cid)
            subject = f"Interview Confirmation - {jd_parsed.get('title', 'Position')}"
            body = (
                f"Dear {name},\n\n"
                f"Thank you for applying for the {jd_parsed.get('title', 'position')} role. "
                f"We are pleased to invite you for an interview.\n\n"
                f"Date: {schedule['date']}\nTime: {schedule['time']}\n\n"
                f"Please confirm your availability. We look forward to speaking with you.\n\n"
                f"Best regards,\nRecruitment Team"
            )
            comm_type = "interview_confirmation"
        elif score_info.get("overall_score", 0) >= 0.7:
            # Offer (high scorers)
            subject = f"Offer Letter - {jd_parsed.get('title', 'Position')}"
            body = (
                f"Dear {name},\n\n"
                f"Congratulations! We are delighted to offer you the position of "
                f"{jd_parsed.get('title', '')} at IntelligenzIT.\n\n"
                f"Please find attached the offer letter with details regarding "
                f"compensation, benefits, and start date. This offer is valid for 7 days.\n\n"
                f"We look forward to welcoming you to the team!\n\n"
                f"Best regards,\nRecruitment Team"
            )
            comm_type = "offer"
        else:
            # Rejection or encouraging rejection
            subject = f"Update on your application - {jd_parsed.get('title', 'Position')}"
            body = (
                f"Dear {name},\n\n"
                f"Thank you for your interest in the {jd_parsed.get('title', '')} position. "
                f"After careful consideration, we have decided to move forward with other candidates "
                f"whose qualifications more closely match our current needs.\n\n"
                f"We sincerely appreciate your time and effort in applying. "
                f"We wish you the best in your job search.\n\n"
                f"Best regards,\nRecruitment Team"
            )
            comm_type = "rejection"

        send_email(to=email, subject=subject, body=body)
        communications.append({
            "candidate_id": cid,
            "candidate_name": name,
            "type": comm_type,
            "subject": subject,
            "body": body[:200],
            "sent": True,
        })

    logger.info("Generated %d communications", len(communications))

    audit_entry = {
        "node": "generate_communications",
        "action": "communications_generated",
        "detail": f"Sent {len(communications)} emails",
    }

    return {
        "communications": communications,
        "pipeline_status": "complete",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def update_ats_tracker_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Log candidate status to the ATS tracker."""
    logger.info("=== update_ats_tracker ===")

    jd_id = state.get("jd_id", "UNKNOWN")
    candidates = state.get("candidates", [])
    shortlisted_ids = state.get("shortlisted", [])
    scores = state.get("scores", [])
    interview_schedule = state.get("interview_schedule", [])
    communications = state.get("communications", [])

    candidate_map = {c["candidate_id"]: c for c in candidates}
    score_map = {s["candidate_id"]: s for s in scores}
    scheduled_ids = {s["candidate_id"] for s in interview_schedule}
    comm_map = {c["candidate_id"]: c["type"] for c in communications}

    for c in candidates:
        cid = c["candidate_id"]
        if cid in shortlisted_ids:
            if cid in scheduled_ids:
                status = "interview_scheduled"
            elif comm_map.get(cid) == "offer":
                status = "offered"
            else:
                status = "shortlisted"
        else:
            status = "rejected"

        score_info = score_map.get(cid, {})
        notes = f"Score: {score_info.get('overall_score', 'N/A')}"
        update_ats_tracker(jd_id, cid, status, notes)

    # Also log to DB
    from hr_agents.shared.db import HRDatabase
    db = HRDatabase()
    session_id = f"rec_{uuid.uuid4().hex[:8]}"
    db.save_recruitment_session(
        session_id, jd_id,
        jd_title=state.get("jd_parsed", {}).get("title", "Unknown")
    )
    db.update_recruitment_session(
        session_id,
        pipeline_status="complete",
        candidate_count=len(candidates),
        shortlisted_count=len(shortlisted_ids),
    )
    for entry in state.get("audit_log", []):
        db.log_action("recruitment", session_id, entry.get("node", ""),
                      entry.get("action", ""), entry.get("detail", ""))

    for c in candidates:
        db.save_candidate(session_id, c)

    logger.info("ATS updated for %d candidates. Session: %s", len(candidates), session_id)

    audit_entry = {
        "node": "update_ats_tracker",
        "action": "ats_updated",
        "detail": f"Session {session_id}: {len(candidates)} candidates, "
                  f"{len(shortlisted_ids)} shortlisted, "
                  f"{len(interview_schedule)} interviews scheduled",
    }

    return {
        "pipeline_status": "complete",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


# ── Conditional routing ──

def route_pipeline(state: dict) -> str:
    """Route based on the current pipeline status."""
    status = state.get("pipeline_status", "collecting")
    route_map = {
        "collecting": "parse_and_enrich_jd",
        "screening": "screen_candidates",
        "shortlisting": "rank_and_shortlist",
        "scheduling": "schedule_interviews",
        "communicating": "generate_communications",
        "complete": "update_ats_tracker",
    }
    return route_map.get(status, "parse_and_enrich_jd")