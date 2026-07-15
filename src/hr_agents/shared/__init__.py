"""HR Agents — shared state schemas, type definitions, and constants."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── HR Operations Agent State ──

class HROperationsState(BaseModel):
    """State for the HR Operations Agent workflow."""
    employee_id: str = ""
    query_text: str = ""
    query_type: Optional[str] = None  # policy | leave | payroll | letter | complaint | other
    urgency: Optional[str] = None     # low | medium | high
    kb_results: list[dict] = Field(default_factory=list)
    leave_balance: Optional[dict] = None
    response_draft: Optional[str] = None
    confidence_score: Optional[float] = None
    requires_human: bool = False
    resolution_status: str = "pending"  # pending | auto_resolved | escalated | failed
    audit_log: list[dict] = Field(default_factory=list)


# ── Recruitment (ATS) Agent State ──

class RecruitmentState(BaseModel):
    """State for the HR Recruitment ATS Agent workflow."""
    jd_id: str = ""
    jd_raw: str = ""
    jd_parsed: dict = Field(default_factory=dict)  # skills, experience, qualifications, etc.
    candidates: list[dict] = Field(default_factory=list)  # parsed resumes
    scores: list[dict] = Field(default_factory=list)  # candidate_id → score breakdown
    shortlisted: list[str] = Field(default_factory=list)  # candidate IDs
    interview_schedule: list[dict] = Field(default_factory=list)
    communications: list[dict] = Field(default_factory=list)  # candidate_id → letter type
    pipeline_status: str = "collecting"  # collecting | screening | shortlisting | scheduling | complete
    audit_log: list[dict] = Field(default_factory=list)


# ── Shared types ──

class AuditEntry(BaseModel):
    action: str
    node: str
    detail: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()