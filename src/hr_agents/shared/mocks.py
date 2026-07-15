"""Mock external APIs — Slack, Email, Calendar — for testing and demo purposes."""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger("hr_agents.mocks")


# ── Mock Slack ──

def send_slack_message(channel: str, message: str, user: str = "HR Bot") -> dict:
    """Simulate sending a Slack message."""
    logger.info("[MOCK Slack] To: #%s | From: %s | %s", channel, user, message[:80])
    return {
        "ok": True,
        "channel": channel,
        "ts": datetime.datetime.now().isoformat(),
        "message": message,
    }


def get_slack_inbox(user_id: str, limit: int = 5) -> list[dict]:
    """Simulate fetching recent Slack DMs to HR."""
    msgs = [
        {"from": "E001", "text": "I need information about annual leave policy", "ts": "2026-07-14T09:00:00"},
        {"from": "E002", "text": "Can I get a salary certificate for my loan application?", "ts": "2026-07-14T10:30:00"},
        {"from": "E003", "text": "I want to report a workplace harassment issue", "ts": "2026-07-14T11:00:00"},
    ]
    return msgs[:limit]


# ── Mock Email ──

def send_email(to: str, subject: str, body: str) -> dict:
    """Simulate sending an email."""
    logger.info("[MOCK Email] To: %s | Subject: %s", to, subject)
    return {
        "ok": True,
        "to": to,
        "subject": subject,
        "sent_at": datetime.datetime.now().isoformat(),
    }


# ── Mock Calendar ──

def get_calendar_slots(calendar_id: str = "primary", days: int = 14) -> list[dict]:
    """Simulate fetching available calendar slots."""
    today = datetime.date.today()
    slots = []
    for d in range(1, days + 1):
        date = today + datetime.timedelta(days=d)
        if date.weekday() < 5:  # Weekdays only
            for hour in [9, 10, 11, 14, 15, 16]:
                slots.append({
                    "date": date.isoformat(),
                    "time": f"{hour:02d}:00",
                    "duration_minutes": 60,
                    "available": True,
                })
    return slots


def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
    description: str = "",
) -> dict:
    """Simulate creating a calendar event."""
    logger.info("[MOCK Calendar] Event: %s | %s → %s | Attendees: %s",
                summary, start_time, end_time, attendees)
    return {
        "ok": True,
        "event_id": f"event_{datetime.datetime.now().timestamp():.0f}",
        "summary": summary,
        "start": start_time,
        "end": end_time,
        "attendees": attendees,
        "html_link": f"https://calendar.google.com/event?id=mock_{datetime.datetime.now().timestamp():.0f}",
    }


# ── Mock HRIS (HR Information System) ──

_EMPLOYEES = {
    "E001": {
        "name": "Alice Johnson",
        "email": "alice.johnson@intelligenzit.com",
        "department": "Engineering",
        "leave_balance": {"annual": 12, "sick": 8, "personal": 3},
        "manager": "M001",
    },
    "E002": {
        "name": "Bob Smith",
        "email": "bob.smith@intelligenzit.com",
        "department": "Marketing",
        "leave_balance": {"annual": 8, "sick": 5, "personal": 2},
        "manager": "M002",
    },
    "E003": {
        "name": "Carol Davis",
        "email": "carol.davis@intelligenzit.com",
        "department": "HR",
        "leave_balance": {"annual": 15, "sick": 10, "personal": 5},
        "manager": "M003",
    },
    "E004": {
        "name": "Diana Patel",
        "email": "diana.patel@intelligenzit.com",
        "department": "Finance",
        "leave_balance": {"annual": 10, "sick": 6, "personal": 2},
        "manager": "M001",
    },
}


def get_employee(employee_id: str) -> dict | None:
    """Look up employee info from mock HRIS."""
    return _EMPLOYEES.get(employee_id)


def get_leave_balance(employee_id: str) -> dict | None:
    """Get leave balance for an employee."""
    emp = get_employee(employee_id)
    if emp:
        return {"employee_id": employee_id, "employee_name": emp["name"], **emp["leave_balance"]}
    return None


# ── Mock Notion / ATS Tracker ──

def update_ats_tracker(jd_id: str, candidate_id: str, status: str, notes: str = "") -> dict:
    """Simulate updating an ATS kanban board."""
    logger.info("[MOCK ATS] JD: %s | Candidate: %s | Status: %s", jd_id, candidate_id, status)
    return {
        "ok": True,
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "status": status,
        "updated_at": datetime.datetime.now().isoformat(),
    }