"""HR Operations Agent — node functions for the LangGraph StateGraph workflow."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from hr_agents.shared.llm import LLMClient, get_llm
from hr_agents.shared.mocks import (
    get_employee,
    get_leave_balance,
    send_slack_message,
)
from hr_agents.ops.rag import PolicyRAG

logger = logging.getLogger("hr_agents.ops")


def receive_query_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Entry point: parse incoming employee query and extract employee info."""
    logger.info("=== receive_query ===")

    employee_id = state.get("employee_id", "E001")
    query_text = state.get("query_text", "")

    if not query_text:
        query_text = "I need information about the annual leave policy"
        logger.warning("No query_text provided, using default: %s", query_text)

    emp = get_employee(employee_id)
    logger.info("Employee: %s (%s)", emp["name"] if emp else "Unknown", employee_id)

    audit_entry = {
        "node": "receive_query",
        "action": "query_received",
        "detail": f"Employee {employee_id}: {query_text[:100]}",
    }

    return {
        "employee_id": employee_id,
        "query_text": query_text,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def classify_query_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Classify the employee's query intent and urgency via LLM."""
    logger.info("=== classify_query ===")

    if llm is None:
        llm = get_llm()

    query_text = state.get("query_text", "")
    prompt = (
        f"Classify the following HR query. Respond ONLY with a JSON object "
        f"containing 'query_type' (one of: policy, leave, payroll, letter, complaint, other) "
        f"and 'urgency' (one of: low, medium, high).\n\n"
        f"Query: {query_text}\n\n"
        f"JSON:"
    )

    response = llm.invoke(prompt, system_prompt="You are an HR query classifier. Output JSON only.")
    logger.info("LLM classification: %s", response)

    try:
        parsed = json.loads(response)
        query_type = parsed.get("query_type", "other")
        urgency = parsed.get("urgency", "medium")
    except (json.JSONDecodeError, TypeError):
        query_type = "other"
        urgency = "medium"

    audit_entry = {
        "node": "classify_query",
        "action": "query_classified",
        "detail": f"type={query_type}, urgency={urgency}",
    }

    return {
        "query_type": query_type,
        "urgency": urgency,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def lookup_policy_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Search HR policy knowledge base for relevant policies."""
    logger.info("=== lookup_policy ===")

    rag = PolicyRAG()
    query_text = state.get("query_text", "")
    results = rag.search(query_text, k=3)

    logger.info("Found %d policy results", len(results))
    for r in results:
        logger.info("  - %s (score: %.3f)", r["title"], r["score"])

    audit_entry = {
        "node": "lookup_policy",
        "action": "policy_searched",
        "detail": f"Found {len(results)} policies",
    }

    return {
        "kb_results": results,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def check_leave_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Check leave balance from the HR system."""
    logger.info("=== check_leave ===")

    employee_id = state.get("employee_id", "")
    leave_balance = get_leave_balance(employee_id)

    logger.info("Leave balance for %s: %s", employee_id, leave_balance)

    audit_entry = {
        "node": "check_leave",
        "action": "leave_checked",
        "detail": f"Balance: {leave_balance}",
    }

    return {
        "leave_balance": leave_balance or {},
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def route_other_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Handle 'other' / unclassified queries — escalate to senior HR."""
    logger.info("=== route_other (escalate) ===")

    query_text = state.get("query_text", "")
    escalation_msg = (
        f"⚠️ *Query Escalated to Senior HR*\n"
        f"Employee: {state.get('employee_id', 'Unknown')}\n"
        f"Query: {query_text}\n"
        f"Urgency: {state.get('urgency', 'medium')}\n\n"
        f"This query could not be auto-classified and requires manual review."
    )
    send_slack_message("senior-hr", escalation_msg)

    audit_entry = {
        "node": "route_other",
        "action": "escalated",
        "detail": "Query escalated to senior HR",
    }

    return {
        "response_draft": escalation_msg,
        "requires_human": True,
        "resolution_status": "escalated",
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def draft_response_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Draft a response using LLM based on query type, KB results, and leave balance."""
    logger.info("=== draft_response ===")

    if llm is None:
        llm = get_llm()

    query_type = state.get("query_type", "other")
    kb_results = state.get("kb_results", [])
    leave_balance = state.get("leave_balance", {})
    query_text = state.get("query_text", "")

    context_parts = [f"Original Query: {query_text}"]
    context_parts.append(f"Query Type: {query_type}")

    if kb_results:
        context_parts.append("Relevant Policies:")
        for r in kb_results:
            context_parts.append(f"  - {r['title']}:\n{r['content'][:300]}")

    if leave_balance:
        context_parts.append(f"Employee Leave Balance: {leave_balance}")

    prompt = (
        "You are an HR assistant. Draft a professional, helpful response to the employee's query. "
        "Be concise but thorough. Use the following context to inform your response.\n\n"
        + "\n\n".join(context_parts) +
        "\n\nDraft response:"
    )

    response = llm.invoke(prompt, system_prompt="You are a professional HR assistant.")

    confidence = 0.85 if query_type in ("policy", "leave") else 0.70

    audit_entry = {
        "node": "draft_response",
        "action": "response_drafted",
        "detail": f"confidence={confidence}",
    }

    return {
        "response_draft": response,
        "confidence_score": confidence,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def review_and_send_node(state: dict, llm: LLMClient | None = None) -> dict:
    """HITL gate: auto-send if confidence > 0.85, otherwise mark as requiring human review."""
    logger.info("=== review_and_send ===")

    confidence = state.get("confidence_score", 0.0)
    response_draft = state.get("response_draft", "(no draft)")
    employee_id = state.get("employee_id", "Unknown")
    query_type = state.get("query_type", "other")

    if confidence >= 0.85:
        # Auto-send
        emp = get_employee(employee_id)
        email = emp["email"] if emp else f"{employee_id}@intelligenzit.com"
        send_slack_message(
            f"employee-{employee_id}",
            f"*HR Response* ({query_type}):\n{response_draft}",
        )
        resolution_status = "auto_resolved"
        logger.info("Auto-sent to %s (confidence=%.2f)", email, confidence)
    else:
        # Flag for human review
        send_slack_message(
            "hr-review-queue",
            f"*⚠️ Needs Human Review*\n"
            f"Employee: {employee_id}\n"
            f"Type: {query_type}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Draft: {response_draft[:200]}",
        )
        resolution_status = "escalated"
        logger.info("Flagged for human review (confidence=%.2f)", confidence)

    audit_entry = {
        "node": "review_and_send",
        "action": "reviewed",
        "detail": f"confidence={confidence:.2f}, status={resolution_status}",
    }

    return {
        "resolution_status": resolution_status,
        "requires_human": confidence < 0.85,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def log_and_notify_node(state: dict, llm: LLMClient | None = None) -> dict:
    """Log the resolution to the audit trail and notify the employee."""
    logger.info("=== log_and_notify ===")

    from hr_agents.shared.db import HRDatabase

    session_id = f"ops_{uuid.uuid4().hex[:8]}"
    db = HRDatabase()

    db.save_ops_session(session_id, state.get("employee_id", ""), state.get("query_text", ""))
    db.update_ops_session(
        session_id,
        query_type=state.get("query_type"),
        urgency=state.get("urgency"),
        resolution_status=state.get("resolution_status"),
        response_draft=state.get("response_draft"),
    )

    for entry in state.get("audit_log", []):
        db.log_action("hr_ops", session_id, entry.get("node", ""), entry.get("action", ""), entry.get("detail", ""))

    logger.info("Session %s logged. Final status: %s", session_id, state.get("resolution_status"))

    audit_entry = {
        "node": "log_and_notify",
        "action": "session_logged",
        "detail": f"session_id={session_id}, status={state.get('resolution_status')}",
    }

    return {
        "resolution_status": state.get("resolution_status", "failed"),
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


# ── Conditional routing ──

def route_by_query_type(state: dict) -> str:
    """Route to the appropriate handler based on classified query type."""
    qtype = state.get("query_type", "other")
    route_map = {
        "policy": "lookup_policy",
        "leave": "check_leave",
        "payroll": "lookup_policy",
        "letter": "lookup_policy",
        "complaint": "lookup_policy",
        "other": "route_other",
    }
    return route_map.get(qtype, "route_other")


def route_to_draft_or_escalate(state: dict) -> str:
    """After handler, route to draft_response or directly to review_and_send."""
    if state.get("resolution_status") == "escalated":
        return "review_and_send"
    return "draft_response"