"""HR Operations Agent — LangGraph StateGraph workflow builder."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from typing import Any
CompiledGraph = type(StateGraph(Any))

from hr_agents.ops.nodes import (
    check_leave_node,
    classify_query_node,
    draft_response_node,
    log_and_notify_node,
    lookup_policy_node,
    receive_query_node,
    review_and_send_node,
    route_by_query_type,
    route_to_draft_or_escalate,
    route_other_node,
)
from hr_agents.shared.llm import LLMClient

logger = logging.getLogger("hr_agents.ops.graph")


def build_hr_ops_graph(llm: LLMClient | None = None) -> CompiledGraph:
    """Build and compile the HR Operations Agent StateGraph.

    Graph structure::

        receive_query → classify_query
            ├── policy → lookup_policy → draft_response ──┐
            ├── leave → check_leave → draft_response ─────┤
            ├── payroll → lookup_policy → draft_response ──┤
            ├── letter → lookup_policy → draft_response ───┤
            ├── complaint → lookup_policy → draft_response ─┤
            └── other → route_other ───────────────────────┤
                                                           ▼
                                                  review_and_send
                                                           │
                                                           ▼
                                                  log_and_notify → END
    """
    workflow = StateGraph(dict)  # type: ignore[arg-type]

    # ── Add nodes ──
    workflow.add_node("receive_query", lambda s: receive_query_node(s, llm))
    workflow.add_node("classify_query", lambda s: classify_query_node(s, llm))
    workflow.add_node("lookup_policy", lambda s: lookup_policy_node(s, llm))
    workflow.add_node("check_leave", lambda s: check_leave_node(s, llm))
    workflow.add_node("route_other", lambda s: route_other_node(s, llm))
    workflow.add_node("draft_response", lambda s: draft_response_node(s, llm))
    workflow.add_node("review_and_send", lambda s: review_and_send_node(s, llm))
    workflow.add_node("log_and_notify", lambda s: log_and_notify_node(s, llm))

    # ── Set entry point ──
    workflow.set_entry_point("receive_query")

    # ── Add edges ──
    workflow.add_edge("receive_query", "classify_query")

    # Conditional routing from classify_query
    workflow.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "lookup_policy": "lookup_policy",
            "check_leave": "check_leave",
            "route_other": "route_other",
        },
    )

    # After lookup_policy → draft_response
    workflow.add_edge("lookup_policy", "draft_response")
    # After check_leave → draft_response
    workflow.add_edge("check_leave", "draft_response")

    # Conditional from draft_response: if route_other already escalated, skip to review
    workflow.add_conditional_edges(
        "route_other",
        route_to_draft_or_escalate,
        {
            "draft_response": "draft_response",
            "review_and_send": "review_and_send",
        },
    )

    # draft_response → review_and_send
    workflow.add_edge("draft_response", "review_and_send")

    # review_and_send → log_and_notify → END
    workflow.add_edge("review_and_send", "log_and_notify")
    workflow.add_edge("log_and_notify", END)

    # ── Compile ──
    return workflow.compile()


# ── Convenience runner ──

def run_hr_ops(
    employee_id: str = "E001",
    query_text: str = "I need information about the annual leave policy",
    llm: LLMClient | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the HR Operations Agent with a given query and return the final state."""
    graph = build_hr_ops_graph(llm)

    initial_state: dict[str, Any] = {
        "employee_id": employee_id,
        "query_text": query_text,
        "query_type": None,
        "urgency": None,
        "kb_results": [],
        "leave_balance": None,
        "response_draft": None,
        "confidence_score": None,
        "requires_human": False,
        "resolution_status": "pending",
        "audit_log": [],
    }

    result = graph.invoke(initial_state)
    return result