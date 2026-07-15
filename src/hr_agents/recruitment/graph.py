"""HR Recruitment ATS Agent — LangGraph StateGraph workflow builder."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
CompiledGraph = type(StateGraph(Any))

from hr_agents.recruitment.nodes import (
    collect_applications_node,
    generate_communications_node,
    parse_and_enrich_jd_node,
    rank_and_shortlist_node,
    receive_jd_node,
    route_pipeline,
    schedule_interviews_node,
    screen_candidates_node,
    update_ats_tracker_node,
)
from hr_agents.shared.llm import LLMClient

logger = logging.getLogger("hr_agents.recruitment.graph")


def build_recruitment_graph(llm: LLMClient | None = None) -> CompiledGraph:
    """Build and compile the HR Recruitment ATS Agent StateGraph.

    Graph structure::

        receive_jd → parse_and_enrich_jd → collect_applications
            → screen_candidates → rank_and_shortlist → schedule_interviews
            → generate_communications → update_ats_tracker → END
    """
    workflow = StateGraph(dict)  # type: ignore[arg-type]

    # ── Add nodes ──
    workflow.add_node("receive_jd", lambda s: receive_jd_node(s, llm))
    workflow.add_node("parse_and_enrich_jd", lambda s: parse_and_enrich_jd_node(s, llm))
    workflow.add_node("collect_applications", lambda s: collect_applications_node(s, llm))
    workflow.add_node("screen_candidates", lambda s: screen_candidates_node(s, llm))
    workflow.add_node("rank_and_shortlist", lambda s: rank_and_shortlist_node(s, llm))
    workflow.add_node("schedule_interviews", lambda s: schedule_interviews_node(s, llm))
    workflow.add_node("generate_communications", lambda s: generate_communications_node(s, llm))
    workflow.add_node("update_ats_tracker", lambda s: update_ats_tracker_node(s, llm))

    # ── Set entry point ──
    workflow.set_entry_point("receive_jd")

    # ── Linear pipeline edges ──
    workflow.add_edge("receive_jd", "parse_and_enrich_jd")
    workflow.add_edge("parse_and_enrich_jd", "collect_applications")
    workflow.add_edge("collect_applications", "screen_candidates")
    workflow.add_edge("screen_candidates", "rank_and_shortlist")
    workflow.add_edge("rank_and_shortlist", "schedule_interviews")
    workflow.add_edge("schedule_interviews", "generate_communications")
    workflow.add_edge("generate_communications", "update_ats_tracker")
    workflow.add_edge("update_ats_tracker", END)

    # ── Compile ──
    return workflow.compile()


# ── Convenience runner ──

def run_recruitment(
    jd_raw: str = "",
    jd_id: str = "",
    llm: LLMClient | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the Recruitment ATS Agent with a job description and return the final state."""
    graph = build_recruitment_graph(llm)

    initial_state: dict[str, Any] = {
        "jd_id": jd_id,
        "jd_raw": jd_raw,
        "jd_parsed": {},
        "candidates": [],
        "scores": [],
        "shortlisted": [],
        "interview_schedule": [],
        "communications": [],
        "pipeline_status": "collecting",
        "audit_log": [],
    }

    result = graph.invoke(initial_state)
    return result