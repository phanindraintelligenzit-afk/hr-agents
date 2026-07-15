#!/usr/bin/env python3
"""Demo runner for HR Agents — runs both agents with sample data.

Usage:
    python demo.py              # Run both agents with defaults
    python demo.py --ops        # Run only HR Operations Agent
    python demo.py --recruit    # Run only Recruitment ATS Agent
    python demo.py --verbose    # Show detailed output
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hr_agents import (
    build_hr_ops_graph,
    build_recruitment_graph,
    get_llm,
    run_hr_ops,
    run_recruitment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def pretty_print(obj: dict, title: str = "") -> None:
    """Print a dict as formatted JSON."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    print(json.dumps(obj, indent=2, default=str))


def demo_hr_ops(verbose: bool = False) -> None:
    """Run the HR Operations Agent with various query scenarios."""
    print("\n" + "★" * 70)
    print("  HR OPERATIONS AGENT — DEMO")
    print("★" * 70)

    scenarios = [
        ("E001", "What is the annual leave policy and how many days do I have remaining?"),
        ("E002", "I need a salary certificate for my loan application"),
        ("E003", "I want to report a workplace harassment issue"),
        ("E004", "Can I work from home next week? What's the WFH policy?"),
        ("E001", "Something random that doesn't fit any category"),
    ]

    for i, (emp_id, query) in enumerate(scenarios, 1):
        print(f"\n{'─'*70}")
        print(f"  Scenario {i}: Employee {emp_id}")
        print(f"  Query: \"{query}\"")
        print(f"{'─'*70}")

        try:
            start = time.time()
            result = run_hr_ops(employee_id=emp_id, query_text=query)
            elapsed = time.time() - start

            if verbose:
                pretty_print(result, f"Full State (took {elapsed:.2f}s)")
            else:
                print(f"  Employee:    {result.get('employee_id', 'N/A')}")
                print(f"  Query Type:  {result.get('query_type', 'N/A')}")
                print(f"  Urgency:     {result.get('urgency', 'N/A')}")
                print(f"  Confidence:  {result.get('confidence_score', 'N/A')}")
                print(f"  Status:      {result.get('resolution_status', 'N/A')}")
                print(f"  KB Results:  {len(result.get('kb_results', []))} policies found")
                leaves = result.get('leave_balance', {})
                if leaves:
                    print(f"  Leave Bal:   {leaves}")
                draft = result.get('response_draft', '')
                if draft:
                    print(f"  Response:    {draft[:120]}...")
                print(f"  Audit Steps: {len(result.get('audit_log', []))}")
                print(f"  ⏱ {elapsed:.2f}s")
        except Exception as e:
            logger.error("Scenario %d failed: %s", i, e)
            import traceback
            traceback.print_exc()


def demo_recruitment(verbose: bool = False) -> None:
    """Run the Recruitment ATS Agent with a sample job description."""
    print("\n" + "★" * 70)
    print("  HR RECRUITMENT (ATS) AGENT — DEMO")
    print("★" * 70)

    jd = (
        "We are hiring a Senior Software Engineer for our Engineering team. "
        "The role is fully remote. We need someone with 5+ years of experience in "
        "Python and Django, strong knowledge of REST APIs, PostgreSQL, Docker, and AWS. "
        "Nice to have: Kubernetes, Redis, React. "
        "A Bachelor's in Computer Science or equivalent is required. "
        "AWS Certified Developer or CKA certification is a plus."
    )

    print(f"\n{'─'*70}")
    print(f"  Job Description: \"{jd[:100]}...\"")
    print(f"{'─'*70}")

    try:
        start = time.time()
        result = run_recruitment(jd_raw=jd)
        elapsed = time.time() - start

        if verbose:
            pretty_print(result, f"Full State (took {elapsed:.2f}s)")
        else:
            print(f"\n  JD ID:             {result.get('jd_id', 'N/A')}")
            jd_parsed = result.get('jd_parsed', {})
            if jd_parsed:
                print(f"  Parsed Title:      {jd_parsed.get('title', 'N/A')}")
                print(f"  Location:          {jd_parsed.get('location', 'N/A')}")
                print(f"  Min Experience:    {jd_parsed.get('min_experience_years', 'N/A')} years")
                print(f"  Skills Required:   {jd_parsed.get('skills_required', [])}")
                print(f"  Nice to Have:      {jd_parsed.get('skills_nice_to_have', [])}")

            candidates = result.get('candidates', [])
            print(f"\n  Candidates:        {len(candidates)} parsed")
            for c in candidates:
                print(f"    - {c.get('name', 'N/A'):20s} | {c.get('email', 'N/A'):30s} | "
                      f"Exp: {c.get('experience_years', 0)}y | Skills: {len(c.get('skills', []))}")

            scores = result.get('scores', [])
            print(f"\n  Scores:")
            for s in sorted(scores, key=lambda x: x.get('overall_score', 0), reverse=True):
                breakdown = s.get('score_breakdown', {})
                print(f"    {s.get('name', 'N/A'):20s} | Overall: {s.get('overall_score', 0):.2f} | "
                      f"Skills: {breakdown.get('skills_match', 0):.2f} | "
                      f"Exp: {breakdown.get('experience_level', 0):.2f} | "
                      f"Edu: {breakdown.get('education', 0):.2f}")

            shortlisted = result.get('shortlisted', [])
            print(f"\n  Shortlisted:       {len(shortlisted)} candidates")
            for sid in shortlisted:
                match = next((s for s in scores if s['candidate_id'] == sid), {})
                print(f"    - {match.get('name', sid):20s} (score: {match.get('overall_score', 0):.2f})")

            schedule = result.get('interview_schedule', [])
            print(f"\n  Interviews:        {len(schedule)} scheduled")
            for s in schedule:
                print(f"    - {s.get('candidate_name', 'N/A'):20s} on {s.get('date', 'N/A')} at {s.get('time', 'N/A')}")

            comms = result.get('communications', [])
            print(f"\n  Communications:    {len(comms)} sent")
            for c in comms:
                print(f"    - {c.get('candidate_name', 'N/A'):20s} → {c.get('type', 'N/A')}")

            print(f"\n  Pipeline Status:   {result.get('pipeline_status', 'N/A')}")
            print(f"  Audit Steps:       {len(result.get('audit_log', []))}")
            print(f"  ⏱ {elapsed:.2f}s")
    except Exception as e:
        logger.error("Recruitment demo failed: %s", e)
        import traceback
        traceback.print_exc()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HR Agents Demo Runner")
    parser.add_argument("--ops", action="store_true", help="Run only HR Operations Agent")
    parser.add_argument("--recruit", action="store_true", help="Run only Recruitment ATS Agent")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full state output")
    args = parser.parse_args()

    run_ops = args.ops or not args.recruit
    run_rec = args.recruit or not args.ops

    # Use mock LLM by default (no API key needed)
    llm = get_llm(provider="mock")
    from hr_agents.shared.llm import set_llm
    set_llm(llm)

    if run_ops:
        demo_hr_ops(verbose=args.verbose)
    if run_rec:
        demo_recruitment(verbose=args.verbose)

    print(f"\n{'='*70}")
    print("  Demo complete! Both HR agents ran successfully.")
    print("  To run with real LLM: set HR_LLM_PROVIDER=openai and OPENAI_API_KEY")
    print("  Or: HR_LLM_PROVIDER=ollama (pointing to http://localhost:11434)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()