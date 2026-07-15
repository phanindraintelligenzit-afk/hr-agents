"""Run HR agents demo with isolated paths (no Hermes venv interference)."""
import sys
# Remove Hermes venv paths from sys.path
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

from hr_agents import (
    build_hr_ops_graph, run_hr_ops,
    build_recruitment_graph, run_recruitment,
    HROperationsState, RecruitmentState
)

if __name__ == "__main__":
    # Test HR Ops
    print("=" * 60)
    print("  HR OPERATIONS AGENT — Demo")
    print("=" * 60)
    
    app = build_hr_ops_graph()
    result = run_hr_ops(
        employee_id="EMP001",
        query_text="What is the leave policy for sick days?",
        query_type="policy",
        thread_id="test-ops-001"
    )
    status = result.get("resolution_status", "unknown")
    print(f"\n  Status: {status}")
    print(f"  Response: {str(result.get('response_draft', ''))[:200]}")
    print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
    
    # Test Recruitment
    print("\n" + "=" * 60)
    print("  HR RECRUITMENT AGENT — Demo")
    print("=" * 60)
    
    app = build_recruitment_graph()
    result = run_recruitment(
        jd_text="Senior Python Developer with 5+ years experience in LangGraph and AI agents.",
        thread_id="test-recruit-001"
    )
    status = result.get("pipeline_status", "unknown")
    print(f"\n  Status: {status}")
    candidates = result.get("scores", [])
    if candidates:
        print(f"  Candidates scored: {len(candidates)}")
        for c in candidates[:3]:
            print(f"    - {c.get('name', '?')}: {c.get('total_score', 0):.2f}")
    shortlisted = result.get("shortlisted", [])
    print(f"  Shortlisted: {len(shortlisted)}")
    
    print("\n✅ HR Agents demo completed successfully!")