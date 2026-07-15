"""Run HR Agents demo — clean venv, no Hermes PYTHONPATH interference."""
import sys, os

os.environ.pop("PYTHONPATH", None)
sys.path = [p for p in sys.path if "hermes" not in p.lower()]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hr_agents import (
    build_hr_ops_graph, run_hr_ops,
    build_recruitment_graph, run_recruitment,
)

def main():
    print("=" * 60)
    print("  HR OPERATIONS AGENT - Demo")
    print("=" * 60)

    app = build_hr_ops_graph()
    queries = [
        "What is the leave policy for sick days?",
        "I need a reference letter for my mortgage application",
        "How do I submit a travel reimbursement?",
        "I have a complaint about my manager",
        "What is the work-from-home policy?",
    ]
    for q in queries:
        result = run_hr_ops("EMP001", q, thread_id="test-ops")
        status = result.get("resolution_status", "?")
        response = str(result.get("response_draft", ""))[:120]
        conf = result.get("confidence_score", 0)
        print(f"\n  Q: {q}")
        print(f"  -> [{status.upper()}] (conf={conf:.2f}) {response}")

    print("\n" + "=" * 60)
    print("  HR RECRUITMENT AGENT - Demo")
    print("=" * 60)

    app = build_recruitment_graph()
    result = run_recruitment(
        jd_text="Senior Python Developer with 5+ years experience in LangGraph and AI agents.",
        thread_id="test-recruit-001",
    )
    status = result.get("pipeline_status", "unknown")
    print(f"\n  Pipeline: {status}")

    candidates = result.get("scores", [])
    if candidates:
        print(f"  Candidates scored: {len(candidates)}")
        for c in sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)[:5]:
            name = c.get("name", "?")
            score = c.get("total_score", 0)
            skills = c.get("skills_match", 0)
            exp = c.get("experience_match", 0)
            print(f"    {name:25s} total={score:.2f}  skills={skills:.2f}  exp={exp:.2f}")

    shortlisted = result.get("shortlisted", [])
    print(f"  Shortlisted: {len(shortlisted)} candidates")
    audit = result.get("audit_log", [])
    print(f"  Audit steps: {len(audit)}")

    print("\nDone! HR Agents demo completed.")

if __name__ == "__main__":
    main()