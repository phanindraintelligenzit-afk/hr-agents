"""Run 3 candidate profiles through HR Recruitment ATS Agent against Frontend Engineer JD."""
import os, sys, json

os.environ.pop("PYTHONPATH", None)
sys.path = [p for p in sys.path if "hermes" not in p.lower()]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Must set mock LLM so scoring doesn't try real API calls
os.environ["HR_LLM_PROVIDER"] = "mock"

from hr_agents.recruitment.scoring import score_candidate, rank_candidates, SCORING_WEIGHTS
from hr_agents.shared.llm import LLMClient

# ── Frontend Engineer JD (from our saved template) ──
JD_PARSED = {
    "title": "Frontend Engineer",
    "department": "Engineering",
    "location": "Remote",
    "min_experience_years": 3,
    "skills_required": [
        "React", "TypeScript", "JavaScript", "HTML5", "CSS3",
        "Tailwind CSS", "REST API", "Git", "Responsive Design",
        "State Management"
    ],
    "skills_nice_to_have": [
        "Next.js", "Docker", "Node.js", "Data Visualization",
        "CI/CD", "Python", "AWS"
    ],
    "education_required": "Bachelor's in Computer Science or equivalent",
    "certifications_preferred": [],
    "responsibilities": [
        "Build agent dashboards and configuration UIs",
        "Implement real-time data visualization",
        "Own the UI component library",
        "Design Systems and reusable components"
    ]
}

# ── Candidate Profiles (from the resumes provided) ──
CANDIDATES = [
    {
        "candidate_id": "CAND_SURYARAVI",
        "name": "Surya Ravi Kumar",
        "email": "suryaravikumar1729@gmail.com",
        "phone": "+91 9014916056",
        "skills": [
            "React", "Next.js", "TypeScript", "JavaScript", "HTML5", "CSS3",
            "Tailwind CSS", "Redux Toolkit", "Node.js", "Express.js", "Docker",
            "PostgreSQL", "MongoDB", "AWS", "Git", "REST API",
            "WebSocket", "JWT", "Framer Motion", "Micro Frontend",
            "Webpack", "Vite", "Playwright", "Jest", "Storybook",
            "CI/CD", "GraphQL", "Responsive Design"
        ],
        "experience_years": 4.7,
        "education": "Bachelor of Technology (B.Tech) – Electrical and Electronics Engineering - Vidya Jyothi Institute of Technology",
        "certifications": ["Frontend Developer (React) - HackerRank"],
        "location": "Hyderabad, Telangana",
        "cover_letter": "",
        "raw_text": "Surya Ravi Kumar - Full Stack Engineer with 4+ years"
    },
    {
        "candidate_id": "CAND_PRADEEP",
        "name": "Pradeep Etika",
        "email": "etikapradeep@gmail.com",
        "phone": "+91 9700542486",
        "skills": [
            "React", "TypeScript", "JavaScript", "HTML5", "CSS3",
            "Tailwind CSS", "Redux Toolkit", "Node.js", "Express.js",
            "AWS EC2", "Git", "REST API", "Jest", "MongoDB",
            "Responsive Design", "Firebase", "Vite", "Webpack"
        ],
        "experience_years": 4.6,
        "education": "Bachelor of Technology - Electronics & Communication Engineering - Dhruva Institute of Engineering and Technology, Hyderabad",
        "certifications": [],
        "location": "Hyderabad, Telangana",
        "cover_letter": "",
        "raw_text": "Pradeep Etika - Frontend Engineer with 4+ years at Capgemini"
    },
    {
        "candidate_id": "CAND_RUCHITH",
        "name": "Ruchith Kumar Renikindi",
        "email": "renikindiruchithkumar@gmail.com",
        "phone": "+91 9390181672",
        "skills": [
            "React", "Next.js", "TypeScript", "JavaScript", "HTML5", "CSS3",
            "Tailwind CSS", "Material UI", "Redux", "Context API",
            "Git", "REST API", "Jest", "Responsive Design",
            "Python", "Java", "Webpack"
        ],
        "experience_years": 3.0,
        "education": "B.Tech in Information Technology - Maturi Venkata Subba Rao Engineering College, Hyderabad (CGPA: 7.34)",
        "certifications": [],
        "location": "Hyderabad",
        "cover_letter": "",
        "raw_text": "Ruchith Kumar Renikindi - Frontend Software Engineer with 3 years"
    }
]

llm = LLMClient(provider="mock")

print("=" * 70)
print("  HR RECRUITMENT ATS - Frontend Engineer Candidate Screening")
print("  JD: Frontend Engineer (Remote, 3+ yrs, React/TypeScript/Tailwind)")
print("=" * 70)

# -- Score each candidate --
scores = []
for c in CANDIDATES:
    result = score_candidate(c, JD_PARSED, llm)
    scores.append(result)
    name = c["name"]
    overall = result["overall_score"]
    breakdown = result["score_breakdown"]
    print(f"\n{'-' * 70}")
    print(f"  [CANDIDATE] {name}")
    print(f"  {'-' * 50}")
    for dim, val in breakdown.items():
        weight = SCORING_WEIGHTS.get(dim, 0)
        bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
        contribution = val * weight
        print(f"    {dim:25s} {bar} {val*100:5.1f}%  (wt {weight*100:2.0f}% -> +{contribution*100:5.1f}%)")
    print(f"    {'-' * 50}")
    print(f"    {'OVERALL SCORE':25s} {'#' * int(overall * 20) + '.' * (20 - int(overall * 20))} {overall*100:5.1f}%")
    print(f"    Skills: {', '.join(c['skills'][:8])}...")
    print(f"    Experience: {c['experience_years']} yrs | Education: {c['education'][:60]}")

# -- Ranking & Shortlist --
print(f"\n\n{'=' * 70}")
print("  RANKING & SHORTLIST")
print(f"{'=' * 70}")

ranked = sorted(scores, key=lambda s: s["overall_score"], reverse=True)
shortlisted = rank_candidates(scores, max_shortlist=5, min_score=0.5)

print(f"\n  {'Rank':<6} {'Name':<30} {'Score':<8} {'Verdict':<12}")
print(f"  {'-' * 56}")
for i, s in enumerate(ranked, 1):
    verdict = "SHORTLISTED" if s["candidate_id"] in {sh["candidate_id"] for sh in shortlisted} else "Not selected"
    print(f"  #{i:<4} {s['name']:<30} {s['overall_score']*100:5.1f}%   {verdict}")

print(f"\n  {'-' * 56}")
print(f"  Shortlisted: {len(shortlisted)}/{len(CANDIDATES)} candidates")
print(f"  Cutoff: >=50% overall score")

# -- Interview Schedule Suggestion --
print(f"\n\n{'=' * 70}")
print("  SUGGESTED INTERVIEW SCHEDULE")
print(f"{'=' * 70}")
import datetime
today = datetime.date.today()
for i, s in enumerate(shortlisted, 1):
    interview_date = today + datetime.timedelta(days=i * 2)
    score_pct = s["overall_score"] * 100
    print(f"\n  Interview #{i}: {s['name']}")
    print(f"    Score:      {score_pct:.1f}%")
    print(f"    Suggested:  {interview_date.strftime('%A, %b %d')}")
    print(f"    Panel:      Hiring Manager + Tech Lead")
    if score_pct >= 75:
        print(f"    Priority:   HIGH - Top candidate")
    elif score_pct >= 60:
        print(f"    Priority:   MEDIUM - Strong candidate")
    else:
        print(f"    Priority:   STANDARD")

print(f"\n{'=' * 70}")
print("  DONE! Ready for review.")
print(f"{'=' * 70}")
