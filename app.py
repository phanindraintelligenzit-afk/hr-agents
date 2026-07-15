"""
HR Recruitment ATS Agent — Streamlit Web App
For IntelligenzIT HR Team (Shankar & co)
Powered by LangGraph + 6-dimension scoring engine
"""

import os
import sys
import json
import tempfile
import datetime
from pathlib import Path

os.environ.pop("PYTHONPATH", None)
sys.path = [p for p in sys.path if "hermes" not in p.lower()]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ["HR_LLM_PROVIDER"] = "mock"

import streamlit as st
import pandas as pd

from hr_agents.shared.llm import LLMClient
from hr_agents.recruitment.scoring import score_candidate, rank_candidates, SCORING_WEIGHTS
from hr_agents.recruitment.resume_parser import parse_resume

# ── Page Config ──
st.set_page_config(
    page_title="HR Recruitment ATS Agent — IntelligenzIT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for dark navy + gold brand ──
st.markdown("""
<style>
    .main { background-color: #0a0e27; }
    .stApp { background-color: #0a0e27; }
    h1, h2, h3 { color: #D3A518 !important; font-family: 'Lato', sans-serif !important; }
    .stButton>button {
        background-color: #D3A518;
        color: #00051E;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover { background-color: #e8b820; color: #00051E; }
    .candidate-card {
        background: linear-gradient(135deg, #00051E 0%, #0a1628 100%);
        border: 1px solid #1a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .score-bar { height: 20px; border-radius: 10px; background: #1a2a4a; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #D3A518, #e8b820); transition: width 0.5s ease; }
    .metric-card {
        background: #00051E;
        border: 1px solid #1a2a4a;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { color: #D3A518; font-size: 2rem; font-weight: bold; }
    .metric-label { color: #8899aa; font-size: 0.85rem; }
    .badge-shortlisted {
        background: #1a5a2a;
        color: #8fdf8f;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-rejected {
        background: #5a1a1a;
        color: #df8f8f;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #00051E !important;
        color: #e0e0e0 !important;
        border: 1px solid #1a2a4a !important;
        border-radius: 8px !important;
    }
    .stFileUploader { border: 2px dashed #1a2a4a; border-radius: 12px; padding: 1rem; }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "results" not in st.session_state:
    st.session_state.results = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "mock"

# ── Default JD (Frontend Engineer) ──
DEFAULT_JD = """# Frontend Engineer — Job Description

Position: Frontend Engineer
Location: Remote / Hyderabad
Experience: 3+ years

Responsibilities:
- Build agent dashboards — real-time status views of LangGraph agents
- Develop interactive scoreboards and configuration UIs
- Create polished single-page marketing sites for AI agents
- Build configuration interfaces with drag-and-drop
- Implement real-time data visualization with charts
- Own the UI component library

Required Skills:
- 3+ years in production frontend
- React 18+ with TypeScript — hooks, context, custom hooks
- State management (Zustand, Jotai, or Redux Toolkit)
- CSS/Tailwind — responsive design, dark mode
- REST API integration
- Git/GitHub

Nice-to-Have:
- Next.js or Remix
- Data visualization (Recharts, D3.js)
- Python basics
- Figma
- Agentic AI understanding

Education: Bachelor's in Computer Science or equivalent"""

DEFAULT_JD_FRONTEND = DEFAULT_JD


def parse_jd_with_llm(jd_text: str, llm: LLMClient) -> dict:
    """Parse raw JD text into structured fields."""
    from hr_agents.recruitment.nodes import _fallback_parse_jd
    prompt = (
        "Extract structured information from this job description. "
        "Respond ONLY with a JSON object containing:\n"
        "- title (str)\n- department (str)\n- location (str)\n"
        "- min_experience_years (int)\n- skills_required (list[str])\n"
        "- skills_nice_to_have (list[str])\n- education_required (str)\n"
        "- certifications_preferred (list[str])\n- responsibilities (list[str])\n\n"
        f"Job Description: {jd_text}\n\nJSON:"
    )
    try:
        response = llm.invoke(prompt, system_prompt="You are a JD parser. Output JSON only.")
        parsed = json.loads(response)
        return parsed
    except (json.JSONDecodeError, TypeError):
        return _fallback_parse_jd(jd_text)


def run_screening(jd_parsed: dict, candidates_data: list[dict], llm: LLMClient) -> list[dict]:
    """Score all candidates against the JD."""
    scores = []
    for c in candidates_data:
        result = score_candidate(c, jd_parsed, llm)
        scores.append(result)
    return scores


def display_score_bar(val: float, label: str, weight: float):
    """Show a single dimension score with bar."""
    pct = val * 100
    contrib = val * weight
    cols = st.columns([3, 5, 1, 2])
    with cols[0]:
        st.caption(label)
    with cols[1]:
        bar_html = f"""
        <div class="score-bar">
            <div class="score-fill" style="width:{pct}%"></div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)
    with cols[2]:
        st.caption(f"{pct:.0f}%")
    with cols[3]:
        st.caption(f"(+{contrib*100:.1f}%)")


# ── Sidebar — Configuration ──
with st.sidebar:
    st.markdown("<h2 style='margin-top:0'>🤖 HR ATS Agent</h2>", unsafe_allow_html=True)
    st.caption("IntelligenzIT — Recruitment Screening Engine")

    st.divider()
    st.markdown("### ⚙️ Settings")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["mock", "openai", "openrouter (DeepSeek)"],
        index=0,
        help="'mock' works offline. 'openrouter' uses our DeepSeek model for AI scoring."
    )
    st.session_state.llm_provider = llm_provider

    openrouter_key = None
    if llm_provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif llm_provider == "openrouter (DeepSeek)":
        openrouter_key = st.text_input("OpenRouter API Key", type="password", value="sk-or-v1-...")
        st.caption("Uses deepseek/deepseek-v4-flash for genuine AI-powered resume screening")
        if openrouter_key and openrouter_key != "sk-or-v1-...":
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
            st.success("✅ OpenRouter key saved")
        elif openrouter_key == "sk-or-v1-...":
            st.warning("Paste your actual OpenRouter API key starting with sk-or-v1-...")
    else:
        st.info("Using mock LLM — scores are rule-based. Switch to OpenRouter for AI-enhanced scoring.")

    st.divider()
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Paste a job description** or use the default
    2. **Upload resumes** (PDF, DOCX, or TXT)
    3. Click **Screen Candidates**
    4. Review scores, breakdown, and shortlist
    """)

    st.divider()
    st.markdown("### 💾 Quick Actions")
    if st.session_state.results is not None and not st.session_state.results.empty:
        df = st.session_state.results
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "candidates_scores.csv", "text/csv")

    st.divider()
    st.caption("Built on LangGraph · Powered by IntelligenzIT")


# ── Main Content ──

# Title
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("<h1 style='font-size:3rem; margin:0'>🤖</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0'>HR Recruitment ATS Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8899aa; margin-top:0'>Screen, score, and shortlist candidates — powered by your own AI agent</p>", unsafe_allow_html=True)

# Two-column layout
jd_col, resume_col = st.columns([1, 1])

with jd_col:
    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        "Paste your job description here",
        value=DEFAULT_JD_FRONTEND if not st.session_state.jd_text else st.session_state.jd_text,
        height=300,
        label_visibility="collapsed",
    )
    st.session_state.jd_text = jd_text

    # Quick JD templates
    st.markdown("**Quick Templates:**")
    template_cols = st.columns(3)
    with template_cols[0]:
        if st.button("Frontend Engineer", use_container_width=True, key="jd_fe"):
            st.session_state.jd_text = DEFAULT_JD_FRONTEND
            st.rerun()
    with template_cols[1]:
        if st.button("LangGraph Sr Eng", use_container_width=True, key="jd_lg"):
            st.session_state.jd_text = """# LangGraph Senior Engineer — Job Description

Position: Senior LangGraph Engineer
Location: Remote / Hyderabad
Experience: 4+ years (2+ yrs LangGraph)

Required Skills:
- 4+ years production Python
- 2+ years LangGraph (StateGraph, conditional edges, interrupt(), checkpointing)
- Strong Python: async, type hints, Pydantic
- LLM integration (OpenAI API, Ollama, prompt engineering)
- Agent design patterns (ReAct, tool-use, multi-agent)
- Git/GitHub, CI/CD

Nice-to-Have:
- LangGraph Cloud / LangSmith
- n8n workflow automation
- AWS (Bedrock, EC2, Lambda)
- Enterprise ticketing (Jira, ServiceNow)
- MCP (Model Context Protocol)

Education: Bachelor's in Computer Science or equivalent"""
            st.rerun()
    with template_cols[2]:
        if st.button("Custom JD", use_container_width=True, key="jd_custom"):
            st.session_state.jd_text = ""
            st.rerun()


with resume_col:
    st.markdown("### 📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, or TXT files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            st.caption(f"  ✓ {f.name}")
    else:
        st.info("Upload resumes to screen")

    # Manual candidate entry (for quick testing)
    with st.expander("Or enter candidate details manually"):
        manual_candidates = st.text_area(
            "Paste candidate data (JSON array)",
            placeholder='[{"name": "John Doe", "skills": ["React", "TypeScript"], "experience_years": 5, ...}]',
            height=120,
        )

# ── Screen Button ──
st.divider()
screen_col1, screen_col2, screen_col3 = st.columns([2, 2, 2])
with screen_col2:
    screen_clicked = st.button("🎯 SCREEN CANDIDATES", use_container_width=True, type="primary")

st.divider()

# ── Processing ──
if screen_clicked:
    if not jd_text.strip():
        st.error("Please enter a job description.")
        st.stop()

    llm = None
    if st.session_state.llm_provider == "openrouter (DeepSeek)":
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key and or_key != "sk-or-v1-...":
            llm = LLMClient(
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
            )
    if llm is None:
        llm = LLMClient(provider=st.session_state.llm_provider)

    with st.status("📡 Processing...", expanded=True) as status:
        # Step 1: Parse JD
        st.write("**Step 1/5:** Parsing job description...")
        jd_parsed = parse_jd_with_llm(jd_text, llm)
        st.write(f"✅ JD parsed: **{jd_parsed.get('title', 'Unknown Position')}**")
        st.caption(f"Skills required: {', '.join(jd_parsed.get('skills_required', [])[:8])}")

        # Step 2: Collect candidates from uploaded files
        st.write("**Step 2/5:** Processing resumes...")
        candidates = []

        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    tmp_path = os.path.join(tmpdir, f.name)
                    with open(tmp_path, "wb") as fp:
                        fp.write(f.getbuffer())
                    try:
                        candidate = parse_resume(tmp_path)
                        candidates.append(candidate)
                        st.write(f"  ✓ Parsed: **{candidate.get('name', f.name)}**")
                    except Exception as e:
                        st.warning(f"  ⚠ Failed to parse {f.name}: {e}")
        else:
            st.warning("No resumes uploaded. Using sample candidates for demo.")
            from hr_agents.recruitment.nodes import collect_applications_node
            sample_state = collect_applications_node({"jd_parsed": jd_parsed, "candidates": [], "audit_log": [], "pipeline_status": "collecting"}, llm)
            candidates = sample_state.get("candidates", [])
            for c in candidates:
                st.write(f"  ✓ Demo candidate: **{c.get('name')}**")

        if not candidates:
            st.error("No candidates to screen. Please upload resumes.")
            status.update(label="❌ Failed — no candidates", state="error")
            st.stop()

        # Step 3: Score candidates
        st.write(f"**Step 3/5:** Scoring {len(candidates)} candidates...")
        scores = run_screening(jd_parsed, candidates, llm)
        st.write(f"✅ All candidates scored")

        # Step 4: Rank
        st.write("**Step 4/5:** Ranking and shortlisting...")
        shortlisted = rank_candidates(scores, max_shortlist=10, min_score=0.5)
        st.write(f"✅ **{len(shortlisted)}/{len(candidates)}** candidates shortlisted (cutoff: ≥50%)")

        # Step 5: Build results dataframe
        st.write("**Step 5/5:** Building report...")
        results_data = []
        for s in scores:
            bd = s.get("score_breakdown", {})
            is_shortlisted = s["candidate_id"] in {sh["candidate_id"] for sh in shortlisted}
            # Find the candidate
            cand = next((c for c in candidates if c["candidate_id"] == s["candidate_id"]), {})
            results_data.append({
                "Rank": 0,  # Will be set after sorting
                "Candidate": s["name"],
                "Overall Score": s["overall_score"],
                "Skills Match": bd.get("skills_match", 0),
                "Experience": bd.get("experience_level", 0),
                "Education": bd.get("education", 0),
                "Certifications": bd.get("certifications", 0),
                "Location Fit": bd.get("location_fit", 0),
                "Experience (yrs)": cand.get("experience_years", "?"),
                "Status": "✅ Shortlisted" if is_shortlisted else "❌ Not Selected",
            })

        df = pd.DataFrame(results_data)
        df = df.sort_values("Overall Score", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        df["Overall Score"] = df["Overall Score"].apply(lambda x: f"{x*100:.1f}%")
        for col in ["Skills Match", "Experience", "Education", "Certifications", "Location Fit"]:
            df[col] = df[col].apply(lambda x: f"{x*100:.0f}%")

        st.session_state.results = df
        st.session_state._raw_scores = scores
        st.session_state._candidates = candidates
        status.update(label="✅ Screening Complete!", state="complete")

    st.rerun()


# ── Display Results ──
if st.session_state.results is not None and not st.session_state.results.empty:
    df = st.session_state.results

    # Summary metrics
    st.markdown("## 📊 Screening Results")
    total = len(df)
    shortlisted_count = len(df[df["Status"].str.contains("Shortlisted")])
    avg_score = df["Overall Score"].str.rstrip("%").astype(float).mean()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Candidates Screened</div></div>""", unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{shortlisted_count}</div><div class="metric-label">Shortlisted</div></div>""", unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df) - shortlisted_count}</div><div class="metric-label">Not Selected</div></div>""", unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_score:.1f}%</div><div class="metric-label">Avg Score</div></div>""", unsafe_allow_html=True)

    # Ranking table
    st.markdown("### 🏆 Ranking")
    display_cols = ["Rank", "Candidate", "Overall Score", "Skills Match", "Experience", "Education", "Certifications", "Location Fit", "Status"]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall Score": st.column_config.ProgressColumn("Overall", format="", min_value=0, max_value=100),
            "Status": st.column_config.TextColumn("Verdict"),
        }
    )

    # Per-candidate deep dive
    st.markdown("### 🔍 Candidate Deep Dive")
    selected = st.selectbox("Select a candidate:", df["Candidate"].tolist())
    if selected:
        row = df[df["Candidate"] == selected].iloc[0]

        # Reconstruct the full score data
        full_scores = []
        for s in st.session_state.get("_raw_scores", []):
            if s["name"] == selected:
                full_scores = s
                break

        st.markdown(f"""<div class="candidate-card">""", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {row['Candidate']}")
            st.caption(f"Experience: {row.get('Experience (yrs)', '?')} yrs")
        with col2:
            score_val = float(row["Overall Score"].rstrip("%"))
            st.markdown(f"<div style='text-align:right'><span style='font-size:2.5rem; color:#D3A518; font-weight:bold'>{score_val:.1f}%</span><br><span style='color:#8899aa'>{row['Status']}</span></div>", unsafe_allow_html=True)

        st.divider()

        # Score breakdown with visual bars
        dimensions = [
            ("Skills Match", "skills_match", 0.35),
            ("Experience Level", "experience_level", 0.25),
            ("Education", "education", 0.15),
            ("Certifications", "certifications", 0.10),
            ("Location Fit", "location_fit", 0.10),
            ("Communication", "communication", 0.05),
        ]

        for label, key, weight in dimensions:
            val_str = row.get(label.replace(" Level", "").replace(" Match", ""), "0%").rstrip("%")
            val = float(val_str) / 100
            display_score_bar(val, label, weight)

        st.markdown("</div>", unsafe_allow_html=True)

        # LLM Reasoning (if available)
        reasoning = None
        for s in st.session_state.get("_raw_scores", []):
            if s.get("name") == selected and s.get("llm_reasoning"):
                reasoning = s.get("llm_reasoning")
                break

        if reasoning and reasoning.get("summary") and "no LLM" not in reasoning.get("summary", "").lower():
            st.markdown("### 🧠 AI Analysis")
            with st.expander("View LLM reasoning", expanded=True):
                for dim_key, dim_label in [("skills_match", "Skills"), ("experience_level", "Experience"),
                                           ("education", "Education"), ("certifications", "Certifications"),
                                           ("location_fit", "Location"), ("communication", "Communication")]:
                    if reasoning.get(dim_key):
                        st.markdown(f"**{dim_label}:** {reasoning[dim_key]}")
                st.markdown(f"**Summary:** {reasoning.get('summary', '')}")
        elif reasoning and "rule-based" in reasoning.get("summary", "").lower():
            st.info("Rule-based scoring only. Switch to **OpenRouter (DeepSeek)** in the sidebar for AI-powered analysis.")

        # Suggested interview
        if "Shortlisted" in row["Status"]:
            today = datetime.date.today()
            rank = int(row["Rank"])
            interview_date = today + datetime.timedelta(days=rank * 2)
            st.info(f"📅 **Suggested Interview:** {interview_date.strftime('%A, %b %d, %Y')} | Panel: Hiring Manager + Tech Lead")

    # Download report
    st.divider()
    st.download_button(
        "📥 Download Full Report (CSV)",
        df.to_csv(index=False).encode(),
        f"ats_screening_report_{datetime.date.today().isoformat()}.csv",
        "text/csv",
        use_container_width=True,
    )

elif not screen_clicked:
    # Welcome state
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#8899aa'>
        <h3 style='color:#8899aa'>👆 Start by pasting a job description and uploading resumes</h3>
        <p style='font-size:1.1rem'>The AI agent will screen, score, and shortlist candidates across 6 dimensions</p>
        <p style='font-size:0.9rem; margin-top:2rem'>
            <b>Scoring Dimensions:</b> Skills Match (35%) · Experience Level (25%) · Education (15%)<br>
            Certifications (10%) · Location Fit (10%) · Communication (5%)
        </p>
    </div>
    """, unsafe_allow_html=True)