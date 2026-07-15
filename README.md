# HR Recruitment ATS Agent

AI-powered resume screening for IntelligenzIT HR team. Drag-and-drop resumes, paste a JD, get ranked shortlists.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hr-ats-agent.streamlit.app)

## 📋 Features

- **Upload Resumes**: Drag-and-drop PDF, DOCX, or TXT files
- **Paste Job Descriptions**: Quick templates for common roles
- **AI-Powered Scoring**: 6-dimension evaluation (Skills, Experience, Education, Certifications, Location, Communication)
- **Ranked Shortlists**: Automatic ranking with >50% cutoff
- **Download Reports**: CSV export of all results

## 🧠 Scoring Rubric

| Dimension | Weight |
|-----------|--------|
| Skills Match | 35% |
| Experience Level | 25% |
| Education | 15% |
| Certifications | 10% |
| Location Fit | 10% |
| Communication | 5% |

## 🏗️ Project Structure

```
src/hr_agents/
├── shared/           # LLM client, DB, mock APIs
│   ├── llm.py        # Configurable: mock | openai | ollama
│   ├── db.py         # SQLite audit trail
│   └── mocks.py      # Mock Slack/Email/Calendar/HRIS
├── ops/              # HR Operations Agent
│   ├── graph.py      # LangGraph StateGraph for HR queries
│   ├── nodes.py      # classify → lookup → draft → HITL → send
│   └── rag.py        # FAISS policy KB
└── recruitment/      # HR Recruitment (ATS) Agent
    ├── graph.py      # LangGraph pipeline (8 nodes)
    ├── nodes.py      # receive JD → parse → collect → screen → rank → schedule → comm → track
    ├── scoring.py    # 6-dimension weighted scoring rubric
    └── resume_parser.py  # PDF/DOCX/TXT text extraction

app.py                # Streamlit web interface
run_profiles.py       # CLI runner for custom profiles
start_hr_ats.bat      # Windows launcher
```

## 🔧 Local Development

```bash
# Clone
git clone https://github.com/phanindraintelligenzit-afk/hr-agents.git
cd hr-agents

# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run
streamlit run app.py
```

## 🔐 Environment (Optional)

Set `HR_LLM_PROVIDER=openai` with `OPENAI_API_KEY` for LLM-enhanced scoring.
Default (`mock`) works offline with rule-based scoring.

## 📄 License

MIT — IntelligenzIT
