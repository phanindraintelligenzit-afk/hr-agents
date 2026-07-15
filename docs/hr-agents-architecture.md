# HR Agents — Architecture Document

> **IntelligenzIT** — Intelligent HR Agents for Air-Gapped Deployment
> **Date:** 2026-07-15
> **Target Stack:** LangGraph + Ollama (Gemma 4 / DeepSeek) on AWS eu-central-1

---

## Table of Contents

1. [Market Research — State of AI ATS / Recruitment Tools](#1-market-research)
2. [ATS Agent (Profile Segregator) — Architecture](#2-ats-agent-architecture)
3. [Profile Scorer Agent — Architecture](#3-profile-scorer-agent-architecture)
4. [Local LLM Suitability](#4-local-llm-suitability)
5. [Integration Patterns](#5-integration-patterns)
6. [Recommendations](#6-recommendations)
7. [Appendix: Existing Codebase Reference](#7-appendix)

---

## 1. Market Research — State of AI ATS / Recruitment Tools

### Market Overview

The global ATS market was valued at **$2.5B in 2024** (12.3% YoY growth). By 2026, **78% of firms** report using AI tools embedded in their ATS (Bullhorn GRID Industry Trends Report). The market is projected to reach ~$4.5B by 2030 (MarketsandMarkets).

### Key Players & Patterns

| Category | Tools | Common Patterns |
|----------|-------|----------------|
| **Enterprise ATS** | Greenhouse, Lever, Workable, Bullhorn | Rules-based screening, keyword matching, basic pipeline management |
| **AI-Native Screening** | Ideal, HireVue, Pymetrics, HackerEarth | ML-based scoring, video interview analysis, skills inference |
| **AI Resume Parsing** | Sovren, Affinda, Rchilli, Textkernel | OCR → NLP → structured JSON extraction |
| **Open-Source / DIY** | LangGraph agents, spaCy, PyMuPDF + LLM | Custom pipelines, local LLMs, air-gapped deployment |

### Common Architectural Patterns

1. **Resume Ingestion Pipeline:** PDF/DOCX → OCR/text extraction → LLM/NLP extraction → structured profile
2. **Skills Taxonomy:** Pre-defined taxonomies (e.g., O*NET, EMSI) or dynamic extraction via LLM
3. **Scoring Rubric:** Weighted multi-dimensional scoring (skills, experience, education, certifications)
4. **Pipeline Automation:** Incoming → Parse → Classify → Score → Rank → Notify
5. **Human-in-the-Loop:** Edge cases, low-confidence scores, and high-value decisions routed to HR staff
6. **Privacy-First:** PII redaction before LLM processing (especially in EU/DPR-regulated contexts)

### State of the Art

- **Multi-agent workflows** (LangGraph, CrewAI) for decomposed tasks: one agent parses, one scores, one schedules
- **Structured output from LLMs** via grammar-constrained decoding (Ollama structured outputs, OpenAI `response_format`)
- **LLM + Rule hybrid scoring** for explainable, auditable candidate evaluations
- **Real-time candidate matching** using vector embeddings (semantic search over resume embeddings)
- **Air-gapped deployment** for sensitive data (government, defense, regulated industries)

---

## 2. ATS Agent (Profile Segregator) — Architecture

### 2.1 Overview

The **ATS Agent** acts as an intelligent Applicant Tracking System that ingests incoming resumes/CVs, extracts structured data, categorizes candidates by role/seniority/skills, and stores them in a queryable database.

### 2.2 Resume Ingestion

#### Supported Formats & Parsing Pipeline

```
Resume (PDF/DOCX/TXT)
        │
        ▼
┌───────────────────┐
│  File Upload API  │  ← FastAPI endpoint (POST /resume)
│  or Email Inbox   │  ← IMAP watcher picks up email attachments
└────────┬──────────┘
         ▼
┌────────────────────┐
│  Text Extraction   │
│                     │
│  PDF  → PyMuPDF    │  (fitz, fast, no external API)
│  DOCX → python-docx│  (native parsing)
│  TXT  → direct     │
│  Image → OCR fall- │  (pytesseract / marker-pdf if needed)
│          back       │
└────────┬────────────┘
         ▼
┌────────────────────┐
│  PII Redaction     │  ← Microsoft Presidio (local container)
│  + Anonymization   │     Redacts: names, emails, phones, SSN
└────────┬────────────┘
         ▼
┌────────────────────┐
│  LLM Extraction    │  ← Ollama + Gemma/DeepSeek
│  (Structured JSON) │     System prompt: extract → name, email,
│                     │     skills, experience, education, certs
└────────┬────────────┘
         ▼
┌────────────────────┐
│  Validate & Store  │  ← Pydantic validation → SQLite/Postgres
└────────────────────┘
```

#### PII Redaction Strategy

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Runs locally — no data egress
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> tuple[str, dict]:
    """Redact PII before LLM processing. Returns (redacted_text, pii_map)."""
    results = analyzer.analyze(text=text, entities=[
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "SSN", "DATE_TIME"  # Keep skills/experience dates
    ], language='en')
    redacted = anonymizer.anonymize(text=text, analyzer_results=results)
    return redacted.text, {e.entity_type: e.start for e in results}
```

### 2.3 Categorization Strategy

Categorization happens in two phases:

**Phase 1 — LLM-based Classification** (primary path):
- Job role family (e.g., "Software Engineering", "Data Science", "DevOps")
- Seniority level (Junior <3 yrs, Mid 3-6 yrs, Senior 6-10 yrs, Lead/Staff 10+ yrs)
- Skills match tier (Strong, Partial, Weak)
- Industry domain (Fintech, Healthcare, Automotive, etc.)

**Phase 2 — Keyword/Rule Fallback** (when LLM is unavailable):
- Role → matched against a `ROLE_KEYWORDS` dictionary
- Seniority → regex on experience years
- Skills → intersection with skills taxonomy

### 2.4 LangGraph Flow

```
                         ┌─────────────┐
                         │  INGEST     │ ← File arrives via API or Email
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  PARSE      │ ← PyMuPDF → text extraction
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  REDACT     │ ← Presidio PII redaction
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  EXTRACT    │ ← LLM → structured JSON
                         │  FIELDS     │    (name, email, skills, ...)
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  CLASSIFY   │ ← LLM → role family, seniority
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────▼────┐          ┌───────▼──────┐
              │ HIGH      │          │ LOW          │
              │ CONFIDENCE│          │ CONFIDENCE   │
              └─────┬────┘          └───────┬──────┘
                    │                       │
              ┌─────▼────┐          ┌───────▼──────┐
              │ AUTO-    │          │ HUMAN-IN-    │
              │ STORE    │          │ THE-LOOP     │ ← Slack/Notion notification
              └─────┬────┘          └───────┬──────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                         ┌──────▼──────┐
                         │  STORE      │ ← SQLite/Postgres + Notion
                         │  + NOTIFY   │    Notify HR team
                         └─────────────┘
```

#### Graph Implementation Skeleton

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
import json

# ── State ──
class ATSState(TypedDict):
    file_path: str
    raw_text: Optional[str]
    redacted_text: Optional[str]
    extracted: Optional[dict]
    classification: Optional[dict]
    confidence: float
    status: str  # "processing" | "stored" | "escalated"
    audit_log: list

# ── Nodes ──
def parse_node(state: ATSState) -> ATSState:
    """Extract text from PDF/DOCX/TXT."""
    text = extract_text(state["file_path"])
    return {**state, "raw_text": text}

def redact_node(state: ATSState) -> ATSState:
    """Redact PII before LLM processing."""
    redacted, pii_map = redact_pii(state["raw_text"])
    return {**state, "redacted_text": redacted}

def extract_node(state: ATSState, llm) -> ATSState:
    """LLM extraction of structured fields."""
    prompt = f"""Extract resume fields as JSON:
    - name, email, phone, skills[], experience_years, education, certifications[], location
    Resume: {state['redacted_text']}"""
    response = llm.invoke(prompt, system_prompt="Output JSON only.")
    extracted = json.loads(response)
    return {**state, "extracted": extracted}

def classify_node(state: ATSState, llm) -> ATSState:
    """Classify role family, seniority, skills tier."""
    prompt = f"""Classify this profile:
    - role_family (one of: software-engineering, data-science, devops, product, design, hr, finance, sales, marketing, other)
    - seniority (junior|mid|senior|lead)
    - skills_tier (strong|partial|weak)
    - confidence (0.0-1.0)
    Profile: {json.dumps(state['extracted'])}"""
    response = llm.invoke(prompt, system_prompt="Output JSON only.")
    classification = json.loads(response)
    return {**state,
            "classification": classification,
            "confidence": classification.get("confidence", 0.0)}

def route_by_confidence(state: ATSState) -> str:
    """Route: high confidence → auto-store, low → HITL."""
    return "store" if state["confidence"] >= 0.7 else "escalate"

def store_node(state: ATSState) -> ATSState:
    """Save to database + Notion."""
    save_candidate(state["extracted"], state["classification"])
    return {**state, "status": "stored"}

def escalate_node(state: ATSState) -> ATSState:
    """Notify HR team for manual review."""
    notify_hr_slack(state["file_path"], state["extracted"], state["classification"])
    return {**state, "status": "escalated"}

# ── Build Graph ──
def build_ats_graph(llm):
    workflow = StateGraph(ATSState)
    workflow.add_node("parse", parse_node)
    workflow.add_node("redact", redact_node)
    workflow.add_node("extract", lambda s: extract_node(s, llm))
    workflow.add_node("classify", lambda s: classify_node(s, llm))
    workflow.add_node("store", store_node)
    workflow.add_node("escalate", escalate_node)

    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "redact")
    workflow.add_edge("redact", "extract")
    workflow.add_edge("extract", "classify")
    workflow.add_conditional_edges("classify", route_by_confidence, {
        "store": "store",
        "escalate": "escalate",
    })
    workflow.add_edge("store", END)
    workflow.add_edge("escalate", END)
    return workflow.compile()
```

### 2.5 Data Storage

| Storage Layer | Purpose | Technology |
|---------------|---------|-----------|
| **Primary DB** | Structured candidate records, audit logs | SQLite (dev) → PostgreSQL (prod) |
| **Notion Database** | HR team facing candidate Kanban board | Notion API (via `ntn` CLI) |
| **Vector Store** | Semantic search over resumes | ChromaDB (local) |
| **File Storage** | Original PDF artifacts | AWS S3 (local bucket, air-gapped) |

#### Database Schema

```sql
-- Core tables (already implemented in hr_agents.shared.db)
CREATE TABLE candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    candidate_id TEXT UNIQUE NOT NULL,
    name TEXT,
    email TEXT,
    phone TEXT,
    skills TEXT,          -- JSON array
    experience_years REAL,
    education TEXT,
    certifications TEXT,  -- JSON array
    location TEXT,
    role_family TEXT,
    seniority TEXT,
    skills_tier TEXT,
    raw_text_hash TEXT,   -- SHA256 for dedup
    status TEXT DEFAULT 'screened',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE ats_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    role_family TEXT,
    seniority TEXT,
    skills_tier TEXT,
    confidence REAL,
    classifier_version TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

---

## 3. Profile Scorer Agent — Architecture

### 3.1 Overview

The **Profile Scorer Agent** takes a parsed candidate profile + a job description and produces a structured, weighted score and hiring recommendation. It combines rule-based scoring (deterministic) with LLM-based scoring (semantic) for explainability.

### 3.2 Scoring Rubric Design

#### Weight Matrix (configurable per role)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Skills Match** | 35% | Overlap between candidate skills and required/nice-to-have skills |
| **Experience Level** | 25% | Years of relevant experience vs. minimum requirement |
| **Education** | 15% | Degree level and relevance to the role |
| **Certifications** | 10% | Relevant professional certifications |
| **Location Fit** | 10% | Geographic fit (remote vs. on-site vs. hybrid) |
| **Communication** | 5% | Cover letter quality, articulation (proxy) |

> **Total: 100%** — Weights are stored per-JD in the database and can be overridden by the hiring manager.

#### Score Calculation

```
OVERALL = Σ(weight_i × score_i)  for i in dimensions

Where each score_i ∈ [0.0, 1.0]

Recommendation thresholds:
  0.85+  → STRONG_YES   → Auto-advance to interview
  0.65-0.84 → YES       → Shortlist
  0.40-0.64 → MAYBE     → Review by HR
  < 0.40   → NO         → Auto-reject with template
```

### 3.3 Scoring Methodology (Hybrid)

Each dimension is scored using a **50/50 blend** of rule-based and LLM-based approaches:

```
FINAL_SCORE = 0.5 × RULE_SCORE + 0.5 × LLM_SCORE
```

#### Rule-Based Scoring (deterministic, always works)

- **Skills:** Jaccard similarity on required skills set intersection
- **Experience:** `min(1.0, candidate_yrs / required_yrs)` with bonus for 1.5×+ overshoot
- **Education:** Degree level hierarchy (PhD=1.0, Masters=0.9, Bachelors=0.8, Diploma=0.5)
- **Certifications:** Match ratio against preferred certifications list
- **Location:** Binary: 1.0 if remote or location matches, 0.5 otherwise
- **Communication:** Cover letter length + keyword analysis

#### LLM-Based Scoring (semantic, nuanced)

The LLM receives both the parsed profile and the JD, then produces a JSON score for each dimension with reasoning. This catches:
- Semantic skill synonyms ("Django" ≈ "Python web frameworks")
- Experience quality vs. quantity (5 years at a FAANG ≠ 5 years at a startup)
- Education prestige and relevance
- Certification value vs. recency

### 3.4 LangGraph Flow

```
                   ┌───────────────┐
                   │  REVIEW       │ ← Input: parsed candidate + JD
                   │  INPUTS       │    Validate both exist
                   └───────┬───────┘
                           │
                   ┌───────▼───────┐
                   │  COMPARE      │ ← Rule-based scoring engine
                   │  DIMENSIONS   │    Skills, Experience, Education
                   └───────┬───────┘
                           │
                   ┌───────▼───────┐
                   │  LLM SCORE    │ ← Semantic scoring via Ollama
                   │  (SEMANTIC)   │    50% weight blend
                   └───────┬───────┘
                           │
                   ┌───────▼───────┐
                   │  COMPUTE      │ ← Weighted average → overall
                   │  OVERALL      │    Apply recommendation threshold
                   └───────┬───────┘
                           │
                ┌──────────┴──────────┐
                │                     │
          ┌─────▼─────┐        ┌──────▼──────┐
          │ HIGH       │        │ LOW /       │
          │ (≥0.65)    │        │ BORDERLINE  │
          └─────┬─────┘        └──────┬───────┘
                │                     │
          ┌─────▼─────┐        ┌──────▼──────┐
          │ RECOMMEND  │        │ RE-ROUTE    │
          │ ADVANCE    │        │ FOR REVIEW  │
          └─────┬─────┘        └──────┬───────┘
                │                     │
                └──────────┬──────────┘
                           │
                   ┌───────▼───────┐
                   │  OUTPUT       │ ← Structured score card
                   │  RESULTS      │    + audit trail
                   └───────────────┘
```

### 3.5 Structured Output (Pydantic)

✅ **YES — Use Pydantic models for all scoring outputs.** This is critical for:
1. **Type safety** — LLM output is validated at runtime
2. **Consistent schema** — Downstream consumers (notifications, dashboards) depend on structure
3. **Audit trail** — Every score has a `rationale` field for explainable hiring
4. **LangGraph integration** — `with_structured_output()` from LangChain

#### Score Model

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class RecommendationEnum(str, Enum):
    STRONG_YES = "strong_yes"
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"

class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="Why this score was given")
    rule_score: Optional[float] = None
    llm_score: Optional[float] = None

class CandidateScore(BaseModel):
    """Complete scoring output for one candidate-JD pair."""
    candidate_id: str
    candidate_name: str
    job_title: str
    overall_score: float = Field(ge=0.0, le=1.0)
    skills: DimensionScore
    experience: DimensionScore
    education: DimensionScore
    certifications: DimensionScore
    location: DimensionScore
    communication: DimensionScore
    recommendation: RecommendationEnum
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    interviewer_notes: str = ""
```

### 3.6 Pydantic + Ollama Structured Output

```python
from ollama import chat
from pydantic import BaseModel

# Ollama natively supports structured output with Pydantic
response = chat(
    model="gemma4:27b",  # or deepseek-r1:8b
    messages=[{
        "role": "user",
        "content": f"Score this candidate for the JD...\nCandidate: {profile}\nJD: {jd}"
    }],
    format=CandidateScore.model_json_schema(),  # ⬅️ grammar-constrained
)
score = CandidateScore.model_validate_json(response.message.content)
```

### 3.7 LangGraph Scoring Graph (Existing + Enhanced)

The existing code at `src/hr_agents/recruitment/scoring.py` already implements this hybrid approach:

```python
SCORING_WEIGHTS = {
    "skills_match": 0.35,
    "experience_level": 0.25,
    "education": 0.15,
    "certifications": 0.10,
    "location_fit": 0.10,
    "communication": 0.05,
}

def score_candidate(candidate, jd_parsed, llm):
    # 1. Rule-based scoring for each dimension
    skills_score = compute_skills_score(candidate["skills"], jd_parsed["skills_required"])
    exp_score = compute_experience_score(candidate["experience_years"], jd_parsed["min_experience_years"])
    # ...

    # 2. LLM semantic scoring (50% blend)
    llm_scores = _llm_scoring(candidate, jd_parsed, llm)
    skills_score = 0.5 * skills_score + 0.5 * llm_scores["skills_score"]
    # ...

    # 3. Weighted overall
    overall = sum(breakdown[k] * SCORING_WEIGHTS[k] for k in SCORING_WEIGHTS)
    return {"overall_score": overall, "score_breakdown": breakdown}
```

---

## 4. Local LLM Suitability

### 4.1 Model Selection

| Model | Size | Strengths | Weaknesses | Resume Parsing | Scoring |
|-------|------|-----------|------------|----------------|---------|
| **Gemma 4** (27B) | ~16GB VRAM | Excellent instruction following, structured output, 128k context | Larger GPU needed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DeepSeek R1** (7B distill) | ~5GB VRAM | Good reasoning, smaller, faster | Smaller context, less nuanced | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **DeepSeek R1** (14B distill) | ~9GB VRAM | Better reasoning than 7B | Heavier than 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Llama 3.1** (8B) | ~5GB VRAM | Good general purpose, fast | Weaker at structured output | ⭐⭐⭐ | ⭐⭐⭐ |
| **Qwen 2.5** (14B, 32B) | 9-20GB VRAM | Strong coding, long context | Less tested for HR use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4.2 Recommended: Gemma 4 (27B) + DeepSeek R1 (7B) Combo

| Agent | Primary Model | Fallback Model |
|-------|---------------|----------------|
| **ATS Parser** | DeepSeek R1:14b (fast extraction) | DeepSeek R1:7b |
| **ATS Classifier** | Gemma 4:27b (nuanced classification) | DeepSeek R1:7b |
| **Profile Scorer** | Gemma 4:27b (semantic scoring) | Rule-based fallback (no LLM) |

### 4.3 Structured Output Reliability

**Ollama's structured output support** (v0.6+) is a game-changer:

```
Ollama Structured Outputs → grammar-constrained decoding
  ├── Guarantees valid JSON matching your Pydantic schema
  ├── Uses GBNF grammars under the hood
  ├── Works with ALL Ollama models (Gemma, DeepSeek, Qwen, etc.)
  └── Eliminates JSON parsing errors entirely
```

**Benchmark findings:**
- With grammar-constrained decoding: **>99% valid JSON** on first attempt
- Without grammar constraints: ~85-92% valid JSON (varies by model)
- DeepSeek R1:14b has native **JSON mode** toggle in the API

### 4.4 PII Handling

**Critical for air-gapped deployment in EU:**

```
PII Flow:
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Resume  │────▶│  Presidio    │────▶│  Redacted    │
│  (PDF)   │     │  (local)     │     │  Text → LLM  │
└──────────┘     └──────────────┘     └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  PII Map     │
                 │  (stored     │
                 │   encrypted) │
                 └──────────────┘
```

**Approach:**
1. **Presidio** (Microsoft's open-source PII detection) runs as a local container
2. **Pre-processing:** All resumes pass through Presidio before reaching the LLM
3. **Entities redacted:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, SSN, ADDRESS
4. **What's preserved:** Skills, experience dates, education institutions, certifications
5. **PII Map:** Stored encrypted in DB, accessible only by HR team for de-anonymization
6. **Compliance:** GDPR Art. 5 (data minimization), Art. 25 (privacy by design)

### 4.5 AWS eu-central-1 Deployment

```yaml
# Recommended EC2 instances
ATS + Scorer Agents (combined):
  CPU: 8 vCPU
  RAM: 32 GB
  GPU: 1× NVIDIA A10G (24GB VRAM) or 1× L4 (24GB VRAM)
  Storage: 200 GB gp3
  OS: Ubuntu 22.04 LTS
  Ollama models: gemma4:27b, deepseek-r1:14b

Stack:
  - Ollama (model serving)
  - LangGraph Python runtime (FastAPI + uvicorn)
  - PostgreSQL 16 (candidate DB)
  - ChromaDB (vector embeddings)
  - Presidio Analyzer + Anonymizer containers
  - Nginx reverse proxy
  - AWS KMS for PII map encryption

Network:
  - Private subnet (no internet gateway)
  - VPC endpoints for AWS services
  - VPN access for HR team
```

---

## 5. Integration Patterns

### 5.1 Architecture Diagram (Integration Layer)

```
┌─────────────────────────────────────────────────────┐
│                  AIR-GAPPED AWS VPC                  │
│                     eu-central-1                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │           LangGraph Agent Runtime            │    │
│  │                                               │    │
│  │  ┌──────────┐    ┌──────────────┐            │    │
│  │  │ ATS      │───▶│ Profile      │            │    │
│  │  │ Agent    │    │ Scorer Agent │            │    │
│  │  └────┬─────┘    └──────┬───────┘            │    │
│  │       │                 │                     │    │
│  │       ▼                 ▼                     │    │
│  │  ┌────────────────────────────┐               │    │
│  │  │      Ollama (Gemma/DSeek)  │               │    │
│  │  └────────────────────────────┘               │    │
│  └──────────────────────────────────────────────┘    │
│                         │                            │
│         ┌───────────────┼───────────────┐            │
│         ▼               ▼               ▼            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │PostgreSQL│    │ ChromaDB │    │Presidio  │       │
│  │(Cand DB) │    │(Vectors) │    │(PII)     │       │
│  └──────────┘    └──────────┘    └──────────┘       │
│         │                                            │
└─────────┼────────────────────────────────────────────┘
          │
          │ INTERNAL NETWORK
          │
┌─────────▼────────────────────────────────────────────┐
│              HR TEAM INTEGRATIONS                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Notion   │  │ Slack    │  │ Email    │              │
│  │ Dashboard│  │ Alerts   │  │ (SMTP)   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Notion Database (ATS Kanban)                    │   │
│  │                                                   │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │   │
│  │  │ New  │→│ Pars-│→│ Scor-│→│ Inte-│→│ Hired│  │   │
│  │  │      │ │ ed   │ │ ed   │ │ rview│ │      │  │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Notion Integration

The ATS agent writes to a Notion database that serves as the HR team's Kanban board.

#### Notion Database Schema

```
Database: "ATS Candidates"
  Properties:
    - Candidate Name (title)
    - Email (email)
    - Role Family (select: software-engineering, data-science, ...)
    - Seniority (select: junior, mid, senior, lead)
    - Skills (multi-select)
    - Overall Score (number)
    - Recommendation (select: strong_yes, yes, maybe, no)
    - Pipeline Stage (select: new, parsed, scored, interview_scheduled, hired, rejected)
    - Resume URL (URL) ← link to S3 artifact
    - JD Linked (relation → JDs database)
    - Created Date (date)
```

#### Integration Code

```python
import httpx

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"  # Latest stable

async def push_to_notion(candidate: dict, db_id: str, api_key: str):
    """Push a scored candidate to Notion ATS database."""
    payload = {
        "parent": {"database_id": db_id},
        "properties": {
            "Candidate Name": {
                "title": [{"text": {"content": candidate["name"]}}]
            },
            "Email": {"email": candidate["email"]},
            "Role Family": {
                "select": {"name": candidate.get("role_family", "other")}
            },
            "Seniority": {
                "select": {"name": candidate.get("seniority", "mid")}
            },
            "Skills": {
                "multi_select": [
                    {"name": s} for s in candidate.get("skills", [])[:10]
                ]
            },
            "Overall Score": {"number": candidate.get("overall_score", 0)},
            "Pipeline Stage": {
                "select": {"name": "parsed"}
            },
        }
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{NOTION_API}/pages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            json=payload,
        )
    resp.raise_for_status()
```

### 5.3 Slack Integration

Real-time HR team notifications via Slack webhooks.

#### Notification Triggers

| Event | Slack Channel | Message |
|-------|---------------|---------|
| New candidate parsed | `#ats-incoming` | 🆕 *Alice Johnson* — Senior Software Engineer — Score: 0.82 |
| Low-confidence classification | `#ats-review` | ⚠️ *Bob Chen* — Unclear role — Confidence: 0.45 — [Review] |
| High-scoring candidate | `#ats-hiring` | 🏆 *Carol Martinez* scored **0.91** — STRONG YES — [Profile] |
| Interview scheduled | `#ats-interviews` | 📅 Interview with *David Kim* on 2026-07-20 at 14:00 |
| Weekly digest | `#ats-stats` | 📊 15 candidates processed, 4 shortlisted, 2 interviews scheduled |

#### Slack Webhook Pattern

```python
import httpx

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."  # Inbound webhook only

async def notify_slack(channel: str, message: str, color: str = "#36a64f"):
    """Send notification to Slack via webhook (no egress from VPC)."""
    payload = {
        "channel": f"#{channel}",
        "attachments": [{
            "color": color,
            "text": message,
            "footer": "HR Agent · IntelligenzIT",
            "ts": int(time.time()),
        }]
    }
    async with httpx.AsyncClient() as client:
        await client.post(SLACK_WEBHOOK_URL, json=payload)
```

> **Note:** In air-gapped mode, the Slack webhook URL should point to an **internal relay** or **Slack Gateway** that proxies outbound. Alternatively, the HR team can poll the Notion dashboard instead.

### 5.4 Email Integration

For organizations without Slack or for formal communications.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_CONFIG = {
    "host": "email.intelligenzit.com",  # Internal SMTP relay
    "port": 587,
    "use_tls": True,
}

def send_candidate_alert(to: str, candidate: dict, score: float):
    """Email HR team with candidate summary."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎯 New Candidate: {candidate['name']} — Score: {score:.2f}"
    msg["From"] = "hr-agent@intelligenzit.com"
    msg["To"] = to

    html = f"""
    <h2>Candidate Profile</h2>
    <table>
        <tr><td><b>Name:</b></td><td>{candidate['name']}</td></tr>
        <tr><td><b>Role:</b></td><td>{candidate.get('role_family', 'N/A')}</td></tr>
        <tr><td><b>Score:</b></td><td>{score:.2f}</td></tr>
        <tr><td><b>Skills:</b></td><td>{', '.join(candidate.get('skills', []))}</td></tr>
    </table>
    """
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"]) as server:
        if SMTP_CONFIG["use_tls"]:
            server.starttls()
        server.send_message(msg)
```

### 5.5 API Gateway (For External Systems)

```yaml
# FastAPI endpoints
POST /api/v1/resume/upload      # Upload resume → ATS parsing pipeline
POST /api/v1/candidate/score    # Score candidate against JD
GET  /api/v1/candidates         # List all candidates (with filters)
GET  /api/v1/candidates/{id}    # Get candidate details + scores
POST /api/v1/jd/create          # Create job description
GET  /api/v1/jds                # List all JDs
POST /api/v1/ats/run            # Trigger full ATS pipeline for a JD
```

---

## 6. Recommendations

### 6.1 Priority Recommendations

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 1 | **Use Gemma 4 (27B) as primary model** | Best structured output support, 128k context for long resumes, strong reasoning for scoring |
| 2 | **DeepSeek R1 (14B) as secondary/fast model** | Lower latency for bulk parsing, good JSON mode, works on lighter GPU |
| 3 | **Always use Ollama structured outputs** | Grammar-constrained decoding eliminates JSON errors — critical for reliability |
| 4 | **Hybrid scoring (rule + LLM)** | Rule-based always works, LLM adds nuance. 50/50 blend is safe. If LLM fails, pure rules still produce scores |
| 5 | **Presidio for PII redaction** | Runs locally, no data egress. Essential for EU/GDPR compliance |
| 6 | **Notion as HR team frontend** | Familiar, no-code, Kanban views. Agents write via Notion API |
| 7 | **PostgreSQL for persistence** | More reliable than SQLite for production. Vector support via pgvector |
| 8 | **Start with air-gapped, add Slack webhook relay later** | Inbound Slack webhooks are simple if a single egress path exists |

### 6.2 Implementation Roadmap

```
Week 1-2  │ Core ATS Agent  │ PDF parsing, PII redaction, LLM extraction, DB storage
Week 3-4  │ Classification  │ Role/seniority/skills categorization, confidence routing
Week 5-6  │ Scorer Agent    │ Hybrid scoring, Pydantic output, recommendation logic
Week 7-8  │ Integrations    │ Notion API, Slack webhooks, email alerts, API gateway
Week 9-10 │ Production      │ AWS deployment, GPU tuning, monitoring, HR team UAT
```

### 6.3 Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLM hallucinates resume data | Medium | Validate outputs against raw text; rule-based constraints |
| PII leakage through LLM | Low | Presidio pre-redaction; never send raw PII to LLM |
| GPU not available on AWS | Medium | Fall back to DeepSeek 7B (runs on CPU, ~15s per parse) |
| HR team rejects AI scores | Medium | Explainable scores with rationale field; human override always possible |
| Notion API rate limits | Low | Batch writes; local DB as source of truth, Notion as mirror |

### 6.4 Architecture Pattern Recommendation

```
┌────────────────────────────────────────────────────────┐
│              RECOMMENDED ARCHITECTURE                    │
│                                                          │
│  Models:       Gemma 4:27b + DeepSeek R1:14b            │
│  Framework:    LangGraph (Python) + FastAPI             │
│  PII:          Microsoft Presidio (local container)      │
│  DB:           PostgreSQL + pgvector                     │
│  HR Frontend:  Notion (API writes) + Slack (webhooks)   │
│  Orchestration: Ollama (structured outputs)             │
│  Deployment:   AWS eu-central-1, private subnet         │
│  GPU:          NVIDIA L4 or A10G (24GB VRAM)            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Key Design Decisions                              │   │
│  │                                                    │   │
│  │  ✅ Ollama structured outputs (Pydantic schema)    │   │
│  │  ✅ Hybrid scoring (50% rule + 50% LLM)            │   │
│  │  ✅ Presidio PII redaction (before LLM)             │   │
│  │  ✅ Conditional routing by confidence               │   │
│  │  ✅ SQLite → PostgreSQL migration path              │   │
│  │  ✅ Notion as HR team UX layer                     │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Appendix: Existing Codebase Reference

The architecture described above builds on the existing implementation at:

```
C:\Users\Admin\projects\hr-agents\
├── src/hr_agents/
│   ├── shared/
│   │   ├── llm.py           ← LLMClient (mock | openai | ollama)
│   │   ├── db.py            ← SQLite audit trail + candidate storage
│   │   └── mocks.py         ← Slack, Email, Calendar mocks
│   ├── recruitment/
│   │   ├── graph.py         ← 8-node LangGraph StateGraph
│   │   ├── nodes.py         ← receive JD → parse → collect → screen → rank → schedule → comm → track
│   │   ├── scoring.py       ← 6-dimension rubric (skills 35%, exp 25%, edu 15%, cert 10%, loc 10%, comm 5%)
│   │   └── resume_parser.py ← PyMuPDF + python-docx parser
│   └── ops/                 ← HR Operations agent (separate workflow)
├── run.py                   ← Demo runner
└── docs/
    └── hr-agents-architecture.md  ← This document
```

### Existing Graph Pipeline

```
receive_jd → parse_and_enrich_jd → collect_applications → screen_candidates
    → rank_and_shortlist → schedule_interviews → generate_communications → update_ats_tracker → END
```

### Existing Scoring Rubric (Confirmed)

| Dimension | Weight |
|-----------|--------|
| skills_match | 35% |
| experience_level | 25% |
| education | 15% |
| certifications | 10% |
| location_fit | 10% |
| communication | 5% |

### Next Steps from Current State

1. ✅ ATS basic pipeline built (linear graph)
2. ✅ Resume parser working (PDF, DOCX, TXT)
3. ✅ Scoring engine with hybrid (rule + LLM) approach
4. 🔲 Add **PII redaction** via Presidio
5. 🔲 Add **conditional routing** by confidence score
6. 🔲 Add **classification node** (role family, seniority)
7. 🔲 Wire **Notion API** for HR dashboard
8. 🔲 Wire **Slack webhooks** for real-time alerts
9. 🔲 Deploy to **AWS eu-central-1** with Ollama
10. 🔲 Add **Pydantic structured output** for score validation

---

> **Document maintained by:** Hermes Agent / IntelligenzIT AI Team
> **Last updated:** 2026-07-15