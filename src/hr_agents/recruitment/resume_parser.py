"""Resume parser — extracts structured data from PDF and DOCX resumes using PyMuPDF."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("hr_agents.recruitment.parser")


def parse_resume(file_path: str | Path) -> dict[str, Any]:
    """Parse a resume file (PDF or DOCX) to extract structured candidate info.

    Returns a dict with keys: candidate_id, name, email, phone, skills, experience_years,
    education, certifications, location, cover_letter, raw_text.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = _parse_pdf(path)
    elif suffix == ".docx":
        raw_text = _parse_docx(path)
    elif suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported resume format: {suffix} (supported: .pdf, .docx, .txt)")

    return _extract_fields(raw_text, path.stem)


def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyMuPDF not installed; falling back to raw filename parsing for %s", path)
        return f"Resume: {path.stem}"

    doc = fitz.open(str(path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _parse_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("python-docx not installed; falling back to raw filename parsing for %s", path)
        return f"Resume: {path.stem}"

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_fields(raw_text: str, default_name: str) -> dict[str, Any]:
    """Extract structured fields from raw resume text using regex patterns."""
    email = _extract_email(raw_text)
    phone = _extract_phone(raw_text)
    name = _extract_name(raw_text, default_name)
    skills = _extract_skills(raw_text)
    experience_years = _extract_experience_years(raw_text)
    education = _extract_education(raw_text)
    certifications = _extract_certifications(raw_text)
    location = _extract_location(raw_text)

    return {
        "candidate_id": f"CAND_{default_name[:12]}",
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "certifications": certifications,
        "location": location,
        "cover_letter": "",
        "raw_text": raw_text[:2000],  # Truncate for storage
    }


def _extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "unknown@example.com"


def _extract_phone(text: str) -> str:
    # Matches various phone formats
    patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        r"\d{10,15}",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(0).strip()
    return "unknown"


def _extract_name(text: str, fallback: str) -> str:
    # Heuristic: first non-empty line that looks like a name
    lines = text.strip().split("\n")
    for line in lines[:5]:
        clean = line.strip().strip("*#-").strip()
        if clean and len(clean) < 50 and not re.search(r"@|http|resume|cv|curriculum", clean, re.I):
            return clean
    return fallback.replace("_", " ").replace("-", " ").title()


def _extract_skills(text: str) -> list[str]:
    # Common tech skills to look for
    skill_keywords = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "React", "Angular", "Vue", "Django", "Flask", "FastAPI", "Spring Boot",
        "Node.js", "Express", "GraphQL", "REST API", "PostgreSQL", "MySQL",
        "MongoDB", "Redis", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
        "CI/CD", "Git", "Linux", "Agile", "Scrum", "Machine Learning", "NLP",
        "TensorFlow", "PyTorch", "SQL", "NoSQL", "HTML", "CSS", "Sass",
        "Terraform", "Ansible", "Jenkins", "GitHub Actions",
    ]
    found = [s for s in skill_keywords if s.lower() in text.lower()]
    # Also check for "Skills:" section
    skills_section = re.search(r"(?:Skills|Technical Skills|Core Competencies)[:\s]+(.*?)(?:\n\n|\Z)", text, re.I | re.S)
    if skills_section:
        extra = re.findall(r"[A-Za-z#+]+(?:\.[A-Za-z]+)*(?:\s*\d+\.\d+)?", skills_section.group(1))
        found.extend(e.strip() for e in extra if e.strip() and e.strip() not in found)
    return found[:20]


def _extract_experience_years(text: str) -> float:
    # Look for patterns like "5 years of experience", "8+ years", etc.
    patterns = [
        r"(\d+)\+?\s*(?:years|yrs)\s*(?:of\s*)?(?:experience|exp)",
        r"(?:experience|exp)\s*(?:of\s*)?(\d+)\+?\s*(?:years|yrs)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.I)
        if match:
            return float(match.group(1))
    # Fallback: check total experience section
    exp_section = re.search(r"(?:Experience|Work History|Employment)[:\s]+(.*?)(?:\n\nEducation|\Z)", text, re.I | re.S)
    if exp_section:
        # Count years from date ranges
        years = re.findall(r"(20\d\d)\s*(?:-|–|to)\s*(?:20\d\d|Present|current)", exp_section.group(1), re.I)
        if years:
            return float(len(years))
    return 0.0


def _extract_education(text: str) -> str:
    # Look for degree information
    degrees = re.findall(
        r"(?:B\.?[A-Z]\.?|M\.?[A-Z]\.?|PhD|Ph\.D|Bachelor|Master|Doctorate|"
        r"Bachelors|Masters|BBA|MBA|BSc|MSc|BE|ME|BTech|MTech|BCA|MCA)"
        r"(?:[^.]*?)(?:University|College|Institute|School)(?:[^.]*?)(?:\.|\n)",
        text, re.I
    )
    if degrees:
        return degrees[0].strip()
    # Broader search
    edu_match = re.search(
        r"(?:Education|Academic Background|Qualifications)[:\s]+(.*?)(?:\n\n|\Z)",
        text, re.I | re.S
    )
    return edu_match.group(1).strip()[:200] if edu_match else "Not specified"


def _extract_certifications(text: str) -> list[str]:
    certs = re.findall(
        r"(?:Certified|Certificate|Certification)[:\s]*([A-Za-z0-9\s+]+)",
        text, re.I
    )
    # Also look for specific cert names
    known_certs = [
        "AWS Certified", "Azure", "Google Cloud", "PMP", "PRINCE2",
        "CISSP", "CISM", "CISA", "CEH", "CompTIA", "ITIL",
        "Scrum Master", "SAFe", "TOGAF", "Six Sigma",
    ]
    found = [c.strip() for c in certs if len(c.strip()) > 3]
    for kc in known_certs:
        if kc.lower() in text.lower() and kc not in found:
            found.append(kc)
    return found[:5]


def _extract_location(text: str) -> str:
    # Look for location patterns
    location_match = re.search(
        r"(?:Location|Based in|Located in|Address)[:\s]+([A-Za-z\s,]+?)(?:\n|$)",
        text, re.I
    )
    if location_match:
        return location_match.group(1).strip()
    # City, Country pattern
    city_match = re.search(r"([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)+)", text)
    return city_match.group(1).strip() if city_match else "Remote"


def parse_resumes_batch(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Parse multiple resume files and return a list of candidate dicts."""
    candidates = []
    for fp in file_paths:
        try:
            candidate = parse_resume(fp)
            candidates.append(candidate)
            logger.info("Parsed: %s → %s", fp, candidate.get("name", "Unknown"))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)
    return candidates