"""Configurable LLM client — supports OpenAI, Ollama, and a mock/fallback provider."""

from __future__ import annotations

import json
import os
from typing import Any


class LLMClient:
    """Lightweight LLM client. Configure via env vars or constructor args.

    Provider resolution:
      1. If ``provider`` arg is passed, use it.
      2. Else read ``HR_LLM_PROVIDER`` env var (default: "mock").
      3. "mock" → returns deterministic responses (no external call).
      4. "openai" → uses the OpenAI Python SDK.
      5. "ollama" → uses raw HTTP to a local Ollama instance.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider or os.environ.get("HR_LLM_PROVIDER", "mock")
        self.model = model or os.environ.get("HR_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def invoke(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        if self.provider == "mock":
            return self._mock_invoke(prompt, system_prompt)
        if self.provider == "openai":
            return self._openai_invoke(prompt, system_prompt, **kwargs)
        if self.provider in ("openrouter", "ollama"):
            return self._openrouter_invoke(prompt, system_prompt, **kwargs)
        msg = f"Unknown LLM provider: {self.provider}"
        raise ValueError(msg)

    def _mock_invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return deterministic responses for demo/testing."""
        p_lower = prompt.lower()

        # ── Classification ──
        if "classify" in p_lower or "query type" in p_lower or "intent" in p_lower:
            if "leave" in p_lower or "vacation" in p_lower or "holiday" in p_lower:
                return json.dumps({"query_type": "leave", "urgency": "medium"})
            if "pay" in p_lower or "salary" in p_lower or "payroll" in p_lower:
                return json.dumps({"query_type": "payroll", "urgency": "high"})
            if "policy" in p_lower or "rule" in p_lower or "regulation" in p_lower:
                return json.dumps({"query_type": "policy", "urgency": "low"})
            if "letter" in p_lower or "reference" in p_lower or "certificate" in p_lower:
                return json.dumps({"query_type": "letter", "urgency": "medium"})
            if "complaint" in p_lower or "harass" in p_lower or "grievance" in p_lower:
                return json.dumps({"query_type": "complaint", "urgency": "high"})
            return json.dumps({"query_type": "other", "urgency": "low"})

        # ── Draft response ──
        if "draft" in p_lower or "response" in p_lower:
            return (
                "Thank you for reaching out. Based on our records and applicable policies, "
                "here is the information you requested:\n\n"
                "- **Leave Balance**: You have 12 annual leave days remaining.\n"
                "- **Policy Reference**: Per Section 4.2 of the Employee Handbook, "
                "leave requests require 48 hours' notice.\n\n"
                "If you have further questions, please reply to this message.\n\n"
                "Best regards,\nHR Operations"
            )

        # ── JD parsing ──
        if "parse" in p_lower or "jd" in p_lower or "job description" in p_lower:
            return json.dumps({
                "title": "Senior Software Engineer",
                "department": "Engineering",
                "location": "Remote",
                "min_experience_years": 5,
                "skills_required": [
                    "Python", "Django", "REST APIs", "PostgreSQL",
                    "Docker", "AWS", "CI/CD", "GraphQL"
                ],
                "skills_nice_to_have": ["Kubernetes", "Redis", "React"],
                "education_required": "Bachelor's in Computer Science or equivalent",
                "certifications_preferred": ["AWS Certified Developer", "Certified Kubernetes Administrator"],
                "responsibilities": [
                    "Design and implement scalable backend services",
                    "Lead code reviews and mentor junior engineers",
                    "Collaborate with cross-functional teams"
                ]
            })

        # ── Resume screening / scoring ──
        if "screen" in p_lower or "score" in p_lower or "resume" in p_lower:
            return json.dumps({
                "skills_score": 0.85,
                "experience_score": 0.75,
                "education_score": 0.90,
                "certification_score": 0.60,
                "location_score": 1.0,
                "communication_score": 0.80,
                "overall_score": 0.82,
                "reasoning": "Strong skills match in Python and Django. "
                             "Experience aligns well. Education requirement met."
            })

        # ── Communication draft ──
        if "offer" in p_lower or "rejection" in p_lower or "interview" in p_lower:
            return (
                "Dear Candidate,\n\n"
                "Thank you for your interest in the Senior Software Engineer position. "
                "After careful review, we are pleased to inform you that you have been "
                "shortlisted for the next stage.\n\n"
                "We will contact you shortly to schedule an interview. "
                "Please expect an invitation within 2-3 business days.\n\n"
                "Best regards,\nRecruitment Team"
            )

        # Fallback
        return "Thank you for your query. Our team will review and respond shortly."

    def _openai_invoke(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = OpenAI(api_key=self.api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content or ""

    def _openrouter_invoke(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        """Call an OpenAI-compatible endpoint (OpenRouter, Ollama, etc.)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = OpenAI(
            api_key=self.api_key or "sk-placeholder",
            base_url=self.base_url.rstrip("/") + "/v1" if not self.base_url.endswith("/v1") else self.base_url,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return response.choices[0].message.content or ""


# Module-level singleton for convenience
_default_llm: LLMClient | None = None


def get_llm(**kwargs: Any) -> LLMClient:
    global _default_llm
    if _default_llm is None:
        _default_llm = LLMClient(**kwargs)
    return _default_llm


def set_llm(client: LLMClient) -> None:
    global _default_llm
    _default_llm = client
