"""HR Policy RAG — vector search over HR policy documents using Chroma/FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Lazy imports — these dependencies are heavy; only load when actually used


class PolicyRAG:
    """Vector-search over HR policies using sentence embeddings + FAISS.

    Falls back to keyword matching if the embeddings/faiss packages are not installed.
    """

    def __init__(self, policies_dir: str | Path = "data/policies"):
        self.policies_dir = Path(policies_dir)
        self._policies: list[dict[str, Any]] = []
        self._index: Any = None
        self._embedder: Any = None
        self._ready = False

    def load_policies(self) -> None:
        """Load policy documents from the policies directory."""
        self._policies = []
        if not self.policies_dir.exists():
            self._policies = self._default_policies()
            return

        # Load from markdown files
        for file in sorted(self.policies_dir.glob("*.md")):
            text = file.read_text(encoding="utf-8")
            self._policies.append({
                "title": file.stem.replace("-", " ").title(),
                "file": file.name,
                "content": text,
            })

        if not self._policies:
            self._policies = self._default_policies()

        # Try building FAISS index
        self._build_index()

    def _default_policies(self) -> list[dict[str, Any]]:
        return [
            {
                "title": "Annual Leave Policy",
                "file": "annual-leave.md",
                "content": (
                    "# Annual Leave Policy\n\n"
                    "## Entitlement\n"
                    "- Permanent employees: 20 working days per year\n"
                    "- Probation employees: 1 day per month of service\n"
                    "- Part-time employees: pro-rated based on FTE\n\n"
                    "## Notice Period\n"
                    "- Leave requests must be submitted at least 48 hours in advance\n"
                    "- For leave > 5 consecutive days, 2 weeks' notice required\n\n"
                    "## Carry Forward\n"
                    "- Up to 5 unused days can be carried forward to next year\n"
                    "- Excess days must be encashed or forfeited\n\n"
                    "## Approval\n"
                    "- Leave < 3 days: manager approval\n"
                    "- Leave >= 3 days: manager + HR approval\n"
                ),
            },
            {
                "title": "Sick Leave Policy",
                "file": "sick-leave.md",
                "content": (
                    "# Sick Leave Policy\n\n"
                    "## Entitlement\n"
                    "- 10 paid sick days per year\n"
                    "- Sick leave accrues at 0.83 days per month\n\n"
                    "## Notification\n"
                    "- Must notify manager within 2 hours of start of workday\n"
                    "- Medical certificate required for absences > 2 consecutive days\n\n"
                    "## Extended Sick Leave\n"
                    "- Beyond 10 days: covered under Long-Term Disability policy\n"
                    "- STD: 60% of salary for up to 12 weeks\n"
                ),
            },
            {
                "title": "Work From Home Policy",
                "file": "wfh-policy.md",
                "content": (
                    "# Work From Home Policy\n\n"
                    "## Eligibility\n"
                    "- All employees with > 3 months tenure are eligible\n"
                    "- Up to 2 days per week WFH (manager discretion)\n\n"
                    "## Requirements\n"
                    "- Must have stable internet connection\n"
                    "- Must be reachable during core hours (10am-4pm)\n"
                    "- Must attend all scheduled meetings virtually\n\n"
                    "## Equipment\n"
                    "- Company laptop provided\n"
                    "- Ergonomic chair subsidy available (up to $300)\n"
                ),
            },
            {
                "title": "Maternity & Paternity Leave Policy",
                "file": "parental-leave.md",
                "content": (
                    "# Maternity & Paternity Leave Policy\n\n"
                    "## Maternity Leave\n"
                    "- 26 weeks paid maternity leave\n"
                    "- Can start up to 4 weeks before due date\n"
                    "- Option to extend unpaid for up to 12 additional weeks\n\n"
                    "## Paternity Leave\n"
                    "- 4 weeks paid paternity leave\n"
                    "- Must be taken within 6 months of birth\n\n"
                    "## Adoption Leave\n"
                    "- Same as maternity/paternity leave based on primary/secondary caregiver\n"
                ),
            },
            {
                "title": "Payroll and Salary Policy",
                "file": "payroll.md",
                "content": (
                    "# Payroll and Salary Policy\n\n"
                    "## Pay Schedule\n"
                    "- Monthly payroll: 25th of each month\n"
                    "- If 25th falls on weekend/holiday, payment on last working day before\n\n"
                    "## Salary Components\n"
                    "- Basic: 50% of CTC\n"
                    "- HRA: 20% of CTC\n"
                    "- Allowances: As per grade\n"
                    "- Bonus: Paid in March annually\n\n"
                    "## Tax\n"
                    "- TDS deducted as per applicable tax slab\n"
                    "- Form 16 issued by June 15th each year\n"
                ),
            },
            {
                "title": "Code of Conduct",
                "file": "code-of-conduct.md",
                "content": (
                    "# Code of Conduct\n\n"
                    "## Core Principles\n"
                    "- Respect, Integrity, Transparency, Accountability\n\n"
                    "## Harassment Policy\n"
                    "- Zero tolerance for harassment of any kind\n"
                    "- Report to HR or via anonymous hotline: 1800-HR-HELP\n"
                    "- All reports investigated within 5 working days\n\n"
                    "## Conflicts of Interest\n"
                    "- Must disclose any external business interests\n"
                    "- No gifts valued over $50 from vendors/clients\n"
                ),
            },
        ]

    def _build_index(self) -> None:
        """Build a FAISS index from policy content using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np  # type: ignore[import-untyped]
            import faiss  # type: ignore[import-untyped]

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            texts = [p["content"] for p in self._policies]
            embeddings = self._embedder.encode(texts, normalize_embeddings=True)
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(np.array(embeddings, dtype=np.float32))
            self._ready = True
        except ImportError:
            self._ready = False

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Search policies by query text. Returns top-k policy chunks."""
        if not self._policies:
            self.load_policies()

        if self._ready and self._embedder and self._index is not None:
            return self._vector_search(query, k)

        return self._keyword_search(query, k)

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        import numpy as np

        q_vec = self._embedder.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(q_vec, dtype=np.float32), min(k, len(self._policies)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < 0.1:
                continue
            policy = self._policies[idx]
            results.append({
                "title": policy["title"],
                "file": policy["file"],
                "content": policy["content"],
                "score": float(score),
            })
        return results

    def _keyword_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Fallback: simple keyword overlap scoring."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored = []
        for policy in self._policies:
            content_lower = policy["content"].lower()
            # Count how many query tokens appear in the content
            match_count = sum(1 for t in query_tokens if t in content_lower)
            # Bonus for title match
            title_bonus = 3 if any(t in policy["title"].lower() for t in query_tokens) else 0
            score = (match_count / max(len(query_tokens), 1)) + title_bonus * 0.1
            scored.append((score, policy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"title": p["title"], "file": p["file"], "content": p["content"], "score": round(s, 3)}
            for s, p in scored[:k] if s > 0
        ]