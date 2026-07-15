"""SQLite database wrapper for HR agents — audit logging and persistence."""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any


class HRDatabase:
    """Lightweight SQLite wrapper for HR agent audit trails and state persistence."""

    def __init__(self, db_path: str | Path = "hr_agents.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                node TEXT NOT NULL,
                action TEXT NOT NULL,
                detail TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS hr_ops_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                employee_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT,
                urgency TEXT,
                resolution_status TEXT DEFAULT 'pending',
                response_draft TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS recruitment_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                jd_id TEXT NOT NULL,
                jd_title TEXT,
                pipeline_status TEXT DEFAULT 'collecting',
                candidate_count INTEGER DEFAULT 0,
                shortlisted_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                name TEXT,
                email TEXT,
                score REAL DEFAULT 0,
                score_breakdown TEXT,
                status TEXT DEFAULT 'screened',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

    def log_action(
        self,
        agent_type: str,
        session_id: str,
        node: str,
        action: str,
        detail: str = "",
    ) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT INTO audit_log (agent_type, session_id, node, action, detail) VALUES (?, ?, ?, ?, ?)",
            (agent_type, session_id, node, action, detail),
        )
        conn.commit()

    def save_ops_session(self, session_id: str, employee_id: str, query_text: str) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT OR REPLACE INTO hr_ops_sessions (session_id, employee_id, query_text) VALUES (?, ?, ?)",
            (session_id, employee_id, query_text),
        )
        conn.commit()

    def update_ops_session(
        self,
        session_id: str,
        *,
        query_type: str | None = None,
        urgency: str | None = None,
        resolution_status: str | None = None,
        response_draft: str | None = None,
    ) -> None:
        fields = []
        values = []
        for k, v in [("query_type", query_type), ("urgency", urgency),
                      ("resolution_status", resolution_status), ("response_draft", response_draft)]:
            if v is not None:
                fields.append(f"{k} = ?")
                values.append(v)
        if fields:
            values.append(session_id)
            conn = self._connection()
            conn.execute(f"UPDATE hr_ops_sessions SET {', '.join(fields)} WHERE session_id = ?", values)
            conn.commit()

    def save_recruitment_session(self, session_id: str, jd_id: str, jd_title: str = "") -> None:
        conn = self._connection()
        conn.execute(
            "INSERT OR REPLACE INTO recruitment_sessions (session_id, jd_id, jd_title) VALUES (?, ?, ?)",
            (session_id, jd_id, jd_title),
        )
        conn.commit()

    def update_recruitment_session(
        self,
        session_id: str,
        *,
        pipeline_status: str | None = None,
        candidate_count: int | None = None,
        shortlisted_count: int | None = None,
    ) -> None:
        fields = []
        values = []
        for k, v in [("pipeline_status", pipeline_status),
                      ("candidate_count", candidate_count),
                      ("shortlisted_count", shortlisted_count)]:
            if v is not None:
                fields.append(f"{k} = ?")
                values.append(v)
        if fields:
            values.append(session_id)
            conn = self._connection()
            conn.execute(f"UPDATE recruitment_sessions SET {', '.join(fields)} WHERE session_id = ?", values)
            conn.commit()

    def save_candidate(self, session_id: str, candidate: dict) -> None:
        conn = self._connection()
        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (session_id, candidate_id, name, email, score, score_breakdown, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                candidate.get("candidate_id", ""),
                candidate.get("name", ""),
                candidate.get("email", ""),
                candidate.get("score", 0.0),
                json.dumps(candidate.get("score_breakdown", {})),
                candidate.get("status", "screened"),
            ),
        )
        conn.commit()

    def get_audit_log(self, agent_type: str, limit: int = 20) -> list[dict]:
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE agent_type = ? ORDER BY created_at DESC LIMIT ?",
            (agent_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None