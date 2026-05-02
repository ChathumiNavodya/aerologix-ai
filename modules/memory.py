"""
modules/memory.py
SQLite-backed conversation and analysis memory for AeroLogix AI.
"""

import sqlite3
import json
import datetime
import tempfile
from pathlib import Path
from typing import Optional


def _get_db_path() -> Path:
    """
    Resolve a writable path for the SQLite database.
    Tries the project root first; falls back to a temp directory
    for read-only deployments (e.g. Streamlit Cloud).
    """
    preferred = Path(__file__).parent.parent / "aerologix_memory.db"
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        # Quick write-test
        preferred.touch(exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "aerologix_memory.db"
        return fallback


DB_PATH = _get_db_path()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                filename    TEXT,
                row_count   INTEGER,
                summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                role        TEXT NOT NULL,   -- 'user' | 'assistant'
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS insights (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                category    TEXT NOT NULL,   -- 'delay'|'cargo'|'gate'|'recommendation'
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_insights_session ON insights(session_id);
        """)


def create_session(filename: str, row_count: int, summary: dict) -> int:
    """Create a new analysis session and return its id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (created_at, filename, row_count, summary_json) VALUES (?,?,?,?)",
            (
                _now(),
                filename,
                row_count,
                json.dumps(summary),
            ),
        )
        return cur.lastrowid


def save_message(session_id: int, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, _now()),
        )


def save_insight(session_id: int, category: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO insights (session_id, category, content, created_at) VALUES (?,?,?,?)",
            (session_id, category, content, _now()),
        )


def get_conversation_history(session_id: int, limit: int = 20) -> list[dict]:
    """Return the last `limit` messages for a session as dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]



def get_recent_messages(session_id: int, limit: int = 6) -> list[dict]:
    """Return recent messages with timestamps for visible memory history."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]

def get_recent_sessions(limit: int = 5) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, filename, row_count FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_insights(session_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT category, content, created_at FROM insights WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_summary(session_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT summary_json FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
    if row:
        return json.loads(row["summary_json"])
    return None


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds")