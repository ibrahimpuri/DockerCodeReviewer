"""
SQLite persistence layer for CodeLens.

Thread-safety contract:
  - All writes acquire _write_lock (threading.Lock) before opening a connection.
  - WAL mode allows concurrent readers without blocking writers.
  - Connections are created per-call to avoid sqlite3 cross-thread errors.
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "./data/codelens.db")

_write_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call multiple times."""
    with _write_lock:
        conn = _get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    code         TEXT    NOT NULL,
                    language     TEXT    NOT NULL,
                    ai_tool      TEXT    NOT NULL,
                    is_defective INTEGER NOT NULL,
                    timestamp    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_samples (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    code      TEXT    NOT NULL,
                    language  TEXT    NOT NULL,
                    label     INTEGER NOT NULL,
                    source    TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                );
            """)
            conn.commit()
            logger.info("Database initialised at %s", DB_PATH)
        finally:
            conn.close()


def save_submission(
    code: str,
    language: str,
    ai_tool: str,
    is_defective: bool,
) -> int:
    """Persist one analysis result. Returns the new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn = _get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO submissions (code, language, ai_tool, is_defective, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, language, ai_tool, int(is_defective), ts),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def save_training_sample(
    code: str,
    language: str,
    label: int,
    source: str = "user_feedback",
) -> int:
    """Persist one labelled training sample. label: 0=clean, 1=defective."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    ts = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        conn = _get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO training_samples (code, language, label, source, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, language, label, source, ts),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_submission(submission_id: int) -> dict | None:
    """Fetch a single submission by id. Returns None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_stats() -> dict:
    """Return aggregate counts. Read-only — no lock needed."""
    conn = _get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        training = conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
        defective = conn.execute(
            "SELECT COUNT(*) FROM submissions WHERE is_defective = 1"
        ).fetchone()[0]
        return {
            "total_submissions": total,
            "total_training_samples": training,
            "defective_submissions": defective,
            "clean_submissions": total - defective,
        }
    finally:
        conn.close()


def get_training_samples(limit: int = 10_000) -> list[dict]:
    """Return up to `limit` training samples for fine-tuning."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT code, language, label FROM training_samples "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
