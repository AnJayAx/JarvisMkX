"""
Database layer for Jarvis Mk.X Streamlit App.
Uses SQLite for persistent storage of sessions, messages, PDFs, corrections, feedback.
"""

import sqlite3
import json
import os
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Optional


DB_PATH = "data/jarvis.db"


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            leniency INTEGER DEFAULT 50,
            top_k INTEGER DEFAULT 5
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            generation_time REAL DEFAULT 0.0,
            sources_json TEXT DEFAULT '[]',
            retrieval_methods TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            summary TEXT DEFAULT '',
            num_pages INTEGER DEFAULT 0,
            num_chunks INTEGER DEFAULT 0,
            sections_json TEXT DEFAULT '[]',
            is_active INTEGER DEFAULT 1,
            uploaded_at TEXT NOT NULL,
            removed_at TEXT DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            wrong_answer TEXT NOT NULL,
            correction TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            vote TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)

    conn.commit()
    conn.close()


# ─── Session CRUD ───

def create_session(title: str = "New Chat") -> str:
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return session_id


def get_all_sessions() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_sessions(keyword: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT DISTINCT s.* FROM sessions s
           LEFT JOIN messages m ON s.id = m.session_id
           WHERE s.title LIKE ? OR m.content LIKE ?
           ORDER BY s.updated_at DESC""",
        (f"%{keyword}%", f"%{keyword}%"),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str):
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def update_session_settings(session_id: str, leniency: int, top_k: int):
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET leniency = ?, top_k = ?, updated_at = ? WHERE id = ?",
        (leniency, top_k, datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str):
    # Best-effort cleanup of uploaded files on disk for this session.
    # (DB deletion alone can leave old files around, which is confusing.)
    try:
        upload_dir = os.path.join("data", "uploads", session_id)
        if os.path.isdir(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)
    except Exception:
        pass

    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM pdfs WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM corrections WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_session(session_id: str) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ─── Message CRUD ───

def add_message(session_id: str, role: str, content: str,
                confidence: float = 0.0, generation_time: float = 0.0,
                sources: list = None, retrieval_methods: str = "") -> int:
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources or [])
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO messages
           (session_id, role, content, confidence, generation_time, sources_json, retrieval_methods, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, role, content, confidence, generation_time, sources_json, retrieval_methods, now),
    )
    msg_id = cursor.lastrowid
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id),
    )
    conn.commit()
    conn.close()
    return msg_id


def get_messages(session_id: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['sources'] = json.loads(d.get('sources_json', '[]'))
        results.append(d)
    return results


# ─── PDF CRUD ───

def add_pdf(session_id: str, filename: str, filepath: str,
            summary: str = "", num_pages: int = 0, num_chunks: int = 0,
            sections: list = None) -> str:
    pdf_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn = get_connection()
    conn.execute(
        """INSERT INTO pdfs
           (id, session_id, filename, filepath, summary, num_pages, num_chunks, sections_json, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pdf_id, session_id, filename, filepath, summary, num_pages, num_chunks,
         json.dumps(sections or []), now),
    )
    conn.commit()
    conn.close()
    return pdf_id


def get_active_pdfs(session_id: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM pdfs WHERE session_id = ? AND is_active = 1 ORDER BY uploaded_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['sections'] = json.loads(d.get('sections_json', '[]'))
        results.append(d)
    return results


def remove_pdf(pdf_id: str):
    conn = get_connection()
    conn.execute(
        "UPDATE pdfs SET is_active = 0, removed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), pdf_id),
    )
    conn.commit()
    conn.close()


# ─── Corrections CRUD ───

def add_correction(session_id: str, question: str, wrong_answer: str, correction: str):
    conn = get_connection()
    conn.execute(
        """INSERT INTO corrections (session_id, question, wrong_answer, correction, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, question, wrong_answer, correction, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_corrections(session_id: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM corrections WHERE session_id = ? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Feedback CRUD ───

def add_feedback(session_id: str, message_id: int, vote: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (session_id, message_id, vote, created_at) VALUES (?, ?, ?, ?)",
        (session_id, message_id, vote, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_feedback_stats() -> Dict:
    conn = get_connection()
    up = conn.execute("SELECT COUNT(*) FROM feedback WHERE vote = 'up'").fetchone()[0]
    down = conn.execute("SELECT COUNT(*) FROM feedback WHERE vote = 'down'").fetchone()[0]
    conn.close()
    return {"thumbs_up": up, "thumbs_down": down, "total": up + down}


# Initialize on import
init_db()
