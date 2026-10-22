"""
Database Setup — SQLite
-----------------------
Stores user-reported URLs for the Active Learning retraining pipeline (Advancement #4).
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "phishguard.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema on first run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS url_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT NOT NULL,
            model_prediction TEXT NOT NULL,
            user_label      TEXT NOT NULL,
            phishing_prob   REAL,
            typosquat_score REAL,
            reporter_ip     TEXT,
            vt_verified     INTEGER DEFAULT 0,
            vt_result       TEXT,
            used_in_retrain INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT NOT NULL,
            risk_level      TEXT NOT NULL,
            phishing_prob   REAL,
            typosquat_score REAL,
            model_type      TEXT,
            created_at      TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            samples_used    INTEGER,
            accuracy_before REAL,
            accuracy_after  REAL,
            created_at      TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def log_analysis(url: str, risk_level: str, phishing_prob: float,
                 typosquat_score: float, model_type: str):
    """Log every URL analysis for monitoring."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO analysis_log (url, risk_level, phishing_prob, typosquat_score, model_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (url, risk_level, phishing_prob, typosquat_score, model_type,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def save_report(url: str, model_prediction: str, user_label: str,
                phishing_prob: float, typosquat_score: float, reporter_ip: str = ""):
    """Save a user-submitted phishing report."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO url_reports
        (url, model_prediction, user_label, phishing_prob, typosquat_score, reporter_ip, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (url, model_prediction, user_label, phishing_prob, typosquat_score,
          reporter_ip, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_pending_reports(limit: int = 500):
    """Fetch user reports not yet used for retraining."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM url_reports
        WHERE used_in_retrain = 0
        ORDER BY created_at ASC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_reports_retrained(ids: list):
    """Mark reports as used in retraining."""
    if not ids:
        return
    conn = get_connection()
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE url_reports SET used_in_retrain = 1 WHERE id IN ({placeholders})",
        ids
    )
    conn.commit()
    conn.close()


def get_stats():
    """Return overall system statistics."""
    conn = get_connection()
    stats = {
        "total_analyses":  conn.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0],
        "total_reports":   conn.execute("SELECT COUNT(*) FROM url_reports").fetchone()[0],
        "phishing_caught": conn.execute("SELECT COUNT(*) FROM analysis_log WHERE risk_level='PHISHING'").fetchone()[0],
        "suspicious_found": conn.execute("SELECT COUNT(*) FROM analysis_log WHERE risk_level='SUSPICIOUS'").fetchone()[0],
        "retrains_done":   conn.execute("SELECT COUNT(*) FROM retrain_log").fetchone()[0],
    }
    conn.close()
    return stats
