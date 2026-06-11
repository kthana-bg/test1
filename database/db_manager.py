import sqlite3
import json
import numpy as np
from datetime import datetime
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "visionmate.db")


def get_connection():
    """Open and return a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create all tables if they do not exist yet."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table stores login credentials and profile info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Facial embeddings table stores one embedding per user
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            embedding TEXT NOT NULL,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Health metrics table stores per-second monitoring snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            eye_status TEXT,
            ear_value REAL,
            posture_status TEXT,
            posture_angle REAL,
            health_score REAL,
            active_eye_model TEXT,
            active_posture_model TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str, full_name: str) -> int:
    """
    Insert a new user row.
    Returns the new user's id, or -1 if username already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            (username, password_hash, full_name)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        user_id = -1
    finally:
        conn.close()
    return user_id


def get_user_by_username(username: str) -> dict | None:
    """Return a user row as a dict, or None if not found."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    """Return all registered users (id, username, full_name)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, full_name FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_face_embedding(user_id: int, embedding: np.ndarray) -> bool:
    """
    Save or update the facial embedding for a user.
    Embedding is serialized as a JSON string.
    """
    conn = get_connection()
    cursor = conn.cursor()
    embedding_str = json.dumps(embedding.tolist())
    try:
        cursor.execute("""
            INSERT INTO face_embeddings (user_id, embedding)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET embedding = excluded.embedding,
                                               registered_at = CURRENT_TIMESTAMP
        """, (user_id, embedding_str))
        conn.commit()
        success = True
    except Exception:
        success = False
    finally:
        conn.close()
    return success


def get_face_embedding(user_id: int) -> np.ndarray | None:
    """Return the stored facial embedding array for a user, or None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT embedding FROM face_embeddings WHERE user_id = ?", (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return np.array(json.loads(row["embedding"]))
    return None


def get_all_face_embeddings() -> list[dict]:
    """
    Return all user_id + embedding pairs.
    Used during registration to detect duplicate faces.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "user_id": r["user_id"],
            "embedding": np.array(json.loads(r["embedding"]))
        })
    return result


def save_health_metric(
    user_id: int,
    eye_status: str,
    ear_value: float,
    posture_status: str,
    posture_angle: float,
    health_score: float,
    active_eye_model: str,
    active_posture_model: str
):
    """Insert one health snapshot row for the given user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO health_metrics (
            user_id, timestamp, eye_status, ear_value,
            posture_status, posture_angle, health_score,
            active_eye_model, active_posture_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        datetime.now().isoformat(),
        eye_status,
        ear_value,
        posture_status,
        posture_angle,
        health_score,
        active_eye_model,
        active_posture_model
    ))
    conn.commit()
    conn.close()


def get_health_metrics(user_id: int, hours: int = 1) -> list[dict]:
    """
    Return health metrics for a user over the last N hours.
    Results are ordered by timestamp ascending.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM health_metrics
        WHERE user_id = ?
          AND timestamp >= datetime('now', ? || ' hours')
        ORDER BY timestamp ASC
    """, (user_id, f"-{hours}"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Run initialization when this module is imported
initialize_database()
