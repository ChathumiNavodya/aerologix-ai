import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path("aerologix_users.db")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_users_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'passenger'))
        )
    """)

    # Default demo users. Change passwords later if needed.
    default_users = [
        ("admin", "admin123", "admin"),
        ("passenger", "pass123", "passenger"),
    ]

    for username, password, role in default_users:
        cur.execute(
            """
            INSERT OR IGNORE INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (username, hash_password(password), role),
        )

    conn.commit()
    conn.close()


def register_user(username: str, password: str, role: str) -> tuple[bool, str]:
    username = username.strip().lower()
    role = role.strip().lower()

    if not username or not password:
        return False, "Username and password are required."

    if role not in {"admin", "passenger"}:
        return False, "Invalid role."

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (username, hash_password(password), role),
        )
        conn.commit()
        conn.close()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login_user(username: str, password: str, role: str) -> tuple[bool, str, str | None]:
    username = username.strip().lower()
    role = role.strip().lower()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role FROM users
        WHERE username = ? AND password_hash = ? AND role = ?
        """,
        (username, hash_password(password), role),
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return True, "Login successful.", row[0]

    return False, "Invalid username, password, or role.", None
