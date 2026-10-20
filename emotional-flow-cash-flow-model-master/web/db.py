# -*- coding: utf-8 -*-
"""用户存储 (SQLite, 内置零依赖)"""
import os
import sqlite3
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);
"""


def _conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute(_INIT_SQL)


def get_user(username: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(uid: int):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def create_user(username: str, password_hash: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, password_hash, datetime.now().isoformat()),
        )
        return cur.lastrowid


def update_last_login(uid: int):
    with _conn() as c:
        c.execute("UPDATE users SET last_login_at=? WHERE id=?",
                  (datetime.now().isoformat(), uid))


def list_users():
    with _conn() as c:
        rows = c.execute("SELECT id, username, created_at, last_login_at FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]
