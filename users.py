"""Who may use the app: one Cassandra row per Google account that signed in."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

PENDING, APPROVED, BLOCKED = "pending", "approved", "blocked"
STATUSES = (PENDING, APPROVED, BLOCKED)


@dataclass
class User:
    email: str
    name: str | None
    picture: str | None
    status: str
    requested_at: datetime | None
    last_login_at: datetime | None
    decided_at: datetime | None = None
    decided_by: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CassandraUserStore:
    """Thin wrapper over the ``users`` table. All emails are stored lower-cased."""

    def __init__(self, session, keyspace: str):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", keyspace):
            raise ValueError(f"invalid keyspace name: {keyspace!r}")
        self.session = session
        self.ks = keyspace

    def ensure_schema(self) -> None:
        self.session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {self.ks} WITH replication = "
            "{'class': 'SimpleStrategy', 'replication_factor': 1}"
        )
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.ks}.users (
                email         text PRIMARY KEY,
                name          text,
                picture       text,
                status        text,
                requested_at  timestamp,
                last_login_at timestamp,
                decided_at    timestamp,
                decided_by    text
            )""")
        self._insert = self.session.prepare(
            f"INSERT INTO {self.ks}.users (email, name, picture, status, requested_at, "
            "last_login_at) VALUES (?, ?, ?, ?, ?, ?) IF NOT EXISTS"
        )
        self._touch = self.session.prepare(
            f"UPDATE {self.ks}.users SET name = ?, picture = ?, last_login_at = ? "
            "WHERE email = ?"
        )
        self._get = self.session.prepare(f"SELECT * FROM {self.ks}.users WHERE email = ?")
        self._set_status = self.session.prepare(
            f"UPDATE {self.ks}.users SET status = ?, decided_at = ?, decided_by = ? "
            "WHERE email = ?"
        )

    def register_login(self, email: str, name: str | None, picture: str | None) -> tuple[User, bool]:
        """Record a sign-in. Returns (user, is_new). New users start pending.

        The insert is a lightweight transaction, so two simultaneous first logins
        still produce exactly one "new user" (and one alert email).
        """
        email = email.lower()
        now = _now()
        applied = self.session.execute(
            self._insert, (email, name, picture, PENDING, now, now)
        ).was_applied
        if not applied:
            self.session.execute(self._touch, (name, picture, now, email))
        return self.get(email), applied

    def get(self, email: str) -> User | None:
        row = self.session.execute(self._get, (email.lower(),)).one()
        return _to_user(row) if row else None

    def list_all(self) -> list[User]:
        return [_to_user(r) for r in self.session.execute(f"SELECT * FROM {self.ks}.users")]

    def set_status(self, email: str, status: str, decided_by: str) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        self.session.execute(self._set_status, (status, _now(), decided_by.lower(), email.lower()))


def _to_user(row) -> User:
    return User(
        email=row.email,
        name=row.name,
        picture=row.picture,
        status=row.status or PENDING,
        requested_at=row.requested_at,
        last_login_at=row.last_login_at,
        decided_at=row.decided_at,
        decided_by=row.decided_by,
    )
