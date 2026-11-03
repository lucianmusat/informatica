from datetime import datetime, timezone

import pytest

from access import Access, check_access
from users import APPROVED, BLOCKED, PENDING, User

ADMIN = "amibadesign@gmail.com"


class FakeStore:
    def __init__(self):
        self.users: dict[str, User] = {}
        self.logins = 0

    def register_login(self, email, name, picture):
        self.logins += 1
        email = email.lower()
        is_new = email not in self.users
        if is_new:
            now = datetime.now(timezone.utc)
            self.users[email] = User(email, name, picture, PENDING, now, now)
        return self.users[email], is_new

    def get(self, email):
        return self.users.get(email.lower())

    def set(self, email, status):
        self.users[email].status = status


def broken():
    raise ConnectionError("cassandra down")


def run(email, store, *, verified=True, register=True, alerts=None):
    return check_access(
        email, verified, lambda: store, ADMIN, register=register, name="N",
        on_new_user=(alerts.append if alerts is not None else None),
    )


def test_new_user_is_pending_and_alerts_once():
    store, alerts = FakeStore(), []
    assert run("bob@gmail.com", store, alerts=alerts) is Access.PENDING
    assert run("bob@gmail.com", store, alerts=alerts) is Access.PENDING
    assert [u.email for u in alerts] == ["bob@gmail.com"]


def test_approved_and_blocked():
    store = FakeStore()
    run("bob@gmail.com", store)
    store.set("bob@gmail.com", APPROVED)
    assert run("bob@gmail.com", store, register=False) is Access.APPROVED
    store.set("bob@gmail.com", BLOCKED)
    assert run("bob@gmail.com", store, register=False) is Access.BLOCKED
    assert not Access.BLOCKED.allowed and not Access.PENDING.allowed


def test_revocation_applies_without_reregistering():
    store = FakeStore()
    run("bob@gmail.com", store)
    store.set("bob@gmail.com", APPROVED)
    logins = store.logins
    store.set("bob@gmail.com", BLOCKED)
    assert run("bob@gmail.com", store, register=False) is Access.BLOCKED
    assert store.logins == logins  # status re-read, no new login recorded


def test_email_is_case_insensitive():
    store = FakeStore()
    run("Bob@Gmail.com", store)
    store.set("bob@gmail.com", APPROVED)
    assert run("BOB@gmail.com", store, register=False) is Access.APPROVED


@pytest.mark.parametrize("email", [ADMIN, ADMIN.upper(), f" {ADMIN} "])
def test_admin_always_allowed_even_when_store_down(email):
    assert check_access(email, True, broken, ADMIN, register=True) is Access.ADMIN
    assert check_access(email, False, broken, ADMIN, register=False) is Access.ADMIN


def test_admin_blocked_row_is_ignored():
    store = FakeStore()
    run(ADMIN, store)
    store.set(ADMIN, BLOCKED)
    assert run(ADMIN, store, register=False) is Access.ADMIN


def test_store_down_fails_closed():
    assert check_access("bob@gmail.com", True, broken, ADMIN, register=True) is Access.UNAVAILABLE
    assert check_access("bob@gmail.com", True, broken, ADMIN, register=False) is Access.UNAVAILABLE


def test_unverified_and_empty_email_denied():
    store = FakeStore()
    assert run("bob@gmail.com", store, verified=False) is Access.UNVERIFIED
    assert run("", store) is Access.UNVERIFIED
    assert store.users == {}


def test_lookalike_admin_email_is_not_admin():
    store = FakeStore()
    assert run("x." + ADMIN, store) is Access.PENDING
    assert run(ADMIN + ".evil.com", store) is Access.PENDING


def test_failing_alert_does_not_block_registration():
    store = FakeStore()

    def boom(_):
        raise RuntimeError("smtp")

    assert check_access("bob@gmail.com", True, lambda: store, ADMIN, register=True,
                        on_new_user=boom) is Access.PENDING
    assert "bob@gmail.com" in store.users
