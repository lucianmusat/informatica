"""The access decision, kept free of Streamlit so it can be unit-tested."""

from __future__ import annotations

import enum
import logging
from typing import Callable

from users import APPROVED, BLOCKED, User

log = logging.getLogger(__name__)


class Access(enum.Enum):
    ADMIN = "admin"
    APPROVED = "approved"
    PENDING = "pending"
    BLOCKED = "blocked"
    UNVERIFIED = "unverified"   # Google says the email is not verified
    UNAVAILABLE = "unavailable"  # user store unreachable: fail closed

    @property
    def allowed(self) -> bool:
        return self in (Access.ADMIN, Access.APPROVED)


def check_access(
    email: str,
    email_verified: bool,
    get_store: Callable[[], object],
    admin_email: str,
    *,
    register: bool,
    name: str | None = None,
    picture: str | None = None,
    on_new_user: Callable[[User], None] | None = None,
) -> Access:
    """Decide whether this signed-in Google account may use the app.

    ``register`` is True on the first check of a browser session: the sign-in is
    recorded (creating a pending request for a new account). Later checks only
    re-read the status, so approving or revoking takes effect on the next rerun.
    ``get_store`` is called lazily so a store that cannot even connect is handled
    like any other failure.
    """
    email = (email or "").strip().lower()
    if not email:
        return Access.UNVERIFIED

    if email == admin_email.lower():
        # Never depend on the database for the admin; recording the login is
        # best effort.
        if register:
            try:
                get_store().register_login(email, name, picture)
            except Exception:
                log.warning("could not record admin login", exc_info=True)
        return Access.ADMIN

    if not email_verified:
        return Access.UNVERIFIED

    try:
        store = get_store()
        if register:
            user, is_new = store.register_login(email, name, picture)
            if is_new and on_new_user is not None:
                try:
                    on_new_user(user)
                except Exception:
                    log.warning("new-user hook failed", exc_info=True)
        else:
            user = store.get(email)
    except Exception:
        log.exception("user store unavailable")
        return Access.UNAVAILABLE

    if user is None:
        return Access.PENDING
    if user.status == APPROVED:
        return Access.APPROVED
    if user.status == BLOCKED:
        return Access.BLOCKED
    return Access.PENDING
