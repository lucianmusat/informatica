"""Email the admin when someone new requests access."""

from __future__ import annotations

import logging
import smtplib
import threading
from email.message import EmailMessage

from config import ALERT_EMAIL, APP_URL, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER
from users import User

log = logging.getLogger(__name__)


def build_message(user: User) -> EmailMessage:
    who = f"{user.name} <{user.email}>" if user.name else user.email
    msg = EmailMessage()
    msg["Subject"] = f"Informatica: access request from {user.name or user.email}"
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL
    msg.set_content(
        f"{who} signed in to Informatica and is waiting for access.\n\n"
        f"Approve or block them here:\n{APP_URL}/access-requests\n"
    )
    return msg


def _send(msg: EmailMessage) -> None:
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
    except Exception:
        log.exception("could not send access-request email")


def alert_new_user(user: User) -> None:
    """Send in the background so a slow SMTP server never delays the page."""
    if not SMTP_PASSWORD:
        log.info("SMTP_PASSWORD not set; skipping alert for %s", user.email)
        return
    threading.Thread(target=_send, args=(build_message(user),), daemon=True).start()
