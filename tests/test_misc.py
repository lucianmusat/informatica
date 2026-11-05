import os
import subprocess
import sys
import tomllib
from pathlib import Path

from notify import build_message
from users import PENDING, User

ROOT = Path(__file__).resolve().parent.parent


def test_alert_email_content():
    msg = build_message(User("bob@gmail.com", "Bob", None, PENDING, None, None))
    assert "Bob" in msg["Subject"]
    assert "bob@gmail.com" in msg.get_content()
    assert "/access-requests" in msg.get_content()


def test_write_secrets_produces_valid_toml(tmp_path):
    # Run a copy so the repo's .streamlit/ is untouched.
    (tmp_path / "write_secrets.py").write_text((ROOT / "write_secrets.py").read_text())
    env = dict(os.environ, GOOGLE_CLIENT_ID="id", GOOGLE_CLIENT_SECRET='s"e\\c',
               COOKIE_SECRET="c", APP_URL="https://example.org/")
    subprocess.run([sys.executable, "write_secrets.py"], cwd=tmp_path, env=env, check=True)
    auth = tomllib.loads((tmp_path / ".streamlit" / "secrets.toml").read_text())["auth"]
    assert auth["redirect_uri"] == "https://example.org/oauth2callback"
    assert auth["client_secret"] == 's"e\\c'
    assert oct((tmp_path / ".streamlit" / "secrets.toml").stat().st_mode)[-3:] == "600"


def test_write_secrets_noop_without_client_id(tmp_path):
    (tmp_path / "write_secrets.py").write_text((ROOT / "write_secrets.py").read_text())
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_CLIENT_ID"}
    subprocess.run([sys.executable, "write_secrets.py"], cwd=tmp_path, env=env, check=True)
    assert not (tmp_path / ".streamlit").exists()


def test_llm_reasoning_parsing(monkeypatch):
    import importlib

    import config

    try:
        for raw, expected in [("true", True), (" False ", False), ("", None), ("maybe", None)]:
            monkeypatch.setenv("LLM_REASONING", raw)
            assert importlib.reload(config).LLM_REASONING is expected
    finally:
        monkeypatch.delenv("LLM_REASONING", raising=False)
        importlib.reload(config)
