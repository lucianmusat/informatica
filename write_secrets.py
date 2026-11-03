"""Write .streamlit/secrets.toml for Streamlit's Google sign-in from env vars.

Run by the container entrypoint, so credentials come from a Kubernetes Secret and
never live in the image or the repo. Does nothing when GOOGLE_CLIENT_ID is unset,
which leaves a hand-written secrets.toml (local development) alone.
"""

import json
import os
import sys
from pathlib import Path

REQUIRED = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "COOKIE_SECRET")


def main() -> int:
    if not os.getenv("GOOGLE_CLIENT_ID"):
        return 0
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        print(f"write_secrets: missing {', '.join(missing)}", file=sys.stderr)
        return 1
    app_url = os.getenv("APP_URL", "https://informatica.lucianmusat.nl").rstrip("/")
    values = {
        "redirect_uri": f"{app_url}/oauth2callback",
        "cookie_secret": os.environ["COOKIE_SECRET"],
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }
    # JSON string escaping is valid TOML basic-string escaping.
    body = "[auth]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items())
    path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(body)
    path.chmod(0o600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
