"""Optional Telegram messages for unattended runs. Silent no-op unless TG_BOT_TOKEN and TG_CHAT_ID are set.

Setup (2 minutes): write to @BotFather, create a bot, copy its token; send any message to your bot; open
https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}. Then on the server:
    export TG_BOT_TOKEN=...  TG_CHAT_ID=...
Never fails the sprint: every error is swallowed and reported as False.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def _creds():
    t, c = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    return (t, c) if t and c else (None, None)


def enabled() -> bool:
    return _creds()[0] is not None


def send(text: str, timeout: float = 15.0) -> bool:
    token, chat = _creds()
    if not token:
        return False
    try:
        data = urllib.parse.urlencode(
            {"chat_id": chat, "text": text[:4000], "disable_web_page_preview": "true"}
        ).encode()
        with urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=timeout
        ) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception:
        return False


def send_file(path: str, caption: str | None = None, timeout: float = 120.0) -> bool:
    """Bots may send documents up to 50 MB: enough for the results bundle (reports, verdicts, figures)."""
    token, chat = _creds()
    p = Path(path)
    if not token or not p.exists() or p.stat().st_size > 49 * 1024 * 1024:
        return False
    try:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (("chat_id", chat), ("caption", (caption or "")[:1000])):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{p.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode()
            + p.read_bytes()
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception:
        return False
