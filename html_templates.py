from __future__ import annotations

import base64
from pathlib import Path


def _img_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


_BASE_DIR = Path(__file__).parent
_STATIC_DIR = _BASE_DIR / "static"

# Embed avatars as data URIs so they work both locally and in k8s (no static-path weirdness).
BOT_AVATAR = _img_to_data_uri(_STATIC_DIR / "bot.png")
USER_AVATAR = _img_to_data_uri(_STATIC_DIR / "user.png")

css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

/* "typing" / waiting animation */
.typing {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.typing .dot {
  width: 7px;
  height: 7px;
  background: rgba(255,255,255,0.85);
  border-radius: 50%;
  animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing .dot:nth-child(2) { animation-delay: 0.15s; }
.typing .dot:nth-child(3) { animation-delay: 0.30s; }

@keyframes typing-bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.55; }
  40% { transform: translateY(-5px); opacity: 1.0; }
}
</style>
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="{BOT_AVATAR}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="{USER_AVATAR}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

