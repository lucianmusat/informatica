"""Idle summaries: a background thread that summarises chats nobody is using.

A chat that has been idle for SUMMARY_IDLE_MINUTES and has messages its summary
doesn't cover yet gets a new summary of the whole chat. The summary is used when
the chat is reopened (summary + the last few exchanges instead of the whole
chat) and when its window is cut (see conversation.fit_window).

The request is the chat's exact last prompt plus its answer plus one "summarise"
message, so if llama.cpp still holds that prompt in its cache, the summary only
costs the writing. It goes to the always-on backend with the same model and load
options as chat (the LLM object is the app's own), so it never starts a second
Ollama runner.

The GPU serves one request at a time, so the thread only starts a summary when
no informatica answer is in progress, and does one at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from langchain_core.messages import HumanMessage

import conversation
import prompts

log = logging.getLogger("informatica.summary")

# After a failed summary, leave that chat alone for this long.
RETRY_AFTER = timedelta(minutes=30)


class Activity:
    """Counts informatica answers in progress, across sessions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = 0

    def __enter__(self):
        with self._lock:
            self._running += 1
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._running -= 1

    @property
    def busy(self) -> bool:
        return self._running > 0


def _utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:  # the driver returns naive UTC
        value = value.replace(tzinfo=timezone.utc)
    return value


class Summarizer:
    def __init__(self, store, llm, *, idle_minutes: int, keep_exchanges: int,
                 is_busy: Callable[[], bool], now: Callable[[], datetime] | None = None):
        self.store = store
        self.llm = llm
        self.idle = timedelta(minutes=idle_minutes)
        # A chat no longer than what a reopen keeps anyway gains nothing from a summary.
        self.min_messages = 2 * keep_exchanges + 1
        self.is_busy = is_busy
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._failed: dict = {}

    def due(self) -> list:
        """Idle chats with unsummarised messages, longest idle first."""
        now = self.now()
        chats = [
            c for c in self.store.scan_for_summary()
            if c.message_count >= self.min_messages
            and c.message_count > c.summary_upto
            and c.updated_at is not None and _utc(c.updated_at) <= now - self.idle
            and self._failed.get(c.chat_id, now) <= now
        ]
        return sorted(chats, key=lambda c: _utc(c.updated_at))

    def summarise(self, owner: str, chat_id) -> bool:
        chat = self.store.load(owner, chat_id)
        if chat is None or chat.message_count <= chat.summary_upto:
            return False
        payload = conversation.model_messages(chat.messages[chat.window_start:], chat.window_summary)
        payload.append(HumanMessage(content=prompts.SUMMARY_REQUEST))
        started = time.monotonic()
        reply = self.llm.invoke(payload)
        text = (reply.content or "").strip()
        if not text:
            raise RuntimeError("empty summary")
        usage = conversation.usage_from(getattr(reply, "response_metadata", None))
        stored = self.store.set_summary(owner, chat_id, text, chat.message_count)
        log.info(
            "chat %s: summarised %d messages in %.1fs (prompt %d tokens read in %.1fs, %d written)%s",
            chat_id, chat.message_count, time.monotonic() - started, usage["prompt_tokens"],
            usage["prompt_seconds"], usage["output_tokens"], "" if stored else "; chat was deleted",
        )
        return stored

    def run_once(self) -> int:
        """Summarise every due chat while informatica is idle. Returns the count."""
        done = 0
        for chat in self.due():
            if self.is_busy():
                break
            try:
                done += self.summarise(chat.owner, chat.chat_id)
            except Exception:
                log.warning("chat %s: summary failed", chat.chat_id, exc_info=True)
                self._failed[chat.chat_id] = self.now() + RETRY_AFTER
        return done

    def start(self, interval_seconds: int) -> threading.Thread:
        def loop() -> None:
            while True:
                time.sleep(interval_seconds)
                try:
                    self.run_once()
                except Exception:
                    log.warning("summary scan failed", exc_info=True)

        thread = threading.Thread(target=loop, daemon=True, name="chat-summaries")
        thread.start()
        return thread
