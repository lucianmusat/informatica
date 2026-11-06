from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import conversation
import prompts
from chats import CassandraChatStore, Chat
from summarizer import Activity, Summarizer
from test_chats import FakeSession

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


class FakeLLM:
    def __init__(self, reply="They discussed coffee.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.fail:
            raise ConnectionError("ollama down")
        return SimpleNamespace(content=self.reply + "\n", response_metadata={"prompt_eval_count": 50})


@pytest.fixture
def store():
    s = CassandraChatStore(FakeSession(), "informatica", keep=5, pinned_max=20)
    s.ensure_schema()
    return s


def saved_chat(store, exchanges, idle_minutes, owner="bob@x.com"):
    chat = Chat(owner=owner)
    for i in range(exchanges):
        chat.messages += [conversation.user_message(f"q{i}", []), {"role": "assistant", "content": f"a{i}"}]
    store.save(chat)
    row = store.session.rows[(owner, chat.chat_id)]
    row["updated_at"] = (NOW - timedelta(minutes=idle_minutes)).replace(tzinfo=None)  # driver style
    return chat


def summarizer(store, llm, busy=False):
    return Summarizer(store, llm, idle_minutes=15, keep_exchanges=2,
                      is_busy=lambda: busy, now=lambda: NOW)


def test_only_idle_long_unsummarised_chats_are_due(store):
    idle = saved_chat(store, 3, idle_minutes=20)
    saved_chat(store, 3, idle_minutes=5)    # still active
    saved_chat(store, 2, idle_minutes=60)   # short: a reopen keeps all of it anyway
    s = summarizer(store, FakeLLM())
    assert [c.chat_id for c in s.due()] == [idle.chat_id]


def test_summary_request_continues_the_last_prompt(store):
    chat = saved_chat(store, 3, idle_minutes=20)
    llm = FakeLLM()
    assert summarizer(store, llm).run_once() == 1
    payload = llm.calls[0]
    assert payload[:-1] == conversation.model_messages(chat.messages)
    assert payload[-1].content == prompts.SUMMARY_REQUEST
    loaded = store.load("bob@x.com", chat.chat_id)
    assert (loaded.summary, loaded.summary_upto) == ("They discussed coffee.", 6)
    # Covered now: not due again until the chat grows.
    assert summarizer(store, llm).due() == []


def test_uses_the_window_and_its_frozen_summary(store):
    chat = saved_chat(store, 4, idle_minutes=20)
    chat.window_start, chat.window_summary = 4, "older"
    store.save(chat)
    store.session.rows[("bob@x.com", chat.chat_id)]["updated_at"] = (NOW - timedelta(hours=1)).replace(tzinfo=None)
    llm = FakeLLM()
    summarizer(store, llm).run_once()
    assert llm.calls[0][:-1] == conversation.model_messages(chat.messages[4:], "older")


def test_waits_while_an_answer_is_in_progress(store):
    saved_chat(store, 3, idle_minutes=20)
    llm = FakeLLM()
    assert summarizer(store, llm, busy=True).run_once() == 0
    assert llm.calls == []


def test_failure_backs_off(store):
    saved_chat(store, 3, idle_minutes=20)
    s = summarizer(store, FakeLLM(fail=True))
    assert s.run_once() == 0
    assert s.due() == []


def test_activity_counts_nested_answers():
    a = Activity()
    with a:
        with a:
            assert a.busy
        assert a.busy
    assert not a.busy


# -- reopening ------------------------------------------------------------------

def chat_of(exchanges, summary="", upto=0, window_start=0):
    c = Chat(owner="bob@x.com", summary=summary, summary_upto=upto, window_start=window_start)
    for i in range(exchanges):
        c.messages += [conversation.user_message(f"q{i}", []), {"role": "assistant", "content": f"a{i}"}]
    return c


def test_reopen_starts_from_summary_and_last_two_exchanges():
    c = chat_of(10, summary="s", upto=20)
    assert conversation.reopen_window(c, 2)
    assert (c.window_start, c.window_summary) == (16, "s")
    payload = conversation.window_payload(c, conversation.user_message("new", []))
    assert payload[0].content.endswith("s") and payload[1].content == "q8"


def test_reopen_never_drops_turns_the_summary_misses():
    c = chat_of(10, summary="s", upto=10)
    conversation.reopen_window(c, 2)
    assert c.window_start == 10


def test_reopen_without_summary_or_short_chat():
    c = chat_of(10)
    assert not conversation.reopen_window(c, 2) and c.window_start == 0
    c = chat_of(2, summary="s", upto=4)
    assert not conversation.reopen_window(c, 2)  # whole chat kept; no summary needed
    assert (c.window_start, c.window_summary) == (0, "")


def test_reopen_never_moves_the_window_back():
    c = chat_of(10, summary="s", upto=20, window_start=18)
    conversation.reopen_window(c, 2)
    assert c.window_start == 18
