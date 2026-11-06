import re
import time
import uuid
from types import SimpleNamespace

import pytest

from chats import CassandraChatStore, Chat, make_title

COLUMNS = ["owner", "chat_id", "title", "created_at", "updated_at", "pinned", "messages",
           "message_count", "prompt_tokens", "window_start", "window_summary", "summary",
           "summary_upto"]


class Result(list):
    def __init__(self, rows=(), applied=True):
        super().__init__(rows)
        self.was_applied = applied

    def one(self):
        return self[0] if self else None


class FakeSession:
    """Just enough CQL for chats.py: prepared statements are their query text."""

    def __init__(self):
        self.rows: dict[tuple, dict] = {}

    def prepare(self, query):
        return " ".join(query.split())

    def _row(self, data, columns=COLUMNS):
        return SimpleNamespace(**{c: data.get(c) for c in columns})

    def execute(self, query, params=()):
        q = " ".join(query.split())
        if q.startswith("CREATE"):
            return Result()
        if q.startswith("UPDATE"):
            sets = re.findall(r"(\w+) = \?", q.split(" WHERE ")[0])
            key = tuple(params[len(sets):])
            if q.endswith("IF EXISTS") and key not in self.rows:
                return Result(applied=False)
            row = self.rows.setdefault(key, {"owner": key[0], "chat_id": key[1]})
            row.update(zip(sets, params))
            return Result()
        if q.startswith("DELETE"):
            if "chat_id" in q:
                self.rows.pop(tuple(params), None)
            else:
                for key in [k for k in self.rows if k[0] == params[0]]:
                    del self.rows[key]
            return Result()
        if q.startswith("SELECT"):
            cols = q[len("SELECT "):q.index(" FROM")]
            columns = COLUMNS if cols == "*" else [c.strip() for c in cols.split(",")]
            matches = [r for k, r in self.rows.items()
                       if not params or k[0] == params[0] and (len(params) == 1 or k[1] == params[1])]
            return Result(self._row(r, columns) for r in matches)
        raise AssertionError(q)


@pytest.fixture
def store():
    s = CassandraChatStore(FakeSession(), "informatica", keep=5, pinned_max=2)
    s.ensure_schema()
    return s


def new_chat(store, owner="bob@x.com", question="q"):
    chat = Chat(owner=owner)
    chat.messages = [{"role": "user", "content": question}, {"role": "assistant", "content": "a"}]
    store.save(chat)
    time.sleep(0.001)  # distinct updated_at
    return chat


def test_save_load_round_trip(store):
    chat = Chat(owner="bob@x.com")
    chat.messages = [{"role": "user", "content": "Wat is één?", "excerpts": [["a.pdf", "t", 0.2]]},
                     {"role": "assistant", "content": "Een.\n", "usage": {"prompt_tokens": 9}}]
    chat.prompt_tokens, chat.window_start = 9, 0
    store.save(chat)
    loaded = store.load("bob@x.com", chat.chat_id)
    assert loaded.messages == chat.messages
    assert loaded.title == "Wat is één?"
    assert loaded.prompt_tokens == 9 and loaded.created_at
    assert store.load("eve@x.com", chat.chat_id) is None


def test_empty_chat_is_never_saved(store):
    store.save(Chat(owner="bob@x.com"))
    assert store.list_for("bob@x.com") == []


def test_keeps_five_most_recent_unpinned(store):
    chats = [new_chat(store, question=f"q{i}") for i in range(7)]
    assert [c.title for c in store.list_for("bob@x.com")] == ["q6", "q5", "q4", "q3", "q2"]
    # Continuing an old chat moves it to the top.
    store.save(chats[2])
    assert store.list_for("bob@x.com")[0].chat_id == chats[2].chat_id


def test_pinned_chats_do_not_count_and_sort_first(store):
    first = new_chat(store, question="first")
    assert store.set_pinned("bob@x.com", first.chat_id, True)
    for i in range(6):
        new_chat(store, question=f"q{i}")
    listed = store.list_for("bob@x.com")
    assert listed[0].title == "first" and listed[0].pinned
    assert len(listed) == 6  # 1 pinned + 5


def test_unpinned_chat_is_dropped_at_next_save_when_oldest(store):
    first = new_chat(store, question="first")
    store.set_pinned("bob@x.com", first.chat_id, True)
    for i in range(5):
        new_chat(store, question=f"q{i}")
    store.set_pinned("bob@x.com", first.chat_id, False)
    assert len(store.list_for("bob@x.com")) == 6  # nothing dropped on unpin
    new_chat(store, question="next")
    titles = [c.title for c in store.list_for("bob@x.com")]
    assert "first" not in titles and len(titles) == 5


def test_pin_cap(store):
    chats = [new_chat(store, question=f"q{i}") for i in range(3)]
    assert store.set_pinned("bob@x.com", chats[0].chat_id, True)
    assert store.set_pinned("bob@x.com", chats[1].chat_id, True)
    assert not store.set_pinned("bob@x.com", chats[2].chat_id, True)
    assert not store.set_pinned("bob@x.com", uuid.uuid1(), False)  # unknown chat: no row created
    assert len(store.list_for("bob@x.com")) == 3


def test_users_only_see_and_delete_their_own(store):
    mine = new_chat(store, owner="bob@x.com")
    theirs = new_chat(store, owner="eve@x.com")
    assert [c.chat_id for c in store.list_for("bob@x.com")] == [mine.chat_id]
    store.delete("bob@x.com", theirs.chat_id)
    assert store.load("eve@x.com", theirs.chat_id) is not None
    store.delete_all("bob@x.com")
    assert store.list_for("bob@x.com") == []
    assert len(store.list_for("eve@x.com")) == 1


def test_summary_does_not_resurrect_a_deleted_chat(store):
    chat = new_chat(store)
    store.delete("bob@x.com", chat.chat_id)
    assert not store.set_summary("bob@x.com", chat.chat_id, "s", 2)
    assert store.list_for("bob@x.com") == []


def test_save_keeps_the_summary(store):
    chat = new_chat(store)
    store.set_summary("bob@x.com", chat.chat_id, "about q", 2)
    store.save(chat)  # the session's copy has no summary
    loaded = store.load("bob@x.com", chat.chat_id)
    assert (loaded.summary, loaded.summary_upto) == ("about q", 2)


def test_make_title():
    assert make_title("  Hello\n world ") == "Hello world"
    long = "How do I descale the espresso machine without damaging the boiler seals?"
    title = make_title(long)
    assert title.endswith("…") and len(title) <= 51 and long.startswith(title[:-1])
