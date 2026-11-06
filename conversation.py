"""What gets sent to the model for a chat, kept free of Streamlit for testing.

A chat is a list of plain dicts, so it can be stored as JSON:

    {"role": "user", "content": question, "excerpts": [[file, text, distance], ...]}
    {"role": "assistant", "content": answer, "caption": "...", "usage": {...}}

User and assistant messages strictly alternate (a question is added only
together with its answer), so every even index is a user message.

``usage`` holds Ollama's real token counts for the turn (see ``usage_from``),
plus the size in characters of the prompt they were measured on and the window
it was built from.

Only a window of the chat goes to the model: ``chat.messages[window_start:]``,
behind the system prompt and ``window_summary``. It only ever grows at the end,
so every prompt continues the previous one and llama.cpp reads just the new
turn. When the prompt passes the budget, ``fit_window`` cuts the oldest turns in
one go, down to well under the budget; that costs one full re-read, and then
the window grows at the end again for thousands of tokens.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import prompts


def user_message(question: str, excerpts) -> dict:
    return {"role": "user", "content": question, "excerpts": [list(e) for e in excerpts]}


def model_message(message: dict) -> BaseMessage:
    if message["role"] == "user":
        return HumanMessage(content=prompts.user_content(message["content"], message.get("excerpts")))
    return AIMessage(content=message["content"])


def model_messages(messages: list[dict], summary: str = "") -> list[BaseMessage]:
    """System prompt (plus the summary of older turns, if any) and ``messages``."""
    return [SystemMessage(content=prompts.system_prompt(summary))] + [model_message(m) for m in messages]


def payload_chars(payload: list[BaseMessage]) -> int:
    return sum(len(m.content) for m in payload)


def window_payload(chat, asked: dict) -> list[BaseMessage]:
    """The prompt for the next answer: the chat's window plus the new question."""
    return model_messages(chat.messages[chat.window_start:] + [asked], chat.window_summary)


# Used until a real prompt size has been measured for the chat. On the low side
# for English with Qwen's tokenizer, so estimates err towards cutting early.
DEFAULT_CHARS_PER_TOKEN = 3.5


def _last_usage(messages: list[dict]) -> dict | None:
    for message in reversed(messages):
        if message["role"] == "assistant" and (message.get("usage") or {}).get("prompt_tokens"):
            return message["usage"]
    return None


def chars_per_token(messages: list[dict]) -> float:
    usage = _last_usage(messages)
    if usage and usage.get("prompt_chars"):
        return usage["prompt_chars"] / usage["prompt_tokens"]
    return DEFAULT_CHARS_PER_TOKEN


def estimate_tokens(chars: int, ratio: float) -> int:
    return int(chars / ratio) + 1


def prompt_tokens(chat, asked: dict) -> int:
    """Size of the next prompt.

    When the last answer was produced from the current window, this is real:
    Ollama's count for that prompt, plus the answer it wrote, plus an estimate
    for the new question only. Otherwise (no answer yet, or the window moved
    since) the whole prompt is estimated.
    """
    ratio = chars_per_token(chat.messages)
    last = chat.messages[-1] if chat.messages else None
    usage = (last or {}).get("usage") or {}
    if (usage.get("prompt_tokens") and usage.get("window_start") == chat.window_start
            and usage.get("summary_chars", 0) == len(chat.window_summary)):
        new = len(model_message(asked).content)
        return usage["prompt_tokens"] + usage.get("output_tokens", 0) + estimate_tokens(new, ratio)
    return estimate_tokens(payload_chars(window_payload(chat, asked)), ratio)


def fit_window(chat, asked: dict, budget: int, trim_to: int, summary: str = "") -> bool:
    """Move ``window_start`` forward when the next prompt would pass ``budget``.

    The oldest whole exchanges are dropped until the estimate is at most
    ``trim_to``, and ``summary`` (the chat's latest idle summary, possibly
    lagging behind the cut) goes in front of what is left. If even the last
    exchange doesn't fit ``trim_to`` it is kept as long as it fits the budget.
    Returns True when the window moved.
    """
    if prompt_tokens(chat, asked) <= budget:
        return False

    ratio = chars_per_token(chat.messages)
    new_summary = summary or chat.window_summary
    fixed = len(prompts.system_prompt(new_summary)) + len(model_message(asked).content)
    sizes = [len(model_message(m).content) for m in chat.messages]

    def tokens_from(start: int) -> int:
        return estimate_tokens(fixed + sum(sizes[start:]), ratio)

    # Exchanges start at even indices. Drop at least one; keep the last one if
    # at all possible, and only drop everything when even that is over budget.
    starts = range(chat.window_start + 2, len(chat.messages) - 1, 2)
    start = next((s for s in starts if tokens_from(s) <= trim_to), None)
    if start is None:
        start = next((s for s in starts if tokens_from(s) <= budget), len(chat.messages))
    chat.window_start = start
    chat.window_summary = new_summary
    return True


def reopen_window(chat, keep_exchanges: int) -> bool:
    """Start a reopened chat from its summary and the last ``keep_exchanges``.

    Turns after ``summary_upto`` are never dropped (the summary may be older
    than the chat), and the window never moves back. Returns True when the
    window changed; the first answer then reads the whole, smaller, prompt.
    """
    if not chat.summary:
        return False
    keep_from = max(len(chat.messages) - 2 * keep_exchanges, 0)
    start = max(chat.window_start, min(chat.summary_upto, keep_from))
    start -= start % 2
    if start == 0:
        return False  # the whole chat fits; the summary would only repeat it
    if start == chat.window_start and chat.window_summary == chat.summary:
        return False
    chat.window_start, chat.window_summary = start, chat.summary
    return True


def usage_from(metadata: dict | None, chat=None, payload: list[BaseMessage] | None = None) -> dict:
    """Token counts and timings from the final chunk of an Ollama stream.

    Ollama reports the *whole* prompt in prompt_eval_count even when llama.cpp
    served most of it from its cache (checked on 0.40.8: the same 320-token
    prompt reports 320 both times, only prompt_eval_duration drops), so it is
    the real prompt size.
    """
    metadata = metadata or {}
    ns = 1e9
    window = {}
    if chat is not None and payload is not None:
        window = {
            "prompt_chars": payload_chars(payload),
            "window_start": chat.window_start,
            "summary_chars": len(chat.window_summary),
        }
    return window | {
        "prompt_tokens": int(metadata.get("prompt_eval_count") or 0),
        "output_tokens": int(metadata.get("eval_count") or 0),
        "prompt_seconds": round((metadata.get("prompt_eval_duration") or 0) / ns, 2),
        "output_seconds": round((metadata.get("eval_duration") or 0) / ns, 2),
        "load_seconds": round((metadata.get("load_duration") or 0) / ns, 2),
    }
