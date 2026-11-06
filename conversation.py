"""What gets sent to the model for a chat, kept free of Streamlit for testing.

A chat is a list of plain dicts, so it can be stored as JSON:

    {"role": "user", "content": question, "excerpts": [[file, text, distance], ...]}
    {"role": "assistant", "content": answer, "caption": "...", "usage": {...}}

User and assistant messages strictly alternate (a failed answer removes its
question again), so every even index is a user message.

``usage`` holds Ollama's real token counts for the turn (see ``usage_from``).
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


def usage_from(metadata: dict | None) -> dict:
    """Token counts and timings from the final chunk of an Ollama stream.

    Ollama reports the *whole* prompt in prompt_eval_count even when llama.cpp
    served most of it from its cache (checked on 0.40.8: the same 320-token
    prompt reports 320 both times, only prompt_eval_duration drops), so it is
    the real prompt size.
    """
    metadata = metadata or {}
    ns = 1e9
    return {
        "prompt_tokens": int(metadata.get("prompt_eval_count") or 0),
        "output_tokens": int(metadata.get("eval_count") or 0),
        "prompt_seconds": round((metadata.get("prompt_eval_duration") or 0) / ns, 2),
        "output_seconds": round((metadata.get("eval_duration") or 0) / ns, 2),
        "load_seconds": round((metadata.get("load_duration") or 0) / ns, 2),
    }
