"""Helpers for talking to Ollama: reading /api/ps and /api/tags, and the HTTP
transport that keeps chat requests compatible with OpenClaw's."""

from __future__ import annotations

import json

import httpx

# Ollama starts a separate llama-server runner per combination of load options,
# and the request field "shift" is one of them (true adds --context-shift).
# OpenClaw sends shift=false; ollama-python (0.6.x) cannot send the field at all,
# so Ollama defaulted to true and swapped the shared model out and back in
# whenever informatica and OpenClaw took turns. Disabling it here makes both
# clients load the identical runner.
NO_SHIFT_PATHS = ("/api/chat", "/api/generate")


def with_shift_disabled(body: dict) -> dict:
    return {**body, "shift": False}


class NoShiftTransport(httpx.BaseTransport):
    """Adds ``"shift": false`` to Ollama chat/generate requests."""

    def __init__(self, inner: httpx.BaseTransport | None = None):
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path in NO_SHIFT_PATHS:
            body = json.loads(request.read() or b"{}")
            headers = [(k, v) for k, v in request.headers.multi_items()
                       if k.lower() not in ("content-length", "content-type")]
            request = httpx.Request(
                request.method, request.url, headers=headers,
                json=with_shift_disabled(body), extensions=request.extensions,
            )
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def _names(model: str) -> set[str]:
    return {model, f"{model}:latest"}


def model_pulled(tags: dict, model: str) -> bool:
    """The model is on disk (``/api/tags``)."""
    return any(m.get("name") in _names(model) for m in tags.get("models", []))


def model_loaded(ps: dict, model: str, num_ctx: int) -> bool:
    """The model is already in memory with the context size we will ask for.

    Ollama reloads a model whose num_ctx differs from the request, so a copy
    loaded with another context size counts as not loaded.
    """
    for m in ps.get("models", []):
        if m.get("name") in _names(model) or m.get("model") in _names(model):
            ctx = m.get("context_length")
            if ctx is None or ctx == num_ctx:
                return True
    return False
