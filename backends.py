"""Pure helpers for reading Ollama's /api/ps and /api/tags responses."""

from __future__ import annotations


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
