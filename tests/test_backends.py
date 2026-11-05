from backends import model_loaded, model_pulled

PS = {"models": [{"name": "qwen3.8:27b", "model": "qwen3.8:27b", "context_length": 16384}]}


def test_loaded_with_matching_context():
    assert model_loaded(PS, "qwen3.8:27b", 16384)


def test_loaded_with_other_context_counts_as_cold():
    assert not model_loaded(PS, "qwen3.8:27b", 4096)


def test_other_or_no_model_loaded():
    assert not model_loaded(PS, "qwen3:4b-instruct", 16384)
    assert not model_loaded({"models": []}, "qwen3.8:27b", 16384)
    assert not model_loaded({}, "qwen3.8:27b", 16384)


def test_latest_suffix():
    assert model_loaded({"models": [{"name": "foo:latest"}]}, "foo", 1)
    assert model_pulled({"models": [{"name": "foo:latest"}]}, "foo")
    assert not model_pulled({"models": [{"name": "foobar"}]}, "foo")


def test_no_shift_transport_adds_shift_false_to_chat_only():
    import json

    import httpx

    from backends import NoShiftTransport

    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json={})

    client = httpx.Client(base_url="http://ollama", transport=NoShiftTransport(httpx.MockTransport(handler)))
    client.post("/api/chat", json={"model": "m", "options": {"num_ctx": 32768}})
    client.post("/api/generate", json={"model": "m"})
    client.post("/api/embed", json={"model": "e", "input": "x"})
    client.get("/api/ps")

    assert seen[0] == ("/api/chat", {"model": "m", "options": {"num_ctx": 32768}, "shift": False})
    assert seen[1] == ("/api/generate", {"model": "m", "shift": False})
    assert seen[2] == ("/api/embed", {"model": "e", "input": "x"})
    assert seen[3] == ("/api/ps", None)
