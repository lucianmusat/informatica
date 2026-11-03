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
