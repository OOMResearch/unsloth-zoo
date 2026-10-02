"""Fast, offline unit tests for unsloth_zoo.mlx.utils._prepare_dataset's
process-level cache (docs/mlx_lm-OPTIMIZATION_PLAN.md item 3.1,
unsloth-studio-test-harness). Monkeypatches the real tokenizer-facing
helpers (normalize_mlx_chat_template/collect_mlx_texts/encode_mlx_text)
with cheap stubs and a call counter, so these tests exercise only
_prepare_dataset's own formatting-loop/cache logic, not real tokenization
(covered elsewhere)."""

from __future__ import annotations

import pytest

import unsloth_zoo.mlx.utils as mlx_utils


class _FakeTokenizer:
    eos_token_id = 99


class _FakeNamedTokenizer:
    """Stands in for mlx_lm.tokenizer_utils.TokenizerWrapper: a real
    tokenizer reloaded via a fresh from_pretrained(...) call gets a new
    object (new id()) but the same name_or_path -- confirmed for real
    against two separate FastMLXModel.from_pretrained("Qwen/Qwen2.5-0.5B")
    calls. Each instance here is deliberately distinct (no shared id())."""

    eos_token_id = 99

    def __init__(self, name_or_path, vocab_size=1000):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size


@pytest.fixture(autouse=True)
def _clear_prepare_dataset_cache():
    mlx_utils._PREPARE_DATASET_CACHE.clear()
    yield
    mlx_utils._PREPARE_DATASET_CACHE.clear()


@pytest.fixture(autouse=True)
def _stub_tokenizer_helpers(monkeypatch):
    calls = {"collect": 0}

    def _fake_normalize(tokenizer, **kwargs):
        return tokenizer

    def _fake_collect(target, item, *, dataset_text_field="text", is_vlm=False):
        calls["collect"] += 1
        return [item[dataset_text_field]]

    def _fake_encode(tokenizer, text, state=None, *, add_special_tokens=None):
        return [ord(c) for c in text[:4]]

    monkeypatch.setattr(mlx_utils, "normalize_mlx_chat_template", _fake_normalize)
    monkeypatch.setattr(mlx_utils, "collect_mlx_texts", _fake_collect)
    monkeypatch.setattr(mlx_utils, "encode_mlx_text", _fake_encode)
    return calls


def test_identical_calls_return_same_object_and_skip_reformatting(_stub_tokenizer_helpers):
    tokenizer = _FakeTokenizer()
    rows = [{"text": "hello"}, {"text": "world"}]

    first = mlx_utils._prepare_dataset(list(rows), tokenizer, dataset_text_field="text")
    assert _stub_tokenizer_helpers["collect"] == len(rows)

    second = mlx_utils._prepare_dataset(list(rows), tokenizer, dataset_text_field="text")
    assert second is first
    # No new collect_mlx_texts calls on the cache hit.
    assert _stub_tokenizer_helpers["collect"] == len(rows)


def test_different_content_misses_cache(_stub_tokenizer_helpers):
    tokenizer = _FakeTokenizer()
    first = mlx_utils._prepare_dataset(
        [{"text": "hello"}], tokenizer, dataset_text_field="text"
    )
    second = mlx_utils._prepare_dataset(
        [{"text": "goodbye"}], tokenizer, dataset_text_field="text"
    )
    assert second is not first
    assert _stub_tokenizer_helpers["collect"] == 2


def test_different_tokenizer_identity_misses_cache(_stub_tokenizer_helpers):
    """Two tokenizer objects with no name_or_path fall back to raw
    identity -- a real cache miss, not a silent false hit."""
    rows = [{"text": "hello"}]
    first = mlx_utils._prepare_dataset(list(rows), _FakeTokenizer(), dataset_text_field="text")
    second = mlx_utils._prepare_dataset(list(rows), _FakeTokenizer(), dataset_text_field="text")
    assert second is not first
    assert _stub_tokenizer_helpers["collect"] == 2


def test_reloaded_tokenizer_with_same_name_or_path_hits_cache(_stub_tokenizer_helpers):
    """Regression test for a real bug found while validating this cache
    through the actual harness (docs/mlx_lm-OPTIMIZATION_PLAN.md item 3.1,
    2026-10-01): a benchmark harness rebuilds the model/tokenizer fresh
    every repetition, so id(tokenizer) differs every single call even
    though it's "the same" tokenizer reloaded -- keying on raw identity
    made every repetition a guaranteed cache miss, defeating the cache
    for its primary intended use case. name_or_path is stable across such
    reloads; two distinct objects sharing it must still hit."""
    rows = [{"text": "hello"}]
    first = mlx_utils._prepare_dataset(
        list(rows), _FakeNamedTokenizer("Qwen/Qwen2.5-0.5B"), dataset_text_field="text"
    )
    second = mlx_utils._prepare_dataset(
        list(rows), _FakeNamedTokenizer("Qwen/Qwen2.5-0.5B"), dataset_text_field="text"
    )
    assert second is first
    assert _stub_tokenizer_helpers["collect"] == 1


def test_different_name_or_path_misses_cache(_stub_tokenizer_helpers):
    rows = [{"text": "hello"}]
    first = mlx_utils._prepare_dataset(
        list(rows), _FakeNamedTokenizer("Qwen/Qwen2.5-0.5B"), dataset_text_field="text"
    )
    second = mlx_utils._prepare_dataset(
        list(rows), _FakeNamedTokenizer("Qwen/Qwen3-0.6B"), dataset_text_field="text"
    )
    assert second is not first
    assert _stub_tokenizer_helpers["collect"] == 2


def test_different_dataset_text_field_misses_cache(_stub_tokenizer_helpers):
    tokenizer = _FakeTokenizer()
    rows_a = [{"text": "hello", "alt": "hello"}]
    first = mlx_utils._prepare_dataset(list(rows_a), tokenizer, dataset_text_field="text")
    second = mlx_utils._prepare_dataset(list(rows_a), tokenizer, dataset_text_field="alt")
    assert second is not first


def test_cache_is_bounded(_stub_tokenizer_helpers):
    tokenizer = _FakeTokenizer()
    for i in range(mlx_utils._PREPARE_DATASET_CACHE_MAX_ENTRIES + 5):
        mlx_utils._prepare_dataset(
            [{"text": f"row-{i}"}], tokenizer, dataset_text_field="text"
        )
    assert len(mlx_utils._PREPARE_DATASET_CACHE) <= mlx_utils._PREPARE_DATASET_CACHE_MAX_ENTRIES


def test_one_shot_iterable_is_not_exhausted_before_formatting(_stub_tokenizer_helpers):
    """_prepare_dataset must materialize a one-shot iterable once and reuse
    those same rows for both the cache-key fingerprint and the real
    formatting loop -- not consume it twice."""
    tokenizer = _FakeTokenizer()

    def _one_shot():
        yield {"text": "hello"}
        yield {"text": "world"}

    result = mlx_utils._prepare_dataset(_one_shot(), tokenizer, dataset_text_field="text")
    assert len(result) == 2
    assert _stub_tokenizer_helpers["collect"] == 2
