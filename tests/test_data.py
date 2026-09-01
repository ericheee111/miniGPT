from __future__ import annotations

import gc
import json
import tempfile
import weakref
from importlib import import_module
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from minigpt import data, story_data
from minigpt.tokenizer import (
    BPE_COUNT_SPECIAL_TOKENS,
    BPE_MAX_VOCAB_SIZE,
    BPE_SPECIAL_TOKENS,
    BPETokenizer,
    StoryForgeTokenizerError,
    load_tokenizer,
)

MetadataValue: TypeAlias = str | int | float
TokenTestSource: TypeAlias = npt.NDArray[np.uint16] | list[int]

_STORY_CORPUS: Final = [
    "hello world",
    "hello hello",
    "world of warcraft",
    "the quick brown fox",
    "  leading and trailing  ",
    "tab\there",
    "uni  café",
    "你好 世界",
]


def _reference_batches(
    tokens: npt.NDArray[np.uint16],
    *,
    batch_size: int,
    block_size: int,
    seed: int,
    count: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reproduce the pre-Stage-8 row-wise batching algorithm independently."""
    rng = np.random.default_rng(seed)
    reference: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        starts = rng.integers(
            0,
            tokens.size - block_size,
            size=batch_size,
            dtype=np.int64,
        )
        windows = np.empty((batch_size, block_size + 1), dtype=np.int64)
        for index in range(batch_size):
            start_index = int(cast("np.int64", starts[index]))
            windows[index] = tokens[start_index : start_index + block_size + 1]
        reference.append(
            (
                torch.tensor(windows[:, :-1], dtype=torch.long),
                torch.tensor(windows[:, 1:], dtype=torch.long),
            ),
        )
    return reference


def _token_test_source(source_kind: str, tmp_path: Path) -> TokenTestSource:
    """Build equivalent ndarray, sequence, or read-only mmap test input."""
    tokens = np.arange(257, dtype=np.uint16)
    if source_kind == "ndarray":
        return tokens
    if source_kind == "sequence":
        return tokens.tolist()
    token_path = tmp_path / "tokens.bin"
    writable = np.memmap(token_path, dtype=np.uint16, mode="w+", shape=tokens.shape)
    writable[:] = tokens
    writable.flush()
    del writable
    return cast(
        "npt.NDArray[np.uint16]",
        np.memmap(token_path, dtype=np.uint16, mode="r", shape=tokens.shape),
    )


def test_tokenizer_round_trip() -> None:
    # Given: a tokenizer built from representative text.
    text = "cab\n"
    tokenizer = data.CharTokenizer.from_text(text)

    # When: the text is encoded and decoded.
    decoded = tokenizer.decode(tokenizer.encode(text))

    # Then: no information is lost.
    assert decoded == text
    assert tokenizer.vocab_size == 4


def test_tokenizer_rejects_unknown_character() -> None:
    # Given: a tokenizer whose vocabulary does not contain "z".
    tokenizer = data.CharTokenizer.from_text("abc")

    # When: unknown text is encoded.
    with pytest.raises(
        data.UnknownCharacterError,
        match=r"'z'.*position 1",
    ) as error_info:
        _ = tokenizer.encode("az")

    # Then: the typed error identifies the character and position.
    assert error_info.value.character == "z"
    assert error_info.value.position == 1


def test_tokenizer_save_load_preserves_vocabulary(tmp_path: Path) -> None:
    # Given: a tokenizer persisted as JSON.
    tokenizer = data.CharTokenizer.from_text("bca")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    # When: the tokenizer is loaded.
    loaded = data.CharTokenizer.load(tokenizer_path)

    # Then: token IDs and decoded text remain stable.
    assert loaded.encode("abc") == tokenizer.encode("abc")
    assert loaded.decode([0, 1, 2]) == tokenizer.decode([0, 1, 2])


def test_download_text_fetches_source(tmp_path: Path) -> None:
    # Given: a local URL and an empty raw-data destination.
    source_path = tmp_path / "source.txt"
    _ = source_path.write_text("first version", encoding="utf-8")
    destination = tmp_path / "raw" / "input.txt"

    # When: the source is downloaded.
    result = data.download_text(source_path.as_uri(), destination)

    # Then: the destination contains the source text.
    assert result == destination
    assert destination.read_text(encoding="utf-8") == "first version"


def test_download_text_reuses_existing_file(tmp_path: Path) -> None:
    # Given: an existing destination and a source that must not be accessed.
    destination = tmp_path / "raw" / "input.txt"
    destination.parent.mkdir(parents=True)
    _ = destination.write_text("cached version", encoding="utf-8")

    # When: download is requested again.
    result = data.download_text("https://invalid.example/unused.txt", destination)

    # Then: the cached destination is returned unchanged.
    assert result == destination
    assert destination.read_text(encoding="utf-8") == "cached version"


def test_prepare_tiny_shakespeare_writes_split_and_metadata(tmp_path: Path) -> None:
    # Given: an existing raw corpus that must be reused.
    raw_path = tmp_path / "raw" / "input.txt"
    raw_path.parent.mkdir(parents=True)
    _ = raw_path.write_text("abcdeabcde", encoding="utf-8")

    # When: the corpus is prepared with a 90/10 split.
    prepared = data.prepare_tiny_shakespeare(
        data_dir=tmp_path,
        source_url="https://invalid.example/unused.txt",
    )

    # Then: tokens, tokenizer, and reproducibility metadata are persisted.
    train_tokens = cast("npt.NDArray[np.uint16]", np.load(prepared.train_path))
    val_tokens = cast("npt.NDArray[np.uint16]", np.load(prepared.val_path))
    metadata = cast(
        "dict[str, MetadataValue]",
        json.loads(prepared.metadata_path.read_text(encoding="utf-8")),
    )
    assert train_tokens.shape == (9,)
    assert val_tokens.shape == (1,)
    assert prepared.tokenizer_path.is_file()
    assert metadata["train_tokens"] == 9
    assert metadata["val_tokens"] == 1
    assert metadata["split_ratio"] == 0.9
    assert metadata["source_url"] == "https://invalid.example/unused.txt"


def test_token_batcher_returns_shifted_cpu_batches() -> None:
    # Given: sequential token IDs with enough room for several windows.
    tokens = np.arange(32, dtype=np.uint16)
    batcher = data.TokenBatcher(tokens, batch_size=4, block_size=8, seed=7)

    # When: a training batch is sampled.
    x, y = batcher.next_batch()

    # Then: shapes, device, integer dtype, and next-token alignment are correct.
    assert x.shape == y.shape == (4, 8)
    assert x.device.type == y.device.type == "cpu"
    assert x.dtype == y.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_token_batcher_is_reproducible_for_fixed_seed() -> None:
    # Given: two independent batchers with identical data and seed.
    tokens = np.arange(64, dtype=np.uint16)
    first = data.TokenBatcher(tokens, batch_size=3, block_size=6, seed=42)
    second = data.TokenBatcher(tokens, batch_size=3, block_size=6, seed=42)

    # When: each batcher samples its first batch.
    first_batch = first.next_batch()
    second_batch = second.next_batch()

    # Then: both x and y are identical.
    assert torch.equal(first_batch[0], second_batch[0])
    assert torch.equal(first_batch[1], second_batch[1])


@pytest.mark.parametrize("source_kind", ["mmap", "ndarray", "sequence"])
def test_token_batcher_matches_pre_optimization_batches(
    source_kind: str,
    tmp_path: Path,
) -> None:
    # Given: each supported source form and an independent implementation of the old algorithm.
    source = _token_test_source(source_kind, tmp_path)
    reference_tokens = np.arange(257, dtype=np.uint16)
    expected = _reference_batches(
        reference_tokens,
        batch_size=7,
        block_size=19,
        seed=314,
        count=4,
    )
    batcher = data.TokenBatcher(source, batch_size=7, block_size=19, seed=314)

    # When: several batches are sampled from the same RNG trajectory.
    actual = [batcher.next_batch() for _ in range(4)]

    # Then: every token remains exactly equal to the pre-optimization path.
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert torch.equal(actual_batch[0], expected_batch[0])
        assert torch.equal(actual_batch[1], expected_batch[1])


def test_token_batcher_restores_next_batch_rng_state() -> None:
    # Given: a batcher whose RNG state is captured before sampling.
    tokens = np.arange(257, dtype=np.uint16)
    batcher = data.TokenBatcher(tokens, batch_size=5, block_size=17, seed=91)
    state = batcher.capture_random_state()
    expected = batcher.next_batch()

    # When: another batch advances the RNG before the captured state is restored.
    _ = batcher.next_batch()
    batcher.restore_random_state(state)
    actual = batcher.next_batch()

    # Then: the restored next batch is token-for-token identical.
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_token_batcher_preserves_read_only_source() -> None:
    # Given: an immutable token array.
    tokens = np.arange(257, dtype=np.uint16)
    tokens.setflags(write=False)
    original = tokens.copy()
    batcher = data.TokenBatcher(tokens, batch_size=5, block_size=17, seed=12)

    # When: multiple batches are sampled.
    _ = batcher.next_batch()
    _ = batcher.next_batch()

    # Then: the source remains read-only and byte-for-byte unchanged.
    assert not tokens.flags.writeable
    assert tokens.tolist() == original.tolist()


def test_token_batcher_retains_mmap_source_without_full_dtype_copy(tmp_path: Path) -> None:
    # Given: a read-only uint16 mmap retained only by the batcher after construction.
    token_path = tmp_path / "tokens.bin"
    tokens = np.arange(257, dtype=np.uint16)
    writable = np.memmap(token_path, dtype=np.uint16, mode="w+", shape=tokens.shape)
    writable[:] = tokens
    writable.flush()
    del writable
    source = cast(
        "npt.NDArray[np.uint16]",
        np.memmap(token_path, dtype=np.uint16, mode="r", shape=tokens.shape),
    )
    source_reference = weakref.ref(source)
    batcher = data.TokenBatcher(source, batch_size=5, block_size=17, seed=12)

    # When: the caller releases its source reference.
    del source
    _ = gc.collect()

    # Then: the batcher still retains the original mmap-backed source and remains usable.
    assert source_reference() is not None
    _ = batcher.next_batch()


def test_token_batcher_returns_shifted_views_of_one_batch_owner() -> None:
    # Given: a CPU batcher.
    batcher = data.TokenBatcher(
        np.arange(257, dtype=np.uint16),
        batch_size=5,
        block_size=17,
        seed=12,
    )

    # When: a batch is sampled.
    x, y = batcher.next_batch()

    # Then: shifted non-contiguous views share the one call-owned tensor allocation.
    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert not x.is_contiguous()
    assert not y.is_contiguous()
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_token_batcher_next_call_does_not_modify_prior_results() -> None:
    # Given: a sampled batch and immutable snapshots of both returned tensors.
    batcher = data.TokenBatcher(
        np.arange(257, dtype=np.uint16),
        batch_size=5,
        block_size=17,
        seed=12,
    )
    first_x, first_y = batcher.next_batch()
    expected_x = first_x.clone()
    expected_y = first_y.clone()

    # When: the batcher samples its next batch.
    _ = batcher.next_batch()

    # Then: the earlier call still owns unchanged results.
    assert torch.equal(first_x, expected_x)
    assert torch.equal(first_y, expected_y)


# ---------------------------------------------------------------------------
# Story Forge tokenizer v2: backward-compatible char tokenizer and ByteLevel BPE
# ---------------------------------------------------------------------------


def _story_tokenizer() -> BPETokenizer:
    """Train a small deterministic BPE tokenizer for schema-level tests."""
    return BPETokenizer.train_from_iterator(iter(_STORY_CORPUS), vocab_size=600)


def _require_tokenizers() -> None:
    """Fail with a clear message when the optional backend is unavailable."""
    try:
        _ = import_module("tokenizers")
    except ModuleNotFoundError as error:
        pytest.skip(f"tokenizers optional dependency unavailable: {error}")


def test_char_tokenizer_round_trip_and_artifact_bytes_unchanged(tmp_path: Path) -> None:
    # Given: the original v1 artifact shape and a representative vocabulary.
    text = "cab\n"
    tokenizer = data.CharTokenizer.from_text(text)

    # When: the tokenizer round-trips and persists the v1 artifact.
    decoded = tokenizer.decode(tokenizer.encode(text))
    artifact_path = tmp_path / "tokenizer.json"
    tokenizer.save(artifact_path)

    # Then: round-trip and the exact v1 artifact remain unchanged.
    assert decoded == text
    loaded = cast(
        "dict[str, object]",
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    assert loaded == {"version": 1, "vocabulary": ["\n", "a", "b", "c"]}


def test_char_tokenizer_legacy_loader_preserves_ids(tmp_path: Path) -> None:
    # Given: a v1 artifact written by the legacy char loader.
    tokenizer = data.CharTokenizer.from_text("bca")
    artifact_path = tmp_path / "tokenizer.json"
    tokenizer.save(artifact_path)

    # When: the artifact is loaded through the legacy entrypoint.
    loaded = data.CharTokenizer.load(artifact_path)

    # Then: the loader preserves token IDs and round-trip semantics.
    assert loaded.encode("abc") == tokenizer.encode("abc")
    assert loaded.decode([0, 1, 2]) == tokenizer.decode([0, 1, 2])


def test_char_tokenizer_protocol_properties() -> None:
    # Given: a char tokenizer that participates in the common protocol.
    tokenizer = data.CharTokenizer.from_text("abc")

    # When: the protocol labels and special-token IDs are inspected.
    # Then: the char backend reports its labels and defines no special tokens.
    assert tokenizer.tokenizer_type == "char"
    assert tokenizer.model_family == "char_gpt"
    assert tokenizer.bos_token_id is None
    assert tokenizer.eos_token_id is None
    assert tokenizer.pad_token_id is None
    assert tokenizer.special_token_id("<bos>") is None


def test_char_tokenizer_decode_accepts_skip_special_tokens_keyword() -> None:
    # Given: a char tokenizer whose decode keeps its historical positional contract.
    tokenizer = data.CharTokenizer.from_text("abc")

    # When: decode is invoked with the protocol keyword and without it.
    positional = tokenizer.decode([0, 1, 2])
    keyword = tokenizer.decode([0, 1, 2], skip_special_tokens=True)

    # Then: both signatures decode identically.
    assert positional == keyword == "abc"


def test_bpe_whitespace_and_unicode_round_trip() -> None:
    # Given: a trained ByteLevel BPE tokenizer.
    _require_tokenizers()
    tokenizer = _story_tokenizer()

    # When: representative strings exchange through encode/decode.
    samples = [
        "",
        "a",
        "hello world",
        "  leading and trailing  ",
        "tab\there",
        "newline\nbreak",
        "café résumé",
        "你好 世界",
    ]
    round_tripped = [tokenizer.decode(tokenizer.encode(sample)) for sample in samples]

    # Then: spacing and arbitrary UTF-8 survive byte-for-byte.
    assert round_tripped == samples


def test_bpe_special_token_exact_ids() -> None:
    # Given: a trained BPE tokenizer with the fixed special-token contract.
    _require_tokenizers()
    tokenizer = _story_tokenizer()

    # When: every special token is resolved through the tokenizer.
    resolved = [tokenizer.special_token_id(token) for token in BPE_SPECIAL_TOKENS]

    # Then: IDs match the stable 0..16 ordering.
    assert resolved == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.pad_token_id == 1


def test_bpe_no_automatic_bos_eos_and_special_skip_decode() -> None:
    # Given: a trained BPE tokenizer.
    _require_tokenizers()
    tokenizer = _story_tokenizer()

    # When: a skeleton is framed by the caller, not by the tokenizer.
    encoded = tokenizer.encode("<bos>hello<eos>")

    # Then: encode inserts nothing; decode skips specials by default and restores them otherwise.
    assert encoded == [2, *tokenizer.encode("hello"), 3]
    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.decode(encoded, skip_special_tokens=False) == "<bos>hello<eos>"


def test_bpe_training_is_deterministic_and_save_is_byte_identical(tmp_path: Path) -> None:
    # Given: two independent trainings over the same ordered corpus.
    _require_tokenizers()
    first = _story_tokenizer()
    second = _story_tokenizer()

    # When: both are persisted.
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first.save(first_path)
    second.save(second_path)

    # Then: training is deterministic and artifacts are byte-identical.
    assert first.native_json == second.native_json
    assert first_path.read_bytes() == second_path.read_bytes()


def test_bpe_v2_schema_round_trip(tmp_path: Path) -> None:
    # Given: a persisted schema-v2 artifact.
    _require_tokenizers()
    tokenizer = _story_tokenizer()
    artifact_path = tmp_path / "tokenizer.json"
    tokenizer.save(artifact_path)

    # When: the artifact is loaded through the dispatching loader.
    loaded = load_tokenizer(artifact_path)

    # Then: the loaded tokenizer preserves the v2 fields and semantics.
    assert loaded.tokenizer_type == "bpe"
    assert loaded.model_family == "story_forge"
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.decode(loaded.encode("hello world")) == "hello world"


def _write_v2_document(tmp_path: Path, *, mutate: dict[str, object] | None = None) -> Path:
    """Write a valid v2 document, optionally merged with field overrides."""
    _require_tokenizers()
    tokenizer = _story_tokenizer()
    artifact_path = tmp_path / "tokenizer.json"
    tokenizer.save(artifact_path)
    if mutate is None:
        return artifact_path
    document = cast(
        "dict[str, object]",
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    for key, value in mutate.items():
        document[key] = value
    _ = artifact_path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ({"extra": 1}, "unknown fields"),
        ({"schema_version": 3}, "schema_version must be 2"),
        ({"tokenizer_type": "char"}, "tokenizer_type must be 'bpe'"),
        ({"model_family": "other"}, "model_family must be 'story_forge'"),
        ({"vocab_size": True}, "vocab_size must be an integer"),
        ({"vocab_size": 17}, "vocab_size is out of range"),
    ],
)
def test_bpe_v2_schema_rejects_top_level_problems(
    tmp_path: Path,
    mutate: dict[str, object],
    match: str,
) -> None:
    # Given: a v2 document with one invalid top-level field.
    artifact_path = _write_v2_document(tmp_path, mutate=mutate)

    # When: the document is loaded.
    # Then: a typed error rejects the malformed document.
    with pytest.raises(data.TokenizerFormatError, match=match):
        _ = load_tokenizer(artifact_path)


def test_bpe_v2_schema_rejects_unknown_training_field(tmp_path: Path) -> None:
    # Given: a v2 document with an unknown nested training field.
    artifact_path = _write_v2_document(tmp_path)
    document = cast(
        "dict[str, object]",
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    training = cast("dict[str, object]", document["training"])
    training["extra"] = 1
    _ = artifact_path.write_text(json.dumps(document), encoding="utf-8")

    # When: the document is loaded.
    # Then: the unknown nested field is rejected.
    with pytest.raises(data.TokenizerFormatError, match="unknown fields"):
        _ = load_tokenizer(artifact_path)


def test_bpe_v2_schema_rejects_special_token_reorder(tmp_path: Path) -> None:
    # Given: a v2 document whose special-token list is reordered.
    artifact_path = _write_v2_document(tmp_path)
    document = cast(
        "dict[str, object]",
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    specials = cast("list[object]", document["special_tokens"])
    specials[0], specials[1] = specials[1], specials[0]
    _ = artifact_path.write_text(json.dumps(document), encoding="utf-8")

    # When: the document is loaded.
    # Then: the wrong order is rejected.
    with pytest.raises(data.TokenizerFormatError, match="wrong token"):
        _ = load_tokenizer(artifact_path)


def test_bpe_v2_schema_rejects_native_vocab_mismatch(tmp_path: Path) -> None:
    # Given: a v2 document whose top-level vocab_size disagrees with native vocab.
    artifact_path = _write_v2_document(tmp_path, mutate={"vocab_size": 500})

    # When: the document is loaded.
    # Then: the mismatch is rejected.
    with pytest.raises(data.TokenizerFormatError, match="does not match native vocabulary size"):
        _ = load_tokenizer(artifact_path)


def test_bpe_v2_schema_rejects_malformed_native_json(tmp_path: Path) -> None:
    # Given: a v2 document whose native_tokenizer is not a JSON object.
    artifact_path = _write_v2_document(tmp_path, mutate={"native_tokenizer": "not-an-object"})

    # When: the document is loaded.
    # Then: the malformed native payload is rejected.
    with pytest.raises(data.TokenizerFormatError, match="native_tokenizer must be an object"):
        _ = load_tokenizer(artifact_path)


def test_bpe_v2_schema_rejects_non_bpe_native_model(tmp_path: Path) -> None:
    # Given: a v2 document whose native model is not BPE.
    artifact_path = _write_v2_document(tmp_path)
    document = cast(
        "dict[str, object]",
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    native = cast("dict[str, object]", document["native_tokenizer"])
    model = cast("dict[str, object]", native["model"])
    model["type"] = "WordPiece"
    _ = artifact_path.write_text(json.dumps(document), encoding="utf-8")

    # When: the document is loaded.
    # Then: the unsupported native model is rejected.
    with pytest.raises(data.TokenizerFormatError, match="must use a BPE model"):
        _ = load_tokenizer(artifact_path)


def test_bpe_training_rejects_invalid_vocab_targets() -> None:
    # Given: invalid vocab-size targets at and beyond the uint16 boundary.
    _require_tokenizers()

    # When/Then: out-of-range and bool-as-int targets are rejected.
    with pytest.raises(ValueError, match="vocab_size must be an integer"):
        _ = BPETokenizer.train_from_iterator(  # type: ignore[arg-type]
            iter(_STORY_CORPUS),
            vocab_size=True,
        )
    with pytest.raises(ValueError, match="must be in"):
        _ = BPETokenizer.train_from_iterator(
            iter(_STORY_CORPUS),
            vocab_size=BPE_COUNT_SPECIAL_TOKENS,
        )
    with pytest.raises(ValueError, match="must be in"):
        _ = BPETokenizer.train_from_iterator(
            iter(_STORY_CORPUS),
            vocab_size=BPE_MAX_VOCAB_SIZE + 1,
        )


def test_bpe_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: the optional backend unavailable at import time.
    def forbid_import(_name: str) -> object:
        reason = "No module named 'tokenizers'"
        raise ModuleNotFoundError(reason, name="tokenizers")

    monkeypatch.setattr("minigpt.tokenizer.import_module", forbid_import)

    # When: a BPE operation is requested.
    # Then: an actionable error names the optional story extra.
    with pytest.raises(StoryForgeTokenizerError, match=r"\[story\]"):
        _ = _story_tokenizer()


def test_bpe_import_is_lazy_for_char_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a char-only load that must not import the optional backend.
    def forbid_import(_name: str) -> object:
        reason = "char path imported the optional tokenizers backend"
        raise AssertionError(reason)

    monkeypatch.setattr("minigpt.tokenizer.import_module", forbid_import)
    tokenizer = data.CharTokenizer.from_text("abc")

    # When: a v1 document is written and dispatched.
    with tempfile.TemporaryDirectory() as raw_dir:
        char_path = Path(raw_dir) / "tokenizer.json"
        tokenizer.save(char_path)
        loaded = load_tokenizer(char_path)

    # Then: dispatch succeeds without importing the optional backend.
    assert loaded.decode(loaded.encode("abc")) == "abc"


def test_bpe_atomic_write_cleans_up_on_injected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a destination directory and a tokenizer ready to save.
    _require_tokenizers()
    tokenizer = _story_tokenizer()
    destination = tmp_path / "tokenizer.json"

    # When: atomic replacement fails mid-write.
    def failing_replace(source: Path, target: Path) -> None:
        del source, target
        message = "injected replace failure"
        raise OSError(message)

    monkeypatch.setattr("minigpt.tokenizer._atomic_replace", failing_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        tokenizer.save(destination)

    # Then: no temporary file or partial destination is left behind.
    remaining = list(tmp_path.iterdir())
    assert destination not in remaining
    assert not any(path.name.endswith(".tmp") for path in remaining)


def test_load_tokenizer_rejects_ambiguous_schema(tmp_path: Path) -> None:
    # Given: a document carrying both v1 and v2 version markers.
    artifact_path = tmp_path / "tokenizer.json"
    _ = artifact_path.write_text(
        json.dumps({"version": 1, "schema_version": 2}),
        encoding="utf-8",
    )

    # When: the document is loaded.
    # Then: the ambiguous document is rejected.
    with pytest.raises(data.TokenizerFormatError, match="ambiguous"):
        _ = load_tokenizer(artifact_path)


def test_load_tokenizer_dispatches_char_and_bpe(tmp_path: Path) -> None:
    # Given: one v1 char artifact and one v2 BPE artifact.
    _require_tokenizers()
    char = data.CharTokenizer.from_text("xyz")
    char_path = tmp_path / "char.json"
    char.save(char_path)
    bpe = _story_tokenizer()
    bpe_path = tmp_path / "bpe.json"
    bpe.save(bpe_path)

    # When: each artifact is dispatched by the loader.
    char_loaded = load_tokenizer(char_path)
    bpe_loaded = load_tokenizer(bpe_path)

    # Then: the loader returns the correct backend with round-trip intact.
    assert char_loaded.tokenizer_type == "char"
    assert char_loaded.decode(char_loaded.encode("xyz")) == "xyz"
    assert bpe_loaded.tokenizer_type == "bpe"
    assert bpe_loaded.decode(bpe_loaded.encode("hello world")) == "hello world"


# ---------------------------------------------------------------------------
# Story Forge deterministic data preparation (SimpleStories Parquet)
# ---------------------------------------------------------------------------


def _require_story_data() -> None:
    """Fail with a clear message when the optional story data backends are unavailable."""
    try:
        _ = import_module("pyarrow")
    except ModuleNotFoundError as error:
        pytest.skip(f"pyarrow optional dependency unavailable: {error}")
    try:
        _ = import_module("huggingface_hub")
    except ModuleNotFoundError as error:
        pytest.skip(f"huggingface_hub optional dependency unavailable: {error}")


# One unambiguous alias per canonical label used to build deficit-free fixtures.
_WORLD_PHRASE: Final = {
    "space": "space",
    "forest": "forest",
    "robot": "robot",
    "mystery": "mystery",
}
_TONE_PHRASE: Final = {
    "adventurous": "action packed",
    "mysterious": "mysterious",
    "warm": "heartwarming",
    "funny": "humorous",
}
_THEME_PHRASE: Final = {
    "discovery": "discovery",
    "friendship": "friendship",
    "logic": "logic",
    "courage": "courage",
}


class _PaTableFactory(Protocol):
    def from_pylist(self, rows: list[dict[str, object]]) -> object: ...

    def from_pydict(self, mapping: dict[str, list[object]]) -> object: ...


class _PaParquetWriter(Protocol):
    def write_table(self, table: object, where: str) -> None: ...


class _PaModule(Protocol):
    Table: _PaTableFactory
    parquet: _PaParquetWriter


def _story_rows() -> list[dict[str, object]]:
    """Return a deterministic, deficit-free fixture covering every cell.

    Each of the 16 ``(world, tone)`` buckets contributes one story per theme,
    yielding 64 distinct generation IDs so both splits are non-empty.
    """
    return [
        {
            "generation_id": f"gid-{world}-{tone}-{theme}",
            "story": (
                f"A {_WORLD_PHRASE[world]} story with a "
                f"{_TONE_PHRASE[tone]} mood about {_THEME_PHRASE[theme]}."
            ),
            "topic": _WORLD_PHRASE[world],
            "theme": _THEME_PHRASE[theme],
            "style": _TONE_PHRASE[tone],
        }
        for world in story_data.WORLDS
        for tone in story_data.TONES
        for theme in story_data.THEMES
    ]


def _pyarrow_module() -> _PaModule:
    """Return the optional pyarrow module typed through a minimal Protocol."""
    _ = import_module("pyarrow.parquet")
    return cast("_PaModule", cast("object", import_module("pyarrow")))


def _write_story_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a tiny pyarrow Parquet fixture with scalar string columns."""
    _require_story_data()
    pa = _pyarrow_module()
    pa.parquet.write_table(pa.Table.from_pylist(rows), str(path))


def test_story_forge_constants_and_corrected_sha256() -> None:
    # Given: the reviewed upstream identity.
    # When/Then: the exact pinned values are preserved.
    assert story_data.SIMPLE_STORIES_REPO == "SimpleStories/SimpleStories"
    assert story_data.SIMPLE_STORIES_REVISION == "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
    assert story_data.SIMPLE_STORIES_FILENAME == "processed.parquet"
    assert story_data.SIMPLE_STORIES_SIZE == 431_432_698
    assert (
        story_data.SIMPLE_STORIES_SHA256
        == "83ad95336a6b7a028be86a12c63facb3956097fe6177b577337510eeb5735938"
    )
    assert story_data.SIMPLE_STORIES_LICENSE == "MIT"
    assert "arXiv:2504.09184" in story_data.SIMPLE_STORIES_CITATION


def test_resolve_story_labels_maps_every_dimension() -> None:
    # Given: unambiguous, fully mappable scalar metadata for each canonical world.
    cases = [
        ("space", "space", "discovery", "action packed"),
        ("forest", "forest", "friendship", "heartwarming"),
        ("robot", "robotics", "logic", "humorous"),
        ("mystery", "detective", "courage", "mysterious"),
    ]
    # When: each row is resolved into canonical labels.
    for expected_world, topic, theme, style in cases:
        labels = story_data.resolve_story_labels("gid", topic, theme, style)

        # Then: the world maps deterministically and the shared row is eligible.
        assert labels is not None
        assert labels.world == expected_world
        assert labels.tone in story_data.TONES
        assert labels.theme in story_data.THEMES


def test_resolve_story_labels_deterministic_tie_break_is_order_independent() -> None:
    # Given: a row whose topic maps to two worlds.
    topic = "space robots"
    first = story_data.resolve_story_labels("gid", topic, "Logic", "funny")
    second = story_data.resolve_story_labels("gid", topic, "Logic", "funny")

    # When/Then: the tie-break is deterministic and independent of Python set order.
    assert first is not None
    assert second is not None
    assert first.world == second.world


def test_partition_for_is_seed_salted_and_deterministic() -> None:
    # Given: the same generation_id under two seeds.
    first = story_data.partition_for(seed=1, generation_id="gid-x")
    again = story_data.partition_for(seed=1, generation_id="gid-x")

    # When/Then: the partition is deterministic for a seed.
    assert first == again
    assert first in {"train", "val"}


def test_compute_quotas_equal_spread_within_availability() -> None:
    # Given: plenty of capacity in every cell.
    availability = dict.fromkeys([(w, t) for w in story_data.WORLDS for t in story_data.TONES], 100)

    # When: a quota is computed.
    quota = story_data.compute_quotas(160, availability)

    # Then: each of the 16 cells receives exactly 10.
    assert sum(quota.values()) == 160
    assert all(value == 10 for value in quota.values())


def test_compute_quotas_raises_on_insufficient_capacity() -> None:
    # Given: total capacity below the requested count.
    availability = dict.fromkeys([(w, t) for w in story_data.WORLDS for t in story_data.TONES], 1)

    # When/Then: the actionable insufficiency error is raised.
    with pytest.raises(story_data.InsufficientStoriesError, match="not enough eligible"):
        _ = story_data.compute_quotas(100, availability)


def test_story_forge_prepare_writes_deterministic_artifacts(tmp_path: Path) -> None:
    # Given: a local Parquet fixture with enough rows for a tiny selection.
    _require_story_data()
    rows = _story_rows()
    source = tmp_path / "fixture.parquet"
    _write_story_parquet(source, rows)
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"

    # When: preparation runs twice with identical settings into two directories.
    first = story_data.prepare_simple_stories(
        output_dir=output_a,
        source_parquet=source,
        train_stories=2,
        val_stories=1,
        vocab_size=600,
        seed=20260901,
    )
    second = story_data.prepare_simple_stories(
        output_dir=output_b,
        source_parquet=source,
        train_stories=2,
        val_stories=1,
        vocab_size=600,
        seed=20260901,
    )

    # Then: all four artifacts are byte-identical across the two directories.
    for name in ("train.npy", "val.npy", "tokenizer.json", "metadata.json"):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()
    assert first.train_path.is_file()
    assert second.val_path.is_file()


def test_story_forge_outputs_are_uint16_and_token_batcher_compatible(tmp_path: Path) -> None:
    # Given: a prepared Story Forge output.
    _require_story_data()
    rows = _story_rows()
    source = tmp_path / "fixture.parquet"
    _write_story_parquet(source, rows)
    output = tmp_path / "out"

    # When: preparation runs.
    prepared = story_data.prepare_simple_stories(
        output_dir=output,
        source_parquet=source,
        train_stories=3,
        val_stories=1,
        vocab_size=600,
        seed=7,
    )

    # Then: arrays are flat uint16 and reusable by the existing TokenBatcher.
    train = cast("npt.NDArray[np.uint16]", np.load(prepared.train_path))
    val = cast("npt.NDArray[np.uint16]", np.load(prepared.val_path))
    assert train.dtype == np.uint16
    assert val.dtype == np.uint16
    assert train.ndim == 1
    assert val.ndim == 1
    assert train.size > 0
    assert val.size > 0
    _ = data.TokenBatcher(train, batch_size=2, block_size=16, seed=1)
    _ = data.TokenBatcher(val, batch_size=1, block_size=16, seed=1)


def test_story_forge_metadata_has_no_paths_or_timestamps(tmp_path: Path) -> None:
    # Given: a prepared output.
    _require_story_data()
    rows = _story_rows()
    source = tmp_path / "fixture.parquet"
    _write_story_parquet(source, rows)
    output = tmp_path / "out"

    # When: preparation runs and the metadata is reloaded.
    prepared = story_data.prepare_simple_stories(
        output_dir=output,
        source_parquet=source,
        train_stories=2,
        val_stories=1,
        vocab_size=600,
        seed=3,
    )
    document = cast(
        "dict[str, object]",
        json.loads(prepared.metadata_path.read_text(encoding="utf-8")),
    )

    # Then: no absolute path, timestamp, hostname, or PID leaks into metadata.
    serialized = json.dumps(document)
    assert str(output) not in serialized
    assert str(tmp_path) not in serialized
    assert "timestamp" not in serialized.lower()
    source = cast("dict[str, object]", document["source"])
    passes = cast("dict[str, object]", document["passes"])
    assert source["kind"] == "local_fixture"
    assert document["no_overlap"] is True
    assert passes["counts_match"] is True
    assert document["deterministic"] is True


def test_story_forge_official_source_mismatch_rejected() -> None:
    # Given: an observed size/hash that differs from the pinned identity.
    # When/Then: the verifier rejects it.
    with pytest.raises(story_data.SourceMismatchError, match="source mismatch"):
        story_data.verify_official_source(size=1, sha256_hex="0" * 64)


def test_story_forge_local_fixture_permitted(tmp_path: Path) -> None:
    # Given: a tiny local fixture.
    _require_story_data()
    rows = _story_rows()
    source = tmp_path / "fixture.parquet"
    _write_story_parquet(source, rows)

    # When: the source is resolved as a local fixture.
    resolved = story_data._resolve_source(source)  # pyright: ignore[reportPrivateUsage]

    # Then: it is marked local_fixture and its identity is measured, not enforced.
    assert resolved.kind == "local_fixture"
    assert resolved.size_bytes > 0
    assert len(resolved.sha256) == 64


def test_story_forge_missing_optional_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: pyarrow unavailable at import time.
    def forbid_import(name: str) -> object:
        if name == "pyarrow":
            msg = "No module named 'pyarrow'"
            raise ModuleNotFoundError(msg, name="pyarrow")
        return import_module(name)

    monkeypatch.setattr("minigpt.story_data.import_module", forbid_import)

    # When: preparation attempts to open a source.
    # Then: an actionable story extra error is raised.
    with pytest.raises(story_data.StoryOptionalDependencyError, match=r"\[story\]"):
        _ = story_data._open_parquet(Path("unused.parquet"))  # pyright: ignore[reportPrivateUsage]


def test_story_forge_schema_rejects_non_string_columns(tmp_path: Path) -> None:
    # Given: a selected column of a non-string type.
    _require_story_data()
    pa = _pyarrow_module()

    source = tmp_path / "string.parquet"
    pa.parquet.write_table(
        pa.Table.from_pydict(
            {
                "generation_id": ["a"],
                "story": ["x"],
                "topic": ["y"],
                "theme": ["z"],
                "style": ["w"],
            }
        ),
        str(source),
    )
    reader = story_data._open_parquet(source)  # pyright: ignore[reportPrivateUsage]
    assert reader is not None

    # When: a selected column's type is changed to non-string.
    non_string = tmp_path / "int.parquet"
    pa.parquet.write_table(
        pa.Table.from_pydict(
            {
                "generation_id": [7],
                "story": ["x"],
                "topic": [1],
                "theme": ["z"],
                "style": ["w"],
            }
        ),
        str(non_string),
    )

    # Then: the non-string selected column is rejected.
    with pytest.raises(story_data.StoryDataError, match="must be a string"):
        _ = story_data._open_parquet(non_string)  # pyright: ignore[reportPrivateUsage]


def test_story_forge_framing_uses_exact_bos_control_story_eos_ids() -> None:
    # Given: a tokenizer and one record with unambiguous labels.
    _require_tokenizers()
    tokenizer = _story_tokenizer()
    record = (0, "gid-frame", "A space story.", "space", "adventurous", "discovery")

    # When: the story is framed.
    frame = story_data._story_token_ids(tokenizer, record)  # pyright: ignore[reportPrivateUsage]

    # Then: the frame is `[bos, world_space, tone_adventurous, theme_discovery, story, ... , eos]`.
    assert frame[0] == tokenizer.bos_token_id
    assert frame[1] == tokenizer.special_token_id("<world_space>")
    assert frame[2] == tokenizer.special_token_id("<tone_adventurous>")
    assert frame[3] == tokenizer.special_token_id("<theme_discovery>")
    assert frame[4] == tokenizer.special_token_id("<story>")
    assert frame[-1] == tokenizer.eos_token_id


def test_story_forge_atomic_publish_rolls_back_on_injected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a staging and an existing output directory with a sentinel marker.
    staging = tmp_path / "out.staging"
    staging.mkdir()
    _ = (staging / "train.npy").write_bytes(b"s")
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "marker.txt"
    _ = sentinel.write_text("keep", encoding="utf-8")

    # When: the atomic swap fails midway after moving the output to backup.
    real_swap = story_data._atomic_swap  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def failing_swap(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "injected swap failure"
            raise OSError(msg)
        real_swap(source, target)

    monkeypatch.setattr("minigpt.story_data._atomic_swap", failing_swap)

    with pytest.raises(OSError, match="injected swap failure"):
        story_data._publish(staging, output)  # pyright: ignore[reportPrivateUsage]

    # Then: the prior output is restored and no backup residue remains.
    assert output.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / ".out.backup").exists()


def test_story_forge_split_selection_is_order_independent(tmp_path: Path) -> None:
    # Given: the same rows in forward and reversed order.
    _require_story_data()
    rows = _story_rows()
    forward_path = tmp_path / "forward.parquet"
    reversed_path = tmp_path / "reversed.parquet"
    _write_story_parquet(forward_path, rows)
    _write_story_parquet(reversed_path, list(reversed(rows)))

    # When: an identical preparation runs against both files.
    forward = story_data.prepare_simple_stories(
        output_dir=tmp_path / "fwd",
        source_parquet=forward_path,
        train_stories=4,
        val_stories=2,
        vocab_size=600,
        seed=42,
    )
    reversed_out = story_data.prepare_simple_stories(
        output_dir=tmp_path / "rev",
        source_parquet=reversed_path,
        train_stories=4,
        val_stories=2,
        vocab_size=600,
        seed=42,
    )

    # Then: selection is order-independent; the ID hashes match across both runs.
    forward_meta = cast(
        "dict[str, object]",
        json.loads(forward.metadata_path.read_text(encoding="utf-8")),
    )
    reversed_meta = cast(
        "dict[str, object]",
        json.loads(reversed_out.metadata_path.read_text(encoding="utf-8")),
    )
    forward_train = cast("dict[str, object]", forward_meta["train"])
    reversed_train = cast("dict[str, object]", reversed_meta["train"])
    assert forward_train["generation_ids_sha256"] == reversed_train["generation_ids_sha256"]
