from __future__ import annotations

import gc
import json
import weakref
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from minigpt import data

if TYPE_CHECKING:
    from pathlib import Path

MetadataValue: TypeAlias = str | int | float
TokenTestSource: TypeAlias = npt.NDArray[np.uint16] | list[int]


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
