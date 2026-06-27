import json
from pathlib import Path

import numpy as np
import pytest
import torch

import minigpt.data as data


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
        tokenizer.encode("az")

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
    source_path.write_text("first version", encoding="utf-8")
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
    destination.write_text("cached version", encoding="utf-8")

    # When: download is requested again.
    result = data.download_text("https://invalid.example/unused.txt", destination)

    # Then: the cached destination is returned unchanged.
    assert result == destination
    assert destination.read_text(encoding="utf-8") == "cached version"


def test_prepare_tiny_shakespeare_writes_split_and_metadata(tmp_path: Path) -> None:
    # Given: an existing raw corpus that must be reused.
    raw_path = tmp_path / "raw" / "input.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("abcdeabcde", encoding="utf-8")

    # When: the corpus is prepared with a 90/10 split.
    prepared = data.prepare_tiny_shakespeare(
        data_dir=tmp_path,
        source_url="https://invalid.example/unused.txt",
    )

    # Then: tokens, tokenizer, and reproducibility metadata are persisted.
    train_tokens = np.load(prepared.train_path)
    val_tokens = np.load(prepared.val_path)
    metadata = json.loads(prepared.metadata_path.read_text(encoding="utf-8"))
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
