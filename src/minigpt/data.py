"""Prepare Tiny Shakespeare and provide its character tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np

from minigpt.batching import TokenBatcher

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

__all__ = (
    "TINY_SHAKESPEARE_URL",
    "CharTokenizer",
    "DatasetDownloadError",
    "InvalidCorpusError",
    "InvalidTokenIdError",
    "PreparedDataset",
    "TokenBatcher",
    "TokenizerFormatError",
    "UnknownCharacterError",
    "download_text",
    "prepare_tiny_shakespeare",
)

TINY_SHAKESPEARE_URL: Final = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TRAIN_SPLIT_RATIO: Final = 0.9
TOKENIZER_FORMAT_VERSION: Final = 1
DOWNLOAD_TIMEOUT_SECONDS: Final = 30.0
MINIMUM_CORPUS_TOKENS: Final = 2
DEFAULT_DATA_DIR: Final = Path("data")
_MEMORY_SOURCE: Final = Path("<memory>")

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class _BinaryResponse(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def read(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class UnknownCharacterError(ValueError):
    """Report text that cannot be represented by a character vocabulary."""

    character: str
    position: int

    @override
    def __str__(self) -> str:
        """Render the unknown character and its position."""
        return f"unknown character {self.character!r} at position {self.position}"


@dataclass(frozen=True, slots=True)
class InvalidTokenIdError(ValueError):
    """Report a token ID outside the tokenizer vocabulary."""

    token_id: int
    position: int
    vocab_size: int

    @override
    def __str__(self) -> str:
        """Render the invalid token ID and vocabulary range."""
        return (
            f"token ID {self.token_id} at position {self.position} is outside "
            f"[0, {self.vocab_size})"
        )


@dataclass(frozen=True, slots=True)
class TokenizerFormatError(ValueError):
    """Report an invalid persisted tokenizer document."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the tokenizer path and validation failure."""
        return f"invalid tokenizer file {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class DatasetDownloadError(RuntimeError):
    """Report a failed raw-dataset download with its source and destination."""

    url: str
    destination: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed download source and destination."""
        return f"failed to download {self.url} to {self.destination}: {self.reason}"


@dataclass(frozen=True, slots=True)
class InvalidCorpusError(ValueError):
    """Report a corpus that cannot produce train and validation splits."""

    token_count: int

    @override
    def __str__(self) -> str:
        """Render the corpus size required for a non-empty split."""
        return f"corpus needs at least {MINIMUM_CORPUS_TOKENS} tokens, received {self.token_count}"


@dataclass(frozen=True, slots=True)
class CharTokenizer:
    """Map a fixed character vocabulary to stable integer token IDs."""

    _vocabulary: tuple[str, ...]
    _character_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the vocabulary and build its immutable reverse lookup."""
        if not self._vocabulary:
            raise TokenizerFormatError(_MEMORY_SOURCE, "vocabulary must not be empty")
        if len(set(self._vocabulary)) != len(self._vocabulary):
            raise TokenizerFormatError(_MEMORY_SOURCE, "vocabulary contains duplicates")
        if any(len(character) != 1 for character in self._vocabulary):
            raise TokenizerFormatError(
                _MEMORY_SOURCE,
                "each vocabulary item must be one character",
            )
        mapping = MappingProxyType(
            {character: token_id for token_id, character in enumerate(self._vocabulary)}
        )
        object.__setattr__(self, "_character_to_id", mapping)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a deterministic vocabulary sorted by Unicode code point."""
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        """Return the number of representable characters."""
        return len(self._vocabulary)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, rejecting unknown characters."""
        token_ids: list[int] = []
        for position, character in enumerate(text):
            try:
                token_ids.append(self._character_to_id[character])
            except KeyError:
                raise UnknownCharacterError(character, position) from None
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back to text, rejecting IDs outside the vocabulary."""
        characters: list[str] = []
        for position, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_size:
                raise InvalidTokenIdError(token_id, position, self.vocab_size)
            characters.append(self._vocabulary[token_id])
        return "".join(characters)

    def save(self, path: Path) -> None:
        """Persist the exact vocabulary needed to preserve token ID meaning."""
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": TOKENIZER_FORMAT_VERSION,
            "vocabulary": list(self._vocabulary),
        }
        _ = path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        """Load and validate a persisted tokenizer vocabulary."""
        try:
            document = cast(
                "JsonValue",
                json.loads(path.read_text(encoding="utf-8")),
            )
        except json.JSONDecodeError as error:
            raise TokenizerFormatError(path, str(error)) from error
        if not isinstance(document, dict):
            raise TokenizerFormatError(path, "top-level JSON value must be an object")
        if document.get("version") != TOKENIZER_FORMAT_VERSION:
            raise TokenizerFormatError(path, "unsupported format version")
        raw_vocabulary = document.get("vocabulary")
        if not isinstance(raw_vocabulary, list):
            raise TokenizerFormatError(path, "vocabulary must be a list")
        vocabulary: list[str] = []
        for item in raw_vocabulary:
            if not isinstance(item, str):
                raise TokenizerFormatError(path, "vocabulary items must be strings")
            vocabulary.append(item)
        return cls(tuple(vocabulary))


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Describe the persisted artifacts produced by data preparation."""

    raw_path: Path
    train_path: Path
    val_path: Path
    tokenizer_path: Path
    metadata_path: Path


def download_text(url: str, destination: Path) -> Path:
    """Download UTF-8 text once and reuse an existing destination."""
    if destination.is_file():
        return destination
    scheme = urlsplit(url).scheme
    if scheme not in {"https", "file"}:
        raise DatasetDownloadError(url, destination, f"unsupported URL scheme {scheme!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = cast(
            "_BinaryResponse",
            urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS),  # noqa: S310
        )
        with response:
            content = response.read()
        _ = content.decode("utf-8")
        _ = destination.write_bytes(content)
    except (OSError, UnicodeError) as error:
        raise DatasetDownloadError(url, destination, str(error)) from error
    return destination


def prepare_tiny_shakespeare(
    data_dir: Path = DEFAULT_DATA_DIR,
    source_url: str = TINY_SHAKESPEARE_URL,
) -> PreparedDataset:
    """Download, tokenize, split, and persist the Tiny Shakespeare corpus."""
    raw_path = download_text(source_url, data_dir / "raw" / "input.txt")
    text = raw_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    if tokens.size < MINIMUM_CORPUS_TOKENS:
        raise InvalidCorpusError(int(tokens.size))

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    tokenizer_path = processed_dir / "tokenizer.json"
    metadata_path = processed_dir / "metadata.json"

    split_index = int(tokens.size * TRAIN_SPLIT_RATIO)
    np.save(train_path, tokens[:split_index])
    np.save(val_path, tokens[split_index:])
    tokenizer.save(tokenizer_path)
    metadata = {
        "format_version": 1,
        "source_url": source_url,
        "raw_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "dtype": str(tokens.dtype),
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": int(tokens.size),
        "train_tokens": split_index,
        "val_tokens": int(tokens.size) - split_index,
        "split_ratio": TRAIN_SPLIT_RATIO,
    }
    _ = metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PreparedDataset(
        raw_path=raw_path,
        train_path=train_path,
        val_path=val_path,
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
    )
