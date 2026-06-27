from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

TINY_SHAKESPEARE_URL: Final = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
TRAIN_SPLIT_RATIO: Final = 0.9
TOKENIZER_FORMAT_VERSION: Final = 1
DOWNLOAD_TIMEOUT_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class UnknownCharacterError(ValueError):
    """Report text that cannot be represented by a character vocabulary."""

    character: str
    position: int

    def __str__(self) -> str:
        return f"unknown character {self.character!r} at position {self.position}"


@dataclass(frozen=True, slots=True)
class InvalidTokenIdError(ValueError):
    """Report a token ID outside the tokenizer vocabulary."""

    token_id: int
    position: int
    vocab_size: int

    def __str__(self) -> str:
        return (
            f"token ID {self.token_id} at position {self.position} is outside "
            f"[0, {self.vocab_size})"
        )


@dataclass(frozen=True, slots=True)
class TokenizerFormatError(ValueError):
    """Report an invalid persisted tokenizer document."""

    path: Path
    reason: str

    def __str__(self) -> str:
        return f"invalid tokenizer file {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class DatasetDownloadError(RuntimeError):
    """Report a failed raw-dataset download with its source and destination."""

    url: str
    destination: Path
    reason: str

    def __str__(self) -> str:
        return f"failed to download {self.url} to {self.destination}: {self.reason}"


@dataclass(frozen=True, slots=True)
class InvalidCorpusError(ValueError):
    """Report a corpus that cannot produce train and validation splits."""

    token_count: int

    def __str__(self) -> str:
        return f"corpus needs at least 2 tokens, received {self.token_count}"


@dataclass(frozen=True, slots=True)
class InvalidBatchConfigurationError(ValueError):
    """Report batch dimensions or token data that cannot form a batch."""

    reason: str

    def __str__(self) -> str:
        return f"invalid token batch configuration: {self.reason}"


@dataclass(frozen=True, slots=True)
class CharTokenizer:
    """Map a fixed character vocabulary to stable integer token IDs."""

    _vocabulary: tuple[str, ...]
    _character_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self._vocabulary:
            raise TokenizerFormatError(Path("<memory>"), "vocabulary must not be empty")
        if len(set(self._vocabulary)) != len(self._vocabulary):
            raise TokenizerFormatError(Path("<memory>"), "vocabulary contains duplicates")
        if any(len(character) != 1 for character in self._vocabulary):
            raise TokenizerFormatError(Path("<memory>"), "each vocabulary item must be one character")
        mapping = MappingProxyType(
            {character: token_id for token_id, character in enumerate(self._vocabulary)}
        )
        object.__setattr__(self, "_character_to_id", mapping)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
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
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """Load and validate a persisted tokenizer vocabulary."""
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
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
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
            content = response.read()
        content.decode("utf-8")
        destination.write_bytes(content)
    except (OSError, UnicodeError) as error:
        raise DatasetDownloadError(url, destination, str(error)) from error
    return destination


def prepare_tiny_shakespeare(
    data_dir: Path = Path("data"),
    source_url: str = TINY_SHAKESPEARE_URL,
) -> PreparedDataset:
    """Download, tokenize, split, and persist the Tiny Shakespeare corpus."""
    raw_path = download_text(source_url, data_dir / "raw" / "input.txt")
    text = raw_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    if tokens.size < 2:
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
    metadata_path.write_text(
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


class TokenBatcher:
    """Sample mutable RNG-driven next-token batches from a one-dimensional corpus."""

    __slots__ = ("_batch_size", "_block_size", "_device", "_rng", "_tokens")

    def __init__(
        self,
        tokens: npt.ArrayLike,
        *,
        batch_size: int,
        block_size: int,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        token_array = np.asarray(tokens, dtype=np.int64)
        if token_array.ndim != 1:
            raise InvalidBatchConfigurationError("tokens must be one-dimensional")
        if batch_size <= 0:
            raise InvalidBatchConfigurationError("batch_size must be positive")
        if block_size <= 0:
            raise InvalidBatchConfigurationError("block_size must be positive")
        if token_array.size <= block_size:
            raise InvalidBatchConfigurationError(
                f"need more than block_size={block_size} tokens, received {token_array.size}"
            )
        self._tokens = token_array
        self._batch_size = batch_size
        self._block_size = block_size
        self._rng = np.random.default_rng(seed)
        self._device = torch.device(device)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return input tokens and their one-position-right training targets."""
        start_limit = self._tokens.size - self._block_size
        starts = self._rng.integers(0, start_limit, size=self._batch_size)
        windows = np.stack(
            [
                self._tokens[int(start) : int(start) + self._block_size + 1]
                for start in starts
            ]
        )
        x = torch.from_numpy(windows[:, :-1].copy()).to(self._device)
        y = torch.from_numpy(windows[:, 1:].copy()).to(self._device)
        return x, y
