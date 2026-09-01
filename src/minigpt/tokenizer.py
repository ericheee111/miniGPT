"""Project-owned tokenizer contracts: character v1 and ByteLevel BPE v2.

The character tokenizer stays in :mod:`minigpt.data` for historical import
compatibility. This module adds the common :class:`TokenizerProtocol`, the
optional Story Forge ByteLevel BPE implementation, and schema-v2 persistence.

The BPE path depends on the optional ``tokenizers`` package. It is imported
lazily through :func:`_backend` so character-only imports, help output, and
data preparation work without the optional dependency installed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from typing_extensions import override

from minigpt.data import CharTokenizer, JsonValue, TokenizerFormatError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = (
    "BPE_COUNT_SPECIAL_TOKENS",
    "BPE_MAX_VOCAB_SIZE",
    "BPE_SPECIAL_TOKENS",
    "BPETokenizer",
    "StoryForgeTokenizerError",
    "TokenizerProtocol",
    "load_tokenizer",
)

BPE_SCHEMA_VERSION: Final = 2
BPE_TOKENIZER_TYPE: Final = "bpe"
BPE_MODEL_FAMILY: Final = "story_forge"
BPE_COUNT_SPECIAL_TOKENS: Final = 17
BPE_MAX_VOCAB_SIZE: Final = 65535
BPE_MIN_FREQUENCY: Final = 2
BPE_MAX_TOKEN_LENGTH: Final = 24
_BPE_ASK_INSTALL: Final = 'python -m pip install -e ".[story]"'
_BPE_MISSING_REASON: Final = "BPE tokenizer requires the optional `tokenizers` package"

BPE_SPECIAL_TOKENS: Final = (
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<story>",
    "<world_space>",
    "<world_forest>",
    "<world_robot>",
    "<world_mystery>",
    "<tone_adventurous>",
    "<tone_mysterious>",
    "<tone_warm>",
    "<tone_funny>",
    "<theme_discovery>",
    "<theme_friendship>",
    "<theme_logic>",
    "<theme_courage>",
)
BPE_SPECIAL_TOKEN_IDS: Final = {
    token: token_id for token_id, token in enumerate(BPE_SPECIAL_TOKENS)
}

_TRAINING_KEYS: Final = frozenset(
    {
        "vocab_size_target",
        "min_frequency",
        "max_token_length",
        "normalizer",
        "pre_tokenizer",
        "decoder",
    }
)
_SPECIAL_KEYS: Final = frozenset({"token", "id"})
_SCHEMA_KEYS: Final = frozenset(
    {
        "schema_version",
        "tokenizer_type",
        "model_family",
        "vocab_size",
        "special_tokens",
        "native_tokenizer",
        "training",
    }
)


class TokenizerProtocol(Protocol):
    """Describe the minimal tokenizer contract shared by char and BPE backends."""

    @property
    def vocab_size(self) -> int:
        """Return the number of distinct token IDs."""
        ...

    @property
    def tokenizer_type(self) -> str:
        """Return the tokenizer family label."""
        ...

    @property
    def model_family(self) -> str:
        """Return the model-family label."""
        ...

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token ID, or ``None``."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID, or ``None``."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID, or ``None``."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs into text, skipping special tokens by default."""
        ...

    def special_token_id(self, token: str) -> int | None:
        """Return the ID of one special token, or ``None``."""
        ...


# -- Optional native ``tokenizers`` backend ---------------------------------
#
# ``tokenizers`` is optional and absent from core/``dev`` installs. A static
# import would break character-only type-checking in CI. These local Protocols
# describe the minimal native surface this module touches; import is deferred.


class _NativeEncoding(Protocol):
    ids: list[int]


class _NativeTokenizer(Protocol):
    pre_tokenizer: object
    decoder: object

    def train_from_iterator(
        self,
        texts: list[str],
        *,
        trainer: object,
        length: int,
    ) -> None: ...

    def to_str(self) -> str: ...

    def encode(self, text: str) -> _NativeEncoding: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

    def get_vocab(self) -> dict[str, int]: ...

    def get_vocab_size(self) -> int: ...

    def token_to_id(self, token: str) -> int | None: ...


class _NativeTokenizerClass(Protocol):
    def __call__(self, model: object) -> _NativeTokenizer: ...

    def from_str(self, content: str) -> _NativeTokenizer: ...


class _NativeByteLevel(Protocol):
    def __call__(self, *, add_prefix_space: bool) -> object: ...

    def alphabet(self) -> list[str]: ...


class _NativeModelsModule(Protocol):
    def BPE(self) -> object: ...  # noqa: N802 - native class name.


class _NativeTrainersModule(Protocol):
    def BpeTrainer(  # noqa: N802 - native class name.
        self,
        *,
        vocab_size: int,
        special_tokens: list[str],
        min_frequency: int,
        max_token_length: int,
        initial_alphabet: list[str],
    ) -> object: ...


class _NativePreTokenizersModule(Protocol):
    @property
    def ByteLevel(self) -> _NativeByteLevel: ...  # noqa: N802 - native class name.


class _NativeDecodersModule(Protocol):
    def ByteLevel(self, *, add_prefix_space: bool) -> object: ...  # noqa: N802


class _TokenizersModule(Protocol):
    @property
    def Tokenizer(self) -> _NativeTokenizerClass: ...  # noqa: N802 - native class name.

    @property
    def models(self) -> _NativeModelsModule: ...

    @property
    def trainers(self) -> _NativeTrainersModule: ...

    @property
    def pre_tokenizers(self) -> _NativePreTokenizersModule: ...

    @property
    def decoders(self) -> _NativeDecodersModule: ...


def _backend() -> _TokenizersModule:
    """Import the optional native backend, failing with an actionable hint."""
    try:
        module = import_module("tokenizers")
    except ImportError as error:
        raise StoryForgeTokenizerError(_BPE_MISSING_REASON) from error
    return cast("_TokenizersModule", cast("object", module))


@dataclass(frozen=True, slots=True)
class StoryForgeTokenizerError(RuntimeError):
    """Describe why a Story Forge BPE operation cannot proceed."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the actionable installation hint alongside the reason."""
        return f"{self.reason}; install with: {_BPE_ASK_INSTALL}"


def _invalid_value(reason: str) -> None:
    raise ValueError(reason)


def _is_int(value: object) -> bool:
    """Reject booleans and non-integers alike."""
    return isinstance(value, int) and not isinstance(value, bool)


def _as_object(value: JsonValue, path: Path, reason: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TokenizerFormatError(path, reason)
    return value


def _reject_unknown_keys(
    document: dict[str, JsonValue],
    allowed: frozenset[str],
    path: Path,
    scope: str,
) -> None:
    unknown = set(document) - allowed
    if unknown:
        label = ", ".join(sorted(unknown))
        raise TokenizerFormatError(path, f"{scope} has unknown fields: {label}")


def _validate_native_vocabulary(vocab: dict[str, JsonValue], path: Path) -> None:
    """Require stable special IDs and a collision-free integer ID range."""
    if len(vocab) > BPE_MAX_VOCAB_SIZE:
        raise TokenizerFormatError(path, f"vocabulary exceeds {BPE_MAX_VOCAB_SIZE} tokens")

    for expected_index, token in enumerate(BPE_SPECIAL_TOKENS):
        raw_id = vocab.get(token)
        if not _is_int(raw_id):
            raise TokenizerFormatError(path, f"special token {token!r} is missing")
        if cast("int", raw_id) != expected_index:
            raise TokenizerFormatError(
                path,
                f"special token {token!r} must have ID {expected_index}",
            )

    observed_ids: list[int] = []
    for token, raw_id in vocab.items():
        if not token:
            raise TokenizerFormatError(path, "vocabulary keys must be non-empty strings")
        if not _is_int(raw_id):
            raise TokenizerFormatError(path, f"vocabulary ID for {token!r} must be an integer")
        token_id = cast("int", raw_id)
        if token_id < 0 or token_id >= BPE_MAX_VOCAB_SIZE:
            raise TokenizerFormatError(path, f"vocabulary ID for {token!r} is out of range")
        observed_ids.append(token_id)

    if len(observed_ids) != len(set(observed_ids)):
        raise TokenizerFormatError(path, "vocabulary contains duplicate token IDs")


def _validate_native_structure(native: dict[str, JsonValue], path: Path) -> dict[str, JsonValue]:
    """Require a ByteLevel BPE native object without byte fallback or normalization."""
    model = _as_object(native.get("model"), path, "native_tokenizer.model must be an object")
    if model.get("type") != "BPE":
        raise TokenizerFormatError(path, "native_tokenizer must use a BPE model")
    if model.get("byte_fallback") is not False:
        raise TokenizerFormatError(path, "native_tokenizer must not enable byte fallback")

    pre_tokenizer = _as_object(
        native.get("pre_tokenizer"),
        path,
        "native_tokenizer.pre_tokenizer must be an object",
    )
    if pre_tokenizer.get("type") != "ByteLevel":
        raise TokenizerFormatError(path, "native_tokenizer.pre_tokenizer must be ByteLevel")
    decoder = _as_object(
        native.get("decoder"),
        path,
        "native_tokenizer.decoder must be an object",
    )
    if decoder.get("type") != "ByteLevel":
        raise TokenizerFormatError(path, "native_tokenizer.decoder must be ByteLevel")
    if native.get("normalizer") is not None:
        raise TokenizerFormatError(path, "native_tokenizer.normalizer must be null")
    return model


def _validate_native_document(document: JsonValue, path: Path) -> dict[str, JsonValue]:
    """Validate the parsed native tokenizer JSON and return it unchanged."""
    native = _as_object(document, path, "native_tokenizer must be an object")
    model = _validate_native_structure(native, path)
    vocab = _as_object(model.get("vocab"), path, "native_tokenizer vocab must be an object")
    _validate_native_vocabulary(vocab, path)
    return native


def _parse_training(document: JsonValue, path: Path) -> tuple[int, int, int]:
    training = _as_object(document, path, "training must be an object")
    _reject_unknown_keys(training, _TRAINING_KEYS, path, "training")
    raw_target = training.get("vocab_size_target")
    raw_min = training.get("min_frequency")
    raw_max = training.get("max_token_length")
    if not _is_int(raw_target):
        raise TokenizerFormatError(path, "training.vocab_size_target must be an integer")
    if not _is_int(raw_min):
        raise TokenizerFormatError(path, "training.min_frequency must be an integer")
    if not _is_int(raw_max):
        raise TokenizerFormatError(path, "training.max_token_length must be an integer")
    target = cast("int", raw_target)
    minimum = cast("int", raw_min)
    maximum = cast("int", raw_max)
    if target <= BPE_COUNT_SPECIAL_TOKENS or target > BPE_MAX_VOCAB_SIZE:
        raise TokenizerFormatError(path, "training.vocab_size_target is out of range")
    if minimum < 1:
        raise TokenizerFormatError(path, "training.min_frequency must be positive")
    if maximum < 1:
        raise TokenizerFormatError(path, "training.max_token_length must be positive")
    _require_training_strings(training, path)
    return target, minimum, maximum


def _require_training_strings(training: dict[str, JsonValue], path: Path) -> None:
    if training.get("normalizer") != "none":
        raise TokenizerFormatError(path, "training.normalizer must be 'none'")
    if training.get("pre_tokenizer") != "bytelevel":
        raise TokenizerFormatError(path, "training.pre_tokenizer must be 'bytelevel'")
    if training.get("decoder") != "bytelevel":
        raise TokenizerFormatError(path, "training.decoder must be 'bytelevel'")


def _parse_special_tokens(document: dict[str, JsonValue], path: Path) -> None:
    raw_specials = document.get("special_tokens")
    if not isinstance(raw_specials, list):
        raise TokenizerFormatError(path, "special_tokens must be a list")
    if len(raw_specials) != BPE_COUNT_SPECIAL_TOKENS:
        raise TokenizerFormatError(path, "special_tokens must list exactly 17 entries")
    for expected_index, (raw_entry, token) in enumerate(
        zip(raw_specials, BPE_SPECIAL_TOKENS, strict=True)
    ):
        entry = _as_object(raw_entry, path, "special_tokens entries must be objects")
        _reject_unknown_keys(entry, _SPECIAL_KEYS, path, "special_tokens entry")
        if entry.get("token") != token:
            raise TokenizerFormatError(path, f"special token {expected_index} has wrong token")
        raw_id = entry.get("id")
        if not _is_int(raw_id) or cast("int", raw_id) != expected_index:
            raise TokenizerFormatError(path, f"special token {expected_index} has wrong ID")


def _parse_v2_document(document: JsonValue, path: Path) -> BPETokenizer:
    top = _as_object(document, path, "top-level tokenizer document must be an object")
    _reject_unknown_keys(top, _SCHEMA_KEYS, path, "tokenizer document")

    raw_version = top.get("schema_version")
    if not _is_int(raw_version) or cast("int", raw_version) != BPE_SCHEMA_VERSION:
        raise TokenizerFormatError(path, "schema_version must be 2")
    if top.get("tokenizer_type") != BPE_TOKENIZER_TYPE:
        raise TokenizerFormatError(path, "tokenizer_type must be 'bpe'")
    if top.get("model_family") != BPE_MODEL_FAMILY:
        raise TokenizerFormatError(path, "model_family must be 'story_forge'")

    raw_vocab_size = top.get("vocab_size")
    if not _is_int(raw_vocab_size):
        raise TokenizerFormatError(path, "vocab_size must be an integer")
    vocab_size = cast("int", raw_vocab_size)
    if vocab_size <= BPE_COUNT_SPECIAL_TOKENS or vocab_size > BPE_MAX_VOCAB_SIZE:
        raise TokenizerFormatError(path, "vocab_size is out of range")

    native = _validate_native_document(top.get("native_tokenizer"), path)
    native_vocab = _as_object(
        _as_object(native.get("model"), path, "native_tokenizer.model must be an object").get(
            "vocab"
        ),
        path,
        "native_tokenizer vocab must be an object",
    )
    if len(native_vocab) != vocab_size:
        raise TokenizerFormatError(path, "vocab_size does not match native vocabulary size")

    _parse_special_tokens(top, path)
    target, minimum, maximum = _parse_training(top.get("training"), path)

    native_json = json.dumps(native, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return BPETokenizer(
        native_json=native_json,
        vocab_size_target=target,
        min_frequency=minimum,
        max_token_length=maximum,
    )


def _render_v2(document: JsonValue) -> str:
    """Render canonical, deterministic schema-v2 JSON bytes (UTF-8 + LF)."""
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _atomic_replace(source: Path, target: Path) -> None:
    _ = source.replace(target)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write one UTF-8/LF canonical JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class BPETokenizer:
    """Story Forge ByteLevel BPE tokenizer with a strict, deterministic v2 schema."""

    native_json: str
    vocab_size_target: int
    min_frequency: int
    max_token_length: int

    def __post_init__(self) -> None:
        """Validate the embedded native document and training integers."""
        document = cast("JsonValue", json.loads(self.native_json))
        _ = _validate_native_document(document, Path("<memory>"))
        if self.vocab_size_target <= BPE_COUNT_SPECIAL_TOKENS:
            raise TokenizerFormatError(Path("<memory>"), "vocab_size_target is out of range")
        if self.vocab_size_target > BPE_MAX_VOCAB_SIZE:
            raise TokenizerFormatError(Path("<memory>"), "vocab_size_target is out of range")
        if self.min_frequency < 1:
            raise TokenizerFormatError(Path("<memory>"), "min_frequency must be positive")
        if self.max_token_length < 1:
            raise TokenizerFormatError(Path("<memory>"), "max_token_length must be positive")

    def _native_document(self) -> dict[str, JsonValue]:
        document = cast("JsonValue", json.loads(self.native_json))
        if not isinstance(document, dict):
            raise TokenizerFormatError(Path("<memory>"), "native document must be an object")
        return document

    def _native(self) -> _NativeTokenizer:
        return _backend().Tokenizer.from_str(self.native_json)

    @property
    def vocab_size(self) -> int:
        """Return the native vocabulary size recorded at construction."""
        native = self._native_document()
        model = cast("dict[str, JsonValue]", native.get("model"))
        vocab = cast("dict[str, JsonValue]", model.get("vocab"))
        return len(vocab)

    @property
    def tokenizer_type(self) -> str:
        """Return the persisted tokenizer family label."""
        return BPE_TOKENIZER_TYPE

    @property
    def model_family(self) -> str:
        """Return the persisted model-family label."""
        return BPE_MODEL_FAMILY

    @property
    def bos_token_id(self) -> int:
        """Return the fixed beginning-of-sequence token ID."""
        return BPE_SPECIAL_TOKEN_IDS["<bos>"]

    @property
    def eos_token_id(self) -> int:
        """Return the fixed end-of-sequence token ID."""
        return BPE_SPECIAL_TOKEN_IDS["<eos>"]

    @property
    def pad_token_id(self) -> int:
        """Return the fixed padding token ID."""
        return BPE_SPECIAL_TOKEN_IDS["<pad>"]

    def special_token_id(self, token: str) -> int | None:
        """Return the fixed ID for one registered special token, or ``None``."""
        return BPE_SPECIAL_TOKEN_IDS.get(token)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs without automatic BOS/EOS insertion."""
        return self._native().encode(text).ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs through the ByteLevel decoder."""
        return self._native().decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterator[str],
        *,
        vocab_size: int,
        min_frequency: int = BPE_MIN_FREQUENCY,
        max_token_length: int = BPE_MAX_TOKEN_LENGTH,
    ) -> BPETokenizer:
        """Train a deterministic ByteLevel BPE tokenizer from caller-ordered text.

        The iterator is materialized exactly once in the order supplied by the
        caller; the native backend performs no parallel ordering. Automatic
        BOS/EOS insertion is disabled so sequence framing stays with the caller.
        """
        if not _is_int(vocab_size):
            _invalid_value("vocab_size must be an integer")
        if vocab_size <= BPE_COUNT_SPECIAL_TOKENS or vocab_size > BPE_MAX_VOCAB_SIZE:
            _invalid_value(
                f"vocab_size must be in ({BPE_COUNT_SPECIAL_TOKENS}, {BPE_MAX_VOCAB_SIZE}]"
            )
        if not _is_int(min_frequency) or min_frequency < 1:
            _invalid_value("min_frequency must be a positive integer")
        if not _is_int(max_token_length) or max_token_length < 1:
            _invalid_value("max_token_length must be a positive integer")

        backend = _backend()
        byte_level = backend.pre_tokenizers.ByteLevel
        tokenizer_object = backend.Tokenizer(backend.models.BPE())
        tokenizer_object.pre_tokenizer = byte_level(add_prefix_space=False)
        tokenizer_object.decoder = backend.decoders.ByteLevel(add_prefix_space=False)
        trainer = backend.trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(BPE_SPECIAL_TOKENS),
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            initial_alphabet=byte_level.alphabet(),
        )
        materialized = list(texts)
        tokenizer_object.train_from_iterator(
            materialized, trainer=trainer, length=len(materialized)
        )

        native_json = tokenizer_object.to_str()
        native = cast("JsonValue", json.loads(native_json))
        _ = _validate_native_document(native, Path("<memory>"))
        canonical = json.dumps(native, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return cls(
            native_json=canonical,
            vocab_size_target=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
        )

    def save(self, path: Path) -> None:
        """Persist the deterministic schema-v2 artifact via atomic replacement."""
        _atomic_write_text(path, _render_v2(self._document()))

    def _document(self) -> dict[str, JsonValue]:
        special_tokens: list[JsonValue] = [
            {"token": token, "id": token_id} for token, token_id in BPE_SPECIAL_TOKEN_IDS.items()
        ]
        return {
            "schema_version": BPE_SCHEMA_VERSION,
            "tokenizer_type": BPE_TOKENIZER_TYPE,
            "model_family": BPE_MODEL_FAMILY,
            "vocab_size": self.vocab_size,
            "special_tokens": special_tokens,
            "native_tokenizer": self._native_document(),
            "training": {
                "vocab_size_target": self.vocab_size_target,
                "min_frequency": self.min_frequency,
                "max_token_length": self.max_token_length,
                "normalizer": "none",
                "pre_tokenizer": "bytelevel",
                "decoder": "bytelevel",
            },
        }


def load_tokenizer(path: Path) -> TokenizerProtocol:
    """Dispatch a persisted v1 char or v2 BPE tokenizer document.

    Documents carrying both ``version`` and ``schema_version`` are rejected as
    ambiguous. Char-only loading never imports the optional ``tokenizers``
    package; only BPE encoding/decoding/training operations trigger that import.
    """
    try:
        document = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise TokenizerFormatError(path, str(error)) from error
    if not isinstance(document, dict):
        raise TokenizerFormatError(path, "top-level JSON value must be an object")

    has_v1 = "version" in document
    has_v2 = "schema_version" in document
    if has_v1 and has_v2:
        raise TokenizerFormatError(
            path,
            "ambiguous tokenizer document: both version and schema_version present",
        )
    if has_v2:
        return _parse_v2_document(document, path)
    if has_v1:
        return CharTokenizer.load(path)
    raise TokenizerFormatError(
        path, "unsupported tokenizer document: missing version or schema_version"
    )
