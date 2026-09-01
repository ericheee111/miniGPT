"""Deterministic SimpleStories data preparation for the Story Forge family.

This module turns the pinned, MIT-licensed SimpleStories Parquet dataset into the
train/validation token arrays and BPE tokenizer that later Story Forge training
uses. Every step is reproducible: the source is identity-bound, label mapping and
splits are hash-derived from ``generation_id`` (never from row order), quotas are
computed deterministically, and output artifacts are byte-identical on re-run.

The optional ``pyarrow`` and ``huggingface_hub`` backends are imported lazily so
that character-only imports, CLI help output, and the Tiny Shakespeare path never
require the ``[story]`` extra. No pandas is used and the 2.1M-row source is streamed
in Parquet batches rather than materialized in memory.
"""

from __future__ import annotations

import heapq
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from minigpt.tokenizer import (
    BPE_COUNT_SPECIAL_TOKENS,
    BPE_MAX_VOCAB_SIZE,
    BPETokenizer,
    load_tokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

__all__ = (
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_TOKEN_LENGTH",
    "DEFAULT_MIN_FREQUENCY",
    "MAPPING_VERSION",
    "MAX_STORY_CHARS",
    "MODEL_FAMILY",
    "SCHEMA_VERSION",
    "SIMPLE_STORIES_CITATION",
    "SIMPLE_STORIES_FILENAME",
    "SIMPLE_STORIES_LICENSE",
    "SIMPLE_STORIES_REPO",
    "SIMPLE_STORIES_REVISION",
    "SIMPLE_STORIES_SHA256",
    "SIMPLE_STORIES_SIZE",
    "THEMES",
    "TONES",
    "VALIDATION_PERCENT",
    "WORLDS",
    "DuplicateGenerationIdError",
    "InsufficientStoriesError",
    "PreparedStories",
    "SourceInfo",
    "SourceMismatchError",
    "StoryDataError",
    "StoryLabels",
    "StoryOptionalDependencyError",
    "compute_quotas",
    "normalize_story_text",
    "partition_for",
    "prepare_simple_stories",
    "resolve_story_labels",
    "selection_rank_for",
    "verify_official_source",
)

# -- Pinned upstream identity -------------------------------------------------
# These constants bind the exact reviewed SimpleStories revision. Never float on
# ``main``: evidence identity must be a full commit, filename, size, and SHA-256.
SIMPLE_STORIES_REPO: Final = "SimpleStories/SimpleStories"
SIMPLE_STORIES_REVISION: Final = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
SIMPLE_STORIES_FILENAME: Final = "processed.parquet"
SIMPLE_STORIES_SIZE: Final = 431_432_698
SIMPLE_STORIES_SHA256: Final = "83ad95336a6b7a028be86a12c63facb3956097fe6177b577337510eeb5735938"
SIMPLE_STORIES_LICENSE: Final = "MIT"
SIMPLE_STORIES_CITATION: Final = (
    'Finke et al., "Parameterized Synthetic Text Generation with SimpleStories," '
    "arXiv:2504.09184 (2025)"
)

# -- Preparation defaults ------------------------------------------------------
SCHEMA_VERSION: Final = 1
MAPPING_VERSION: Final = 1
MODEL_FAMILY: Final = "story_forge"
VALIDATION_PERCENT: Final = 5
DEFAULT_MIN_FREQUENCY: Final = 2
DEFAULT_MAX_TOKEN_LENGTH: Final = 24
DEFAULT_BATCH_SIZE: Final = 8192
# Fixed, documented upper bound for a single story after LF normalization and
# outer-whitespace stripping. Longer rows are rejected to bound per-row work.
MAX_STORY_CHARS: Final = 100_000

# Selected columns, in a fixed order. The published ``processed.parquet`` stores
# ``topic``, ``theme``, and ``style`` as scalar strings (verified from the Arrow
# schema embedded in the Parquet footer), not as lists or structs.
_COLUMNS: Final = ("generation_id", "story", "topic", "theme", "style")

_ASK_INSTALL: Final = 'python -m pip install -e ".[story]"'

# -- Label vocabulary ----------------------------------------------------------
# Canonical control labels map 1:1 onto the BPE special tokens ``<world_*>``,
# ``<tone_*>``, and ``<theme_*>`` defined by the tokenizer module.
WORLDS: Final = ("space", "forest", "robot", "mystery")
TONES: Final = ("adventurous", "mysterious", "warm", "funny")
THEMES: Final = ("discovery", "friendship", "logic", "courage")

# Alias phrases are already normalized: lowercase, hyphen/underscore turned into
# spaces, whitespace collapsed. Source metadata is normalized the same way and
# matched as whole normalized phrases (word-boundary aware).
WORLD_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "space": (
        "space",
        "universe",
        "galaxy",
        "galaxies",
        "astronaut",
        "astronauts",
        "planet",
        "planets",
        "alien",
        "aliens",
        "constellation",
        "constellations",
        "cosmos",
        "cosmic",
    ),
    "forest": (
        "forest",
        "forests",
        "woodland",
        "woodlands",
        "tree",
        "trees",
        "plant",
        "plants",
        "plant life",
        "nature",
        "wildlife",
        "ecosystem",
        "ecosystems",
        "conservation",
        "biodiversity",
    ),
    "robot": (
        "robot",
        "robots",
        "robotics",
        "artificial intelligence",
        "technology",
        "machine",
        "machines",
        "automation",
        "computer science",
        "engineering",
    ),
    "mystery": (
        "detective",
        "detectives",
        "mystery",
        "mysteries",
        "crime",
        "crimes",
        "clue",
        "clues",
        "investigation",
        "investigations",
        "secret",
        "secrets",
        "suspense",
    ),
}

TONE_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "adventurous": (
        "action packed",
        "adventurous",
        "fast paced",
        "heroic",
        "exciting",
        "adventure",
        "exploration",
        "quests",
        "courage",
        "bravery",
    ),
    "mysterious": (
        "mysterious",
        "suspenseful",
        "sinister",
        "cryptic",
        "eerie",
        "haunting",
        "dark",
        "mystery",
        "curiosity",
    ),
    "warm": (
        "heartwarming",
        "empathetic",
        "cozy",
        "friendly",
        "wholesome",
        "uplifting",
        "compassionate",
        "kindness",
        "friendship",
        "love",
        "empathy",
        "compassion",
    ),
    "funny": (
        "humorous",
        "funny",
        "playful",
        "absurd",
        "sarcastic",
        "ironic",
        "comedic",
        "witty",
    ),
}

THEME_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "discovery": (
        "discovery",
        "exploration",
        "curiosity",
        "learning",
        "knowledge",
        "education",
        "science",
        "wonder",
    ),
    "friendship": (
        "friendship",
        "kindness",
        "love",
        "empathy",
        "compassion",
        "collaboration",
        "community",
        "teamwork",
        "helping others",
    ),
    "logic": (
        "problem solving",
        "critical thinking",
        "logic",
        "strategy",
        "reasoning",
        "creativity",
        "patience",
        "resourcefulness",
    ),
    "courage": (
        "courage",
        "bravery",
        "perseverance",
        "resilience",
        "determination",
        "overcoming challenges",
        "strength",
    ),
}

_ALIAS_TABLES: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = {
    "world": WORLD_ALIASES,
    "tone": TONE_ALIASES,
    "theme": THEME_ALIASES,
}
# Field(s) feeding each dimension: world <- topic; tone <- style + theme;
# theme <- theme + topic. Order is fixed but tie-breaking is hash-based.
_DIMENSION_SOURCES: Final[Mapping[str, tuple[str, ...]]] = {
    "world": ("topic",),
    "tone": ("style", "theme"),
    "theme": ("theme", "topic"),
}


# -- Errors --------------------------------------------------------------------
class StoryDataError(ValueError):
    """Base class for deterministic SimpleStories preparation failures."""


class StoryOptionalDependencyError(StoryDataError):
    """Raise when preparation needs the optional ``[story]`` dependencies."""

    def __init__(self) -> None:
        """Render the actionable install hint for the missing ``[story]`` extra."""
        message = (
            "SimpleStories preparation requires the optional pyarrow and "
            f"huggingface_hub dependencies; install with: {_ASK_INSTALL}"
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SourceMismatchError(StoryDataError):
    """Raise when the downloaded official source does not match its identity."""

    size: int
    sha256_hex: str

    @override
    def __str__(self) -> str:
        """Render the expected versus received source identity."""
        return (
            "official SimpleStories source mismatch: expected size "
            f"{SIMPLE_STORIES_SIZE} and SHA-256 {SIMPLE_STORIES_SHA256}, "
            f"received {self.size} and {self.sha256_hex}"
        )


@dataclass(frozen=True, slots=True)
class InsufficientStoriesError(StoryDataError):
    """Raise when eligible capacity cannot satisfy the requested quotas."""

    required: int
    available: int

    @override
    def __str__(self) -> str:
        """Render the requested versus available eligible capacity."""
        return f"not enough eligible stories: requested {self.required}, available {self.available}"


@dataclass(frozen=True, slots=True)
class DuplicateGenerationIdError(StoryDataError):
    """Raise when a duplicated ``generation_id`` reaches the selected sets."""

    generation_ids: list[str]

    @override
    def __str__(self) -> str:
        """Render the duplicated identifiers."""
        return (
            "selected records contain duplicate generation_id values: "
            f"{', '.join(sorted(set(self.generation_ids)))}"
        )


# -- Data shapes ---------------------------------------------------------------
Bucket: TypeAlias = tuple[str, str]
Partition: TypeAlias = str  # "train" | "val"
# rank, generation_id, story, world, tone, theme
SelectedRow: TypeAlias = tuple[int, str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class StoryLabels:
    """Canonical world/tone/theme labels resolved for one story row."""

    world: str
    tone: str
    theme: str


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Describe the resolved parsed source and its measured identity."""

    path: Path
    kind: str  # "official_pinned" | "local_fixture"
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PreparedStories:
    """Describe the persisted output produced by Story Forge preparation."""

    output_dir: Path
    train_path: Path
    val_path: Path
    tokenizer_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class _ResolvedRow:
    """One eligible, fully classified story row."""

    generation_id: str
    story: str
    world: str
    tone: str
    theme: str
    partition: Partition
    rank: int


@dataclass(frozen=True, slots=True)
class _Counts:
    """Pass-level counters for eligible rows and skip reasons."""

    scanned: int
    eligible: Counter[Partition]
    skip_reasons: Counter[str]
    bucket_counts: Mapping[Partition, Counter[Bucket]]


# -- Optional native backends ---------------------------------------------------
# ``pyarrow`` and ``huggingface_hub`` ship no type stubs, so imports are deferred
# and the needed surface is described with small local Protocols.


class _RecordBatch(Protocol):
    def to_pydict(self) -> dict[str, list[object]]: ...


class _Field(Protocol):
    name: str
    type: object


class _Schema(Protocol):
    def field(self, name: str) -> _Field: ...


class _ParquetFile(Protocol):
    @property
    def schema_arrow(self) -> _Schema: ...

    def iter_batches(
        self,
        *,
        batch_size: int,
        columns: list[str] | None = None,
    ) -> Iterator[_RecordBatch]: ...


class _ParquetModule(Protocol):
    def ParquetFile(self, source: str) -> _ParquetFile: ...  # noqa: N802 - native class name.


class _TypesModule(Protocol):
    def is_string(self, value: object) -> bool: ...


class _PyarrowModule(Protocol):
    @property
    def parquet(self) -> _ParquetModule: ...

    @property
    def types(self) -> _TypesModule: ...


class _HubModule(Protocol):
    def hf_hub_download(self, repo_id: str, filename: str, *, revision: str) -> str: ...


def _pyarrow() -> _PyarrowModule:
    """Import the optional Parquet backend with an actionable failure hint."""
    try:
        module = import_module("pyarrow")
    except ImportError as error:
        raise StoryOptionalDependencyError from error
    return cast("_PyarrowModule", cast("object", module))


def _hub() -> _HubModule:
    """Import the optional Hugging Face Hub backend with an actionable hint."""
    try:
        module = import_module("huggingface_hub")
    except ImportError as error:
        raise StoryOptionalDependencyError from error
    return cast("_HubModule", cast("object", module))


# -- Pure helpers (labels, split, selection) -----------------------------------
def normalize_story_text(text: str) -> str:
    """Normalize CRLF/CR to LF and strip outer whitespace without inner edits."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_phrase(phrase: str) -> str:
    """Lowercase and collapse a metadata phrase, folding hyphen/underscore to space."""
    text = phrase.lower().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Return whether ``phrase`` appears as a whole phrase in ``normalized_text``."""
    return f" {phrase} " in f" {normalized_text} "


def _digest_int(*parts: object) -> int:
    """Derive a stable integer from SHA-256 over a deterministic joined message."""
    message = "\n".join(str(part) for part in parts).encode("utf-8")
    return int(sha256(message).hexdigest(), 16)


def _map_dimension(
    dimension: str,
    generation_id: str,
    normalized_sources: Mapping[str, str],
) -> str | None:
    """Map a dimension's source field(s) to one canonical label deterministically."""
    table = _ALIAS_TABLES[dimension]
    source_fields = _DIMENSION_SOURCES[dimension]
    labels: set[str] = set()
    for label, phrases in table.items():
        if any(
            _contains_phrase(normalized_sources[field], phrase)
            for field in source_fields
            for phrase in phrases
        ):
            labels.add(label)
    if not labels:
        return None
    if len(labels) == 1:
        return next(iter(labels))
    # Deterministic tie-break, independent of Python set ordering.
    return min(labels, key=lambda label: _digest_int(generation_id, dimension, label))


def resolve_story_labels(
    generation_id: str,
    topic: str,
    theme: str,
    style: str,
) -> StoryLabels | None:
    """Resolve world/tone/theme labels or return ``None`` when ineligible.

    World derives from ``topic``; tone derives from ``style`` plus ``theme``;
    theme derives from ``theme`` plus ``topic``. A row is eligible only when every
    dimension maps. Multi-label ties resolve by the smallest SHA-256 of
    ``(generation_id, dimension, candidate)``, independent of Python set order.
    """
    sources: dict[str, str] = {
        "topic": _normalize_phrase(topic),
        "theme": _normalize_phrase(theme),
        "style": _normalize_phrase(style),
    }
    world = _map_dimension("world", generation_id, sources)
    if world is None:
        return None
    tone = _map_dimension("tone", generation_id, sources)
    if tone is None:
        return None
    mapped_theme = _map_dimension("theme", generation_id, sources)
    if mapped_theme is None:
        return None
    return StoryLabels(world=world, tone=tone, theme=mapped_theme)


def partition_for(seed: int, generation_id: str) -> Partition:
    """Place one row into the fixed validation partition by hashed split."""
    digest = _digest_int(seed, generation_id, "split")
    return "val" if digest % 100 < VALIDATION_PERCENT else "train"


def selection_rank_for(seed: int, generation_id: str) -> int:
    """Return the order-independent selection rank for one row."""
    return _digest_int(seed, generation_id, "select")


# -- Quota computation ----------------------------------------------------------
def compute_quotas(  # noqa: C901, PLR0912 - four documented redistribution steps.
    desired: int,
    availability: Mapping[Bucket, int],
) -> dict[Bucket, int]:
    """Compute deterministic bucket quotas honoring availability.

    Quotas are filled in the documented steps: equal per-cell quotas with a
    lexicographic integer remainder; cap to availability; fill each world toward
    an equal world marginal from same-world spare cells; then fill any remaining
    global deficit from all spare cells in lexicographic water-filling order.
    Raises :class:`InsufficientStoriesError` when total capacity is too small.
    """
    if desired <= 0:
        raise StoryDataError(_DESIRED_POSITIVE_REASON)
    buckets: list[Bucket] = [(world, tone) for world in WORLDS for tone in TONES]
    avail = {bucket: availability.get(bucket, 0) for bucket in buckets}
    if any(value < 0 for value in avail.values()):
        raise StoryDataError(_AVAILABILITY_NEGATIVE_REASON)
    if sum(avail.values()) < desired:
        raise InsufficientStoriesError(desired, sum(avail.values()))
    quota = dict.fromkeys(buckets, 0)

    cell_base, cell_rem = divmod(desired, len(buckets))
    for index, bucket in enumerate(buckets):
        quota[bucket] = min(cell_base + (1 if index < cell_rem else 0), avail[bucket])

    world_base, world_rem = divmod(desired, len(WORLDS))
    for world_index, world in enumerate(WORLDS):
        target = world_base + (1 if world_index < world_rem else 0)
        cells = [(world, tone) for tone in TONES]
        current = sum(quota[cell] for cell in cells)
        for cell in cells:
            if current >= target:
                break
            take = min(avail[cell] - quota[cell], target - current)
            quota[cell] += take
            current += take

    deficit = desired - sum(quota.values())
    while deficit > 0:
        progress = False
        for bucket in buckets:
            if deficit <= 0:
                break
            if avail[bucket] - quota[bucket] > 0:
                quota[bucket] += 1
                deficit -= 1
                progress = True
        if not progress:
            break
    if sum(quota.values()) < desired:
        raise InsufficientStoriesError(desired, sum(avail.values()))
    return quota


# -- Row classification ---------------------------------------------------------
# Skip-reason identifiers are stable strings so both streaming passes compare
# exactly and report matching counts.
_SKIP_INVALID_ID: Final = "invalid_generation_id"
_SKIP_INVALID_STORY: Final = "invalid_story_type"
_SKIP_EMPTY_STORY: Final = "empty_story"
_SKIP_TOO_LONG: Final = "story_too_long"
_SKIP_NO_WORLD: Final = "no_world_label"
_SKIP_NO_TONE: Final = "no_tone_label"
_SKIP_NO_THEME: Final = "no_theme_label"


def _classify_row(  # noqa: PLR0911, PLR0913, PLR0917 - raw Parquet row shape.
    seed: int,
    generation_id: object,
    story: object,
    topic: object,
    theme: object,
    style: object,
) -> tuple[_ResolvedRow | None, str | None]:
    """Validate and classify one Parquet row into an eligible row or a skip reason."""
    if not isinstance(generation_id, str) or not generation_id:
        return None, _SKIP_INVALID_ID
    if not isinstance(story, str):
        return None, _SKIP_INVALID_STORY
    normalized = normalize_story_text(story)
    if not normalized:
        return None, _SKIP_EMPTY_STORY
    if len(normalized) > MAX_STORY_CHARS:
        return None, _SKIP_TOO_LONG
    topic_text = topic if isinstance(topic, str) else ""
    theme_text = theme if isinstance(theme, str) else ""
    style_text = style if isinstance(style, str) else ""
    labels = resolve_story_labels(generation_id, topic_text, theme_text, style_text)
    if labels is None:
        # Distinguish the missing dimension for actionable reporting.
        world = resolve_world(generation_id, topic_text)
        if world is None:
            return None, _SKIP_NO_WORLD
        tone = resolve_tone(generation_id, theme_text, style_text)
        if tone is None:
            return None, _SKIP_NO_TONE
        return None, _SKIP_NO_THEME
    partition = partition_for(seed, generation_id)
    rank = selection_rank_for(seed, generation_id)
    return (
        _ResolvedRow(
            generation_id=generation_id,
            story=normalized,
            world=labels.world,
            tone=labels.tone,
            theme=labels.theme,
            partition=partition,
            rank=rank,
        ),
        None,
    )


def resolve_world(generation_id: str, topic: str) -> str | None:
    """Resolve the world label from a scalar topic phrase."""
    sources = {"topic": _normalize_phrase(topic)}
    return _map_dimension("world", generation_id, sources)


def resolve_tone(generation_id: str, theme: str, style: str) -> str | None:
    """Resolve the tone label from scalar style and theme phrases."""
    sources = {"style": _normalize_phrase(style), "theme": _normalize_phrase(theme)}
    return _map_dimension("tone", generation_id, sources)


ColumnsRow: TypeAlias = tuple[list[object], list[object], list[object], list[object], list[object]]


def _open_parquet(source_path: Path) -> _ParquetFile:
    """Open the source and require string-typed selected columns."""
    pa = _pyarrow()
    reader = pa.parquet.ParquetFile(str(source_path))
    schema = reader.schema_arrow
    for name in _COLUMNS:
        try:
            field = schema.field(name)
        except KeyError as error:
            raise StoryDataError(_missing_column_reason(name)) from error
        if not pa.types.is_string(field.type):
            raise StoryDataError(_non_string_column_reason(name))
    return reader


def _iter_columns(source_path: Path, batch_size: int) -> Iterator[ColumnsRow]:
    """Yield selected Parquet columns batch by batch without loading all rows."""
    reader = _open_parquet(source_path)
    try:
        for batch in reader.iter_batches(batch_size=batch_size, columns=list(_COLUMNS)):
            data = batch.to_pydict()
            yield (
                data["generation_id"],
                data["story"],
                data["topic"],
                data["theme"],
                data["style"],
            )
    finally:
        del reader


def _scan_counts(source_path: Path, seed: int, batch_size: int) -> _Counts:
    """Run one streaming pass, returning eligible and skip-reason counters."""
    scanned = 0
    eligible: Counter[Partition] = Counter()
    skip_reasons: Counter[str] = Counter()
    bucket_counts: dict[Partition, Counter[Bucket]] = {
        "train": Counter(),
        "val": Counter(),
    }
    for generation_ids, stories, topics, themes, styles in _iter_columns(source_path, batch_size):
        for gid, story, topic, theme, style in zip(
            generation_ids, stories, topics, themes, styles, strict=True
        ):
            scanned += 1
            resolved, reason = _classify_row(seed, gid, story, topic, theme, style)
            if resolved is None:
                skip_reasons[cast("str", reason)] += 1
                continue
            bucket_counts[resolved.partition][(resolved.world, resolved.tone)] += 1
            eligible[resolved.partition] += 1
    return _Counts(
        scanned=scanned,
        eligible=eligible,
        skip_reasons=skip_reasons,
        bucket_counts=bucket_counts,
    )


# -- Selection (pass 2) ---------------------------------------------------------
HeapEntry: TypeAlias = tuple[int, SelectedRow]


def _select_records(
    source_path: Path,
    seed: int,
    batch_size: int,
    train_quota: Mapping[Bucket, int],
    val_quota: Mapping[Bucket, int],
) -> tuple[list[SelectedRow], list[SelectedRow], _Counts]:
    """Run the second streaming pass, keeping only quota-bounded smallest ranks.

    Peak retained memory is proportional to the quotas, not to the dataset size.
    """
    scanned = 0
    eligible: Counter[Partition] = Counter()
    skip_reasons: Counter[str] = Counter()
    bucket_counts: dict[Partition, Counter[Bucket]] = {
        "train": Counter(),
        "val": Counter(),
    }
    heaps: dict[tuple[Partition, Bucket], list[HeapEntry]] = {}
    quota_by_partition: dict[Partition, Mapping[Bucket, int]] = {
        "train": train_quota,
        "val": val_quota,
    }

    for generation_ids, stories, topics, themes, styles in _iter_columns(source_path, batch_size):
        for gid, story, topic, theme, style in zip(
            generation_ids, stories, topics, themes, styles, strict=True
        ):
            scanned += 1
            resolved, reason = _classify_row(seed, gid, story, topic, theme, style)
            if resolved is None:
                skip_reasons[cast("str", reason)] += 1
                continue
            bucket = (resolved.world, resolved.tone)
            bucket_counts[resolved.partition][bucket] += 1
            eligible[resolved.partition] += 1
            quota = quota_by_partition[resolved.partition].get(bucket, 0)
            if quota <= 0:
                continue
            key = (resolved.partition, bucket)
            heap = heaps.setdefault(key, [])
            record: SelectedRow = (
                resolved.rank,
                resolved.generation_id,
                resolved.story,
                resolved.world,
                resolved.tone,
                resolved.theme,
            )
            # Max-heap on negated rank keeps the smallest positive ranks.
            heapq.heappush(heap, (-resolved.rank, record))
            if len(heap) > quota:
                _ = heapq.heappop(heap)

    train: list[SelectedRow] = []
    val: list[SelectedRow] = []
    for (partition, _bucket), heap in heaps.items():
        target = train if partition == "train" else val
        target.extend(record for _neg_rank, record in heap)

    counts = _Counts(
        scanned=scanned,
        eligible=eligible,
        skip_reasons=skip_reasons,
        bucket_counts=bucket_counts,
    )
    return train, val, counts


def _finalize_records(records: list[SelectedRow]) -> list[SelectedRow]:
    """Reject duplicate IDs and return records sorted by selection rank."""
    ids = [record[1] for record in records]
    if len(set(ids)) != len(ids):
        raise DuplicateGenerationIdError(ids)
    return sorted(records, key=lambda record: record[0])


# -- Encoding -------------------------------------------------------------------
def _story_token_ids(tokenizer: BPETokenizer, record: SelectedRow) -> list[int]:
    """Frame one story as ``<bos> <world> <tone> <theme> <story> tokens <eos>``."""
    _rank, _gid, story, world, tone, theme = record
    world_id = tokenizer.special_token_id(f"<world_{world}>")
    tone_id = tokenizer.special_token_id(f"<tone_{tone}>")
    theme_id = tokenizer.special_token_id(f"<theme_{theme}>")
    story_id = tokenizer.special_token_id("<story>")
    if world_id is None or tone_id is None or theme_id is None or story_id is None:
        raise StoryDataError(_MISSING_SPECIAL_REASON)
    return [
        tokenizer.bos_token_id,
        world_id,
        tone_id,
        theme_id,
        story_id,
        *tokenizer.encode(story),
        tokenizer.eos_token_id,
    ]


def _encode_records(
    records: Sequence[SelectedRow],
    tokenizer: BPETokenizer,
) -> npt.NDArray[np.uint16]:
    """Encode sequential records into one flat ``uint16`` token array."""
    token_ids: list[int] = []
    for record in records:
        token_ids.extend(_story_token_ids(tokenizer, record))
    return np.asarray(token_ids, dtype=np.uint16)


# -- Metadata -------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, streaming it in fixed-size chunks."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SAVE_TOKEN_ARRAY = cast(
    "Callable[[Path, npt.NDArray[np.uint16]], None]",
    cast("object", np.save),
)


def _artifact_hashes(
    staging: Path,
    train_array: npt.NDArray[np.uint16],
    val_array: npt.NDArray[np.uint16],
) -> dict[str, dict[str, object]]:
    """Measure the persisted artifacts from the in-memory arrays plus file hashes."""
    train = staging / "train.npy"
    val = staging / "val.npy"
    tokenizer = staging / "tokenizer.json"
    return {
        "train.npy": {
            "dtype": str(train_array.dtype),
            "tokens": int(train_array.size),
            "bytes": int(train_array.nbytes),
            "sha256": sha256_file(train),
        },
        "val.npy": {
            "dtype": str(val_array.dtype),
            "tokens": int(val_array.size),
            "bytes": int(val_array.nbytes),
            "sha256": sha256_file(val),
        },
        "tokenizer.json": {
            "bytes": int(tokenizer.stat().st_size),
            "sha256": sha256_file(tokenizer),
        },
    }


def _dimension_counts(
    records: Sequence[SelectedRow],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    buckets: Counter[Bucket] = Counter()
    worlds: Counter[str] = Counter()
    tones: Counter[str] = Counter()
    themes: Counter[str] = Counter()
    for record in records:
        _rank, _gid, _story, world, tone, theme = record
        buckets[(world, tone)] += 1
        worlds[world] += 1
        tones[tone] += 1
        themes[theme] += 1
    return (
        {f"{w}/{t}": count for (w, t), count in sorted(buckets.items())},
        dict(sorted(worlds.items())),
        dict(sorted(tones.items())),
        dict(sorted(themes.items())),
    )


def _ids_sha256(records: Sequence[SelectedRow]) -> str:
    ids = sorted(record[1] for record in records)
    return sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _records_summary(records: Sequence[SelectedRow], requested: int) -> dict[str, object]:
    buckets, worlds, tones, themes = _dimension_counts(records)
    return {
        "requested": requested,
        "selected": len(records),
        "generation_ids_sha256": _ids_sha256(records),
        "buckets": buckets,
        "worlds": worlds,
        "tones": tones,
        "themes": themes,
    }


def _counts_document(counts: _Counts) -> dict[str, object]:
    return {
        "scanned": counts.scanned,
        "eligible": dict(sorted(counts.eligible.items())),
        "skip_reasons": dict(sorted(counts.skip_reasons.items())),
    }


def _counts_match(first: _Counts, second: _Counts) -> bool:
    return (
        first.scanned == second.scanned
        and first.eligible == second.eligible
        and first.skip_reasons == second.skip_reasons
    )


def _build_metadata(  # noqa: PLR0913, PLR0917 - one canonical document shape.
    source: SourceInfo,
    parameters: Mapping[str, object],
    pass_one: _Counts,
    pass_two: _Counts,
    train_records: Sequence[SelectedRow],
    val_records: Sequence[SelectedRow],
    tokenizer: BPETokenizer,
    artifacts: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """Assemble the canonical, deterministic metadata document (no paths/timestamps)."""
    no_overlap = not {record[1] for record in train_records}.intersection(
        record[1] for record in val_records
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "model_family": MODEL_FAMILY,
        "source": {
            "repo": SIMPLE_STORIES_REPO,
            "revision": SIMPLE_STORIES_REVISION,
            "filename": SIMPLE_STORIES_FILENAME,
            "license": SIMPLE_STORIES_LICENSE,
            "citation": SIMPLE_STORIES_CITATION,
            "kind": source.kind,
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
        },
        "parameters": dict(sorted(parameters.items())),
        "passes": {
            "pass_one": _counts_document(pass_one),
            "pass_two": _counts_document(pass_two),
            "counts_match": _counts_match(pass_one, pass_two),
        },
        "train": _records_summary(train_records, cast("int", parameters["train_stories"])),
        "val": _records_summary(val_records, cast("int", parameters["val_stories"])),
        "no_overlap": no_overlap,
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "vocab_size_target": tokenizer.vocab_size_target,
            "min_frequency": tokenizer.min_frequency,
            "max_token_length": tokenizer.max_token_length,
            "special_tokens": (
                [f"<world_{world}>" for world in WORLDS]
                + [f"<tone_{tone}>" for tone in TONES]
                + [f"<theme_{theme}>" for theme in THEMES]
            ),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "artifacts": {
            name: dict(sorted(value.items())) for name, value in sorted(artifacts.items())
        },
        "deterministic": True,
        "performance_claim": "bounded_non_performance",
    }


# -- Atomic output --------------------------------------------------------------
def _atomic_swap(source: Path, target: Path) -> None:
    """Atomically move one directory into place; a monkeypatch seam for tests."""
    _ = source.replace(target)


def _publish(staging: Path, output: Path) -> None:
    """Swap a fully built staging directory into place, rollback-safe."""
    backup = output.with_name(f".{output.name}.backup")
    try:
        if output.exists():
            if backup.exists():
                shutil.rmtree(backup)
            _atomic_swap(output, backup)
        _atomic_swap(staging, output)
    except BaseException:
        if backup.exists() and not output.exists():
            _atomic_swap(backup, output)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _validate_staging(staging: Path, expected: Mapping[str, object]) -> None:
    """Reload tokenizer and arrays to confirm staged output is self-consistent."""
    tokenizer = load_tokenizer(staging / "tokenizer.json")
    if tokenizer.vocab_size != expected["vocab_size"]:
        raise StoryDataError(_STAGED_VOCAB_REASON)
    for name, expected_tokens in (
        ("train.npy", expected["train_tokens"]),
        ("val.npy", expected["val_tokens"]),
    ):
        array = cast("npt.NDArray[np.uint16]", np.load(staging / name))
        if array.dtype != np.uint16 or array.ndim != 1:
            raise StoryDataError(_flat_uint16_reason(name))
        if int(array.size) != expected_tokens:
            raise StoryDataError(_token_count_reason(name))


# -- Main entry -----------------------------------------------------------------
_DESIRED_POSITIVE_REASON: Final = "desired story count must be positive"
_AVAILABILITY_NEGATIVE_REASON: Final = "availability counts must be non-negative"
_MISSING_SPECIAL_REASON: Final = "tokenizer is missing a Story Forge special token"
_STAGED_VOCAB_REASON: Final = "staged tokenizer vocab size differs from expectation"
_TRAIN_POSITIVE_REASON: Final = "train_stories must be a positive integer"
_VAL_POSITIVE_REASON: Final = "val_stories must be a positive integer"
_MIN_FREQUENCY_REASON: Final = "min_frequency must be a positive integer"
_MAX_TOKEN_LENGTH_REASON: Final = "max_token_length must be a positive integer"  # noqa: S105
_BATCH_SIZE_REASON: Final = "batch_size must be a positive integer"
_SEED_REASON: Final = "seed must be a non-negative integer"
_VOCAB_INT_REASON: Final = "vocab_size must be an integer"


def _missing_column_reason(name: str) -> str:
    return f"source parquet is missing required column {name!r}"


def _non_string_column_reason(name: str) -> str:
    return f"source parquet column {name!r} must be a string type"


def _flat_uint16_reason(name: str) -> str:
    return f"staged {name} is not a flat uint16 array"


def _token_count_reason(name: str) -> str:
    return f"staged {name} token count differs from expectation"


def _vocab_range_reason() -> str:
    return f"vocab_size must be in ({BPE_COUNT_SPECIAL_TOKENS}, {BPE_MAX_VOCAB_SIZE}]"


def _invalid_value(reason: str) -> NoReturn:
    raise StoryDataError(reason)


def _validate_options(  # noqa: PLR0913 - one CLI parameter set.
    *,
    train_stories: int,
    val_stories: int,
    vocab_size: int,
    min_frequency: int,
    max_token_length: int,
    batch_size: int,
    seed: int,
) -> None:
    """Validate preparation options before any source access or scan."""
    if train_stories <= 0:
        _invalid_value(_TRAIN_POSITIVE_REASON)
    if val_stories <= 0:
        _invalid_value(_VAL_POSITIVE_REASON)
    if min_frequency <= 0:
        _invalid_value(_MIN_FREQUENCY_REASON)
    if max_token_length <= 0:
        _invalid_value(_MAX_TOKEN_LENGTH_REASON)
    if batch_size <= 0:
        _invalid_value(_BATCH_SIZE_REASON)
    if seed < 0:
        _invalid_value(_SEED_REASON)
    if vocab_size <= BPE_COUNT_SPECIAL_TOKENS or vocab_size > BPE_MAX_VOCAB_SIZE:
        _invalid_value(_vocab_range_reason())


def prepare_simple_stories(  # noqa: PLR0913 - public preparation parameter set.
    *,
    output_dir: Path,
    train_stories: int,
    val_stories: int,
    vocab_size: int,
    seed: int,
    source_parquet: Path | None = None,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PreparedStories:
    """Prepare deterministic SimpleStories train/validation artifacts.

    Two streaming Parquet passes count eligible rows and select the smallest
    hashed ranks without loading the dataset into memory. Selected records are
    sorted by rank, a BPE tokenizer is trained only on training stories in that
    order, and the four artifacts are written atomically and validated.
    """
    _validate_options(
        train_stories=train_stories,
        val_stories=val_stories,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        batch_size=batch_size,
        seed=seed,
    )

    source = _resolve_source(source_parquet)
    pass_one = _scan_counts(source.path, seed, batch_size)
    train_quota = compute_quotas(train_stories, pass_one.bucket_counts["train"])
    val_quota = compute_quotas(val_stories, pass_one.bucket_counts["val"])

    train_records, val_records, pass_two = _select_records(
        source.path, seed, batch_size, train_quota, val_quota
    )
    train_records = _finalize_records(train_records)
    val_records = _finalize_records(val_records)

    train_ids = {record[1] for record in train_records}
    val_ids = {record[1] for record in val_records}
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise DuplicateGenerationIdError(sorted(overlap))

    tokenizer = BPETokenizer.train_from_iterator(
        (record[2] for record in train_records),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
    )
    train_array = _encode_records(train_records, tokenizer)
    val_array = _encode_records(val_records, tokenizer)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    succeeded = False
    try:
        _SAVE_TOKEN_ARRAY(staging / "train.npy", train_array)
        _SAVE_TOKEN_ARRAY(staging / "val.npy", val_array)
        tokenizer.save(staging / "tokenizer.json")

        parameters: dict[str, object] = {
            "seed": seed,
            "train_stories": train_stories,
            "val_stories": val_stories,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "max_token_length": max_token_length,
            "batch_size": batch_size,
            "max_story_chars": MAX_STORY_CHARS,
            "validation_percent": VALIDATION_PERCENT,
        }
        artifacts = _artifact_hashes(staging, train_array, val_array)
        metadata = _build_metadata(
            source,
            parameters,
            pass_one,
            pass_two,
            train_records,
            val_records,
            tokenizer,
            artifacts,
        )
        _ = (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        _validate_staging(
            staging,
            {
                "vocab_size": tokenizer.vocab_size,
                "train_tokens": int(train_array.size),
                "val_tokens": int(val_array.size),
            },
        )
        _publish(staging, output_dir)
        succeeded = True
    finally:
        if not succeeded and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return PreparedStories(
        output_dir=output_dir,
        train_path=output_dir / "train.npy",
        val_path=output_dir / "val.npy",
        tokenizer_path=output_dir / "tokenizer.json",
        metadata_path=output_dir / "metadata.json",
    )


def _resolve_source(source_parquet: Path | None) -> SourceInfo:
    """Resolve the source file, enforcing identity for the official download."""
    if source_parquet is not None:
        if not source_parquet.is_file():
            reason = f"local source parquet does not exist: {source_parquet}"
            raise StoryDataError(reason)
        size = int(source_parquet.stat().st_size)
        return SourceInfo(
            path=source_parquet,
            kind="local_fixture",
            size_bytes=size,
            sha256=sha256_file(source_parquet),
        )
    path = Path(
        _hub().hf_hub_download(
            SIMPLE_STORIES_REPO,
            SIMPLE_STORIES_FILENAME,
            revision=SIMPLE_STORIES_REVISION,
        )
    )
    size = int(path.stat().st_size)
    digest = sha256_file(path)
    verify_official_source(size, digest)
    return SourceInfo(path=path, kind="official_pinned", size_bytes=size, sha256=digest)


def verify_official_source(size: int, sha256_hex: str) -> None:
    """Reject a downloaded source whose size or hash differs from the pinned identity."""
    if size != SIMPLE_STORIES_SIZE or sha256_hex != SIMPLE_STORIES_SHA256:
        raise SourceMismatchError(size, sha256_hex)
