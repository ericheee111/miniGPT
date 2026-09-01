"""Story Forge controls, framing, and deterministic evaluation cases.

This module provides the user-facing story-control surface: validated world,
tone, and theme selectors, deterministic control-prefix tokenization, a
context-aware prompt framer with left-truncation history management, and a
fixed sixteen-case evaluation battery.

The framing function accepts only a Story Forge BPE tokenizer
(``model_family == "story_forge"``); character tokenizers are explicitly
rejected. Control tokens are resolved through ``special_token_id()`` and are
never encoded as ordinary text. The caller supplies an opening text that may
contain arbitrary Unicode but must not contain ``<...>`` control syntax or
NUL/control characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import compile as compile_pattern
from typing import Final, cast
from unicodedata import category

from typing_extensions import override

from minigpt.story_data import THEMES, TONES, WORLDS
from minigpt.tokenizer import BPE_MODEL_FAMILY, TokenizerProtocol

__all__ = (
    "MAX_OPENING_CHARS",
    "STORY_EVALUATION_CASES",
    "FramedStoryPrompt",
    "StoryControlError",
    "StoryControls",
    "StoryEvaluationCase",
    "StoryFramingError",
    "frame_story_prompt",
    "story_control_prefix_ids",
)

# Upper bound on the caller-supplied opening text, measured in Unicode
# characters. Bounded to reject unreasonably large inputs before encoding.
MAX_OPENING_CHARS: Final = 10_000

_CONTROL_SYNTAX_PATTERN: Final = compile_pattern(r"<[^<>]*>")


# -- Errors --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryControlError(ValueError):
    """Report an invalid world, tone, or theme control selector."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed control constraint."""
        return f"invalid story control: {self.reason}"


@dataclass(frozen=True, slots=True)
class StoryFramingError(ValueError):
    """Report an opening text that cannot be framed for generation."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the framing failure."""
        return f"story framing failed: {self.reason}"


def _invalid_control(dimension: str, value: str, allowed: tuple[str, ...]) -> str:
    return f"{dimension} {value!r} is not one of: {', '.join(allowed)}"


def _control_type_reason(dimension: str) -> str:
    return f"{dimension} must be a string"


def _missing_special(token: str) -> str:
    return f"tokenizer is missing required special token {token!r}"


def _wrong_family(actual: str) -> str:
    return f"tokenizer model_family must be {BPE_MODEL_FAMILY!r}, got {actual!r}"


_OPENING_NOT_STRING: Final = "opening must be a string"
_OPENING_EMPTY: Final = "opening must not be empty"
_OPENING_CONTROL_SYNTAX: Final = "opening must not contain <...> control syntax"
_OPENING_NUL: Final = "opening must not contain NUL characters"
_OPENING_CONTROL_CHARS: Final = "opening must not contain control characters"
_RESERVED_NOT_INTEGER: Final = "reserved_generation_tokens must be an integer"
_RESERVED_NEGATIVE: Final = "reserved_generation_tokens must be non-negative"
_RESERVED_WITHOUT_CONTEXT: Final = "reserved_generation_tokens requires max_context_tokens"
_CONTEXT_NOT_INTEGER: Final = "max_context_tokens must be an integer"
_CONTEXT_NOT_POSITIVE: Final = "max_context_tokens must be positive"


# -- Controls ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryControls:
    """Validated world, tone, and theme selectors for a Story Forge prompt."""

    world: str
    tone: str
    theme: str

    def __post_init__(self) -> None:
        """Reject any selector outside the current canonical vocabularies."""
        for dimension, value in (
            ("world", self.world),
            ("tone", self.tone),
            ("theme", self.theme),
        ):
            if type(value) is not str:
                reason = _control_type_reason(dimension)
                raise StoryControlError(reason)
        if self.world not in WORLDS:
            raise StoryControlError(_invalid_control("world", self.world, WORLDS))
        if self.tone not in TONES:
            raise StoryControlError(_invalid_control("tone", self.tone, TONES))
        if self.theme not in THEMES:
            raise StoryControlError(_invalid_control("theme", self.theme, THEMES))


# -- Framing -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FramedStoryPrompt:
    """Describe a fully framed Story Forge prompt ready for generation."""

    token_ids: tuple[int, ...]
    """Complete token-ID sequence: control prefix followed by opening tokens."""

    control_prefix_length: int
    """Number of leading tokens belonging to the control prefix."""

    truncated: bool
    """Whether the opening was left-truncated to fit the context budget."""

    retained_history_tokens: int
    """Number of opening tokens retained after any truncation."""


def _require_special(tokenizer: TokenizerProtocol, token: str) -> int:
    """Resolve one registered special token or raise a framing error."""
    resolved = tokenizer.special_token_id(token)
    if resolved is None:
        reason = _missing_special(token)
        raise StoryFramingError(reason)
    return resolved


def story_control_prefix_ids(
    tokenizer: TokenizerProtocol,
    controls: StoryControls | None = None,
) -> tuple[int, ...]:
    """Return the five canonical control-prefix token IDs.

    When ``controls`` is omitted, the deterministic default is space,
    adventurous, and discovery. Training, serving, and evaluation callers
    share this source of truth so control-token ordering stays consistent.
    """
    if tokenizer.model_family != BPE_MODEL_FAMILY:
        reason = _wrong_family(tokenizer.model_family)
        raise StoryFramingError(reason)
    resolved = controls or StoryControls(
        world="space",
        tone="adventurous",
        theme="discovery",
    )
    return _build_prefix(tokenizer, resolved)


def _build_prefix(
    tokenizer: TokenizerProtocol,
    controls: StoryControls,
) -> tuple[int, ...]:
    """Build the five-token control prefix for the resolved selectors."""
    bos = _require_special(tokenizer, "<bos>")
    world = _require_special(tokenizer, f"<world_{controls.world}>")
    tone = _require_special(tokenizer, f"<tone_{controls.tone}>")
    theme = _require_special(tokenizer, f"<theme_{controls.theme}>")
    story = _require_special(tokenizer, "<story>")
    return (bos, world, tone, theme, story)


def _opening_too_long(count: int) -> str:
    return f"opening exceeds {MAX_OPENING_CHARS} characters (received {count})"


def _context_too_small(max_tokens: int, prefix_length: int) -> str:
    return (
        f"max_context_tokens ({max_tokens}) is smaller than "
        f"the control prefix ({prefix_length} tokens)"
    )


def _budget_exhausted(
    max_tokens: int,
    reserved: int,
    available: int,
    prefix_length: int,
) -> str:
    return (
        f"max_context_tokens ({max_tokens}) minus "
        f"reserved_generation_tokens ({reserved}) "
        f"leaves {available} tokens, which is smaller than the "
        f"control prefix ({prefix_length} tokens)"
    )


def _available_prompt_budget(
    max_context_tokens: int | None,
    reserved_generation_tokens: int,
    prefix_length: int,
) -> int | None:
    """Validate context arguments and return the prompt-token budget."""
    if type(reserved_generation_tokens) is not int:
        raise StoryFramingError(_RESERVED_NOT_INTEGER)
    if reserved_generation_tokens < 0:
        raise StoryFramingError(_RESERVED_NEGATIVE)
    if max_context_tokens is None:
        if reserved_generation_tokens != 0:
            raise StoryFramingError(_RESERVED_WITHOUT_CONTEXT)
        return None
    if type(max_context_tokens) is not int:
        raise StoryFramingError(_CONTEXT_NOT_INTEGER)
    if max_context_tokens <= 0:
        raise StoryFramingError(_CONTEXT_NOT_POSITIVE)
    if max_context_tokens < prefix_length:
        reason = _context_too_small(max_context_tokens, prefix_length)
        raise StoryFramingError(reason)
    available = max_context_tokens - reserved_generation_tokens
    if available < prefix_length:
        reason = _budget_exhausted(
            max_context_tokens,
            reserved_generation_tokens,
            available,
            prefix_length,
        )
        raise StoryFramingError(reason)
    return available


def _validate_opening(opening: str) -> None:
    """Reject non-text, empty, injected, control, and oversized opening text."""
    if type(opening) is not str:
        raise StoryFramingError(_OPENING_NOT_STRING)
    if not opening or not opening.strip():
        raise StoryFramingError(_OPENING_EMPTY)
    if len(opening) > MAX_OPENING_CHARS:
        reason = _opening_too_long(len(opening))
        raise StoryFramingError(reason)
    if _CONTROL_SYNTAX_PATTERN.search(opening) is not None:
        raise StoryFramingError(_OPENING_CONTROL_SYNTAX)
    for code_point in opening:
        if code_point == "\x00":
            raise StoryFramingError(_OPENING_NUL)
        if category(code_point) == "Cc" and code_point not in ("\n", "\r", "\t"):
            raise StoryFramingError(_OPENING_CONTROL_CHARS)


def frame_story_prompt(
    tokenizer: TokenizerProtocol,
    controls: StoryControls,
    opening: str,
    *,
    max_context_tokens: int | None = None,
    reserved_generation_tokens: int = 0,
) -> FramedStoryPrompt:
    """Frame a Story Forge prompt with context-aware left-truncation.

    The control prefix (``<bos> <world_*> <tone_*> <theme_*> <story>``) is
    always preserved in full. The caller-supplied opening is tokenized and
    appended after the prefix. When ``max_context_tokens`` is set and the
    combined length would exceed the budget minus ``reserved_generation_tokens``,
    the opening is truncated from the left at a token boundary; the control
    prefix is never truncated.

    The tokenizer must be a Story Forge BPE instance. Character tokenizers
    are rejected because they lack the required special tokens.

    Returns a :class:`FramedStoryPrompt` with the complete token-ID tuple,
    the control-prefix length, whether truncation occurred, and how many
    opening tokens were retained.
    """
    if tokenizer.model_family != BPE_MODEL_FAMILY:
        reason = _wrong_family(tokenizer.model_family)
        raise StoryFramingError(reason)

    prefix = story_control_prefix_ids(tokenizer, controls)
    prefix_length = len(prefix)

    _validate_opening(opening)

    opening_ids = tokenizer.encode(opening)
    opening_length = len(opening_ids)

    available = _available_prompt_budget(
        max_context_tokens,
        reserved_generation_tokens,
        prefix_length,
    )
    if available is not None:
        opening_budget = available - prefix_length
        if opening_length > opening_budget:
            retained = opening_ids[-opening_budget:] if opening_budget > 0 else []
            retained_count = len(retained)
            return FramedStoryPrompt(
                token_ids=(*prefix, *retained),
                control_prefix_length=prefix_length,
                truncated=True,
                retained_history_tokens=retained_count,
            )

    return FramedStoryPrompt(
        token_ids=(*prefix, *opening_ids),
        control_prefix_length=prefix_length,
        truncated=False,
        retained_history_tokens=opening_length,
    )


# -- Evaluation cases ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryEvaluationCase:
    """One deterministic Story Forge evaluation case."""

    id: str
    """Stable, human-readable identifier."""

    world: str
    tone: str
    theme: str
    opening: str
    """Original story opening text."""

    seed: int
    """Deterministic sampling seed for reproducibility."""


# Sixteen original cases covering every (world, tone) combination with a
# rotating theme index so each of the four themes appears exactly four times.
# No copyrighted IP, character names, or franchise references.
_STORY_EVALUATION_CASES_DATA: tuple[dict[str, object], ...] = (
    # space x adventurous
    {
        "id": "sf-space-adventurous-discovery",
        "world": "space",
        "tone": "adventurous",
        "theme": "discovery",
        "opening": (
            "The cargo shuttle drifted past the third beacon, "
            "its hull humming with a frequency no one on board recognized."
        ),
        "seed": 2026090100,
    },
    # space x mysterious
    {
        "id": "sf-space-mysterious-friendship",
        "world": "space",
        "tone": "mysterious",
        "theme": "friendship",
        "opening": (
            "A faint signal pulsed from the dark side of the moon, "
            "and only the youngest crew member seemed to hear it."
        ),
        "seed": 2026090101,
    },
    # space x warm
    {
        "id": "sf-space-warm-logic",
        "world": "space",
        "tone": "warm",
        "theme": "logic",
        "opening": (
            "Captain Osei poured two cups of reconstituted tea and slid "
            "one across the navigation table toward her first officer."
        ),
        "seed": 2026090102,
    },
    # space x funny
    {
        "id": "sf-space-funny-courage",
        "world": "space",
        "tone": "funny",
        "theme": "courage",
        "opening": (
            "The ship's vending machine had been dispensing the same "
            "mystery sandwich for three parsecs, and nobody dared complain."
        ),
        "seed": 2026090103,
    },
    # forest x adventurous
    {
        "id": "sf-forest-adventurous-friendship",
        "world": "forest",
        "tone": "adventurous",
        "theme": "friendship",
        "opening": (
            "Two saplings stood where the old trail map said there "
            "should be a river, and the compass needle spun lazily."
        ),
        "seed": 2026090104,
    },
    # forest x mysterious
    {
        "id": "sf-forest-mysterious-logic",
        "world": "forest",
        "tone": "mysterious",
        "theme": "logic",
        "opening": (
            "Every morning the moss grew into the same spiral pattern, "
            "and no one could explain why the birds avoided it."
        ),
        "seed": 2026090105,
    },
    # forest x warm
    {
        "id": "sf-forest-warm-courage",
        "world": "forest",
        "tone": "warm",
        "theme": "courage",
        "opening": (
            "The old caretaker left a lantern at the trailhead every "
            "dusk, even though no hiker had been lost in thirty years."
        ),
        "seed": 2026090106,
    },
    # forest x funny
    {
        "id": "sf-forest-funny-discovery",
        "world": "forest",
        "tone": "funny",
        "theme": "discovery",
        "opening": (
            "A squirrel dropped an acorn on the surveyor's clipboard and "
            "seemed genuinely offended when it rolled away."
        ),
        "seed": 2026090107,
    },
    # robot x adventurous
    {
        "id": "sf-robot-adventurous-logic",
        "world": "robot",
        "tone": "adventurous",
        "theme": "logic",
        "opening": (
            "Unit Seven rolled off the assembly line with a toolkit "
            "nobody had ordered and a set of instructions written in crayon."
        ),
        "seed": 2026090108,
    },
    # robot x mysterious
    {
        "id": "sf-robot-mysterious-courage",
        "world": "robot",
        "tone": "mysterious",
        "theme": "courage",
        "opening": (
            "The maintenance log showed the same error code every night "
            "at 03:17, but the diagnostic panel was always clean by morning."
        ),
        "seed": 2026090109,
    },
    # robot x warm
    {
        "id": "sf-robot-warm-discovery",
        "world": "robot",
        "tone": "warm",
        "theme": "discovery",
        "opening": (
            "The teaching assistant bot had been quietly learning "
            "lullabies from the children it was supposed to tutor."
        ),
        "seed": 2026090110,
    },
    # robot x funny
    {
        "id": "sf-robot-funny-friendship",
        "world": "robot",
        "tone": "funny",
        "theme": "friendship",
        "opening": (
            "The delivery drone insisted on signing for its own packages, "
            "which confused every warehouse clerk in the district."
        ),
        "seed": 2026090111,
    },
    # mystery x adventurous
    {
        "id": "sf-mystery-adventurous-courage",
        "world": "mystery",
        "tone": "adventurous",
        "theme": "courage",
        "opening": (
            "The lock on the archive room had been changed three times "
            "this month, and the archivist kept finding new keys in her pocket."
        ),
        "seed": 2026090112,
    },
    # mystery x mysterious
    {
        "id": "sf-mystery-mysterious-discovery",
        "world": "mystery",
        "tone": "mysterious",
        "theme": "discovery",
        "opening": (
            "A folded note appeared under the door each Tuesday, always "
            "in the same handwriting, always asking the same impossible question."
        ),
        "seed": 2026090113,
    },
    # mystery x warm
    {
        "id": "sf-mystery-warm-friendship",
        "world": "mystery",
        "tone": "warm",
        "theme": "friendship",
        "opening": (
            "The retired detective still received birthday cards from "
            "every witness she had ever interviewed, even the uncooperative ones."
        ),
        "seed": 2026090114,
    },
    # mystery x funny
    {
        "id": "sf-mystery-funny-logic",
        "world": "mystery",
        "tone": "funny",
        "theme": "logic",
        "opening": (
            "The suspect's alibi was airtight except for the part where "
            "he claimed to have been at a restaurant that closed in 2019."
        ),
        "seed": 2026090115,
    },
)

STORY_EVALUATION_CASES: Final[tuple[StoryEvaluationCase, ...]] = tuple(
    StoryEvaluationCase(
        id=cast("str", entry["id"]),
        world=cast("str", entry["world"]),
        tone=cast("str", entry["tone"]),
        theme=cast("str", entry["theme"]),
        opening=cast("str", entry["opening"]),
        seed=cast("int", entry["seed"]),
    )
    for entry in _STORY_EVALUATION_CASES_DATA
)
