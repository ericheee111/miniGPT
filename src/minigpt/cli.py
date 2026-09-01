"""Expose the installable miniGPT command-line interface without eager optional imports."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from minigpt._version import __version__

CommandMain = Callable[[Sequence[str] | None], int]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Describe one lazily imported repository command."""

    name: str
    module_name: str
    help_text: str
    optional_modules: frozenset[str] = frozenset()
    extras: str = ".[serve]"
    dependency_label: str = "serving"


_COMMANDS = (
    CommandSpec("prepare-data", "prepare_data", "prepare tokenizer and train/validation arrays"),
    CommandSpec(
        "prepare-stories",
        "prepare_stories",
        "prepare deterministic SimpleStories Story Forge data",
        frozenset({"huggingface_hub", "pyarrow", "tokenizers"}),
        ".[story]",
        "story",
    ),
    CommandSpec("train", "train", "train or exactly resume a GPT experiment"),
    CommandSpec("generate", "generate", "generate text from a checkpoint"),
    CommandSpec("simulate", "simulate_serving", "run a deterministic serving simulation"),
    CommandSpec(
        "serve",
        "serve",
        "run the optional HTTP/SSE completion service",
        frozenset({"fastapi", "httpx", "uvicorn"}),
    ),
    CommandSpec(
        "demo-serve",
        "minigpt.public_demo",
        "run the restricted public portfolio demo",
        frozenset({"fastapi", "uvicorn"}),
    ),
    CommandSpec("verify", "minigpt.project_doctor", "verify the repository and evidence chain"),
)
_COMMAND_BY_NAME = {item.name: item for item in _COMMANDS}


def _root_help() -> str:
    width = max(len(item.name) for item in _COMMANDS)
    commands = "\n".join(f"  {item.name:<{width}}  {item.help_text}" for item in _COMMANDS)
    return (
        "miniGPT — CPU-first GPT training, inference, serving, and evidence lab\n\n"
        "Usage:\n"
        "  minigpt [--version]\n"
        "  minigpt <command> [arguments]\n\n"
        "Commands:\n"
        f"{commands}\n\n"
        "Run 'minigpt <command> --help' for command-specific options.\n"
    )


def _load_command(spec: CommandSpec) -> CommandMain:
    try:
        module = importlib.import_module(spec.module_name)
    except ModuleNotFoundError as error:
        if error.name in spec.optional_modules:
            reason = (
                f"command {spec.name!r} requires optional {spec.dependency_label} dependencies; "
                f'install with: python -m pip install -e "{spec.extras}"'
            )
            raise RuntimeError(reason) from error
        raise
    target = getattr(module, "main", None)
    if not callable(target):
        reason = f"command module {spec.module_name!r} does not expose callable main()"
        raise TypeError(reason)
    return cast("CommandMain", target)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command while preserving each existing parser and exit contract."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_root_help(), end="")  # noqa: T201 - command-line boundary
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(__version__)  # noqa: T201 - command-line boundary
        return 0
    command_name = arguments.pop(0)
    spec = _COMMAND_BY_NAME.get(command_name)
    if spec is None:
        print(f"unknown command: {command_name}\n", file=sys.stderr)  # noqa: T201
        print(_root_help(), end="", file=sys.stderr)  # noqa: T201
        return 2
    try:
        target = _load_command(spec)
    except (RuntimeError, TypeError) as error:
        print(str(error), file=sys.stderr)  # noqa: T201 - command-line error boundary
        return 2
    return target(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
