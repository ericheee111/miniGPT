from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from minigpt import __version__, cli

if TYPE_CHECKING:
    import pytest


def test_root_help_lists_stable_commands(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Usage:" in captured.out
    for command in ("prepare-data", "train", "generate", "simulate", "serve", "verify"):
        assert command in captured.out
    assert captured.err == ""


def test_version_path_does_not_import_command_modules(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbid_import(_name: str) -> object:
        reason = "version path imported a command module"
        raise AssertionError(reason)

    monkeypatch.setattr(importlib, "import_module", forbid_import)

    exit_code = cli.main(["--version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == __version__
    assert captured.err == ""


def test_command_dispatch_preserves_remaining_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def command_main(arguments: list[str] | None = None) -> int:
        observed.extend(arguments or [])
        return 17

    module = SimpleNamespace(main=command_main)

    def load_command_module(_name: str) -> object:
        return module

    monkeypatch.setattr(importlib, "import_module", load_command_module)

    exit_code = cli.main(["train", "--config", "configs/char_gpt_smoke.yaml"])

    assert exit_code == 17
    assert observed == ["--config", "configs/char_gpt_smoke.yaml"]


def test_optional_serve_dependency_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_optional(_name: str) -> object:
        reason = "No module named 'fastapi'"
        raise ModuleNotFoundError(reason, name="fastapi")

    monkeypatch.setattr(importlib, "import_module", missing_optional)

    exit_code = cli.main(["serve", "--help"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "optional serving dependencies" in captured.err
    assert ".[serve]" in captured.err


def test_unknown_command_returns_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["unknown-command"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown command" in captured.err
    assert "Usage:" in captured.err


def test_python_module_entrypoint_is_installation_safe() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "minigpt", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__
    assert completed.stderr == ""


def test_version_subprocess_does_not_eagerly_import_http_stack() -> None:
    script = (
        "import json,sys; import minigpt.cli; "
        "raise SystemExit(minigpt.cli.main(['--version']) or "
        "print(json.dumps(sorted(name for name in ('fastapi','uvicorn','httpx') "
        "if name in sys.modules))))"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and inline probe
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    lines = completed.stdout.splitlines()
    assert lines[0] == __version__
    assert json.loads(lines[1]) == []


def test_console_script_metadata_points_to_unified_cli() -> None:
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    matching = [item for item in entry_points if item.name == "minigpt"]

    assert len(matching) == 1
    assert matching[0].value == "minigpt.cli:main"


def test_pyproject_includes_repository_command_modules() -> None:
    document = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'minigpt = "minigpt.cli:main"' in document
    for module in ("prepare_data", "train", "generate", "simulate_serving", "serve"):
        assert f'"{module}"' in document


def test_command_module_contract_is_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def main(arguments: list[str] | None = None) -> int:
        observed.extend(arguments or [])
        return 0

    def load_command_module(_name: str) -> object:
        return SimpleNamespace(main=main)

    monkeypatch.setattr(importlib, "import_module", load_command_module)

    assert cli.main(["generate", "--help"]) == 0
    assert observed == ["--help"]
