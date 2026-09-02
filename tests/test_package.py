from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import final

import pytest
from typing_extensions import override

import minigpt
from minigpt import story_forge_product_evidence
from minigpt._version import __version__


@final
class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.ids: set[str] = set()

    @override
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag
        for name, value in attrs:
            if name in {"href", "src"} and value is not None:
                self.urls.append(value)
            if name == "id" and value is not None:
                self.ids.add(value)


def test_package_can_be_imported() -> None:
    # Given: the project has been installed in editable mode.
    # When: Python imports the public package.
    package = minigpt

    # Then: the package exposes the single authored version value.
    assert package.__version__ == __version__


@pytest.mark.parametrize(
    "value",
    [
        "http://demo.example",
        "https://user:password@demo.example",
        "https://demo.example/api",
        "https://demo.example?api=1",
        "https://demo.example#fragment",
        " https://demo.example",
    ],
)
def test_public_site_api_base_rejects_every_non_origin_value(
    value: str,
    tmp_path: Path,
) -> None:
    # Given: a Pages build input that is not an empty value or an HTTPS origin.
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "site"

    # When: the real build entrypoint receives the invalid origin.
    result = _run_site_builder(
        project_root,
        ["--output", str(output), "--api-base", value],
    )

    # Then: it fails before publishing output and names the invalid boundary.
    assert result.returncode != 0
    assert "DEMO_API_BASE" in result.stderr
    assert not output.exists()


def test_public_site_api_base_accepts_offline_and_normalizes_https(tmp_path: Path) -> None:
    # Given: the two supported Pages configuration shapes.
    project_root = Path(__file__).resolve().parents[1]
    offline_output = tmp_path / "offline"
    online_output = tmp_path / "online"

    # When: the real build entrypoint renders each shape.
    offline = _run_site_builder(
        project_root,
        ["--output", str(offline_output), "--api-base", ""],
    )
    online = _run_site_builder(
        project_root,
        [
            "--output",
            str(online_output),
            "--api-base",
            "https://Demo.Example:443/",
        ],
    )

    # Then: the output is stable and carries no path component.
    assert offline.returncode == 0, offline.stderr
    assert online.returncode == 0, online.stderr
    assert 'apiBase: ""' in (offline_output / "config.js").read_text(encoding="utf-8")
    assert 'apiBase: "https://demo.example"' in (online_output / "config.js").read_text(
        encoding="utf-8"
    )


def test_public_site_build_is_offline_safe_deterministic_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no API variable and an unrelated environment secret marker.
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "site"
    sentinel = "must-never-enter-static-output"
    monkeypatch.delenv("DEMO_API_BASE", raising=False)
    monkeypatch.setenv("UNRELATED_DEPLOYMENT_SECRET", sentinel)

    # When: the same managed output is built twice.
    first_build = _run_site_builder(
        project_root,
        ["--source", str(project_root / "web"), "--output", str(output)],
    )
    assert first_build.returncode == 0, first_build.stderr
    first = _directory_bytes(output)
    second_build = _run_site_builder(
        project_root,
        ["--source", str(project_root / "web"), "--output", str(output)],
    )
    assert second_build.returncode == 0, second_build.stderr
    second = _directory_bytes(output)

    # Then: config is offline-only, contents are byte-identical, and no secret leaked.
    assert first == second
    assert 'apiBase: ""' in (output / "config.js").read_text(encoding="utf-8")
    assert sentinel.encode("utf-8") not in b"".join(second.values())
    assert (output / ".nojekyll").is_file()


def test_public_site_build_json_encodes_config_and_preserves_relative_assets(
    tmp_path: Path,
) -> None:
    # Given: an HTTPS Tailscale Funnel API origin and the repository web source.
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "site"

    # When: the deterministic Pages site is built.
    result = _run_site_builder(
        project_root,
        [
            "--source",
            str(project_root / "web"),
            "--output",
            str(output),
            "--api-base",
            "https://minigpt-demo.example-tailnet.ts.net",
        ],
    )
    assert result.returncode == 0, result.stderr
    config = (output / "config.js").read_text(encoding="utf-8")
    index = (output / "index.html").read_text(encoding="utf-8")
    parser = _AssetParser()
    parser.feed(index)

    # Then: config is generated data and every local asset works under /miniGPT/.
    assert 'apiBase: "https://minigpt-demo.example-tailnet.ts.net"' in config
    assert "__DEMO_API_BASE_JSON__" not in config
    local_assets = [url for url in parser.urls if url.startswith("./")]
    assert local_assets == ["./styles.css", "./config.js", "./app.js"]
    assert all((output / url.removeprefix("./")).is_file() for url in local_assets)
    expected_scenarios = {
        "automatic_prefix_cache.json",
        "continuous_batching.json",
        "kv_preemption.json",
        "lazy_reservation.json",
    }
    assert {path.name for path in (output / "data").glob("*.json")} == expected_scenarios
    assert all(url.removeprefix("#") in parser.ids for url in parser.urls if url.startswith("#"))
    assert 'href="/' not in index
    assert 'src="/' not in index


def test_public_site_client_exposes_safe_story_prediction_and_systems_labs() -> None:
    # Given: the audited Story Forge browser source used by GitHub Pages.
    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "web" / "app.js").read_text(encoding="utf-8")
    index = (project_root / "web" / "index.html").read_text(encoding="utf-8")

    # When/Then: untrusted user/model output remains text-only and cancellable.
    assert "innerHTML" not in app_source
    assert "textContent" in app_source
    assert "replaceChildren" in app_source
    assert "ngrok-skip-browser-warning" not in app_source
    assert "response.body" in app_source
    assert ".getReader()" in app_source
    assert "AbortController" in app_source
    assert "info.streaming_enabled === true" in app_source
    assert "stream: state.streamingEnabled && elements.storyStream.checked" in app_source
    assert 'id="story-stream"' in index

    # And: all three product surfaces and their bounded contracts are wired.
    assert "BRANCH_COUNT = 3" in app_source
    assert "MAX_ROUNDS = 4" in app_source
    assert "demo/story/branches" in app_source
    assert "demo/predict/next" in app_source
    assert "demo/predict/score" in app_source
    assert "./data/${name}.json" in app_source
    assert "Story Forge" in index
    assert "Prediction Lab" in index
    assert "Systems Lab" in index
    assert 'id="story-history"' in index
    assert 'id="candidate-list"' in index
    assert 'id="request-lanes"' in index

    # And: polling is visibility-aware, bounded, and stores no prompt or browser identity.
    assert "HEALTH_INTERVAL_MS = 60_000" in app_source
    assert "document.hidden" in app_source
    assert "localStorage" not in app_source
    assert "sessionStorage" not in app_source
    assert "document.cookie" not in app_source
    assert "Content-Security-Policy" in index
    assert 'name="referrer" content="no-referrer"' in index
    assert "Backend offline" in app_source
    assert "Number(elements." not in app_source
    for control_id in ("story-seed", "story-max-tokens", "prediction-top-k"):
        attributes = index.split(f'id="{control_id}"', maxsplit=1)[1].split(">", maxsplit=1)[0]
        assert "required" in attributes
    assert "No prompt storage." in index
    assert "browser keeps story history only in memory" in index
    assert "Controlled story continuation, not general chat." in index
    assert "not a factual assistant" in index


def test_public_site_css_preserves_touch_targets_and_light_theme_contrast() -> None:
    # Given: the responsive Story Forge stylesheet used by the public portfolio.
    styles = (Path(__file__).resolve().parents[1] / "web" / "styles.css").read_text(
        encoding="utf-8"
    )

    # When/Then: primary links, form controls, chips, and replay controls keep large targets.
    for contract in (
        ".brand {\n  display: inline-flex;\n  align-items: center;\n  min-height: 44px;",
        (
            ".site-header nav a,\n.header-link {\n  display: inline-flex;\n"
            "  align-items: center;\n  min-height: 44px;"
        ),
        ".button,\n.icon-button {\n  display: inline-flex;\n  align-items: center;",
        ".chip-row span {\n  display: inline-flex;\n  align-items: center;\n  min-height: 44px;",
        "input,\nselect {\n  min-height: 44px;",
        ".world-card {\n  position: relative;\n  display: flex;\n  min-height: 160px;",
        ".scenario-button {\n  display: grid;",
        ".evidence-links a {\n  display: inline-flex;\n  align-items: center;\n  min-height: 44px;",
        ".site-footer a {\n  display: inline-flex;\n  align-items: center;\n  min-height: 44px;",
    ):
        assert contract in styles
    assert "min-height: 58px;" in styles

    # And: light-theme colors, reduced motion, and responsive layouts remain explicit.
    assert "@media (prefers-color-scheme: light)" in styles
    assert "--text: #102b2c;" in styles
    assert "--muted: #526c70;" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ".site-header nav { display: none; }" in styles
    assert ".branch-grid, .temperature-grid { grid-template-columns: 1fr; }" in styles
    assert ".world-grid { grid-template-columns: 1fr; }" in styles


def test_pages_workflow_uses_variable_boundary_and_main_only_push() -> None:
    # Given: the committed GitHub Pages workflow.
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml"
    ).read_text(encoding="utf-8")

    # When/Then: deployment uses official actions and a build-process environment variable.
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "DEMO_API_BASE: ${{ inputs.demo_api_base || vars.DEMO_API_BASE }}" in workflow
    assert 'description: "Public demo HTTPS origin; empty builds offline-only"' in workflow
    assert "python scripts/build_public_demo_site.py --output _site" in workflow

    # And: automatic deployment is restricted to main while manual dispatch remains possible.
    assert "branches:\n      - main" in workflow
    assert "workflow_dispatch:" in workflow


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _run_site_builder(
    project_root: Path,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(project_root / "scripts" / "build_public_demo_site.py"),
            *arguments,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_systems_lab_assets_builder(
    project_root: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(project_root / "scripts" / "build_systems_lab_assets.py"),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )


def test_story_forge_launch_scripts_are_loopback_scoped_and_non_destructive() -> None:
    # Given: the dedicated Story Forge Windows launchers.
    root = Path(__file__).resolve().parents[1]
    start = (root / "scripts" / "start_story_forge_demo.ps1").read_text(encoding="utf-8")
    stop = (root / "scripts" / "stop_story_forge_demo.ps1").read_text(encoding="utf-8")

    # When/Then: dedicated port, loopback-only bind, and no CodexPro/ngrok touch.
    assert "127.0.0.1" in start
    assert "8001" in start
    assert "Port 8000 is reserved" in start
    assert "SkipFunnel" in start
    assert "ngrok" not in start.lower()
    assert "ngrok" not in stop.lower()
    assert "tailscale down" not in stop.lower()
    assert "DisableFunnel" in stop
    assert "runtime-state.json" in start
    assert "runtime-state.json" in stop


def test_story_forge_model_validator_binds_external_artifact_identity() -> None:
    # Given: the pre-cutover checkpoint validator.
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "validate_story_forge_model.py").read_text(encoding="utf-8")

    # When/Then: it verifies both external files, model family, context, and parameters.
    for contract in (
        "checkpoint-sha256",
        "tokenizer-sha256",
        "story_forge",
        "parameter_count",
        "block_size",
    ):
        assert contract in source


def test_story_forge_systems_assets_are_deterministic(tmp_path: Path) -> None:
    # Given: the source-bound recorded Systems Lab builder.
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_run = _run_systems_lab_assets_builder(root, first)
    second_run = _run_systems_lab_assets_builder(root, second)
    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr

    # When/Then: both builds have the same exact membership and bytes.
    first_files = {path.name: path.read_bytes() for path in first.glob("*.json")}
    second_files = {path.name: path.read_bytes() for path in second.glob("*.json")}
    assert first_files == second_files
    assert set(first_files) == {
        "automatic_prefix_cache.json",
        "continuous_batching.json",
        "kv_preemption.json",
        "lazy_reservation.json",
    }


def test_story_forge_product_evidence_binds_declared_commit_not_worktree(
    tmp_path: Path,
) -> None:
    # Given: every reviewed product source exists in a small committed fixture repository.
    git = shutil.which("git")
    assert git is not None
    repository = tmp_path / "repository"
    repository.mkdir()
    commands: tuple[tuple[str, ...], ...] = (
        (git, "init", "-q"),
        (git, "config", "user.name", "miniGPT Test"),
        (git, "config", "user.email", "test@example.invalid"),
    )
    for command in commands:
        _ = subprocess.run(  # noqa: S603 - resolved Git executable
            command,
            cwd=repository,
            check=True,
            capture_output=True,
        )
    source_paths = story_forge_product_evidence._SOURCE_PATHS  # pyright: ignore[reportPrivateUsage]
    committed_contents: dict[str, bytes] = {}
    for relative in source_paths:
        content = f"committed:{relative}\n".encode()
        committed_contents[relative] = content
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        _ = source.write_bytes(content)
    _ = subprocess.run(  # noqa: S603 - resolved Git executable
        (git, "add", "."),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    _ = subprocess.run(  # noqa: S603 - resolved Git executable
        (git, "commit", "-q", "-m", "fixture"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    source_commit = subprocess.check_output(  # noqa: S603 - resolved Git executable
        (git, "rev-parse", "HEAD"),
        cwd=repository,
        text=True,
        encoding="utf-8",
    ).strip()

    # When: one working-tree source is changed after the declared source commit.
    changed_relative = source_paths[0]
    changed_path = repository / changed_relative
    _ = changed_path.write_bytes(b"unreviewed-working-tree-content\n")
    records = story_forge_product_evidence._source_file_records(  # pyright: ignore[reportPrivateUsage]
        repository,
        source_commit,
    )
    records_by_path = {
        str(record["path"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }

    # Then: Evidence hashes the committed bytes, not the unreviewed worktree mutation.
    record = records_by_path[changed_relative]
    committed_digest = hashlib.sha256(committed_contents[changed_relative]).hexdigest()
    changed_digest = hashlib.sha256(changed_path.read_bytes()).hexdigest()
    assert record["sha256"] == committed_digest
    assert record["sha256"] != changed_digest
