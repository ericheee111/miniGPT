from __future__ import annotations

import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import final

import pytest
from typing_extensions import override

import minigpt
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
    assert all(url.removeprefix("#") in parser.ids for url in parser.urls if url.startswith("#"))
    assert 'href="/' not in index
    assert 'src="/' not in index


def test_public_site_client_uses_safe_streaming_and_bounded_offline_polling() -> None:
    # Given: the audited browser source used by GitHub Pages.
    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "web" / "app.js").read_text(encoding="utf-8")
    index = (project_root / "web" / "index.html").read_text(encoding="utf-8")

    # When/Then: output stays text-only and API calls carry no provider-specific header.
    assert "innerHTML" not in app_source
    assert "textContent" in app_source
    assert "ngrok-skip-browser-warning" not in app_source
    assert "response.body" in app_source
    assert ".getReader()" in app_source
    assert "AbortController" in app_source
    assert "info.streaming_enabled === true" in app_source
    assert "stream: state.streamingEnabled && elements.stream.checked" in app_source
    assert '<input id="stream" name="stream" type="checkbox" disabled>' in index
    non_stream_source = app_source.split("async function readNonStream", maxsplit=1)[1].split(
        "async function refreshMetrics", maxsplit=1
    )[0]
    assert "firstTokenAt" not in non_stream_source

    # And: polling is visibility-aware, no faster than 60 seconds, and stores no prompt.
    assert "HEALTH_INTERVAL_MS = 60_000" in app_source
    assert "document.hidden" in app_source
    assert "localStorage" not in app_source
    assert "sessionStorage" not in app_source
    assert "document.cookie" not in app_source
    assert "Content-Security-Policy" in index
    assert 'name="referrer" content="no-referrer"' in index
    assert "Static example" in index
    assert "Backend offline" in app_source
    assert 'elements.outputMode.textContent === "Static example"' in app_source
    assert "Static example shown until you run the local CPU model." in app_source
    assert "Partial model output" in app_source
    assert "No model output" in app_source
    assert app_source.count("valueAsNumber") == 3
    assert "Number(elements." not in app_source
    assert '<pre id="output" tabindex="0">' in index
    assert 'id="output" tabindex="0" aria-live' not in index
    for control_id in ("max-tokens", "temperature", "seed"):
        attributes = index.split(f'id="{control_id}"', maxsplit=1)[1].split(">", maxsplit=1)[0]
        assert "required" in attributes
    assert "does not intentionally\n                persist prompts" in index
    assert "Nothing is stored in this browser" not in index


def test_public_site_css_preserves_touch_targets_and_light_theme_contrast() -> None:
    # Given: the responsive stylesheet used by the public portfolio.
    styles = (Path(__file__).resolve().parents[1] / "web" / "styles.css").read_text(
        encoding="utf-8"
    )

    # When/Then: interactive controls and link groups keep 44px touch targets.
    for contract in (
        ".brand {\n  display: inline-flex;\n  min-height: 44px;",
        ".site-header nav a,\n.header-link {\n  display: inline-flex;\n  min-height: 44px;",
        'input[type="number"] {\n  min-height: 44px;',
        ".preset {\n  min-height: 44px;",
        (
            ".toggle-field {\n  display: flex;\n  grid-column: 1 / -1;\n"
            "  gap: 0.6rem;\n  min-height: 44px;"
        ),
        ".evidence-grid article a {\n  display: inline-flex;\n  min-height: 44px;",
        ".site-footer a {\n  display: inline-flex;\n  min-height: 44px;",
    ):
        assert contract in styles

    # And: the audited light-theme subtle text color remains above 4.5:1.
    assert "--subtle: #5b7065;" in styles
    assert ".site-header nav {\n    grid-row: 2;\n    grid-column: 1 / -1;" in styles
    assert "overflow-x: auto;" in styles
    assert "scroll-margin-top: 90px;" in styles


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
