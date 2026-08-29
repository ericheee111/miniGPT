from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import final

import pytest
from scripts.build_public_demo_site import build_site, normalize_api_base
from scripts.build_public_demo_site import main as build_site_main
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
def test_public_site_api_base_rejects_every_non_origin_value(value: str) -> None:
    # Given: a Pages build input that is not an empty value or an HTTPS origin.
    # When/Then: normalization rejects it before any static output is published.
    with pytest.raises(ValueError, match="DEMO_API_BASE"):
        _ = normalize_api_base(value)


def test_public_site_api_base_accepts_offline_and_normalizes_https() -> None:
    # Given: the two supported Pages configuration shapes.
    # When: API Base normalization runs.
    offline = normalize_api_base("")
    online = normalize_api_base("https://Demo.Example:443/")

    # Then: the output is stable and carries no path component.
    assert offline == ""
    assert online == "https://demo.example"


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
    assert (
        build_site_main(
            [
                "--source",
                str(project_root / "web"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    first = _directory_bytes(output)
    assert build_site(source=project_root / "web", output=output, api_base="") is None
    second = _directory_bytes(output)

    # Then: config is offline-only, contents are byte-identical, and no secret leaked.
    assert first == second
    assert 'apiBase: ""' in (output / "config.js").read_text(encoding="utf-8")
    assert sentinel.encode("utf-8") not in b"".join(second.values())
    assert (output / ".nojekyll").is_file()


def test_public_site_build_json_encodes_config_and_preserves_relative_assets(
    tmp_path: Path,
) -> None:
    # Given: an HTTPS ngrok-style API origin and the repository web source.
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "site"

    # When: the deterministic Pages site is built.
    build_site(
        source=project_root / "web",
        output=output,
        api_base="https://example.ngrok-free.app",
    )
    config = (output / "config.js").read_text(encoding="utf-8")
    index = (output / "index.html").read_text(encoding="utf-8")
    parser = _AssetParser()
    parser.feed(index)

    # Then: config is generated data and every local asset works under /miniGPT/.
    assert 'apiBase: "https://example.ngrok-free.app"' in config
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

    # When/Then: output stays text-only and every API call carries the ngrok header.
    assert "innerHTML" not in app_source
    assert "textContent" in app_source
    assert '"ngrok-skip-browser-warning": "1"' in app_source
    assert "response.body" in app_source
    assert ".getReader()" in app_source
    assert "AbortController" in app_source

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


def test_pages_workflow_uses_variable_boundary_and_main_only_push() -> None:
    # Given: the committed GitHub Pages workflow.
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml"
    ).read_text(encoding="utf-8")

    # When/Then: deployment uses official actions and a build-process environment variable.
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "DEMO_API_BASE: ${{ vars.DEMO_API_BASE }}" in workflow
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
