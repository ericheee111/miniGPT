import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
README_PATH = PROJECT_ROOT / "README.md"
REQUIRED_COMMANDS = (
    "python prepare_data.py",
    "python train.py",
    "python generate.py",
    "python benchmark.py",
    "python profile_model.py",
)
REQUIRED_HEADINGS = (
    "## 项目概览",
    "## English Summary",
    "## Quick Start",
    "## 核心架构",
    "## 性能分析",
    "## Roadmap",
    "## Resume",
)
STAGE_6_CONTRACTS = (
    "Python 3.11",
    "Python 3.14",
    "lr_decay_steps",
    "--run-until-step",
    "checkpoint format v2",
    "v1",
    "LegacyCheckpointResumeError",
    "Windows",
    "Linux",
)
REFERENCE_RESULT_LINK = "docs/results/reference-training/README.md"
BENCHMARK_V2_CONTRACTS = (
    "benchmark_v2.py",
    "compare_benchmarks.py",
    "profile_benchmark_v2.py",
    "fresh process",
    "final_rss_mib",
    "peak_rss_mib",
    "peak_rss_scope: worker_lifetime",
    "stable",
    "unstable",
    "insufficient_samples",
    "shared CI runner",
    "intra-op",
    "inter-op",
    "strictly greater than",
    "--policy configs/benchmark_v2_comparison.yaml",
    "0 = `pass`",
    "1 = 输入、schema、I/O 或证据损坏",
    "2 = `fail`",
    "3 = `not_comparable`",
)
BENCHMARK_V2_LINKS = (
    "configs/benchmark_v2_comparison.yaml",
    "configs/benchmark_v2_smoke.yaml",
    "configs/benchmark_v2_reference.yaml",
    "docs/superpowers/specs/2026-07-28-stage7b-cpu-benchmark-v2-design.md",
    "docs/superpowers/plans/2026-07-28-stage7b-cpu-benchmark-v2.md",
)


def test_readme_documents_stable_user_contracts() -> None:
    # Given: the repository README.
    readme = README_PATH.read_text(encoding="utf-8")

    # When: required sections and commands are inspected.
    # Then: every stable public workflow is documented.
    assert all(heading in readme for heading in REQUIRED_HEADINGS)
    assert all(command in readme for command in REQUIRED_COMMANDS)
    assert "ruff check src tests" in readme
    assert "basedpyright" in readme
    assert "pytest" in readme
    assert "项目状态" in readme
    assert "已结项" in readme
    assert "Stage 1" in readme
    assert "Stage 21" in readme
    assert "PagedKVCachePool" in readme
    assert "minigpt verify --mode release --require-clean" in readme
    assert "](docs/PROJECT_COMPLETION.md)" in readme


def test_readme_documents_stage_6_resume_and_portability_contracts() -> None:
    # Given: the repository README after correctness and portability hardening.
    readme = README_PATH.read_text(encoding="utf-8")

    # When/Then: each proven public contract is present and the old override is absent.
    assert all(contract in readme for contract in STAGE_6_CONTRACTS)
    assert "--resume checkpoints/smoke/latest.pt --max-steps" not in readme


def test_readme_links_generated_reference_training_evidence() -> None:
    # Given: the repository README and committed reference-training evidence.
    readme = README_PATH.read_text(encoding="utf-8")

    # When/Then: readers can reach the generated report from the project overview.
    assert f"]({REFERENCE_RESULT_LINK})" in readme
    assert (PROJECT_ROOT / REFERENCE_RESULT_LINK).is_file()


def test_readme_documents_benchmark_v2_methodology_and_entrypoints() -> None:
    # Given: the repository README after Benchmark v2 is published.
    readme = README_PATH.read_text(encoding="utf-8")

    # When/Then: canonical commands and interpretation safeguards remain discoverable.
    assert all(contract in readme for contract in BENCHMARK_V2_CONTRACTS)


def test_readme_links_benchmark_v2_public_material() -> None:
    # Given: the documented Benchmark v2 local resources.
    readme = README_PATH.read_text(encoding="utf-8")

    # When/Then: each public resource is linked and present in this checkout.
    for target in BENCHMARK_V2_LINKS:
        assert f"]({target})" in readme
        assert (PROJECT_ROOT / target).is_file()


def test_readme_local_markdown_links_exist() -> None:
    # Given: local Markdown links in the README.
    readme = README_PATH.read_text(encoding="utf-8")
    targets = cast(
        "list[str]",
        re.findall(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)", readme),
    )

    # When: every local target is resolved from the project root.
    missing = [target for target in targets if not (PROJECT_ROOT / target).exists()]

    # Then: the README contains no broken local links.
    assert missing == []


def test_readme_documents_public_playground_without_claiming_it_is_live() -> None:
    # Given: the post-v1 repository README.
    readme = README_PATH.read_text(encoding="utf-8")

    # When/Then: deployment, character-level scope, and manual setup remain explicit.
    assert "## Public Playground / Deployment" in readme
    assert "minigpt demo-serve" in readme
    assert "字符级文本续写" in readme
    assert "不是通用问答助手" in readme
    assert "GitHub Pages 尚需 repository owner 配置" in readme
    assert "不能视为已经在线" in readme
    assert "](docs/PUBLIC_DEMO_DEPLOYMENT.md)" in readme
    assert "](docs/PUBLIC_DEMO_THREAT_MODEL.md)" in readme


def test_public_demo_deployment_and_threat_documents_cover_operational_contracts() -> None:
    # Given: the deployment guide and threat model shipped with the static portfolio.
    deployment = (PROJECT_ROOT / "docs" / "PUBLIC_DEMO_DEPLOYMENT.md").read_text(encoding="utf-8")
    completion = (PROJECT_ROOT / "docs" / "PROJECT_COMPLETION.md").read_text(encoding="utf-8")
    threat_model = (PROJECT_ROOT / "docs" / "PUBLIC_DEMO_THREAT_MODEL.md").read_text(
        encoding="utf-8"
    )

    # When/Then: the complete zero-cost operator path is documented.
    for contract in (
        "DEMO_API_BASE",
        "GitHub Actions",
        "start_public_demo_tailscale.ps1",
        "stop_public_demo_tailscale.ps1",
        "Task Scheduler",
        "tailscale funnel status --json",
        "global_requests_per_hour",
        "global_generated_tokens_per_day",
        "轮换 public URL",
        "完全卸载",
        "Tailscale Funnel",
        "zrok",
        "Cloudflare Quick Tunnel",
        "不支持 Server-Sent Events",
        "没有 24/7 SLA",
        "不要向 Demo 输入",
        "非商业个人作品集",
        "不声明生产安全",
    ):
        assert contract in deployment

    # And: operator examples use an explicit placeholder, never a tracked host path.
    assert "<REPO_ROOT>" in deployment
    assert "D:\\Projects\\miniGPT" not in deployment
    assert "C:\\Users\\" not in deployment
    assert "本轮 feature branch 不合入 `main`" not in deployment
    assert "本轮 feature branch 不合入 `main`" not in completion

    # And: high-risk trust boundaries and cleanup obligations remain reviewable.
    for contract in (
        "CORS 被误当成认证",
        "X-Forwarded-For",
        "global limiter",
        "无限生成/SSE",
        "client disconnect",
        "Prompt 进入日志",
        "模型输出 XSS",
        "API Base JavaScript 注入",
        "Actions expression injection",
        "意外监听 `0.0.0.0`",
        "queue/rate race",
        "Stage 21 Evidence",
    ):
        assert contract in threat_model


def test_public_demo_launcher_is_loopback_scoped_and_never_accepts_token() -> None:
    # Given: the Windows launcher and its non-secret environment example.
    launcher = (PROJECT_ROOT / "scripts" / "start_public_demo_tailscale.ps1").read_text(
        encoding="utf-8"
    )
    stopper = (PROJECT_ROOT / "scripts" / "stop_public_demo_tailscale.ps1").read_text(
        encoding="utf-8"
    )
    example = (PROJECT_ROOT / ".env.public-demo.example").read_text(encoding="utf-8")

    # When/Then: startup, readiness, status parsing, logs, and exact cleanup are present.
    for contract in (
        ".venv\\Scripts\\python.exe",
        "MINIGPT_CHECKPOINT",
        "MINIGPT_TOKENIZER",
        "PUBLIC_ORIGIN",
        "DEMO_ENABLED",
        '"127.0.0.1"',
        "http://127.0.0.1:8000/healthz",
        '@("status", "--json")',
        '@("funnel", "status", "--json")',
        '@("funnel", "--bg", "--yes", "8000")',
        "Get-MiniGPTFunnel",
        "Read-RuntimeState",
        "Get-ManagedBackendProcess",
        "outputs\\public-demo",
        "-WindowStyle Hidden",
        "Stop-Process -Id $Process.Id",
    ):
        assert contract in launcher

    # And: neither script can touch another tunnel provider or the Tailscale node state.
    scripts = f"{launcher}\n{stopper}".lower()
    assert "ngrok" not in scripts
    assert "127.0.0.1:4040" not in scripts
    assert "127.0.0.1:8787" not in scripts
    assert "tailscale down" not in scripts
    assert "stop-process -name" not in scripts
    assert '@("funnel", "--https=443", "off")' in stopper
    assert "0.0.0.0" not in launcher  # noqa: S104 - assert the unsafe bind is absent
    assert "New-NetFirewallRule" not in launcher
    assert "portproxy" not in launcher
    assert example.splitlines() == [
        "PUBLIC_ORIGIN=https://ericheee111.github.io",
        "DEMO_ENABLED=1",
        "MINIGPT_CHECKPOINT=checkpoints/reference/latest.pt",
        "MINIGPT_TOKENIZER=data/processed/tokenizer.json",
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell launcher tests require Windows")
def test_public_demo_tailscale_status_parsing_and_preflight_failures_are_mocked() -> None:
    # Given: mocked Tailscale status documents and the dot-sourceable launcher functions.
    funnel_status = json.dumps(
        {
            "TCP": {"443": {"HTTPS": True}},
            "Web": {
                "minigpt-demo.example-tailnet.ts.net:443": {
                    "Handlers": {"/": {"Proxy": "http://127.0.0.1:8000"}}
                }
            },
            "AllowFunnel": {"minigpt-demo.example-tailnet.ts.net:443": True},
        }
    )
    logged_out = json.dumps({"BackendState": "NeedsLogin", "Self": {"Online": False}})
    conflicting_funnel = json.dumps(
        {
            "Web": {
                "other.example-tailnet.ts.net:443": {
                    "Handlers": {"/": {"Proxy": "http://127.0.0.1:9000"}}
                }
            },
            "AllowFunnel": {"other.example-tailnet.ts.net:443": True},
        }
    )

    # When: parser, login validation, and CLI discovery run without invoking real Tailscale.
    parsed = _run_powershell_launcher_probe(
        funnel_status,
        "$result = Get-MiniGPTFunnel -StatusJson $mockJson; $result | ConvertTo-Json -Compress",
    )
    login_error = _run_powershell_launcher_probe(
        logged_out,
        (
            "try { Assert-TailscaleLoggedIn -StatusJson $mockJson } "
            "catch { Write-Output $_.Exception.Message }"
        ),
    )
    missing_cli = _run_powershell_launcher_probe(
        "{}",
        (
            'try { Resolve-TailscaleCommand -Name "minigpt-definitely-missing.exe" } '
            "catch { Write-Output $_.Exception.Message }"
        ),
    )
    reuse_action = _run_powershell_launcher_probe(
        funnel_status,
        "Get-MiniGPTFunnelAction -StatusJson $mockJson",
    )
    create_action = _run_powershell_launcher_probe(
        "{}",
        "Get-MiniGPTFunnelAction -StatusJson $mockJson",
    )
    conflict_error = _run_powershell_launcher_probe(
        conflicting_funnel,
        (
            "try { Get-MiniGPTFunnelAction -StatusJson $mockJson } "
            "catch { Write-Output $_.Exception.Message }"
        ),
    )

    # Then: the ts.net origin is exact and missing login/CLI fail with actionable errors.
    assert parsed.returncode == 0, parsed.stderr
    document = cast("dict[str, object]", json.loads(parsed.stdout))
    assert document == {
        "HostPort": "minigpt-demo.example-tailnet.ts.net:443",
        "PublicUrl": "https://minigpt-demo.example-tailnet.ts.net",
        "Proxy": "http://127.0.0.1:8000",
    }
    assert "not logged in and online" in login_error.stdout
    assert "not installed or is not available on PATH" in missing_cli.stdout
    assert reuse_action.stdout.strip() == "reuse"
    assert create_action.stdout.strip() == "create"
    assert "port 443 already serves another target" in conflict_error.stdout


def _run_powershell_launcher_probe(
    mock_json: str,
    probe: str,
) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(mock_json.encode()).decode("ascii")
    launcher = str(PROJECT_ROOT / "scripts" / "start_public_demo_tailscale.ps1").replace("'", "''")
    command = (
        f". '{launcher}'; "
        f"$mockJson = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')); "
        f"{probe}"
    )
    powershell = (
        Path(os.environ.get("SYSTEMROOT", "C:\\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    return subprocess.run(  # noqa: S603
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
