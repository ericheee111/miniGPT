import re
from pathlib import Path
from typing import cast

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
    threat_model = (PROJECT_ROOT / "docs" / "PUBLIC_DEMO_THREAT_MODEL.md").read_text(
        encoding="utf-8"
    )

    # When/Then: the complete zero-cost operator path is documented.
    for contract in (
        "ngrok config add-authtoken",
        "DEMO_API_BASE",
        "GitHub Actions",
        "start_public_demo.ps1",
        "ngrok-skip-browser-warning",
        "Task Scheduler",
        "Ctrl+C",
        "Usage",
        "轮换 public URL",
        "完全卸载",
        "Tailscale Funnel",
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
    launcher = (PROJECT_ROOT / "scripts" / "start_public_demo.ps1").read_text(encoding="utf-8")
    example = (PROJECT_ROOT / ".env.public-demo.example").read_text(encoding="utf-8")

    # When/Then: startup, readiness, management API, logs, and exact cleanup are present.
    for contract in (
        ".venv\\Scripts\\python.exe",
        "MINIGPT_CHECKPOINT",
        "MINIGPT_TOKENIZER",
        "PUBLIC_ORIGIN",
        "DEMO_ENABLED",
        '"127.0.0.1"',
        "http://127.0.0.1:8000/healthz",
        "http://127.0.0.1:4040/api/tunnels",
        "outputs\\public-demo",
        "-WindowStyle Hidden",
        "Stop-Process -Id $Process.Id",
    ):
        assert contract in launcher

    # And: credentials, firewall, port mapping, and public bind are absent.
    assert "authtoken" not in launcher.lower()
    assert "0.0.0.0" not in launcher  # noqa: S104 - assert the unsafe bind is absent
    assert "New-NetFirewallRule" not in launcher
    assert "portproxy" not in launcher
    assert example.splitlines() == [
        "PUBLIC_ORIGIN=https://ericheee111.github.io",
        "DEMO_ENABLED=1",
        "MINIGPT_CHECKPOINT=checkpoints/reference/latest.pt",
        "MINIGPT_TOKENIZER=data/processed/tokenizer.json",
    ]
