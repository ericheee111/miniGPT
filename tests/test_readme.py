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
