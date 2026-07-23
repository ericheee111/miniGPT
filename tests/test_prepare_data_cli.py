import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).parents[1]


def test_prepare_data_cli_writes_all_artifacts() -> None:
    # Given: a local UTF-8 corpus and an empty output directory.
    workspace = PROJECT_ROOT / "outputs" / f"test-prepare-data-{uuid4().hex}"
    workspace.mkdir(parents=True)
    try:
        source = workspace / "source.txt"
        _ = source.write_text("abcdeabcde", encoding="utf-8")
        data_dir = workspace / "data"

        # When: data preparation runs through the public CLI.
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "prepare_data.py",
                "--data-dir",
                str(data_dir),
                "--source-url",
                source.as_uri(),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        # Then: every durable artifact exists and is reported.
        assert completed.returncode == 0, completed.stderr
        assert "train.npy" in completed.stdout
        assert "val.npy" in completed.stdout
        assert (data_dir / "raw" / "input.txt").is_file()
        assert (data_dir / "processed" / "train.npy").is_file()
        assert (data_dir / "processed" / "val.npy").is_file()
        assert (data_dir / "processed" / "tokenizer.json").is_file()
        assert (data_dir / "processed" / "metadata.json").is_file()
    finally:
        shutil.rmtree(workspace)


def test_prepare_data_cli_help_lists_boundary_options() -> None:
    # Given: the public data preparation entrypoint.
    # When: its help page is requested.
    completed = subprocess.run(
        [sys.executable, "prepare_data.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: both supported boundary options are documented.
    assert completed.returncode == 0, completed.stderr
    assert "--data-dir" in completed.stdout
    assert "--source-url" in completed.stdout
