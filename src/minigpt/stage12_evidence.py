"""Generate and verify Stage 12 HTTP serving evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.data import JsonValue

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NUMBER = 12


@dataclass(frozen=True, slots=True)
class Stage12EvidenceVerificationError(ValueError):
    """Report missing, unexpected, or hash-mismatched Stage 12 evidence."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 12 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage12EvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> EvidenceDocument:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("EvidenceDocument", raw)


def _write_json(path: Path, document: EvidenceDocument) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_stage12_evidence(
    *,
    benchmark_path: Path,
    api_examples_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build a compact HTTP package from fresh benchmark and correctness evidence."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "http_benchmark.json": benchmark_path,
        "api_examples.json": api_examples_path,
        "lifecycle.json": lifecycle_path,
    }
    for source in inputs.values():
        if not source.is_file():
            _invalid(f"evidence input does not exist: {source}")

    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, source in inputs.items():
        _ = shutil.copyfile(source, evidence_root / name)

    benchmark = _read_json(evidence_root / "http_benchmark.json")
    api_examples = _read_json(evidence_root / "api_examples.json")
    lifecycle = _read_json(evidence_root / "lifecycle.json")
    summary = _summary(
        source_commit=source_commit,
        benchmark=benchmark,
        api_examples=api_examples,
        lifecycle=lifecycle,
    )
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, benchmark, api_examples),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "stage": 12,
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_stage12_evidence(package_root)
    return package_root


def verify_stage12_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact package membership, sizes, hashes, and source identity."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NUMBER:
        _invalid("manifest stage must be 12")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        _invalid("manifest source_commit must be non-empty")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        _invalid("manifest artifacts must be a list")
    expected = _manifest_entries(cast("list[object]", entries))
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        _invalid(f"artifact membership differs: {missing=}, {unexpected=}")
    for relative, (size, digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    summary = _read_json(package_root / "summary.json")
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    return manifest


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[str, object]", raw_entry)
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            _invalid("manifest artifact entry fields are invalid")
        expected[relative] = (size, digest)
    return expected


def _summary(
    *,
    source_commit: str,
    benchmark: EvidenceDocument,
    api_examples: EvidenceDocument,
    lifecycle: EvidenceDocument,
) -> EvidenceDocument:
    cases = benchmark.get("cases")
    measurements = benchmark.get("measurements")
    engine = benchmark.get("engine")
    if not isinstance(cases, list) or not isinstance(measurements, list):
        _invalid("benchmark cases and measurements must be lists")
    if not isinstance(engine, dict):
        _invalid("benchmark engine snapshot must be an object")
    errors = 0
    cancellations = 0
    for raw_case in cast("list[object]", cases):
        if not isinstance(raw_case, dict):
            _invalid("benchmark cases must be objects")
        case = cast("dict[str, object]", raw_case)
        case_errors = case.get("http_error_count")
        case_cancellations = case.get("cancellation_count")
        if not isinstance(case_errors, int) or not isinstance(case_cancellations, int):
            _invalid("benchmark error and cancellation counts must be integers")
        errors += case_errors
        cancellations += case_cancellations
    lifecycle_exit = lifecycle.get("exit_code")
    if lifecycle_exit != 0:
        _invalid("lifecycle correctness command did not pass")
    if not isinstance(api_examples.get("non_stream"), dict):
        _invalid("non-stream API evidence is missing")
    if not isinstance(api_examples.get("streaming"), dict):
        _invalid("streaming API evidence is missing")
    return {
        "schema_version": 1,
        "stage": 12,
        "source_commit": source_commit,
        "openai_compatible_subset": True,
        "chat_completions": False,
        "paged_attention": False,
        "bpe": False,
        "gpu_or_distributed": False,
        "benchmark_scope": "http_end_to_end",
        "benchmark_cases": len(cases),
        "benchmark_requests": len(measurements),
        "http_error_count": errors,
        "cancellation_count": cancellations,
        "benchmark_duration_seconds": benchmark.get("duration_seconds"),
        "peak_active_requests": cast("dict[str, JsonValue]", engine).get("peak_active_requests"),
        "average_decode_batch_size": cast("dict[str, JsonValue]", engine).get(
            "average_decode_batch_size"
        ),
        "average_prefill_batch_size": cast("dict[str, JsonValue]", engine).get(
            "average_prefill_batch_size"
        ),
        "lifecycle_test_exit_code": lifecycle_exit,
    }


def _readme(
    summary: EvidenceDocument,
    benchmark: EvidenceDocument,
    api_examples: EvidenceDocument,
) -> str:
    cases = cast("list[EvidenceDocument]", benchmark["cases"])
    non_stream = cast("EvidenceDocument", api_examples["non_stream"])
    streaming = cast("EvidenceDocument", api_examples["streaming"])
    concurrent = cast("EvidenceDocument", api_examples["concurrent"])
    performance_caveat = "are descriptive for this machine and are not a speedup claim or shared-CI performance truth."  # noqa: E501
    benchmark_header = "| Concurrency | Prompts | Stream | req/s | tokens/s | TTFT P50/P95/P99 | TPOT P50/P95/P99 | E2E P50/P95/P99 | Errors |"  # noqa: E501
    lines = [
        "# Stage 12 — OpenAI-Compatible HTTP Serving and Streaming",
        "",
        "## Outcome",
        "",
        "Stage 12 exposes the existing Stage 10 FIFO scheduler and Stage 11 executors through",
        "`GET /healthz`, `GET /v1/models`, and an OpenAI-compatible subset of",
        "`POST /v1/completions`. Checkpoint, tokenizer, config, and model load once at startup.",
        "The async HTTP layer never calls the model: it submits commands to one dedicated",
        "`EngineRunner` thread, which is the only owner that calls `ServingEngine` and its model.",
        "",
        "The subset accepts only `model`, `prompt`, `max_tokens`, `temperature`, `stream`, and",
        "`seed`; unsupported OpenAI fields are rejected. Chat Completions, BPE, PagedAttention,",
        "GPU, distributed serving, and new model structures remain out of scope.",
        "",
        "## Ordinary completion",
        "",
        "```console",
        str(non_stream["curl"]),
        "```",
        "",
        "```json",
        json.dumps(non_stream["response"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Streaming SSE",
        "",
        "```console",
        str(streaming["curl"]),
        "```",
        "",
        "```text",
        *cast("list[str]", streaming["sse_lines"]),
        "```",
        "",
        "Each token chunk is produced independently. The final JSON chunk carries `length` and",
        "usage, followed by `[DONE]` only on normal completion. Concatenated token chunks equal",
        "the non-stream result for the same seed. Client disconnect requests engine cancellation;",
        "Stage 10 cancellation releases the KV reservation. A bounded per-stream queue prevents a",
        "slow client from blocking peers; overflow records backpressure, cancels that request, and",
        "does not emit `[DONE]`.",
        "",
        "## Concurrent workload",
        "",
        f"Concurrent example status codes: `{concurrent['status_codes']}`; ",
        f"independent fixed-seed outputs: `{concurrent['texts']}`.",
        "",
        "HTTP concurrency is the number of requests in flight at the API boundary. Continuous",
        "batching is the executor's tensor-level grouping of eligible prefill/decode work after",
        "FIFO admission. The former creates opportunity; it does not guarantee the latter's batch",
        "size.",
        "",
        "## End-to-end HTTP benchmark",
        "",
        "This benchmark includes HTTP validation, JSON/SSE serialization, async-to-thread queues,",
        "the scheduler, and engine execution. It is separate from the Stage 11 executor benchmark,",
        "which isolates executor strategies without HTTP system overhead. These localhost numbers",
        performance_caveat,
        "",
        benchmark_header,
        "|---:|---|---|---:|---:|---|---|---|---:|",
    ]
    lines.extend(_benchmark_row(case) for case in cases)
    lines.extend(
        [
            "",
            "TTFT is time to the first observable token chunk for streaming. For non-streaming,",
            "the first observable completion payload arrives at E2E, so TTFT equals E2E and TPOT",
            "is not observable. TPOT is the mean interval between streamed output tokens. P95/P99",
            "show tail latency and must be interpreted with request count and machine context.",
            "",
            (
                f"Canonical matrix: {summary['benchmark_cases']} cases, "
                f"{summary['benchmark_requests']} requests, "
                f"{summary['http_error_count']} HTTP errors, "
                f"{summary['cancellation_count']} cancellations, peak active "
                f"{summary['peak_active_requests']}, average prefill batch "
                f"{summary['average_prefill_batch_size']}, and average decode batch "
                f"{summary['average_decode_batch_size']}."
            ),
            "",
            "The lifecycle evidence separately covers active/waiting cancellation, real localhost",
            "disconnect, bounded-buffer backpressure, failure isolation, graceful shutdown, and KV",
            "reservation release. Historical Stage evidence was not modified.",
        ]
    )
    return "\n".join(lines) + "\n"


def _benchmark_row(case: EvidenceDocument) -> str:
    ttft = cast("EvidenceDocument", case["ttft_seconds"])
    tpot = cast("EvidenceDocument", case["tpot_seconds"])
    e2e = cast("EvidenceDocument", case["e2e_seconds"])
    return (
        f"| {case['concurrency']} | {case['prompt_kind']} | {case['stream']} | "
        f"{_number(case['requests_per_second'])} | "
        f"{_number(case['generated_tokens_per_second'])} | {_latency(ttft)} | "
        f"{_latency(tpot)} | {_latency(e2e)} | {case['http_error_count']} |"
    )


def _latency(values: EvidenceDocument) -> str:
    return "/".join(_number(values[key]) for key in ("p50", "p95", "p99"))


def _number(value: JsonValue) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.6f}"
