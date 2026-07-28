"""Test the isolated Benchmark v2 worker boundary and environment evidence."""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, NoReturn, cast

import psutil
import pytest
import torch

import minigpt.benchmark_v2_worker as worker_module
from minigpt.benchmark_v2_environment import (
    ProcessMemoryEvidence,
    WorkerEnvironment,
    apply_cpu_affinity,
    read_process_memory,
)
from minigpt.benchmark_v2_types import BenchmarkV2Case
from minigpt.benchmark_v2_worker import (
    WorkerRequest,
    WorkerResult,
    run_worker_request,
    worker_main,
    worker_request_document,
    worker_response_document,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CountingWorkload:
    """Record which workload calls occur before and inside the canonical timer."""

    timer_call_count: Callable[[], int]
    constructed_before_timer: bool
    warmup_calls: int = 0
    measured_calls: int = 0
    parameter_count: int = 123
    tokens_per_step: int = 64

    def step(self) -> None:
        """Count a warmup or measured step from deterministic timer state."""
        if self.timer_call_count() == 0:
            self.warmup_calls += 1
        else:
            self.measured_calls += 1


class AffinityProcess:
    """Provide an in-memory psutil affinity setter/readback."""

    def __init__(self) -> None:
        """Initialize with an arbitrary effective CPU set."""
        self.effective: list[int] = [7]

    def cpu_affinity(self, cpus: list[int] | None = None) -> list[int]:
        """Set affinity when provided and always return the effective set."""
        if cpus is not None:
            self.effective = list(cpus)
        return list(self.effective)


def make_request() -> WorkerRequest:
    """Build one small, fully explicit worker request."""
    return WorkerRequest(
        protocol_version=1,
        case_identity="a" * 64,
        replicate_index=0,
        case=BenchmarkV2Case(
            name="tiny_t1_s8_b2",
            model_name="tiny",
            n_layer=1,
            n_head=1,
            n_embd=8,
            torch_num_threads=1,
            block_size=8,
            batch_size=2,
        ),
        benchmark_seed=1337,
        vocab_size=65,
        warmup_steps=2,
        measurement_steps=3,
        torch_num_interop_threads=1,
        cpu_affinity=None,
        relevant_environment_variables=("OMP_NUM_THREADS",),
    )


def make_environment() -> WorkerEnvironment:
    """Build deterministic environment evidence for the worker boundary test."""
    return WorkerEnvironment(
        platform="test-platform",
        python_version="3.14.0",
        torch_version="test-torch",
        torch_num_threads=1,
        torch_num_interop_threads=1,
        logical_cpu_count=8,
        requested_cpu_affinity=None,
        effective_cpu_affinity=None,
        relevant_environment_variables={"OMP_NUM_THREADS": None},
    )


def assert_success_metadata(result: WorkerResult, request: WorkerRequest) -> None:
    """Assert worker-owned success metadata in both records and JSON."""
    assert result.worker_pid == os.getpid()
    started_at = datetime.fromisoformat(result.started_at_utc)
    ended_at = datetime.fromisoformat(result.ended_at_utc)
    assert started_at.tzinfo is not None
    assert started_at <= ended_at
    assert result.warmup_steps == request.warmup_steps
    payload = worker_response_document(result)
    assert payload["worker_pid"] == os.getpid()
    assert payload["started_at_utc"] == result.started_at_utc
    assert payload["ended_at_utc"] == result.ended_at_utc
    assert payload["warmup_steps"] == request.warmup_steps


@pytest.mark.parametrize("warmup_steps", [0, 2])
def test_worker_keeps_warmup_construction_and_evidence_outside_timer(
    monkeypatch: pytest.MonkeyPatch,
    warmup_steps: int,
) -> None:
    """Measure only fixed training steps after controls, construction, and warmup."""
    # Given: deterministic controls, timer values, workload, and post-timer evidence.
    request = replace(make_request(), warmup_steps=warmup_steps)
    timer_values = iter((10.0, 11.5))
    timer_calls = 0
    controls = {"threads": False, "interop": False, "affinity": False}
    workloads: list[CountingWorkload] = []

    def fake_perf_counter() -> float:
        nonlocal timer_calls
        timer_calls += 1
        return next(timer_values)

    def fake_set_num_threads(_threads: int) -> None:
        controls["threads"] = True

    def fake_set_num_interop_threads(_threads: int) -> None:
        controls["interop"] = True

    def fake_affinity(_requested: tuple[int, ...] | None) -> tuple[int, ...] | None:
        controls["affinity"] = True
        return None

    def fake_factory(_case: BenchmarkV2Case, *, seed: int, vocab_size: int) -> CountingWorkload:
        del seed, vocab_size
        workload = CountingWorkload(
            timer_call_count=lambda: timer_calls,
            constructed_before_timer=timer_calls == 0 and all(controls.values()),
        )
        workloads.append(workload)
        return workload

    def fake_memory() -> ProcessMemoryEvidence:
        assert timer_calls == 2
        return ProcessMemoryEvidence(
            final_rss_mib=128.0,
            peak_rss_mib=160.0,
            peak_rss_method="windows_peak_working_set",
            peak_rss_sampling_interval_ms=None,
        )

    def fake_capture_environment(
        *,
        requested_cpu_affinity: tuple[int, ...] | None,
        effective_cpu_affinity: tuple[int, ...] | None,
        relevant_environment_variables: tuple[str, ...],
    ) -> WorkerEnvironment:
        del requested_cpu_affinity, effective_cpu_affinity, relevant_environment_variables
        return make_environment()

    monkeypatch.setattr(torch, "set_num_threads", fake_set_num_threads)
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        fake_set_num_interop_threads,
    )
    monkeypatch.setattr(worker_module, "apply_cpu_affinity", fake_affinity)
    monkeypatch.setattr(worker_module, "create_training_step_workload", fake_factory)
    monkeypatch.setattr(worker_module, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(worker_module, "read_process_memory", fake_memory)
    monkeypatch.setattr(
        worker_module,
        "capture_worker_environment",
        fake_capture_environment,
    )

    # When: one fresh-process request is run.
    result = run_worker_request(request)

    # Then: setup and warmup precede exactly two timer reads, and RSS follows the timer.
    workload = workloads[0]
    assert workload.constructed_before_timer is True
    assert workload.warmup_calls == request.warmup_steps
    assert workload.measured_calls == request.measurement_steps
    assert result.measurement_steps == request.measurement_steps
    assert result.elapsed_seconds == 1.5
    assert result.step_time_ms == 500.0
    assert result.tokens_per_second == 128.0
    assert_success_metadata(result, request)
    assert result.final_rss_mib > 0
    assert result.peak_rss_method in {
        "windows_peak_working_set",
        "linux_getrusage_ru_maxrss",
    }
    assert result.peak_rss_sampling_interval_ms is None


def test_apply_cpu_affinity_none_reads_back_inherited_effective_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave inherited affinity unchanged while still recording its effective set."""

    # Given: a process inherited an effective affinity from its parent.
    process = AffinityProcess()
    monkeypatch.setattr(psutil, "Process", lambda: process)

    # When: no new affinity is requested.
    effective = apply_cpu_affinity(None)

    # Then: no setter changes the inherited set, but its readback is retained.
    assert effective == (7,)
    assert process.effective == [7]


def test_apply_cpu_affinity_sets_and_reads_back_effective_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the OS affinity readback instead of assuming the request succeeded."""
    # Given: a process with a working affinity setter and readback.
    process = AffinityProcess()
    monkeypatch.setattr(psutil, "Process", lambda: process)

    # When: a logical CPU set is requested.
    effective = apply_cpu_affinity((3, 1))

    # Then: the exact effective OS readback is returned.
    assert effective == (3, 1)
    assert process.effective == [3, 1]


def test_apply_cpu_affinity_propagates_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the worker when the requested affinity cannot be enforced."""

    # Given: the OS denies affinity access.
    access_denied_name = "AccessDenied"
    access_denied = cast("type[Exception]", getattr(psutil, access_denied_name))

    class DeniedProcess:
        def cpu_affinity(self, _cpus: list[int] | None = None) -> list[int]:
            raise access_denied(42)

    monkeypatch.setattr(psutil, "Process", DeniedProcess)

    # When/Then: access denial is not silently converted into unpinned execution.
    with pytest.raises(access_denied):
        _ = apply_cpu_affinity((0,))


def test_read_process_memory_reports_native_peak_without_sampling() -> None:
    """Capture positive final RSS and the platform-native lifetime peak."""
    # Given/When: current-process memory is captured after measurement.
    evidence = read_process_memory()

    # Then: RSS uses MiB and peak RSS declares a native, non-sampled method.
    assert evidence.final_rss_mib > 0
    assert evidence.peak_rss_mib >= evidence.final_rss_mib
    assert evidence.peak_rss_method in {
        "windows_peak_working_set",
        "linux_getrusage_ru_maxrss",
    }
    assert evidence.peak_rss_sampling_interval_ms is None


@pytest.mark.parametrize(
    "stdin_text",
    [
        "{",
        "{}",
        '{"protocol_version":1,"unexpected":true}',
    ],
)
def test_worker_main_returns_compact_json_failure_for_malformed_stdin(
    monkeypatch: pytest.MonkeyPatch,
    stdin_text: str,
) -> None:
    """Reject invalid stdin with one machine-readable object and nonzero status."""
    # Given: malformed JSON or a request that violates the exact protocol schema.
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "stdout", stdout)

    # When: the worker protocol entrypoint handles the request.
    status = worker_main()

    # Then: it emits exactly one compact failure document and returns nonzero.
    payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
    assert status != 0
    assert payload["protocol_version"] == 1
    assert payload["status"] == "error"
    assert payload["worker_pid"] == os.getpid()
    assert datetime.fromisoformat(cast("str", payload["started_at_utc"])).tzinfo is not None
    assert datetime.fromisoformat(cast("str", payload["ended_at_utc"])).tzinfo is not None
    assert payload["case_identity"] is None
    assert payload["case_name"] is None
    assert payload["replicate_index"] is None
    assert isinstance(payload["error_type"], str)
    assert isinstance(payload["message"], str)
    assert "\n" not in stdout.getvalue().rstrip("\n")


def test_worker_main_retains_case_context_after_parsing_zero_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve parsed identity in failures and accept zero warmup consistently."""
    # Given: a valid zero-warmup request whose execution raises an ordinary error.
    request = replace(make_request(), warmup_steps=0)
    captured_requests: list[WorkerRequest] = []
    stdout = io.StringIO()

    def failing_run(parsed: WorkerRequest) -> NoReturn:
        captured_requests.append(parsed)
        msg = "injected execution failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(worker_module, "run_worker_request", failing_run)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps(worker_request_document(request))),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    # When: the strict protocol parses the request before execution fails.
    status = worker_main()

    # Then: zero warmup reaches execution and failure retains orchestration identity.
    payload = cast("dict[str, object]", json.loads(stdout.getvalue()))
    assert captured_requests[0].warmup_steps == 0
    assert status != 0
    assert payload["worker_pid"] == os.getpid()
    assert payload["case_identity"] == request.case_identity
    assert payload["case_name"] == request.case.name
    assert payload["replicate_index"] == request.replicate_index


def test_worker_main_keyboard_interrupt_returns_130_without_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the conventional interrupted status without emitting false success."""
    # Given: a valid request interrupted during execution.
    request = make_request()
    stdout = io.StringIO()

    def interrupted_run(_request: WorkerRequest) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_module, "run_worker_request", interrupted_run)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps(worker_request_document(request))),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    # When: the worker is interrupted.
    status = worker_main()

    # Then: the shell-visible status is 130 and stdout contains no fake JSON result.
    assert status == 130
    assert stdout.getvalue() == ""
