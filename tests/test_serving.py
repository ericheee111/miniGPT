from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest
import torch

from minigpt.layers import KVCache, LayerKVCache
from minigpt.model import GPT, GPTConfig
from minigpt.serving import (
    ContinuousDecodeExecutor,
    DecodeBatchObservation,
    EngineConfig,
    EngineEventType,
    ExecutionResult,
    GenerationRequest,
    ReferenceExecutor,
    RequestState,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True)
class FakeExecutor:
    """Return deterministic tokens without performing tensor-level batching."""

    block_size: int = 8
    failing_ids: set[str] = field(default_factory=set)
    prefill_calls: list[tuple[str, ...]] = field(default_factory=list)
    decode_calls: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        return ()

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        self.prefill_calls.append(tuple(state.request.request_id for state in requests))
        return tuple(self._result(state, latency=0.2) for state in requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        self.decode_calls.append(tuple(state.request.request_id for state in active_requests))
        return tuple(self._result(state, latency=0.1) for state in active_requests)

    def _result(self, state: RequestState, *, latency: float) -> ExecutionResult:
        if state.request.request_id in self.failing_ids:
            return ExecutionResult.failure(state.request.request_id, "injected failure", latency)
        generated_count = len(state.generated_tokens)
        token_id = (state.request.seed + generated_count) % 17
        cache_tokens = min(self.block_size, len(state.request.prompt_tokens) + generated_count)
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=(),
            cache_tokens=cache_tokens,
            latency_seconds=latency,
            used_fallback=cache_tokens == self.block_size and generated_count > 0,
        )


def make_engine(
    *,
    max_active_requests: int = 2,
    max_cached_tokens: int = 32,
    executor: FakeExecutor | ReferenceExecutor | ContinuousDecodeExecutor | None = None,
) -> ServingEngine:
    selected = executor if executor is not None else FakeExecutor()
    return ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=max_cached_tokens,
            ),
            block_size=selected.block_size,
        ),
        executor=selected,
        clock=lambda: 0.0,
    )


def request(
    request_id: str,
    *,
    prompt_tokens: tuple[int, ...] = (1, 2),
    max_new_tokens: int = 2,
    seed: int = 7,
    arrival_time: float = 0.0,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=None,
        seed=seed,
        arrival_time=arrival_time,
    )


def test_fifo_admission_does_not_bypass_the_queue_head() -> None:
    # Given: one active slot and three requests submitted in a known order.
    engine = make_engine(max_active_requests=1)
    for request_id in ("first", "second", "third"):
        engine.submit(request(request_id, max_new_tokens=1))

    # When: ticks admit and complete requests one at a time.
    for tick_time in range(6):
        engine.tick(now=float(tick_time))

    # Then: admission events preserve strict FIFO order.
    admitted = [event for event in engine.events if event.event_type is EngineEventType.ADMITTED]
    assert [event.request_id for event in admitted] == ["first", "second", "third"]
    assert all(event.status is RequestStatus.PREFILLING for event in admitted)
    assert all(event.active_requests == 1 for event in admitted)
    assert all(event.reserved_cache_tokens == 2 for event in admitted)


def test_active_and_reserved_cache_limits_apply_at_admission() -> None:
    # Given: two active slots but capacity for only one three-token reservation.
    engine = make_engine(max_active_requests=2, max_cached_tokens=3)
    engine.submit(request("one", prompt_tokens=(1, 2), max_new_tokens=2))
    engine.submit(request("two", prompt_tokens=(3, 4), max_new_tokens=2))

    # When: the first scheduler tick performs admission.
    engine.tick(now=0.0)

    # Then: cache capacity, rather than the active-slot limit, backpressures the second request.
    assert engine.request_state("one").status is RequestStatus.PREFILLING
    assert engine.request_state("two").status is RequestStatus.WAITING
    assert engine.metrics().reserved_cache_tokens == 3
    assert engine.metrics().waiting_requests == 1


def test_cache_release_allows_waiting_head_to_be_admitted() -> None:
    # Given: the first request occupies the complete cache reservation budget.
    engine = make_engine(max_active_requests=2, max_cached_tokens=2)
    engine.submit(request("one", max_new_tokens=1))
    engine.submit(request("two", max_new_tokens=1))
    engine.tick(now=0.0)

    # When: one tick finishes the first request and the following tick schedules again.
    engine.tick(now=1.0)
    assert engine.metrics().reserved_cache_tokens == 0
    engine.tick(now=2.0)

    # Then: the waiting request is admitted after the release.
    assert engine.request_state("two").status is RequestStatus.PREFILLING
    assert engine.request_state("two").admission_time == 2.0


def test_multiple_requests_decode_once_per_tick_in_stable_order() -> None:
    # Given: two requests that each need one prefill token and two decode tokens.
    engine = make_engine()
    engine.submit(request("a", max_new_tokens=3, seed=10))
    engine.submit(request("b", max_new_tokens=3, seed=20))

    # When: both requests run to completion.
    for tick_time in range(4):
        engine.tick(now=float(tick_time))

    # Then: every iteration emits at most one token per request in stable active order.
    token_events = [
        (event.timestamp, event.request_id)
        for event in engine.events
        if event.event_type is EngineEventType.TOKEN
    ]
    assert token_events == [
        (1.2, "a"),
        (1.2, "b"),
        (2.1, "a"),
        (2.1, "b"),
        (3.1, "a"),
        (3.1, "b"),
    ]


def test_cancellation_covers_waiting_prefill_and_decode_states() -> None:
    # Given: requests deliberately left waiting, prefilling, and decoding.
    engine = make_engine(max_active_requests=2)
    for request_id in ("prefill", "decode", "waiting"):
        engine.submit(request(request_id, max_new_tokens=3))
    engine.tick(now=0.0)
    engine.tick(now=1.0)
    engine.cancel("prefill")
    engine.cancel("decode")
    engine.cancel("waiting")

    # When: cancellation is processed before the next admission or executor call.
    engine.tick(now=2.0)

    # Then: each request is terminal and its reservation is released.
    assert {engine.request_state(name).status for name in ("prefill", "decode", "waiting")} == {
        RequestStatus.CANCELLED
    }
    assert engine.metrics().reserved_cache_tokens == 0
    assert engine.metrics().cancelled_requests == 3


def test_request_can_be_cancelled_after_admission_before_prefill() -> None:
    # Given: admission leaves a request in PREFILLING until the next tick.
    executor = FakeExecutor()
    engine = make_engine(executor=executor)
    engine.submit(request("cancel-me"))
    engine.tick(now=0.0)
    engine.cancel("cancel-me")

    # When: the next tick applies cancellation first.
    engine.tick(now=1.0)

    # Then: the executor never sees the cancelled request.
    assert engine.request_state("cancel-me").status is RequestStatus.CANCELLED
    assert executor.prefill_calls == []


def test_zero_token_request_finishes_without_cache_or_executor() -> None:
    # Given: a request asking for no generated tokens.
    executor = FakeExecutor()
    engine = make_engine(executor=executor)
    engine.submit(request("empty", max_new_tokens=0))

    # When: admission runs.
    engine.tick(now=0.0)

    # Then: it finishes immediately without reserving or executing a cache.
    state = engine.request_state("empty")
    assert state.status is RequestStatus.FINISHED
    assert state.generated_tokens == []
    assert state.admission_time == 0.0
    assert state.finish_time == 0.0
    assert executor.prefill_calls == []
    assert engine.metrics().cached_tokens == 0


def test_request_failure_is_isolated_from_other_active_requests() -> None:
    # Given: one executor result fails while a peer request is valid.
    engine = make_engine(executor=FakeExecutor(failing_ids={"bad"}))
    engine.submit(request("bad", max_new_tokens=1))
    engine.submit(request("good", max_new_tokens=1))

    # When: both reach prefill in the same tick.
    engine.tick(now=0.0)
    engine.tick(now=1.0)

    # Then: only the bad request fails and the peer completes.
    assert engine.request_state("bad").status is RequestStatus.FAILED
    assert engine.request_state("bad").failure_reason == "injected failure"
    assert engine.request_state("good").status is RequestStatus.FINISHED
    assert engine.metrics().failed_requests == 1
    assert engine.metrics().completed_requests == 1


def test_identical_workloads_produce_identical_events_and_tokens() -> None:
    # Given: two independently constructed engines and identical request definitions.
    engines = (make_engine(), make_engine())
    for engine in engines:
        engine.submit(request("a", max_new_tokens=3, seed=100))
        engine.submit(request("b", max_new_tokens=2, seed=200))

    # When: both engines receive the same logical ticks.
    for tick_time in range(4):
        for engine in engines:
            engine.tick(now=float(tick_time))

    # Then: their complete event streams and generated tokens match exactly.
    assert engines[0].events == engines[1].events
    assert engines[0].request_state("a").generated_tokens == [15, 16, 0]
    assert (
        engines[0].request_state("a").generated_tokens
        == engines[1].request_state("a").generated_tokens
    )


def tiny_gpt() -> GPT:
    _ = torch.default_generator.manual_seed(1234)
    model = GPT(
        GPTConfig(
            vocab_size=17,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
    )
    _ = model.eval()
    return model


def run_reference_requests(requests: Sequence[GenerationRequest]) -> ServingEngine:
    executor = ReferenceExecutor(tiny_gpt(), clock=StepClock(step=0.01))
    engine = make_engine(max_active_requests=4, executor=executor)
    for definition in requests:
        engine.submit(definition)
    for tick_time in range(12):
        if engine.is_idle:
            break
        engine.tick(now=float(tick_time))
    return engine


def run_continuous_requests(requests: Sequence[GenerationRequest]) -> ServingEngine:
    executor = ContinuousDecodeExecutor(tiny_gpt(), clock=StepClock(step=0.01))
    engine = make_engine(max_active_requests=4, executor=executor)
    for definition in requests:
        engine.submit(definition)
    for tick_time in range(12):
        if engine.is_idle:
            break
        engine.tick(now=float(tick_time))
    return engine


@dataclass(slots=True)
class StepClock:
    step: float
    current: float = 0.0

    def __call__(self) -> float:
        value = self.current
        self.current += self.step
        return value


def test_request_sampling_rng_is_independent_of_peers() -> None:
    # Given: one target request run alone and alongside a noisy peer.
    target = request("target", prompt_tokens=(1, 2), max_new_tokens=5, seed=42)
    peer = request("peer", prompt_tokens=(3, 4), max_new_tokens=5, seed=999)

    # When: reference execution samples both workloads.
    alone = run_reference_requests((target,))
    together = run_reference_requests((peer, target))

    # Then: peer sampling does not perturb the target generator.
    assert (
        alone.request_state("target").generated_tokens
        == together.request_state("target").generated_tokens
    )


def test_fifo_head_eventually_runs_without_starvation() -> None:
    # Given: three finite requests competing for one active slot.
    engine = make_engine(max_active_requests=1, max_cached_tokens=8)
    for request_id in ("one", "two", "three"):
        engine.submit(request(request_id, max_new_tokens=2))

    # When: the engine runs until no queued or active work remains.
    engine.run_until_idle(start_time=0.0, tick_seconds=1.0, max_ticks=20)

    # Then: all requests complete in FIFO order rather than starving.
    assert engine.is_idle
    assert [engine.request_state(name).status for name in ("one", "two", "three")] == [
        RequestStatus.FINISHED,
        RequestStatus.FINISHED,
        RequestStatus.FINISHED,
    ]
    assert [engine.request_state(name).admission_time for name in ("one", "two", "three")] == [
        0.0,
        3.0,
        6.0,
    ]


def test_metrics_report_queue_ttft_tpot_e2e_and_throughput() -> None:
    # Given: deterministic prefill/decode durations and logical tick timestamps.
    engine = make_engine(max_active_requests=1)
    engine.submit(request("metrics", max_new_tokens=3, arrival_time=0.0))

    # When: the request completes across admission, prefill, and two decode ticks.
    for tick_time in range(4):
        engine.tick(now=float(tick_time))

    # Then: request and aggregate metric definitions are exact.
    metric = engine.request_metrics("metrics")
    assert metric.queue_time_seconds == 0.0
    assert metric.prefill_latency_seconds == 0.2
    assert metric.time_to_first_token_seconds == 1.2
    assert metric.decode_latencies_seconds == (0.1, 0.1)
    assert metric.time_per_output_token_seconds == 0.1
    assert metric.end_to_end_latency_seconds == 3.1
    summary = engine.metrics()
    assert summary.completed_requests == 1
    assert summary.generated_tokens == 3
    assert summary.request_throughput_per_second == 1 / 3.1
    assert summary.token_throughput_per_second == 3 / 3.1
    assert summary.peak_cached_tokens == 4


def test_learned_position_overflow_reuses_stage9_reprefill_fallback() -> None:
    # Given: a prompt already filling the learned-position block and more decode tokens requested.
    engine = run_reference_requests(
        (request("overflow", prompt_tokens=(1, 2, 3, 4), max_new_tokens=3, seed=55),)
    )

    # When: reference execution advances beyond the original cache capacity.
    fallback_events = [
        event
        for event in engine.events
        if event.event_type is EngineEventType.TOKEN and event.used_fallback
    ]

    # Then: decode remains valid by re-prefilling a sliding block-sized context.
    assert engine.request_state("overflow").status is RequestStatus.FINISHED
    assert len(engine.request_state("overflow").generated_tokens) == 3
    assert len(fallback_events) == 2
    assert engine.metrics().peak_cached_tokens == 4


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_continuous_executor_matches_reference_tokens_states_events_and_metrics(
    batch_size: int,
) -> None:
    # Given: mixed prompts, generation lengths, and request-local sampling streams.
    definitions = tuple(
        request(
            f"request-{index}",
            prompt_tokens=tuple(range(1, 2 + (index % 3))),
            max_new_tokens=2 + (index % 3),
            seed=100 + index,
        )
        for index in range(batch_size)
    )

    # When: identical models run through reference and continuous decode executors.
    reference = run_reference_requests(definitions)
    continuous = run_continuous_requests(definitions)

    # Then: tokens, terminal state, and logical event semantics remain identical.
    for definition in definitions:
        request_id = definition.request_id
        assert (
            continuous.request_state(request_id).generated_tokens
            == reference.request_state(request_id).generated_tokens
        )
        assert continuous.request_state(request_id).status is RequestStatus.FINISHED
        assert continuous.request_metrics(request_id) == reference.request_metrics(request_id)
    assert continuous.events == reference.events
    continuous_metrics = continuous.metrics()
    reference_metrics = reference.metrics()
    assert continuous_metrics.completed_requests == reference_metrics.completed_requests
    assert continuous_metrics.generated_tokens == reference_metrics.generated_tokens
    assert continuous_metrics.max_decode_batch_size == batch_size


def test_continuous_batch_order_does_not_change_request_outputs() -> None:
    # Given: the same four request identities in forward and reverse active order.
    definitions = tuple(
        request(
            f"request-{index}",
            prompt_tokens=tuple(range(1, index + 2)),
            max_new_tokens=4,
            seed=200 + index,
        )
        for index in range(4)
    )

    # When: continuous decode batches use opposite row orders.
    forward = run_continuous_requests(definitions)
    reverse = run_continuous_requests(tuple(reversed(definitions)))

    # Then: stable request identity and local generators determine every output.
    for definition in definitions:
        request_id = definition.request_id
        assert (
            forward.request_state(request_id).generated_tokens
            == reverse.request_state(request_id).generated_tokens
        )


def test_continuous_cancellation_does_not_perturb_surviving_request_rng() -> None:
    # Given: one target run alone and with a peer cancelled before its first decode.
    target = request("target", prompt_tokens=(1, 2), max_new_tokens=4, seed=301)
    peer = GenerationRequest(
        request_id="peer",
        prompt_tokens=(3, 4, 5),
        max_new_tokens=4,
        seed=302,
        cancellation_time=2.0,
    )

    # When: both continuous workloads finish.
    alone = run_continuous_requests((target,))
    with_cancel = run_continuous_requests((peer, target))

    # Then: cancellation and peer admission never consume the target generator.
    assert (
        alone.request_state("target").generated_tokens
        == with_cancel.request_state("target").generated_tokens
    )
    assert with_cancel.request_state("peer").status is RequestStatus.CANCELLED


def test_continuous_invalid_cache_fails_only_bad_request() -> None:
    # Given: two valid requests prefetched together, then one caller-owned cache is corrupted.
    executor = ContinuousDecodeExecutor(tiny_gpt(), clock=StepClock(step=0.01))
    engine = make_engine(max_active_requests=2, executor=executor)
    engine.submit(request("bad", max_new_tokens=3, seed=401))
    engine.submit(request("good", max_new_tokens=3, seed=402))
    engine.tick(now=0.0)
    engine.tick(now=1.0)
    bad_state = engine.request_state("bad")
    assert bad_state.kv_cache is not None
    first = bad_state.kv_cache[0]
    bad_state.kv_cache = (LayerKVCache(key=first.key.to(torch.float64), value=first.value),)

    # When: the next decode tick validates requests before assembly.
    engine.tick(now=2.0)

    # Then: the malformed request fails and its valid peer still decodes.
    assert engine.request_state("bad").status is RequestStatus.FAILED
    assert engine.request_state("bad").kv_cache is None
    assert engine.request_state("good").status is RequestStatus.DECODING
    assert len(engine.request_state("good").generated_tokens) == 2
    assert executor.decode_observations[-1].request_ids == ("good",)


def test_continuous_scatter_returns_compact_cache_without_padding_or_mutation() -> None:
    # Given: two decoding states whose compact caches have different true lengths.
    model_instance = tiny_gpt()
    executor = ContinuousDecodeExecutor(model_instance, clock=StepClock(step=0.01))
    states: list[RequestState] = []
    for request_id, prompt in (("short", (1,)), ("long", (2, 3, 4))):
        definition = request(request_id, prompt_tokens=prompt, max_new_tokens=4, seed=500)
        generator = torch.Generator(device="cpu").manual_seed(definition.seed)
        token_ids = torch.tensor((prompt,), dtype=torch.long)
        logits, cache = model_instance.prefill(token_ids)
        token_id = int(logits[:, -1, :].argmax(dim=-1).item())
        states.append(
            RequestState(
                request=definition,
                generator=generator,
                status=RequestStatus.DECODING,
                generated_tokens=[token_id],
                kv_cache=cache,
                cached_tokens=len(prompt),
                reserved_cache_tokens=4,
            )
        )
    old_keys = tuple(cast("KVCache", state.kv_cache)[0].key.clone() for state in states)

    # When: one dense batch is assembled and scattered back to both requests.
    results = executor.decode_batch(states)

    # Then: each result is compact, typed correctly, and caller caches remain unchanged.
    assert [result.cache_tokens for result in results] == [2, 4]
    for state, old_key, result in zip(states, old_keys, results, strict=True):
        assert state.kv_cache is not None
        assert torch.equal(state.kv_cache[0].key, old_key)
        assert result.cache is not None
        assert result.cache[0].length == state.cached_tokens + 1
        assert result.cache[0].key.dtype == model_instance.token_embedding.weight.dtype
        assert result.cache[0].key.device == model_instance.token_embedding.weight.device
        assert result.cache[0].key.shape[0] == 1
    observation = executor.decode_observations[-1]
    assert observation.batch_size == 2
    assert observation.useful_cache_tokens == 4
    assert observation.padded_cache_tokens == 6


def test_continuous_overflow_matches_reference_reprefill_tokens() -> None:
    # Given: a full learned-position window that requires Stage 9 re-prefill every later tick.
    definition = request(
        "overflow",
        prompt_tokens=(1, 2, 3, 4),
        max_new_tokens=3,
        seed=601,
    )

    # When: reference and continuous executors cross the same window boundary.
    reference = run_reference_requests((definition,))
    continuous = run_continuous_requests((definition,))

    # Then: tokens and fallback event semantics are identical.
    assert (
        continuous.request_state("overflow").generated_tokens
        == reference.request_state("overflow").generated_tokens
    )
    assert continuous.events == reference.events
