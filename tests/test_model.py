from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from minigpt import layers, model


def call_gpt(
    gpt: model.GPT,
    token_ids: Tensor,
    targets: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    return cast("tuple[Tensor, Tensor | None]", gpt(token_ids, targets))


def tiny_config() -> model.GPTConfig:
    return model.GPTConfig(
        vocab_size=11,
        block_size=4,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )


def test_layer_norm_normalizes_last_dimension() -> None:
    # Given: non-uniform hidden states and an affine-free custom LayerNorm.
    layer_norm = model.LayerNorm(embedding_dim=4, bias=False)
    hidden_states = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            [[-2.0, 0.0, 2.0, 4.0], [5.0, 5.0, 7.0, 7.0]],
        ]
    )

    # When: normalization is applied across the embedding dimension.
    normalized = cast("Tensor", layer_norm(hidden_states))

    # Then: each token vector has approximately zero mean and unit variance.
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(normalized.var(dim=-1, unbiased=False), torch.ones(2, 2), atol=2e-5)


def test_gpt_forward_returns_expected_logits_shape() -> None:
    # Given: a tiny GPT and a [batch, time] token tensor.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)

    # When: a forward pass runs without training targets.
    logits, loss = call_gpt(gpt, token_ids)

    # Then: every position has one logit per vocabulary item and no loss is computed.
    assert logits.shape == (2, 4, 11)
    assert loss is None


def test_gpt_forward_returns_finite_scalar_loss() -> None:
    # Given: next-token inputs and targets.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    # When: the model evaluates the language-model objective.
    _, loss = call_gpt(gpt, token_ids, targets)

    # Then: cross entropy reduces all batch/time positions to one finite scalar.
    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_training_forward_does_not_construct_inference_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: cache construction is forbidden while the ordinary training API runs.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def reject_cache_construction(*_args: object, **_kwargs: object) -> None:
        msg = "ordinary forward constructed an inference cache"
        raise AssertionError(msg)

    monkeypatch.setattr(layers, "LayerKVCache", reject_cache_construction)

    # When: the unchanged training-facing forward computes logits and loss.
    logits, loss = call_gpt(gpt, token_ids, targets)

    # Then: it never allocates or detaches caller-owned inference cache entries.
    assert logits.shape == (1, 4, 11)
    assert loss is not None


def test_gpt_calls_transformer_block_through_module_protocol() -> None:
    # Given: a GPT block with a registered forward hook.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    hook_outputs: list[Tensor] = []

    def record_output(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor):
            msg = "TransformerBlock hook output must be a tensor"
            raise TypeError(msg)
        hook_outputs.append(output)

    handle = gpt.blocks[0].register_forward_hook(record_output)
    try:
        # When: execution enters through GPT.__call__.
        _ = call_gpt(gpt, token_ids)
    finally:
        handle.remove()

    # Then: the nested block hook observes its output.
    assert len(hook_outputs) == 1
    assert hook_outputs[0].shape == (1, 4, 8)


def test_gpt_rejects_unexpected_module_list_entry() -> None:
    # Given: a GPT whose first block was replaced by an unrelated module.
    gpt = model.GPT(tiny_config())
    gpt.blocks[0] = nn.Identity()
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    # When: execution reaches the invalid architecture.
    with pytest.raises(RuntimeError, match=r"block 0.*Identity") as error_info:
        _ = call_gpt(gpt, token_ids)

    # Then: a dedicated error reports the corrupted block instead of silently skipping it.
    assert type(error_info.value).__name__ == "UnexpectedTransformerBlockError"


def test_gpt_rejects_out_of_range_token_id() -> None:
    # Given: an input containing token ID 11 for a vocabulary of size 11.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 11]], dtype=torch.long)

    # When: the invalid input crosses the model boundary.
    with pytest.raises(model.TokenIdOutOfRangeError, match=r"11.*vocabulary size 11"):
        _ = call_gpt(gpt, token_ids)


def test_gpt_generate_appends_requested_tokens_with_temperature_and_top_k() -> None:
    # Given: a two-token prompt and constrained sampling settings.
    _ = torch.default_generator.manual_seed(7)
    gpt = model.GPT(tiny_config())
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    # When: three new tokens are sampled.
    generated = gpt.generate(prompt, max_new_tokens=3, temperature=0.8, top_k=5)

    # Then: generation preserves the prompt and appends exactly three valid IDs.
    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :2], prompt)
    assert int(generated.min()) >= 0
    assert int(generated.max()) < 11


def test_generate_uses_explicit_generator_without_consuming_global_rng() -> None:
    # Given: a model, a local sample generator, and a captured global RNG state.
    _ = torch.default_generator.manual_seed(23)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    sample_generator = torch.Generator(device="cpu")
    _ = sample_generator.manual_seed(29)
    global_state = torch.get_rng_state().clone()

    # When: generation samples through the explicit generator.
    _ = gpt.generate(
        prompt,
        max_new_tokens=4,
        generator=sample_generator,
    )

    # Then: global Torch randomness is unchanged.
    assert torch.equal(torch.get_rng_state(), global_state)


def test_generate_reproduces_tokens_with_equal_local_generator_states() -> None:
    # Given: one eval model and two independent generators with the same seed.
    _ = torch.default_generator.manual_seed(31)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    first_generator = torch.Generator(device="cpu")
    second_generator = torch.Generator(device="cpu")
    _ = first_generator.manual_seed(37)
    _ = second_generator.manual_seed(37)

    # When: both generators drive the same sampling request.
    first = gpt.generate(prompt, max_new_tokens=5, generator=first_generator)
    second = gpt.generate(prompt, max_new_tokens=5, generator=second_generator)

    # Then: local generator state fully determines the sampled continuation.
    assert torch.equal(first, second)


def test_generate_cached_returns_prompt_unchanged_for_zero_new_tokens() -> None:
    # Given: a batched prompt that is longer than the configured context window.
    gpt = model.GPT(tiny_config())
    prompt = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)

    # When: cached generation is asked to append no tokens.
    generated = gpt.generate_cached(prompt, max_new_tokens=0)

    # Then: it preserves the exact caller tensor without attempting prefill.
    assert generated is prompt


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens", "temperature", "top_k"),
    [
        pytest.param([[1]], 1, 1.0, None, id="short-greedy-domain"),
        pytest.param([[1, 2], [3, 4]], 2, 0.8, 5, id="batch"),
        pytest.param([[1, 2, 3]], 4, 1.2, 7, id="cross-overflow"),
        pytest.param([[1, 2, 3, 4]], 3, 0.9, None, id="full-prompt"),
        pytest.param([[1, 2, 3, 4, 5, 6]], 3, 0.7, 3, id="long-prompt"),
    ],
)
def test_cached_generation_exactly_matches_uncached_sampling(
    prompt: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> None:
    # Given: one eval model and equal local generator states for both generation modes.
    _ = torch.default_generator.manual_seed(59)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    token_ids = torch.tensor(prompt, dtype=torch.long)
    uncached_generator = torch.Generator(device="cpu")
    cached_generator = torch.Generator(device="cpu")
    _ = uncached_generator.manual_seed(61)
    _ = cached_generator.manual_seed(61)

    # When: uncached and cached generation sample the same request.
    uncached = gpt.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=uncached_generator,
    )
    cached = gpt.generate_cached(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=cached_generator,
    )

    # Then: cache use and overflow re-prefill do not change any sampled token.
    assert torch.equal(cached, uncached)
    assert torch.equal(cached_generator.get_state(), uncached_generator.get_state())


def test_generate_cached_does_not_register_cache_as_model_state() -> None:
    # Given: an eval model and its parameter/buffer inventory before generation.
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    state_names = tuple(gpt.state_dict())
    buffer_names = tuple(name for name, _ in gpt.named_buffers())

    # When: cached generation crosses the context boundary.
    _ = gpt.generate_cached(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        max_new_tokens=4,
        generator=torch.Generator(device="cpu").manual_seed(67),
    )

    # Then: no caller-lifetime cache appears in parameters, buffers, or serialized state.
    assert tuple(gpt.state_dict()) == state_names
    assert tuple(name for name, _ in gpt.named_buffers()) == buffer_names


@pytest.mark.parametrize(
    ("max_new_tokens", "temperature", "top_k", "reason"),
    [
        pytest.param(-1, 1.0, None, "non-negative", id="length"),
        pytest.param(1, 0.0, None, "positive", id="temperature"),
        pytest.param(1, 1.0, 0, "positive", id="top-k"),
    ],
)
def test_generate_cached_validates_sampling_configuration(
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    reason: str,
) -> None:
    # Given: a valid prompt and an invalid sampling request.
    gpt = model.GPT(tiny_config())

    # When/Then: cached generation reports the same public configuration boundary.
    with pytest.raises(model.InvalidGenerationConfigError, match=reason):
        _ = gpt.generate_cached(
            torch.tensor([[1]], dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )


def test_causal_mask_prevents_future_tokens_affecting_prefix_logits() -> None:
    # Given: two sequences with the same prefix but different future tokens.
    _ = torch.default_generator.manual_seed(11)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    second = torch.tensor([[1, 2, 8, 9]], dtype=torch.long)

    # When: both sequences pass through the model.
    first_logits, _ = call_gpt(gpt, first)
    second_logits, _ = call_gpt(gpt, second)

    # Then: logits at prefix positions cannot depend on positions to their right.
    assert torch.allclose(first_logits[:, :2], second_logits[:, :2], atol=1e-6)


def test_parameter_count_matches_untied_gpt_formula() -> None:
    # Given: the documented tiny configuration with bias disabled.
    gpt = model.GPT(tiny_config())

    # When: trainable parameters are counted.
    parameter_count = gpt.parameter_count()

    # Then: embeddings, two Blocks, final LayerNorm, and untied LM head total 1,784.
    assert parameter_count == 1_784


@pytest.mark.parametrize("bias", [False, True])
def test_expected_parameter_count_matches_instantiated_gpt_for_both_bias_modes(
    *, bias: bool
) -> None:
    """The pure expected-count helper agrees with the actual GPT parameter inventory."""
    # Given: a small valid GPT configuration whose optional biases are explicit.
    config = model.GPTConfig(
        vocab_size=13,
        block_size=5,
        n_layer=2,
        n_head=1,
        n_embd=6,
        dropout=0.0,
        bias=bias,
    )

    # When: the pure formula and the real model count their parameters.
    expected = model.expected_gpt_parameter_count(config)
    observed = model.GPT(config).parameter_count()

    # Then: the formula covers every trainable scalar without model allocation in callers.
    assert expected == observed


def test_prefill_matches_forward_and_returns_layer_caches() -> None:
    # Given: an eval model and a batched prompt shorter than the context window.
    _ = torch.default_generator.manual_seed(41)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    # When: the prompt is evaluated by ordinary forward and cache prefill.
    expected_logits, _ = call_gpt(gpt, prompt)
    actual_logits, cache = gpt.prefill(prompt)

    # Then: prefill preserves final-position logits and materializes detached per-layer K/V.
    assert torch.equal(actual_logits, expected_logits[:, -1:, :])
    assert len(cache) == tiny_config().n_layer
    for layer_cache in cache:
        assert layer_cache.key.shape == (2, 2, 3, 4)
        assert layer_cache.value.shape == (2, 2, 3, 4)
        assert layer_cache.key.dtype == torch.float32
        assert layer_cache.value.dtype == torch.float32
        assert layer_cache.key.device == prompt.device
        assert layer_cache.value.device == prompt.device
        assert layer_cache.length == 3
        assert not layer_cache.key.requires_grad
        assert not layer_cache.value.requires_grad
    assert model.kv_cache_nbytes(cache) == 2 * 2 * 2 * 2 * 3 * 4 * 4


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batched_prefill_matches_individual_prefill_for_equal_lengths(batch_size: int) -> None:
    # Given: equally sized prompts that can be evaluated either together or one at a time.
    _ = torch.default_generator.manual_seed(42)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    rows = [
        [(row + column) % gpt.config.vocab_size for column in range(3)] for row in range(batch_size)
    ]
    prompts = torch.tensor(
        rows,
        dtype=torch.long,
    )
    original = prompts.clone()

    # When: one padded-prefill call evaluates the complete batch.
    actual_logits, actual_cache = gpt.prefill_batch(
        prompts,
        torch.full((batch_size,), 3, dtype=torch.long),
    )

    # Then: every row is numerically equal to its ordinary single-request prefill.
    for row in range(batch_size):
        expected_logits, expected_cache = gpt.prefill(prompts[row : row + 1])
        torch.testing.assert_close(actual_logits[row : row + 1], expected_logits)
        for actual_layer, expected_layer in zip(actual_cache, expected_cache, strict=True):
            torch.testing.assert_close(actual_layer.key[row : row + 1], expected_layer.key)
            torch.testing.assert_close(actual_layer.value[row : row + 1], expected_layer.value)
    assert torch.equal(prompts, original)


def test_variable_length_batched_prefill_matches_compact_individual_caches() -> None:
    # Given: right-padded prompts with mixed true lengths and non-zero padding tokens.
    _ = torch.default_generator.manual_seed(44)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompts = torch.tensor([[1, 2, 10, 9], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.long)
    lengths = torch.tensor([2, 3, 4], dtype=torch.long)

    # When: the shared Transformer path performs one masked padded prefill.
    actual_logits, dense_cache = gpt.prefill_batch(prompts, lengths)

    # Then: final logits and every valid cache prefix equal ordinary compact prefill exactly.
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        expected_logits, expected_cache = gpt.prefill(prompts[row : row + 1, :length])
        torch.testing.assert_close(actual_logits[row : row + 1], expected_logits)
        for dense_layer, expected_layer in zip(dense_cache, expected_cache, strict=True):
            torch.testing.assert_close(
                dense_layer.key[row : row + 1, :, :length],
                expected_layer.key,
            )
            torch.testing.assert_close(
                dense_layer.value[row : row + 1, :, :length],
                expected_layer.value,
            )
            assert dense_layer.key.shape == (3, 2, 4, 4)
            assert dense_layer.value.shape == (3, 2, 4, 4)


def test_batched_prefill_padding_values_cannot_change_valid_outputs() -> None:
    # Given: two copies of the same valid prompts with different right-padding token values.
    _ = torch.default_generator.manual_seed(46)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    first = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)
    second = torch.tensor([[1, 2, 9, 10], [3, 4, 5, 8]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)

    # When: both padded representations are evaluated.
    first_logits, first_cache = gpt.prefill_batch(first, lengths)
    second_logits, second_cache = gpt.prefill_batch(second, lengths)

    # Then: valid final logits and valid K/V prefixes do not depend on padding token IDs.
    torch.testing.assert_close(first_logits, second_logits)
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        for first_layer, second_layer in zip(first_cache, second_cache, strict=True):
            torch.testing.assert_close(
                first_layer.key[row : row + 1, :, :length],
                second_layer.key[row : row + 1, :, :length],
            )
            torch.testing.assert_close(
                first_layer.value[row : row + 1, :, :length],
                second_layer.value[row : row + 1, :, :length],
            )


@pytest.mark.parametrize(
    ("lengths", "message"),
    [
        (torch.tensor([[2]], dtype=torch.long), "shape"),
        (torch.tensor([2], dtype=torch.int32), "dtype"),
        (torch.tensor([0], dtype=torch.long), "values must be"),
        (torch.tensor([3], dtype=torch.long), "padded prompt length"),
    ],
)
def test_batched_prefill_rejects_invalid_prompt_lengths(
    lengths: torch.Tensor,
    message: str,
) -> None:
    # Given: a valid padded prompt and malformed true-length metadata.
    gpt = model.GPT(tiny_config())

    # When/Then: validation rejects metadata before Transformer execution.
    with pytest.raises(model.InvalidTokenTensorError, match=message):
        _ = gpt.prefill_batch(torch.tensor([[1, 2]], dtype=torch.long), lengths)


def test_single_token_decode_matches_full_forward_for_batch() -> None:
    # Given: two prompts, their cache, and one new token per batch row.
    _ = torch.default_generator.manual_seed(43)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    new_tokens = torch.tensor([[5], [6]], dtype=torch.long)
    _, cache = gpt.prefill(prompt)

    # When: only the new tokens are decoded and the full sequences are evaluated independently.
    actual_logits, next_cache = gpt.decode(new_tokens, cache)
    expected_logits, _ = call_gpt(gpt, torch.cat((prompt, new_tokens), dim=1))

    # Then: incremental final logits and cache growth match full-context semantics.
    torch.testing.assert_close(actual_logits, expected_logits[:, -1:, :], rtol=1e-5, atol=1e-6)
    assert all(layer_cache.length == 3 for layer_cache in next_cache)
    assert all(layer_cache.length == 2 for layer_cache in cache)
    assert all(
        next_layer.key.data_ptr() != old_layer.key.data_ptr()
        for old_layer, next_layer in zip(cache, next_cache, strict=True)
    )


def test_multi_token_decode_uses_offset_causal_mask() -> None:
    # Given: a one-token prompt and three tokens decoded in a single call.
    _ = torch.default_generator.manual_seed(47)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1], [2]], dtype=torch.long)
    new_tokens = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.long)
    _, cache = gpt.prefill(prompt)

    # When: cached decode evaluates all new positions.
    actual_logits, next_cache = gpt.decode(new_tokens, cache)
    expected_logits, _ = call_gpt(gpt, torch.cat((prompt, new_tokens), dim=1))

    # Then: every new position has the same causal result as full forward.
    torch.testing.assert_close(actual_logits, expected_logits[:, 1:, :], rtol=1e-5, atol=1e-6)
    assert all(layer_cache.length == 4 for layer_cache in next_cache)


@pytest.mark.parametrize(
    ("prompt_length", "new_length"),
    [(1, 1), (2, 1), (2, 2), (3, 1)],
)
def test_decode_matches_forward_across_context_lengths(prompt_length: int, new_length: int) -> None:
    # Given: deterministic tokens spanning several valid prompt/decode partitions.
    _ = torch.default_generator.manual_seed(53)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    complete = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)[:, : prompt_length + new_length]
    prompt = complete[:, :prompt_length]
    new_tokens = complete[:, prompt_length:]

    # When: the suffix is decoded from a prefetched prefix.
    _, cache = gpt.prefill(prompt)
    actual_logits, _ = gpt.decode(new_tokens, cache)
    expected_logits, _ = call_gpt(gpt, complete)

    # Then: all suffix logits match the corresponding ordinary forward positions.
    torch.testing.assert_close(
        actual_logits,
        expected_logits[:, prompt_length:, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_prefill_and_decode_caches_never_participate_in_gradients() -> None:
    # Given: a training-mode model whose parameters require gradients.
    gpt = model.GPT(tiny_config())
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    # When: explicit inference APIs build and extend a cache.
    logits, cache = gpt.prefill(prompt)
    decoded_logits, next_cache = gpt.decode(torch.tensor([[3]], dtype=torch.long), cache)

    # Then: outputs and every cache tensor are detached despite the surrounding grad mode.
    assert not logits.requires_grad
    assert not decoded_logits.requires_grad
    for layer_cache in (*cache, *next_cache):
        assert not layer_cache.key.requires_grad
        assert not layer_cache.value.requires_grad


@pytest.mark.parametrize(
    "cache_mutation",
    [
        pytest.param("layer_count", id="layer-count"),
        pytest.param("batch", id="batch"),
        pytest.param("head", id="head"),
        pytest.param("dtype", id="dtype"),
        pytest.param("length", id="length"),
        pytest.param("requires_grad", id="requires-grad"),
    ],
)
def test_decode_rejects_invalid_cache_with_clear_error(cache_mutation: str) -> None:
    # Given: a valid prompt cache modified to violate one explicit cache invariant.
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    _, valid_cache = gpt.prefill(prompt)
    first = valid_cache[0]
    invalid_first = first
    if cache_mutation == "layer_count":
        invalid_cache = valid_cache[:-1]
    else:
        if cache_mutation == "batch":
            invalid_first = replace(first, key=first.key.expand(2, -1, -1, -1))
        elif cache_mutation == "head":
            invalid_first = replace(first, key=first.key[:, :1])
        elif cache_mutation == "dtype":
            invalid_first = replace(first, key=first.key.to(torch.float64))
        elif cache_mutation == "length":
            oversized = torch.zeros(1, 2, 5, 4)
            invalid_first = model.LayerKVCache(key=oversized, value=oversized.clone())
        elif cache_mutation == "requires_grad":
            invalid_first = replace(first, key=first.key.detach().requires_grad_())
        invalid_cache = (invalid_first, *valid_cache[1:])

    # When: cached decode validates the caller-owned cache.
    with pytest.raises(model.InvalidKVCacheError, match=cache_mutation.replace("_", " ")):
        _ = gpt.decode(torch.tensor([[3]], dtype=torch.long), invalid_cache)


def test_decode_rejects_cache_overflow_before_attention() -> None:
    # Given: a cache that already consumes the full learned-position window.
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    _, full_cache = gpt.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    # When: another token is decoded without a window re-prefill.
    with pytest.raises(
        model.InvalidKVCacheError, match=r"cached length 4.*new length 1.*block_size 4"
    ):
        _ = gpt.decode(torch.tensor([[5]], dtype=torch.long), full_cache)


def test_variable_length_batched_decode_matches_individual_decode() -> None:
    # Given: two independently prefetched prompts with different true cache lengths.
    _ = torch.default_generator.manual_seed(59)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    _, short_cache = gpt.prefill(torch.tensor([[1, 2]], dtype=torch.long))
    _, long_cache = gpt.prefill(torch.tensor([[3, 4, 5]], dtype=torch.long))
    dense_cache = tuple(
        model.LayerKVCache(
            key=torch.cat(
                (short_layer.key, torch.zeros_like(long_layer.key[:, :, :1])), dim=2
            ).clone(),
            value=torch.cat(
                (short_layer.value, torch.zeros_like(long_layer.value[:, :, :1])), dim=2
            ).clone(),
        )
        for short_layer, long_layer in zip(short_cache, long_cache, strict=True)
    )
    dense_cache = tuple(
        model.LayerKVCache(
            key=torch.cat((layer.key, long_cache[index].key), dim=0),
            value=torch.cat((layer.value, long_cache[index].value), dim=0),
        )
        for index, layer in enumerate(dense_cache)
    )
    old_keys = tuple(layer.key.clone() for layer in dense_cache)
    new_tokens = torch.tensor([[6], [7]], dtype=torch.long)

    # When: one model call decodes both rows with their real position offsets and validity mask.
    batched_logits, next_dense = gpt.decode_batch(
        new_tokens,
        dense_cache,
        torch.tensor([2, 3], dtype=torch.long),
    )
    short_logits, short_next = gpt.decode(new_tokens[:1], short_cache)
    long_logits, long_next = gpt.decode(new_tokens[1:], long_cache)

    # Then: masked logits match individual decode and the caller-owned padding stays unchanged.
    torch.testing.assert_close(batched_logits[:1], short_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_logits[1:], long_logits, rtol=1e-5, atol=1e-6)
    for index, (old_key, dense_layer, next_layer) in enumerate(
        zip(old_keys, dense_cache, next_dense, strict=True)
    ):
        assert torch.equal(dense_layer.key, old_key)
        assert next_layer.key.shape == (2, 2, 4, 4)
        torch.testing.assert_close(next_layer.key[0, :, :2], short_next[index].key[0, :, :2])
        torch.testing.assert_close(next_layer.key[0, :, -1:], short_next[index].key[0, :, -1:])
        torch.testing.assert_close(next_layer.key[1], long_next[index].key[0])


def test_variable_length_batched_decode_size_one_matches_decode() -> None:
    # Given: one ordinary compact request cache.
    _ = torch.default_generator.manual_seed(61)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    _, cache = gpt.prefill(torch.tensor([[1, 2]], dtype=torch.long))
    token = torch.tensor([[3]], dtype=torch.long)

    # When: it uses ordinary and batched single-token decode paths.
    expected_logits, expected_cache = gpt.decode(token, cache)
    actual_logits, actual_cache = gpt.decode_batch(
        token,
        cache,
        torch.tensor([2], dtype=torch.long),
    )

    # Then: batch size one is numerically and structurally equivalent.
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
    for actual_layer, expected_layer in zip(actual_cache, expected_cache, strict=True):
        torch.testing.assert_close(actual_layer.key, expected_layer.key)
        torch.testing.assert_close(actual_layer.value, expected_layer.value)


def test_variable_length_paged_decode_matches_individual_dense_decode() -> None:
    # Given: two compact historical caches exposed as differently blocked read-only views.
    _ = torch.default_generator.manual_seed(67)
    gpt = model.GPT(tiny_config()).eval()
    _, short_cache = gpt.prefill(torch.tensor([[1, 2]], dtype=torch.long))
    _, long_cache = gpt.prefill(torch.tensor([[3, 4, 5]], dtype=torch.long))

    def cache_view(cache: model.KVCache) -> layers.PagedKVCacheView:
        return tuple(
            layers.PagedLayerKVCacheView(
                key_blocks=tuple(
                    layer.key[0, :, start : start + 2, :] for start in range(0, layer.length, 2)
                ),
                value_blocks=tuple(
                    layer.value[0, :, start : start + 2, :] for start in range(0, layer.length, 2)
                ),
                cache_length=layer.length,
                block_tokens=2,
            )
            for layer in cache
        )

    new_tokens = torch.tensor([[6], [7]], dtype=torch.long)

    # When: one model call traverses block views without assembling historical K/V.
    paged_logits, cache_delta = gpt.decode_paged_batch(
        new_tokens,
        (cache_view(short_cache), cache_view(long_cache)),
        torch.tensor([2, 3], dtype=torch.long),
    )
    short_logits, short_next = gpt.decode(new_tokens[:1], short_cache)
    long_logits, long_next = gpt.decode(new_tokens[1:], long_cache)

    # Then: logits and only the newly projected K/V token match the dense reference.
    torch.testing.assert_close(paged_logits[:1], short_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(paged_logits[1:], long_logits, rtol=1e-5, atol=1e-6)
    for layer_index, delta in enumerate(cache_delta):
        assert delta.key.shape == (2, 2, 1, 4)
        torch.testing.assert_close(delta.key[:1], short_next[layer_index].key[:, :, -1:])
        torch.testing.assert_close(delta.value[:1], short_next[layer_index].value[:, :, -1:])
        torch.testing.assert_close(delta.key[1:], long_next[layer_index].key[:, :, -1:])
        torch.testing.assert_close(delta.value[1:], long_next[layer_index].value[:, :, -1:])


@pytest.mark.parametrize(
    ("cache_lengths", "message"),
    [
        (torch.tensor([[2]], dtype=torch.long), "shape"),
        (torch.tensor([2], dtype=torch.int32), "dtype"),
        (torch.tensor([0], dtype=torch.long), "must be in"),
        (torch.tensor([3], dtype=torch.long), "padded cache length"),
    ],
)
def test_batched_decode_rejects_invalid_real_cache_lengths(
    cache_lengths: torch.Tensor,
    message: str,
) -> None:
    # Given: a valid compact cache and invalid real-length metadata.
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    _, cache = gpt.prefill(torch.tensor([[1, 2]], dtype=torch.long))

    # When/Then: validation rejects the metadata before attention executes.
    with pytest.raises(model.InvalidKVCacheError, match=message):
        _ = gpt.decode_batch(torch.tensor([[3]], dtype=torch.long), cache, cache_lengths)
