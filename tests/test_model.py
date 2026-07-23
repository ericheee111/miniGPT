import pytest
import torch

from minigpt import model


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
    normalized = layer_norm.forward(hidden_states)

    # Then: each token vector has approximately zero mean and unit variance.
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(normalized.var(dim=-1, unbiased=False), torch.ones(2, 2), atol=2e-5)


def test_gpt_forward_returns_expected_logits_shape() -> None:
    # Given: a tiny GPT and a [batch, time] token tensor.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)

    # When: a forward pass runs without training targets.
    logits, loss = gpt.forward(token_ids)

    # Then: every position has one logit per vocabulary item and no loss is computed.
    assert logits.shape == (2, 4, 11)
    assert loss is None


def test_gpt_forward_returns_finite_scalar_loss() -> None:
    # Given: next-token inputs and targets.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    # When: the model evaluates the language-model objective.
    _, loss = gpt.forward(token_ids, targets)

    # Then: cross entropy reduces all batch/time positions to one finite scalar.
    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_gpt_rejects_out_of_range_token_id() -> None:
    # Given: an input containing token ID 11 for a vocabulary of size 11.
    gpt = model.GPT(tiny_config())
    token_ids = torch.tensor([[0, 1, 11]], dtype=torch.long)

    # When: the invalid input crosses the model boundary.
    with pytest.raises(model.TokenIdOutOfRangeError, match=r"11.*vocabulary size 11"):
        _ = gpt.forward(token_ids)


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


def test_causal_mask_prevents_future_tokens_affecting_prefix_logits() -> None:
    # Given: two sequences with the same prefix but different future tokens.
    _ = torch.default_generator.manual_seed(11)
    gpt = model.GPT(tiny_config())
    _ = gpt.eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    second = torch.tensor([[1, 2, 8, 9]], dtype=torch.long)

    # When: both sequences pass through the model.
    first_logits, _ = gpt.forward(first)
    second_logits, _ = gpt.forward(second)

    # Then: logits at prefix positions cannot depend on positions to their right.
    assert torch.allclose(first_logits[:, :2], second_logits[:, :2], atol=1e-6)


def test_parameter_count_matches_untied_gpt_formula() -> None:
    # Given: the documented tiny configuration with bias disabled.
    gpt = model.GPT(tiny_config())

    # When: trainable parameters are counted.
    parameter_count = gpt.parameter_count()

    # Then: embeddings, two Blocks, final LayerNorm, and untied LM head total 1,784.
    assert parameter_count == 1_784
