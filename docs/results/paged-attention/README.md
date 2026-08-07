# Stage 13B — Block-Aware Paged Attention Decode

## Outcome

Normal single-token decode now traverses ordered physical K/V block views directly.
Historical K/V is neither concatenated into a compact cache nor padded densely.
One global softmax covers score chunks in logical token order; value context accumulates
block by block. The model returns one K/V delta per layer and the pool appends it
transactionally. Initial prefill and learned-position overflow re-prefill remain dense.

## Correctness

- `paged-block-size-overflow`: dense/materialized/direct hash `af29e19ddd4b6f484a3988e7682a38abbe03f638cc837dc21627fc44775c7b7f`, zero leaks `True`, overflow fallback `True`.
- `direct-paged-attention`: dense/materialized/direct hash `b874c7576f1f26e0226564895ad03f0734700d2ffa95576d7980366a9f2b1523`, zero leaks `True`, overflow fallback `False`.

A guard replaces `PagedKVCachePool.materialize` with an exception; direct generation
still completes. Generated tokens, terminal states/cancellation, FIFO
admission, logical events, request metrics, and logical cache accounting match both
reference strategies.

## Descriptive CPU benchmark

| Strategy | E2E median (s) | E2E CV |
|---|---:|---:|
| Stage 13A materialized | 0.01778260013088584 | 0.03585036614549277 |
| Stage 13B direct | 0.019080600002780557 | 0.023301498746392008 |

Cache-access loop materialize time: `0.04579310002736747` seconds;
request-view time: `0.03055060002952814` seconds.

These are single-machine descriptive CPU measurements. The verdict is
`descriptive_only` and no speedup is claimed. Stable cross-run and
environment-identity requirements were not used to make a release performance claim.
