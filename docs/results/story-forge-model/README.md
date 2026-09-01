# Story Forge — model, evaluation, and local serving evidence

This package records the 5M-parameter Story Forge training completion,
bounded deterministic evaluation over the 16-case battery, milestone
evaluations, and an isolated local serving smoke. It is a post-v1 research
extension and does not alter historical Stage 7A-21 Evidence.

Source commit: cf296dfc3b028b4b0fd2f7b349ba39607da32773.
Parameter count: 4928144.
Final step: 2999; final validation loss 3.3334 (perplexity 28.03).
Sampling: temperature 0.8, top_k 20.
Cases: 16; EOS hits 5; special-token leaks 0; max immediate loop 0.

## Claim policy

The verdict is descriptive_only. Objective metrics are bounded lexical and
distributional proxies; they do not measure semantic understanding, story
quality, or authorship. No production readiness, general-chat, or universal
speedup claim is made. Checkpoint and canonical data remain external/ignored.
