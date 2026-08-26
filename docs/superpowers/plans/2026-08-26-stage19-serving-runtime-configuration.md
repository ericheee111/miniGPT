# Stage 19 implementation plan

1. Extract serving policy resolution and manifest handling into `src/minigpt/serving_runtime.py`.
2. Make `serve.py` a thin compatibility wrapper and expose the Stage 15–18 options.
3. Pass APC strategy and scheduler fields into the real executor/engine.
4. Add focused runtime, parser, manifest, app, and subprocess tests.
5. Add a canonical runtime example and hash-bound Stage 19 evidence.
6. Run formatting, lint, typing, focused lifecycle, and fresh-checkout verification before committing evidence.
