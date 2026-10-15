# Revision validation notes

- Fine reranks shortlisted Add/Replace **every generation**, regardless of
  `--eval-every`; there is no Coarse-only branch choice.
- `--eval-every=5 --steps=12` detailed-evaluation checkpoints: steps 0,4,9,11.
- Coarse and Fine scoring are never mixed in a single chooseMove comparison.
- `--coarse-power == --fine-power` reuses the exact same diffusion rows and
  stage caches for Fine; unequal schedules never share row-dependent caches.
- `--pair-fingerprint=1` filters only absent *mandatory adjacent fixed token
  pairs*; wildcard gaps impose no adjacency requirement. Tests compare filtered
  and unfiltered vectors on constructed examples.
- `--delta-suffix=1` is on by default. Existing full-versus-delta regression
  tests remain in `--self-test`.
- `--eval-every=0` is rejected.
- **Nim compilation and benchmarks are not verified.** Run `./cms --self-test`
  after compiling.
