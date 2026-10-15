## Regression checklist

- `--eval-every=1` yields `doEvaluate=true` on every generation and the same Fine branch comparison as the base ZIP.
- `--eval-every=5 --steps=12` checkpoints are steps 0,4,9,11; intermediate steps choose moves only from Coarse scores.
- `--eval-power` left unspecified equals `finePower`, independent of CLI argument order.
- `--eval-power=2.0 --fine-power=1.0` uses disjoint caches for the diagnostic and Fine training schedules.
- `training.dat` and Pearson MA are updated only on checkpoints, but `rules.tsv` is saved each generation.
- When `--eval-every > 1`, Fine is not a gate between checkpoints: this may change selected rules and generalization.
- `evalEvery >= 1` and all powers are finite `>= 1` (with `coarse >= fine` only).
- `--eval-every=0` is an error, not a silently invalid or degenerate interval.
