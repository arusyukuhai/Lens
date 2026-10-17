# Incremental caching changes

* Main file: `cms.nim`
* Fast cache: exact starting-rule-chain equality; immutable row-state snapshots every `--cache-stride` rules and shared preceding firing-feature columns.
* Cache separation: rotating train vs fixed reference vs held-out validation. Train cache invalidated on rotation to avoid comparing different row cohorts.
* LRU with approximate memory budget and `--cache-entries` bound.
* Guaranteed impossible-marker scan skipping for prior-rule markers never generated on the current evaluation rows.
* `fullModel=false` skips post-LOO coefficient/prediction work in selection and viral control evaluations.
* `cache_stats.dat` and `--bench-cache` to measure performance and correctness on the target machine.
* Existing 1500 rules, edit operations, three-parent diff3, viral DAG evolution, dataset rotation, and moving-average plotting retained.

Tests in this environment: `verify_prefix_cache.py`, `verify_rotation.py`, `verify_variation.py`, `verify_ga_math.py` passed. Nim compiler not installed; must run `bash check_build.sh` on a machine with Nim 2.x.
