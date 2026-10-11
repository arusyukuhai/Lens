# Lens / Replacer — Efficient Stage-Aware BIPS + Bounded CEGAR

2026-10-11. Based on **Lens_Pearson_BIPS_Validated_CEGAR_20261011_full.zip**. This is a *complete-source ZIP*, not a small patch. The outer goal stays ordinary OLS + held-out Pearson, and the inner string-reconstruction GA, 512-token Embedding LUT, original checkpoint schema, HOF capacity, and legacy Hidden Memory/opcode interpreter all remain compatible.

## Why

The user's 61-generation log displayed `bips-cegar tested=12 accepted=0` on **every generation**, with most mismatches classified as `blockedAtSource`: 732 concrete checks and no accepted BIPS children. The original BIPS proposals used raw-input n-grams even though preceding rewriting rules could destroy their literals before the candidate window. The 12-call CEGAR schedule also often failed to test early placements. The old 'refinement' mainly relocated unchanged chains.

## What this version changes

1. **Exact native prefix snapshots (`prefix_states_native`)**. A new exported `replacer_cpu_prefix_batch` function in the C++ evaluator returns *embedded internal token states immediately before a specified rule index on the first sweep*. It runs the original full program's prefix rather than an artificially truncated Genome, preserving the full program's hidden rewrite phase boundaries and existing `-2/-3` side effects. Snapshots are memoized by `(candidate rule boundary, source example ID)` across proposals in one generation. Python / C++ exact parity is tested at every rule boundary, including Hidden Memory.
2. **Counterexample-guided SOURCE anchor repair**. The first rule's literal is re-synthesized from **actual stage tokens** when an earlier rewrite has destroyed the original input literal. Repaired patterns can contain any legitimate internal 0..511 token ID, and are not re-embedded through the LUT. Only the *proposal's source original* supplies the repair; the inner validation source is never used to invent an anchor.
3. **Counterexample-guided BRIDGE repair**. Compile and simulate the prospective 2–4-rule chain on train-original prefix states, one rule at a time. If a producer consumed a later rule's anchor, replace that later anchor with a surviving witnessed intermediate token. This is bounded, example-guided program repair, not an exhaustive inverse-synthesis solver. Any resulting 2–4-rule sequence is still evaluated as a *whole program* through Native Replacer, including normal multiple rounds.
4. **Fair depth × position evaluation**. For each rule-chain depth (2/3/4), try the top proposal at *every candidate placement* before trying additional distinct proposals at the same position. With 3 depths, 3 placements, and budget >=9, all 9 combinations get an attempt, except that a failed cheap source preflight can be skipped. The source, middle and front windows are now actually covered under typical budgets.
5. **Cheap source feasibility gate** (`--bips-cegar-source-gate`, **on**). A chain that cannot fire through all its stages in any train-original first-sweep state is skipped *before* the expensive full Native child evaluation. Other candidates can use the freed budget. This gate trades completeness for efficiency: later-sweep-only chains can be discarded. Use `--no-bips-cegar-source-gate` for ablation and more exhaustive exploration.
6. **Stage-aware abstraction**. Compare the actual whole-program final-rule firing with the necessary ordered anchors **in the embedded state at insertion**, rather than testing raw-input n-grams. Source/validation separation, limited-source micro-OLS, held-out Pearson + MSE gate and uncertainty exploration are preserved.
7. **Instrumentation**. New console metrics: `prefixRows`, `repaired`, `bridgeFix`, `sourceChains`, `sourceSkipped`, `stageHits`, `front`, and `snapshotFail`. CSV fields: `bips_cegar_prefix_rows`, `bips_cegar_repaired_paths`, `bips_cegar_repaired_bridges`, `bips_cegar_source_chains`, `bips_cegar_source_skipped`, `bips_cegar_stage_abstract_hits`, `bips_cegar_front_evaluations`, `bips_cegar_snapshot_failures`.

## Suggested command

```bash
python main.py \
  --backend cpu-native \
  --learning-mode guided \
  --readout-mode linear-pearson \
  --inference-mode ga \
  --bips-share 0.35 \
  --bips-depth 4 \
  --bips-crossfit \
  --bips-cegar-budget 12 \
  --bips-cegar-probes 16 \
  --bips-cegar-windows 3 \
  --bips-cegar-source-gate \
  --population 96 \
  --rules 1500 \
  --hof-size 4096
```

To disable the cheap first-sweep gate while keeping stage-aware synthesis:

```bash
--no-bips-cegar-source-gate
```

The legacy bounded approximation can be compared by disabling BIPS entirely (`--bips-share 0`) and fixing the same corpus, case rotations, seed and wall-clock budget.

## Validation (this source archive)

- `python -m pytest -q`: **149 passed, 4 skipped, 10 subtests passed**. Four Metal/MPS-specific tests were skipped on the CPU-only runner.
- `python main.py --self-test`: PASS.
- CPU Native first-sweep states matched the Python reference at **all test rule boundaries** for 1, 2 and 3 hidden phases, including an old `-2` capture.
- A synthetic example with `ab` destroyed by an earlier rewrite was repaired by synthesizing from its actual intermediate token; a separate example with the *downstream* anchor consumed by the producer was also repaired.
- 4 training generations on a **synthetic local corpus** completed (32 genomes, 128 rules, 3 originals, 20 samples/case, Native + inner GA). At least one generation produced source-independent micro-validation improvements and forwarded BIPS children for full next-generation scoring. This does **not** establish improvements on the actual user's corpus or long-run generalization.

## Important limitations and interpretation

- The stage snapshot describes the **first sweep**. The exact full child evaluation remains multi-sweep. A valid chain that exists only on a later sweep may be skipped by the default source gate.
- Local chain previews do not replay interleaved Hidden Memory rewrite boundaries *inside the candidate window*. The final Native execution and held-out group checks remain authoritative. New generated BIPS rules do not use pattern `-2/-3`; previously saved rules can still use them.
- `accepted > 0` means a candidate survived source-feasibility and the selected micro validation / exploration scheme. It **does not** show that its *next-generation* fitness improved. Check `bips_prev_improved_d2/d3/d4` and `bips_prev_reward_d2/d3/d4` on actual data.
- `repaired`/`bridgeFix` indicate program repairs attempted, not that all repairs generalized. `valid=A/B` is acceptance on a small, repeatedly used original-specific validation group, not independent final test accuracy.
- Pearson is measured on a held-out original for the outer evaluator. The CEGAR micro-OLS is a cheap approximate readout and only a ranking/preflight. The inner GA is unchanged, including automatic compute-budget clamping.
- If there are fewer than 3 rolling originals, grouped crossfit may use within-source sample partitioning and is not a cross-original generalization guarantee.
- Checkpoints and LUT work in the same token space, with generated private intermediate marker IDs disjoint from explicitly occupied literals and input LUT IDs.
