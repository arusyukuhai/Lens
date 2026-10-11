# Lens: bounded BIPS multi-rule information paths (2026-10-11)

This revision implements a **working bounded 2- and 3-rule synthesis operator** for the outer Lens learning loop. Unlike coordinate edits, each child installs the entire chain *at once* and receives one ordinary full-Genome evaluation at the next generation. The change is limited to **outer rule-structure learning**. **Linear regression + Pearson**, **inner reconstruction GA**, native evaluator, Hall of Fame, existing checkpoint format, and diff3/guided alternatives remain in place.

## What the operator actually implements

1. From clean/noisy training pairs, mine a prefix *witness* containing a corrupted byte, and one or two later literal constraints (backward *requirements*).
2. Score the **joint ordered occurrence** of all anchors against the residual from the current linear readout. Merely activating the first rule does not make the path attractive. The last original chunk is excluded from proposal mining, along with its targets and features.
3. Compile a 2-rule chain:

   ```text
   R1:   prefix        -> MARKER_1
   R2:   MARKER_1, -1, suffix  -> MARKER_2, -1
   ```

   or a 3-rule chain:

   ```text
   R1:   prefix        -> MARKER_1
   R2:   MARKER_1, -1, middle  -> MARKER_2, -1
   R3:   MARKER_2, -1, suffix  -> MARKER_3, -1
   ```

   `-1` captures **and preserves** arbitrarily long intervening context. The downstream rule fires only when all upstream information and its own suffix requirement are satisfied. Each compiled rule obeys the existing non-expansion invariant.
4. Allocate markers from the existing **512-token internal namespace**, excluding all 256 coordinates of the Genome's **injective embedding LUT** and all literal codes referenced by *surviving* rules. Markers referenced only by the 2–3 replaced rules may be reclaimed. Input literals are translated **exactly once** through the Genome LUT. No embedding schema/checkpoint change.
5. Prefer low-utility *contiguous* slots for the chain so the ordered sweep executes R1 before R2/R3. Reserve candidate quota for 3-rule paths. Interleave 2- and 3-rule proposals and evaluation windows so deeper paths are actually tested.
6. Use exact CPU-native/MPS/Python evaluation on the **whole modified Genome** in the next generation, rather than treating the proxy score as ground truth. Old/other rules can modify the intermediate state and thereby invalidate a proxy; this is intentional and is why exact evaluation matters. Ordinary parent/child source-overlap reward and delayed BIPS depth-specific statistics are recorded in history CSV.

## Embedding LUT and special wildcard policy

- **Keep the embedding LUT**, including its normal optional mutations/crossover in historical GA/hybrid modes. Because BIPS works in the active Genome's coordinate system, it remains compatible with a permuted LUT.
- **Use ordinary `-1`** as a capture-and-transfer channel. Other existing replacement transform opcodes retain their semantics for legacy rules and other training strategies.
- **Stop generating new pattern-side `-2/-3` hidden-memory writes**, and stop randomly generating new hidden-memory read opcodes. Old/checkpoint rules may still contain and execute them. This avoids silently breaking historical genomes. This is a *training-generation deprecation*, not deletion from the runtime engine.
- Existing capture references on the **replacement side** (`-2` for `$2`, `-3` for `$3`) remain supported; these are *not* the deprecated pattern wildcard operators.
- This BIPS version uses visible latent markers instead of hidden-memory writes. Other wildcard transform families are **not** yet synthesized as part of its path grammar; a future version can add them with explicit typed transition semantics and exact interventions.

## Run

```bash
python main.py \
  --backend cpu-native \
  --learning-mode guided \
  --readout-mode linear-pearson \
  --inference-mode ga \
  --bips-share 0.35 \
  --bips-depth 3 \
  --bips-candidates 32 \
  --bips-sources 24 \
  --population 96 \
  --rules 1500 \
  --hof-size 4096
```

- `--bips-share 0`: disable BIPS; old guided multi-strategy logic is unchanged.
- `--bips-depth 2`: only two-rule paths.
- `--learning-mode ga`: historical full-GA outer learning; BIPS is not run.
- `--learning-mode hybrid`: split guided share between BIPS and other guided operators, while the outer GA continues in its reserved share.
- `--bips-candidates N`, `--bips-sources N`: cap path search overhead. Mining is deliberately bounded per generation; huge values can make it slow.

Logged fields include `bips_proposals`, `bips_added`, `bips_prev_evaluated_d2/d3`, `bips_prev_improved_d2/d3`, and `bips_prev_reward_d2/d3`.

## Boundaries and research questions

This is **bounded forward/backward witness matching with macro-rule insertion**, not a complete bidirectional program synthesizer, CEGAR engine, AND-OR solver, or formal proof of generalized learning. It cannot yet solve arbitrary long-range dependency structures, automatically derive an optimal intermediate alphabet, synthesize hidden-memory channel operators, or prove whole-program equivalence. Also, a simple joint-occurrence proxy is not an exact estimate of the rule chain's effect after other rewrites. Existing rules' transform opcodes can theoretically produce a chosen latent marker even if no rule contains it literally, so full-program testing remains necessary.

Checkpoints need no conversion, because they already store ordinary Rule objects and the 256-entry LUT. The generation setting does not persist as a special BIPS object; choose CLI flags again when resuming.

## Verification

`python -m pytest -q` tests existing behavior, two- and three-rule transfer, modified/retired-window token selection, source-level holdout isolation, and differential comparison between Python reference and C++ native trajectory evaluators. A 2-generation CPU-native training smoke verifies that BIPS children enter the outer population, and the original inner GA is still used.

This implementation is intended as a safe experimental baseline for comparing BIPS vs single-rule directed learning and GA under identical source-level train/validation splits, wall time, and number of expensive full-Genome evaluations. **No general accuracy/speed advantage has yet been demonstrated** on real workloads.
