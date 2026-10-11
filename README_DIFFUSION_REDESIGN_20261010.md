# Lens / Replacer — Severity-balanced branching diffusion (2026-10-10)

Base: `Lens_Spearman_CPU_no_inference_20261010.zip`. The training objective remains **Spearman only**, the native CPU evaluator and prior optimizations remain unchanged, and the string-side inference GA remains disabled. This update changes the **forward corruption/data-generation process**, not the Replacer engine or the learned reverse process.

## Motivation

Old `make_noise_trajectory` applies a fixed set of `K = round(max_noise * length)` substitutions progressively, then samples `S` uniformly spaced stages along it. When `S > K+1` (often even if `S <= length`), adjacent stages have the same integer corruption count and **identical strings**. For a 96-byte sample, `--samples 100 --max-noise .35`, only ~35 distinct damage levels can exist under the old trajectory; it therefore repeatedly evaluated exact duplicate inputs and created redundant rank ties.

## New corruption process

Let `n` be the original byte length, `S=max(3, --samples)` and `K=min(n-1,max(1,round(max_noise*n)))` for `n>=2` (for a 1-byte sequence, K=1). Preserve **exactly one clean example**; distribute the other `S-1` examples across the integer damage levels `k=1..K`.

- If `S <= K+1`, select `S` **distinct** integer severities evenly between 0 and K (single monotonic scattered-corruption path).
- If `S > K+1`, include all `k=1..K` and distribute additional examples **as independent corrupted strings at the same k**. The clean example occurs only once.
- If a level has two examples, the first uses scattered substitutions; the second uses a contiguous **burst-copy**: overwrite a contiguous span with bytes from another contiguous window of the clean source. The two examples have the same target, `cleanliness = 1-k/n`.
- If a level needs three or more examples, allocate more independent corruption branches, alternating scattered and burst-copy families. Replicas form separate cumulative paths over increasing k.
- Always reject no-op substitutions. If two stochastic branches happen to produce the same full byte string at a given severity, change the newest damaged position to another non-original value to eliminate the collision, as long as the finite state space permits.
- The byte-string length is unchanged for both corruption types. The exact **Hamming distance** to the clean original is k, including for repeated-value source strings and context copying.
- `--diffusion-mode scatter` is an ablation mode: same balanced corruption counts and independent replicas, but all branches use scattered substitution. The default is `--diffusion-mode dual`.

Example: a 96-byte source, S=100, max_noise=0.35 gives `K=34`; **100 unique input strings** now represent 35 severity levels (clean + 34 corrupted levels), with multiple equal-damage examples at some levels. The original code could only represent 35 distinct inputs along one path, and many were identical.

**Important distinction:** Equal Hamming damage is *not* guaranteed to mean equal semantic disruption. This model is a discrete fixed-length corruption curriculum with a measurable target, not a physical diffusion process or a trained DDPM reverse kernel. The Spearman target deliberately gives equal labels to the two damage families to encourage a corruption-type-invariant estimate of corruption severity. Pairwise ridge preserves zero-margin constraints for tied levels. This may make the task harder; old Spearman scores and new ones are not directly comparable.

## Run

```bash
python -m pip install numpy scipy threadpoolctl matplotlib
python main.py --backend cpu-native --cpu-threads 8 --cpu-model-batch 32 \
  --blas-threads 1 --cases 2 --samples 100 --max-noise 0.35 \
  --diffusion-mode dual --population 450 --rules 1500 \
  --hof-size 4096 --hof-eval 32
```

At the first rolling dataset construction, with progress enabled, expect a line like:

```
gen=0: diffusion dual case0 samples=100 distinct=100 severity_levels=35 equal_severity_replicas=65
```

Use `--diffusion-mode scatter` to assess whether burst-copy is helping or hurting. Corpus sampling, rolling one-case replacement, native CPU evaluation, selection, caching, and checkpoints retain their previous structure. The new diffusion parameter is stored in new checkpoint `config` metadata; older checkpoints without it still load. The active evaluation dataset is rebuilt upon resume, as before.

## Verification

- `python main.py --self-test`: PASS.
- `python -m pytest -q`: **122 passed, 4 skipped**, plus 5 parametrized subtests.
- Unit checks cover all-samples uniqueness where mathematically possible, exact Hamming damage/targets, monotone severity, alternate burst geometry, n=1/2 corner cases, invalid modes and noise ranges, rolling case rotation and flattened IDs.
- A 2-generation native CPU Spearman-only smoke run completed using **100 corrupted-trajectory steps on short text** with a check that the objective remains Spearman-only; no inference GA was activated.

Changing the noise curriculum does **not** itself guarantee a speedup or improved generalization. It makes each trajectory evaluation carry new information rather than re-evaluating identical corrupted strings.
