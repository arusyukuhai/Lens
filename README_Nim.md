# Lens / Replacer — greedy DE + rare mutation

This package is based on the **single-pass / skip-oversize** edition. It keeps its one-incumbent, one-rule-at-a-time greedy differential evolution, strict Pareto replacement, one left-to-right Replace pass, 100 diffusion steps, inference GA, gnuplot output, and stage caches.

## New: configurable low-probability mutation

- Each **DE trial offspring** starts as an edit transferred by `diff3`: apply `(donor B -> donor C)` to the current rule A (both its pattern and replacement).
- With probability **5% per trial** (default), it then locally mutates **only one side**, chosen at random (pattern or replacement). The mutation edits a log-uniform number of sequence positions using insert/delete/swap/replace; zero-effect mutations are corrected by a forced token substitution.
- The old **85% per-side mutations** and **9% whole-rule random restarts** during trial generation have been removed.
- Mutation is a **proposal only**. Both population and global-incumbent replacement still require strict Pareto dominance: `(rho >= previousRho && inference >= previousInference) && (rho > previousRho || inference > previousInference)`.
- Initial population diversification is unchanged (mutated seed rules and occasional random rules); the **5% rate only governs trials inside the DE iterations**.

`--mutation-rate X` accepts `0.0..1.0` and defaults to `0.05`. `0` disables mutation *during DE proposals*; `1` applies one mutation to every proposal. The chosen value is persisted with checkpoint parameters.

For a previously saved checkpoint from the same **one-pass, skip-oversize** edition, `--resume` accepts the missing `mutationRate` field and uses the selected new rate from that point onward. The fitness function has not changed. Checkpoints for older two-pass or cropping-based editions remain incompatible.

## Run

```sh
nim c -d:release --opt:speed main.nim
./main --self-test --outdir smoke --mutation-rate 0.05
./main --corpus github-code.txt --rules 16 --population 40 --iterations 40 --mutation-rate 0.05 --outdir run16 --seed 42
python3 test_optimized.py
python3 test_mutation_integration.py
```

Requires `replace.nim` and `ridge.nim`, included. `gnuplot` is optional for PNG rendering. The corpus is not included.

The built-in Nim self-test now includes checks for mutation rate 0 vs 1 and preservation of original donor rules; the Python script checks source integration without replacing compilation/runtime tests. Actual Nim compilation was not available in the packaging environment.
