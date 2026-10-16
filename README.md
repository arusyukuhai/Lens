# Lens Dual-Grid + Online Progressive Pruning

This release is **based on `Lens_DualGrid_Fast_FineEveryGen.zip`**, preserving dual-grid diffusion, per-generation Fine candidate reranking, periodically plotted eval, Ridge regression and the fingerprint/delta suffix implementations. The new optimization progressively culls candidate *patterns* before costly all-row match counting. See `PROGRESSIVE_DESIGN.md` for details, safeguards, and known limitations.

## Usage

```bash
nim c -d:release --opt:speed -d:lensProfile -o:lens cms.nim
./lens --self-test
./lens --coarse-power=2 --fine-power=1.5 --eval-power=2 \
  --eval-every=5 --fine-top=12 --delta-suffix=1 \
  --progressive=1 --progressive-start=64 \
  --progressive-keep=24 --progressive-factor=0.5 \
  --steps=100 --out=lens_progressive
```

To directly compare against the old candidate evaluation, run another instance with `--progressive=0` and a separate `--out` directory. The new **pruning heuristic** can change which rules are discovered; exact Ridge scoring is retained for survivors only.

**Validation notice:** Numerical Python tests and source-level checks passed; no Nim compiler was available to execute `nim c` or `--self-test` here.
