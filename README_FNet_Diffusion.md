# FNet full-trajectory SoftRank diffusion training (v2)

Replace the **old** trainer and `fnet.py` with the two files in this archive. Put `github-code-dataset.txt` in the same folder; the code reads every nonempty `===SPLIT===`-delimited chunk as UTF-8 **bytes**. Chunks are divided into disjoint training and validation sets.

```bash
python -m pip install torch numpy matplotlib
python train_fnet_diffusion.py --data github-code-dataset.txt --device mps \
  --max-tokens 256 --d-model 128 --layers 4 \
  --diffusion-steps 64 --schedule random --eval-every 50
```

## Design

**One entire trajectory is one batch.** For one source window, create its entire sequence of increasingly corrupted versions on CPU, transfer this `[num_stages + 1, seq_len]` tensor to MPS, run **one FNet forward**, combine losses, perform **one backward** and **one AdamW update**. No per-pair GPU forwards, gradient accumulation, microbatches, or `retain_graph=True`. The next window gets a fresh trajectory and a fresh update.

Random schedules choose one of two styles **for every training chunk**:

- `rapid`: relatively few stages, with strongly varying numbers of bytes corrupted per transition; large jumps are common.
- `gradual`: many stages, with smaller, more nearly equal corruption increments.

There is also randomness **within** each style. All stages strictly increase the number of changed bytes, and replacement bytes are guaranteed different from the source. The `--diffusion-steps 64` setting caps the maximum number of transitions rather than forcing 64 transitions in every trajectory. `--diffusion-steps 0` makes each byte a separate transition and overrides the rapid/gradual mix. `--schedule rapid` and `--schedule gradual` can be set for ablation tests. The actual number of states can be smaller than this limit when the source window is shorter.

**Main loss:** differentiable Soft Rank followed by Pearson correlation against the true increasing noise-stage ranks:

```
soft_rank_i = sum_j sigmoid((normalized_score_i-normalized_score_j) / rank_temp)
rank_loss   = 1 - Pearson(soft_rank, [0, 1, ..., stages])
```

The implementation uses standard PyTorch operations rather than depending on torchsort's compiled extension; it runs on MPS and has O(num_states²) rank-comparison memory. Predicted scores are standardized with a minimum standard deviation to prevent giant gradients at initialization; default `--rank-temp 0.75` and `--rank-min-std 0.05` can be tuned. A perfect score ordering is *not guaranteed to make the relaxed loss exactly zero* because it is a differentiable approximation.

**Auxiliary loss:** `BCEWithLogits` predicting which byte positions have been changed, using a linear head on hidden states **immediately before global average pooling**. Global pooling of the same hidden states feeds a scalar noise-severity score. Combined objective:

```
loss = rank_loss + 0.1 * byte_noise_BCE
```

The FNet Fourier transform uses `norm="ortho"` to avoid the amplification and potential NaNs of an unnormalized FFT. Default dropout=0 to make comparisons along one trajectory coherent. Nonfinite loss and gradients trigger an explicit exception rather than poisoning checkpoints. Gradient norm clipping is on by default.

## Monitoring and checkpoints

The output directory is `fnet_diffusion_fullbatch/` by default:

- `spearman.png`: true (non-relaxed) Spearman correlation on fixed **train and held-out validation** trajectories; a second zoomed panel shows `1 - Spearman` on a log scale; third panel shows both train losses and validation auxiliary BCE.
- `history.csv`: evaluation metrics by optimizer step. At step 0, training losses are empty (not NaN): no training has happened yet.
- `best.pt`: highest *validation* Spearman checkpoint.
- `last.pt`: latest checkpoint, with optimizer state and diagnostic history.

Eval schedules use fixed RNG seeds, so the trajectories do not change each time, which makes curves comparable. `--eval-schedule random` uses a fixed mix of the two styles. `--eval-train-chunks 3`, `--eval-val-chunks 6` control the number of diagnostic chunks. `--eval-every 50` controls plot frequency. Fixed train evaluation windows are included in each epoch's training sample.

## Speed, memory and numerical considerations

**The full-trajectory batch consumes memory in proportion to (number of stages) × (FFT-padded sequence length) × (model width) × (number of FNet layers)**, plus the O(stages²) soft-rank matrix. This version is much faster than one forward/backward per adjacent pair, but the full chain must fit in accelerator memory. If out of memory, first lower `--diffusion-steps` to 32 or 16, then `--max-tokens` to 128, then `--d-model` or `--layers`. Unlike the previous version, it intentionally does *not* use backward-per-pair memory limiting.

For MPS, `--d-model` must be a power of 2. Sequence lengths are zero-padded to the next power of 2 internally for the FFT. Real MPS hardware was not available in the test environment, so MPS execution must be verified on your Mac. A CPU smoke test verified one full-trajectory forward/backward, evaluation, loss reporting, and PNG/CSV checkpoint creation.

```bash
# Faster / smaller
python train_fnet_diffusion.py --device mps --max-tokens 128 --d-model 128 \
  --layers 3 --diffusion-steps 32 --eval-every 100

# Full bytewise schedule (memory intensive)
python train_fnet_diffusion.py --device mps --diffusion-steps 0

# Resume a new-format v2 checkpoint with the SAME model architecture options
python train_fnet_diffusion.py --device mps \
  --resume fnet_diffusion_fullbatch/last.pt
```

**Compatibility:** The v1 per-pair trainer and v2 SoftRank trainer use different objectives and checkpoint metadata. The new trainer explicitly refuses v1 checkpoints; start in a fresh output directory. The supplied `fnet.py` encoder architecture is compatible at the module level, but reusing old trained weights is deliberately not supported by `--resume`.
