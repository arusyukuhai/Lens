#!/usr/bin/env python3
"""Train FNet on complete byte-corruption trajectories as ONE batch per chunk.

The main loss is 1 - Pearson(soft_rank(predicted_scores), true_ranks).
The auxiliary loss predicts whether each byte was corrupted before global pooling.
"""
from __future__ import annotations

import argparse
import csv
import math
import mmap
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fnet import FNet

DELIMITER = b'===SPLIT==='


class ChunkCorpus:
    """Index only chunks that fit max_tokens; never truncate or crop them."""
    def __init__(self, path: str, max_tokens: int):
        self.file = open(path, 'rb')
        if os.fstat(self.file.fileno()).st_size == 0:
            raise ValueError('Dataset file is empty')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.spans = []
        self.skipped_too_long = 0
        self.skipped_too_short = 0
        start = 0
        while True:
            stop = self.mm.find(DELIMITER, start)
            if stop == -1:
                stop = len(self.mm)
            length = stop - start
            if length > max_tokens:
                self.skipped_too_long += 1
            elif length < 2 or not self.mm[start:stop].strip():
                self.skipped_too_short += 1
            else:
                self.spans.append((start, stop))
            if stop == len(self.mm):
                break
            start = stop + len(DELIMITER)
        if len(self.spans) < 2:
            self.close()
            raise ValueError(
                f'Only {len(self.spans)} eligible chunks (2 <= byte length <= '
                f'max_tokens={max_tokens}); need at least two for train/validation. '
                f'Skipped long={self.skipped_too_long}, short/empty={self.skipped_too_short}')

    def __len__(self):
        return len(self.spans)

    def read(self, idx: int) -> np.ndarray:
        start, stop = self.spans[int(idx)]
        return np.frombuffer(self.mm[start:stop], dtype=np.uint8).copy()

    def close(self):
        self.mm.close()
        self.file.close()


def noise_distribution(corpus: ChunkCorpus, train_ids: np.ndarray, mode: str,
                       seed: int) -> np.ndarray:
    if mode == 'uniform':
        return np.full(256, 1/256, dtype=np.float64)
    rng = np.random.default_rng(seed + 711)
    chosen = rng.choice(train_ids, size=min(len(train_ids), 4096), replace=False)
    counts = np.ones(256, dtype=np.float64)
    for idx in chosen:
        a, b = corpus.spans[int(idx)]
        counts += np.bincount(np.frombuffer(corpus.mm[a:b], dtype=np.uint8), minlength=256)
    return counts/counts.sum()


def sample_boundaries(n: int, limit: int, rng: np.random.Generator,
                      mode: str) -> tuple[np.ndarray, str]:
    """Rapid: few, lumpy corruption jumps. Gradual: many, small jumps.

    limit=0 requests every single-byte level (n transitions), regardless of mode.
    Distinct integer bounds are essential: the true ranking must have no duplicate stages.
    """
    if n < 2:
        raise ValueError('Need at least two bytes')
    if limit == 0:
        return np.arange(n+1, dtype=np.int32), 'each-byte'
    maximum = min(n, max(2, limit))
    kind = ('rapid' if rng.random() < 0.5 else 'gradual') if mode == 'random' else mode
    if kind == 'rapid':
        # A handful of steps, some replacing much more text than others.
        # At least 4 transitions when allowed: Spearman on only two states
        # has no useful smooth ranking information.
        lo = min(maximum, max(4, maximum//8))
        hi = min(maximum, max(lo, maximum//3))
        steps = int(rng.integers(lo, hi+1)) if hi > lo else lo
        weights = rng.gamma(shape=0.35, scale=1.0, size=steps)
    else:
        lo = max(2, math.ceil(maximum*0.75))
        steps = int(rng.integers(lo, maximum+1)) if maximum > lo else maximum
        weights = rng.gamma(shape=4.0, scale=1.0, size=steps)
    # Require >=1 byte changed at each stage, with random additional changes
    # apportioned according to the gamma weights. This preserves the intended
    # stage count instead of collapsing many rounded, duplicate boundaries.
    counts = np.ones(steps, dtype=np.int32)
    if n > steps:
        counts += rng.multinomial(n-steps, weights/weights.sum()).astype(np.int32)
    bounds = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return bounds, kind


def trajectory(source: np.ndarray, rng: np.random.Generator,
               probs: np.ndarray, stages: int, schedule: str
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Build every diffusion state at once on CPU, then transfer as one batch.

    Once a position changes, it stays corrupted. Replacement byte is guaranteed
    different from its source byte, so ground-truth mask is unambiguous.
    """
    n = len(source)
    bounds, kind = sample_boundaries(n, stages, rng, schedule)
    perm = rng.permutation(n)
    when = np.empty(n, dtype=np.int32)
    when[perm] = np.arange(n, dtype=np.int32)

    repl = rng.choice(256, size=n, p=probs).astype(np.uint8)
    same = repl == source
    # Vectorized rejection, with a guaranteed different fallback for rare matches.
    for _ in range(4):
        if not same.any():
            break
        repl[same] = rng.choice(256, size=int(same.sum()), p=probs).astype(np.uint8)
        same = repl == source
    if same.any():
        repl[same] = ((source[same].astype(np.uint16)+1) % 256).astype(np.uint8)

    masks = when[None, :] < bounds[:, None]      # [num_states, num_tokens]
    states = np.where(masks, repl[None, :], source[None, :]).astype(np.int64)
    return states, masks.astype(np.float32), bounds, kind


class DiffusionFNet(nn.Module):
    def __init__(self, max_tokens: int, d_model: int, layers: int,
                 expansion: int, dropout: float):
        super().__init__()
        if d_model & (d_model-1):
            raise ValueError('d_model must be a power of two for the MPS FFT')
        self.max_tokens = max_tokens
        self.emb = nn.Embedding(256, d_model)
        self.pos = nn.Embedding(max_tokens, d_model)
        self.encoder = FNet(d_model=d_model, expansion_factor=expansion,
                            dropout=dropout, num_layers=layers)
        self.noise_head = nn.Linear(d_model, 1)
        self.score_head = nn.Linear(d_model, 1)
        self.register_buffer('position_ids', torch.arange(max_tokens), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape is [ALL diffusion states, text length].
        _, length = token_batch.shape
        if length > self.max_tokens:
            raise ValueError(f'Input length={length} > max_tokens={self.max_tokens}')
        h = self.dropout(self.emb(token_batch) + self.pos(self.position_ids[:length])[None])
        fft_length = 1 << (length-1).bit_length()
        if fft_length != length:
            h = F.pad(h, (0, 0, 0, fft_length-length))
        h = self.encoder(h)[:, :length, :]
        noise_logits = self.noise_head(h).squeeze(-1)  # BEFORE GAP
        scores = self.score_head(h.mean(dim=1)).squeeze(-1)  # AFTER GAP
        return scores, noise_logits


def soft_rank_pearson_loss(scores: torch.Tensor, temperature: float,
                           min_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Soft Spearman = Pearson(sigmoid-based ranks, true ranks).

    Uses PyTorch ops only: MPS-friendly, without torchsort compiled extensions.
    Pairwise score comparisons cost O(states^2), but only O(states^2) temporary
    storage, *not* O(states^2 * sequence_length * model_width).
    """
    n = scores.numel()
    if n < 2:
        raise ValueError('Need at least 2 states')
    scores = scores.float()
    # Stabilize the soft rank against tiny initial output differences without
    # changing any pair's ordering. min_std avoids huge early gradients.
    centered = scores - scores.mean()
    scaled = centered / centered.square().mean().sqrt().clamp_min(min_std)
    rank = torch.sigmoid((scaled[:, None] - scaled[None, :]) / temperature).sum(dim=1)
    rank_centered = rank - rank.mean()
    true_rank = torch.arange(n, device=scores.device, dtype=torch.float32)
    true_rank = true_rank - true_rank.mean()
    denom = (rank_centered.square().sum().clamp_min(1e-8).sqrt() *
             true_rank.square().sum().sqrt())
    rho = (rank_centered*true_rank).sum()/denom
    return 1.0 - rho, rho


def true_spearman(scores: np.ndarray) -> float:
    """Exact rank correlation; tied predicted scores receive average ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        return float('nan')
    n = len(scores)
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    _, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
    ranks = np.empty(n, dtype=np.float64)
    for start, count in zip(starts, counts):
        ranks[order[start:start+count]] = start + (count+1)/2
    ranks -= ranks.mean()
    true = np.arange(n, dtype=np.float64) - (n-1)/2
    denom = np.linalg.norm(ranks)*np.linalg.norm(true)
    return float(np.dot(ranks, true)/denom) if denom > 0 else 0.0


def evaluate(model: nn.Module, corpus: ChunkCorpus, chosen: np.ndarray,
             max_tokens: int, seed: int, probs: np.ndarray, stages: int,
             schedule: str, device: torch.device) -> tuple[float, float]:
    model.eval()
    rhos = []
    bces = []
    with torch.inference_mode():
        for idx in chosen:
            rng = np.random.default_rng(seed + int(idx)*134775813)
            src = corpus.read(int(idx))
            if len(src) < 2:
                continue
            tokens, labels, _, _ = trajectory(src, rng, probs, stages, schedule)
            token_batch = torch.from_numpy(tokens).to(device)
            mask_batch = torch.from_numpy(labels).to(device)
            scores, logits = model(token_batch)
            if not bool(torch.isfinite(scores).all() and torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite model output during evaluation; check checkpoint/learning rate')
            rhos.append(true_spearman(scores.float().cpu().numpy()))
            bces.append(float(F.binary_cross_entropy_with_logits(logits, mask_batch).item()))
    model.train()
    if not rhos:
        return float('nan'), float('nan')
    return float(np.mean(rhos)), float(np.mean(bces))


def save_history(rows: list[dict], folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'history.csv').open('w', newline='', encoding='utf8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['step', 'train_spearman', 'val_spearman',
                                               'train_soft_rank_loss', 'train_noise_bce',
                                               'val_noise_bce'])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout='constrained')
    steps = [r['step'] for r in rows]
    for key, name in (('train_spearman', 'Training Spearman'),
                      ('val_spearman', 'Validation Spearman')):
        vals = [r[key] if r[key] != '' else np.nan for r in rows]
        axes[0].plot(steps, vals, marker='o', markersize=2.5, label=name)
        axes[1].plot(steps, [max(1e-7, 1-v) if np.isfinite(v) else np.nan for v in vals],
                     label=name)
    axes[0].set(ylabel='True Spearman rho', ylim=(-0.04, 1.04))
    axes[1].set(ylabel='1 - Spearman (log)', yscale='log')
    axes[2].plot(steps, [r['train_soft_rank_loss'] if r['train_soft_rank_loss'] != '' else np.nan
                         for r in rows], label='Train SoftRank + Pearson loss')
    axes[2].plot(steps, [r['train_noise_bce'] if r['train_noise_bce'] != '' else np.nan
                         for r in rows], label='Train noise BCE')
    axes[2].plot(steps, [r['val_noise_bce'] for r in rows], label='Validation noise BCE')
    axes[2].set(xlabel='Optimizer steps (= number of trained chains)', ylabel='Loss', yscale='log')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc='best')
    fig.suptitle('FNet: full-batch diffusion Soft Rank + Pearson')
    temp = folder/'spearman.tmp.png'
    fig.savefig(temp, dpi=140)
    plt.close(fig)
    os.replace(temp, folder/'spearman.png')


def save_checkpoint(path: Path, model, optimizer, args, rows, step, best_val):
    tmp = path.with_suffix('.tmp')
    torch.save({'version': 2, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best_val_spearman': best_val, 'rows': rows,
                'model_args': {key: getattr(args, key) for key in
                               ('max_tokens', 'd_model', 'layers', 'expansion', 'dropout')}}, tmp)
    os.replace(tmp, path)


def arguments():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', default='github-code-dataset.txt')
    p.add_argument('--output', default='fnet_diffusion_fullbatch')
    p.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--expansion', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--diffusion-steps', type=int, default=64,
                   help='Upper bound on transitions; 0 = replace each byte separately')
    p.add_argument('--schedule', choices=['random', 'rapid', 'gradual'], default='random',
                   help='Random mixes large rapid jumps and fine gradual steps')
    p.add_argument('--eval-stages', type=int, default=32)
    p.add_argument('--eval-schedule', choices=['random', 'rapid', 'gradual'], default='random')
    p.add_argument('--rank-temp', type=float, default=0.75)
    p.add_argument('--rank-min-std', type=float, default=0.05)
    p.add_argument('--aux-weight', type=float, default=1.0)
    p.add_argument('--noise-mode', choices=['unigram', 'uniform'], default='unigram')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--chunks-per-epoch', type=int, default=0,
                   help='0 = all chunks; each chunk/trajectory is one optimizer update')
    p.add_argument('--val-fraction', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--adam-eps', type=float, default=1e-6)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-train-chunks', type=int, default=3)
    p.add_argument('--eval-val-chunks', type=int, default=6)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--resume', default='')
    return p.parse_args()


def main():
    args = arguments()
    if args.max_tokens < 2 or args.diffusion_steps < 0 or args.eval_stages < 0:
        raise ValueError('max-tokens >= 2, diffusion/eval-stages >= 0 required')
    if args.rank_temp <= 0 or args.rank_min_std <= 0 or args.lr <= 0:
        raise ValueError('rank-temp, rank-min-std, lr must be positive')
    if args.eval_every < 1 or args.log_every < 1 or args.epochs < 1 or args.chunks_per_epoch < 0:
        raise ValueError('eval-every, log-every, epochs >= 1; chunks-per-epoch >= 0')
    if not 0 < args.val_fraction < 1 or args.aux_weight < 0:
        raise ValueError('val-fraction must be in (0,1) and aux-weight >= 0')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    corpus = ChunkCorpus(args.data, args.max_tokens)
    try:
        print(f'[data] eligible={len(corpus)} chunks '
              f'(2..{args.max_tokens} bytes); skipped_too_long={corpus.skipped_too_long} '
              f'skipped_short_or_empty={corpus.skipped_too_short}', flush=True)
        all_ids = rng.permutation(len(corpus))
        n_val = max(1, min(len(all_ids)-1, round(len(all_ids)*args.val_fraction)))
        val_ids, train_ids = all_ids[:n_val], all_ids[n_val:]
        probs = noise_distribution(corpus, train_ids, args.noise_mode, args.seed)
        model = DiffusionFNet(args.max_tokens, args.d_model, args.layers,
                              args.expansion, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay, eps=args.adam_eps)
        rows: list[dict] = []
        step = 0
        best_val = -float('inf')
        architecture = {key: getattr(args, key) for key in
                        ('max_tokens', 'd_model', 'layers', 'expansion', 'dropout')}
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            if ckpt.get('version') != 2:
                raise ValueError('Old pairwise checkpoints are incompatible: start a new run')
            if ckpt['model_args'] != architecture:
                raise ValueError('Architecture arguments differ from checkpoint')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            step, rows, best_val = ckpt['step'], ckpt['rows'], ckpt['best_val_spearman']
            for name, p in model.named_parameters():
                if not torch.isfinite(p).all().item():
                    raise FloatingPointError(f'NaN/Inf weights in checkpoint: {name}')
        fixed_train = train_ids[:args.eval_train_chunks]
        fixed_val = val_ids[:args.eval_val_chunks]
        print(f'[config] device={device} train={len(train_ids)} val={len(val_ids)} '
              f'len={args.max_tokens} stages<={args.diffusion_steps or "byte count"} '
              f'schedule={args.schedule} rank_temp={args.rank_temp}', flush=True)
        print(f'[model] params={sum(p.numel() for p in model.parameters()):,}; '
              'each entire diffusion chain is ONE batch and ONE backward/step', flush=True)

        # Accumulate only detached small scalars; no persistent computation graphs.
        rank_acc = torch.zeros((), device=device)
        aux_acc = torch.zeros((), device=device)
        logged_steps = 0

        def record():
            nonlocal best_val, rank_acc, aux_acc, logged_steps
            tr, _ = evaluate(model, corpus, fixed_train, args.max_tokens,
                             args.seed+3000, probs, args.eval_stages,
                             args.eval_schedule, device)
            va, vb = evaluate(model, corpus, fixed_val, args.max_tokens,
                              args.seed+6000, probs, args.eval_stages,
                              args.eval_schedule, device)
            rk = float((rank_acc/logged_steps).item()) if logged_steps else ''
            au = float((aux_acc/logged_steps).item()) if logged_steps else ''
            rows.append({'step': step, 'train_spearman': tr, 'val_spearman': va,
                         'train_soft_rank_loss': rk, 'train_noise_bce': au,
                         'val_noise_bce': vb})
            save_history(rows, out)
            improved = math.isfinite(va) and va > best_val
            if improved:
                best_val = va
            save_checkpoint(out/'last.pt', model, optimizer, args, rows, step, best_val)
            if improved:
                save_checkpoint(out/'best.pt', model, optimizer, args, rows, step, best_val)
            print(f'[eval] step={step} train_spearman={tr:.6f} val_spearman={va:.6f} '
                  f'rank_loss={rk if rk != "" else "n/a"} '
                  f'noise_bce={au if au != "" else "n/a"} val_bce={vb:.5f}', flush=True)
            rank_acc = torch.zeros((), device=device)
            aux_acc = torch.zeros((), device=device)
            logged_steps = 0

        if not rows or rows[-1]['step'] != step:
            record()
        for epoch in range(1, args.epochs+1):
            count = len(train_ids) if args.chunks_per_epoch == 0 else args.chunks_per_epoch
            required = fixed_train[:min(count, len(fixed_train))]
            remaining = count - len(required)
            other = train_ids[len(fixed_train):]
            draw = other if len(other) else train_ids
            sampled = np.concatenate([required, rng.choice(draw, size=remaining,
                                    replace=remaining > len(draw))])
            rng.shuffle(sampled)
            model.train()
            for idx in sampled:
                src = corpus.read(int(idx))
                if len(src) < 2:
                    continue
                tokens, labels, bounds, kind = trajectory(src, rng, probs,
                                                          args.diffusion_steps, args.schedule)
                # Whole corruption series on accelerator in ONE forward.
                token_batch = torch.from_numpy(tokens).to(device)
                label_batch = torch.from_numpy(labels).to(device)
                optimizer.zero_grad(set_to_none=True)
                scores, logits = model(token_batch)
                rank_loss, rho_soft = soft_rank_pearson_loss(
                    scores, args.rank_temp, args.rank_min_std)
                aux_loss = F.binary_cross_entropy_with_logits(logits, label_batch)
                loss = rank_loss + args.aux_weight * aux_loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f'Nonfinite loss at optimizer_step={step} chunk={idx} '
                        f'schedule={kind}; reset from clean checkpoint or lower --lr')
                loss.backward()  # one backward for the ENTIRE trajectory
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                   error_if_nonfinite=False)
                if not torch.isfinite(g).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f'Nonfinite gradient at step={step}, norm={g.item()}; '
                                             'try --lr 3e-5 or smaller model')
                optimizer.step()
                step += 1
                # Keep detached diagnostics on device: do not synchronize MPS
                # twice for every chain simply to print losses later.
                rank_acc = rank_acc + rank_loss.detach()
                aux_acc = aux_acc + aux_loss.detach()
                logged_steps += 1
                # Explicitly release batch and autograd graph before next chain.
                del token_batch, label_batch, scores, logits, loss, rank_loss, aux_loss, rho_soft
                if step % args.log_every == 0:
                    print(f'[train] step={step} epoch={epoch} mode={kind} states={len(bounds)} '
                          f'rank_avg={(rank_acc/max(logged_steps,1)).item():.5f} '
                          f'noise_avg={(aux_acc/max(logged_steps,1)).item():.5f}', flush=True)
                if step % args.eval_every == 0:
                    record()
            if rows[-1]['step'] != step:
                record()
            print(f'[epoch] {epoch}/{args.epochs} completed; best val rho={best_val:.6f}',
                  flush=True)
    finally:
        corpus.close()


if __name__ == '__main__':
    main()
