"""Exact *joint sequence-law* pushforward for serial old-Nim byte Replace.

This implementation DOES NOT factorise random output bytes, sample, detach/prune
low-probability worlds, or use a mean-field / soft-greedy surrogate. The law is
represented by all reachable full sequences, with optional complete firing-count
histories. Static finite supports are necessary because general dense categorical
supports grow exponentially. A hard error is raised when a compiled plan exceeds
its explicit computational budget; there is never a silent approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence
import math

import torch
from torch import nn, Tensor

from .old_replace import replace_with_count


def smoothstep5(t: Tensor) -> Tensor:
    z = t.clamp(0, 1)
    return z * z * z * (10 + z * (-15 + 6 * z))


def length_distribution(length: Tensor, capacity: int) -> Tensor:
    """C², valid categorical distribution on lengths 0..capacity."""
    if capacity < 0:
        raise ValueError('capacity must be >= 0')
    if capacity == 0:
        return torch.ones_like(length)[..., None]
    grid = torch.arange(capacity, device=length.device, dtype=length.dtype)
    g = smoothstep5(length[..., None] - grid)
    return torch.cat((1 - g[..., :1], g[..., :-1] - g[..., 1:], g[..., -1:]), -1)


def _supports(rows, vocab: int):
    rows = tuple(tuple(int(z) for z in row) for row in rows)
    for row in rows:
        if not row or len(row) != len(set(row)) or any(z < 0 or z >= vocab for z in row):
            raise ValueError('support must contain unique in-range symbols and cannot be empty')
    return rows


class Operand(nn.Module):
    """Prefix sequence states and their exact independent-symbol weight products."""
    def __init__(self, supports, vocab: int):
        super().__init__()
        self.supports = _supports(supports, vocab)
        self.capacity = len(self.supports)
        self.vocab = vocab
        self.states = tuple(state for k in range(self.capacity + 1)
                            for state in product(*self.supports[:k]))
        # Trie recurrence: each prefix may be generated via one parent and one
        # newly appended token. No per-state dense P x 256 products on GPU.
        for k in range(1, self.capacity + 1):
            suffix = self.supports[k - 1]
            n = math.prod(len(row) for row in self.supports[:k])
            self.register_buffer(f'parent_{k}', torch.arange(n, dtype=torch.long) // len(suffix), persistent=False)
            self.register_buffer(f'channel_{k}', torch.tensor(suffix, dtype=torch.long).repeat(n // len(suffix)), persistent=False)

    def weights(self, probs: Tensor, lengths: Tensor, *, validate=True):
        if probs.shape[-2:] != (self.capacity, self.vocab) or probs.ndim not in (2, 3):
            raise ValueError(f'expected [...,{self.capacity},{self.vocab}] probabilities')
        if lengths.shape[-1] != self.capacity + 1 or lengths.ndim not in (1, 2):
            raise ValueError(f'expected [...,{self.capacity + 1}] length law')
        if probs.device != lengths.device or probs.dtype != lengths.dtype:
            raise ValueError('device/dtype mismatch')
        batched = probs.ndim == 3 or lengths.ndim == 2
        if probs.ndim == 2:
            probs = probs.unsqueeze(0)
        if lengths.ndim == 1:
            lengths = lengths.unsqueeze(0)
        n = torch.broadcast_shapes(probs.shape[:1], lengths.shape[:1])[0]
        probs = probs.expand(n, -1, -1)
        lengths = lengths.expand(n, -1)
        if validate:
            with torch.no_grad():
                if not torch.isfinite(probs).all() or not torch.isfinite(lengths).all():
                    raise ValueError('nonfinite distribution')
                if (probs < 0).any() or not torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-6, rtol=1e-6):
                    raise ValueError('invalid per-position categorical distribution')
                if (lengths < 0).any() or not torch.allclose(lengths.sum(-1), torch.ones_like(lengths[..., 0]), atol=1e-6, rtol=1e-6):
                    raise ValueError('invalid length distribution')
                for k, support in enumerate(self.supports):
                    if probs[..., k, :].sum(-1).sub(probs[..., k, list(support)].sum(-1)).abs().max() > 1e-6:
                        raise ValueError('positive mass outside STATIC supports; expand the supports before compiling')
        prev = probs.sum(dim=(1, 2))[:, None] * 0 + 1
        blocks = [prev * lengths[:, :1]]
        for k in range(1, self.capacity + 1):
            prev = (prev.index_select(1, getattr(self, f'parent_{k}')) *
                    probs[:, k - 1, :].index_select(1, getattr(self, f'channel_{k}')))
            blocks.append(prev * lengths[:, k:k + 1])
        weights = torch.cat(blocks, -1)
        return weights if batched else weights[0]


@dataclass(frozen=True)
class RuleSupport:
    pattern: Sequence[Sequence[int]]
    replacement: Sequence[Sequence[int]]


@dataclass
class RuleDistribution:
    pattern_symbols: Tensor
    pattern_lengths: Tensor
    replacement_symbols: Tensor
    replacement_lengths: Tensor


@dataclass
class ExactLaw:
    """A full correlated distribution on strings, plus optional firing histories."""
    outcomes: tuple[bytes, ...]
    probabilities: Tensor  # [batch, outcome] or [outcome]
    histories: tuple[tuple[int, ...], ...] | None
    initial_lengths: tuple[int, ...] | None

    def text_law(self) -> tuple[tuple[bytes, ...], Tensor]:
        """Marginalise histories without independently marginalising characters."""
        if self.histories is None:
            return self.outcomes, self.probabilities
        keys = tuple(dict.fromkeys(self.outcomes))
        lookup = {v: i for i, v in enumerate(keys)}
        indices = torch.tensor([lookup[v] for v in self.outcomes], device=self.probabilities.device)
        p = self.probabilities
        if p.ndim == 1:
            return keys, p.new_zeros(len(keys)).scatter_add(0, indices, p)
        return keys, p.new_zeros((p.shape[0], len(keys))).scatter_add(1, indices[None].expand(p.shape[0], -1), p)

    def marginal_bytes(self, *, output_capacity=None) -> tuple[Tensor, Tensor, Tensor]:
        """Return byte marginals and position presence, NEVER as full joint law.

        For absent positions the output rows use uniform padding. Do not feed
        these back to further Replace operations; that would lose correlations.
        """
        strings, p = self.text_law()
        p = p if p.ndim == 2 else p.unsqueeze(0)
        capacity = max((len(s) for s in strings), default=0) if output_capacity is None else output_capacity
        if any(len(s) > capacity for s in strings):
            raise ValueError('output_capacity would truncate a reachable outcome')
        flat_index, sources, positions = [], [], []
        for i, s in enumerate(strings):
            for j, c in enumerate(s):
                flat_index.append(j * 256 + c)
                sources.append(i)
                positions.append(j)
        ix = torch.tensor(flat_index, device=p.device, dtype=torch.long)
        src = torch.tensor(sources, device=p.device, dtype=torch.long)
        pos = torch.tensor(positions, device=p.device, dtype=torch.long)
        weights = p.index_select(1, src)
        presence = p.new_zeros((p.shape[0], capacity)).scatter_add(1, pos[None].expand(p.shape[0], -1), weights)
        out = p.new_zeros((p.shape[0], capacity * 256)).scatter_add(1, ix[None].expand(p.shape[0], -1), weights)
        joint = out.reshape(p.shape[0], capacity, 256) + (p.sum(-1, keepdim=True) - presence)[..., None] / 256
        expected_len = (p * p.new_tensor([len(s) for s in strings])).sum(-1)
        if self.probabilities.ndim == 1:
            return joint[0], presence[0], expected_len[0]
        return joint, presence, expected_len

    def feature_tensor(self) -> Tensor:
        if self.histories is None or self.initial_lengths is None:
            raise ValueError('set track_features=True when compiling the engine')
        return self.probabilities.new_tensor([
            [c / max(1, initial) for c in hist]
            for initial, hist in zip(self.initial_lengths, self.histories)
        ])

    def expected_features(self) -> Tensor:
        z = self.feature_tensor()
        return self.probabilities @ z

    def expected_head(self, head: nn.Module, *, per_outcome=False) -> Tensor:
        """Exactly E[head(firing-count vector)], including nonlinear heads."""
        features = self.feature_tensor()
        prediction = head(features)
        if prediction.shape[0] != len(self.outcomes):
            raise ValueError('head needs one value per sequence/history')
        if per_outcome:
            return prediction
        if prediction.ndim == 1:
            return self.probabilities @ prediction
        return self.probabilities @ prediction


class ExactSerialPushforward(nn.Module):
    """Compiled exact serial transition law of the OLD Replace interpreter.

    Each rule's (pattern, replacement) is drawn ONCE PER STRING from its joint
    categorical law, then applied to ALL of that string. This is *not* a new
    independent sample at every replacement occurrence.

    If track_features=True, outcome states include every prior firing count and
    initial sequence length. Merging by text alone would corrupt nonlinear
    prediction heads using these features. An exact trainable predictor can use
    ``law.expected_head(head)``; using head(law.expected_features()) is not exact.
    """
    def __init__(self, x_supports, rules: Sequence[RuleSupport], *, n=15,
                 min_wildcard_len=0, max_output=1048576, track_features=False,
                 max_worlds=100_000, max_states=100_000, max_total_worlds=500_000):
        super().__init__()
        if n < 1 or n > 15:
            raise ValueError('n must be in 1..15 (old opcode -1..-15)')
        if min_wildcard_len < 0 or max_output < len(x_supports):
            raise ValueError('invalid min_wildcard_len/max_output')
        self.x = Operand(x_supports, 256)
        self.patterns = nn.ModuleList()
        self.replacements = nn.ModuleList()
        self.rules = tuple(rules)
        self.track_features = bool(track_features)
        self.max_worlds = max_worlds
        self.max_states = max_states
        self.max_total_worlds = max_total_worlds
        self._max_output = max_output
        self._min_wildcard_len = min_wildcard_len
        for rule in rules:
            self.patterns.append(Operand(rule.pattern, 256 + n))
            self.replacements.append(Operand(rule.replacement, 256 + n))
        # In history-preserving mode, a full state is (bytes, initial length,
        # count-history); otherwise, only the output bytes are kept.
        layer = tuple((bytes(x), len(x), ()) if self.track_features else bytes(x)
                      for x in self.x.states)
        if len(layer) > max_states:
            raise ValueError(f'initial states={len(layer)} exceeds max_states={max_states}')
        self.layer_sizes = [len(layer)]
        self._layer_states = [layer]
        total_worlds = 0
        self.world_counts = []
        for i, (aplan, bplan) in enumerate(zip(self.patterns, self.replacements)):
            world_count = len(layer) * len(aplan.states) * len(bplan.states)
            total_worlds += world_count
            if world_count > max_worlds or total_worlds > max_total_worlds:
                raise ValueError(f'Exact pushforward would require {world_count:,} transition worlds '
                                 f'in rule {i} (total {total_worlds:,}); limits are '
                                 f'{max_worlds:,} per rule and {max_total_worlds:,} total. '
                                 'Reduce STATIC supports / lengths / rules; no approximation is performed.')
            self.world_counts.append(world_count)
            na, nb = len(aplan.states), len(bplan.states)
            lookup = {}
            next_layer = []
            targets = []
            for s in layer:
                x = s[0] if self.track_features else s
                for a in aplan.states:
                    for b in bplan.states:
                        # wildcard 256+r -> negative capture opcode -r-1
                        p = tuple(c if c < 256 else -(c - 256) - 1 for c in a)
                        r = tuple(c if c < 256 else -(c - 256) - 1 for c in b)
                        y, count = replace_with_count(x, p, r, min_wildcard_len=min_wildcard_len,
                                                      max_output=max_output)
                        if self.track_features:
                            key = (y, s[1], s[2] + (count,))
                        else:
                            key = y
                        if key not in lookup:
                            lookup[key] = len(next_layer)
                            next_layer.append(key)
                            if len(next_layer) > max_states:
                                raise ValueError(f'Exact pushforward reaches >{max_states:,} distinct states '
                                                 f'at rule {i}; no probability pruning or approximation is performed')
                        targets.append(lookup[key])
            self.register_buffer(f'target_{i}', torch.tensor(targets, dtype=torch.long), persistent=False)
            layer = tuple(next_layer)
            self._layer_states.append(layer)
            self.layer_sizes.append(len(layer))

    def forward(self, X: Tensor, rho_x: Tensor, rules: Sequence[RuleDistribution], *, validate=True) -> ExactLaw:
        if len(rules) != len(self.patterns):
            raise ValueError('wrong number of rule distributions')
        weight = self.x.weights(X, rho_x, validate=validate)
        batched = weight.ndim == 2
        if not batched:
            weight = weight.unsqueeze(0)
        batch = weight.shape[0]
        for i, (spec, a, b) in enumerate(zip(rules, self.patterns, self.replacements)):
            wa = a.weights(spec.pattern_symbols, spec.pattern_lengths, validate=validate)
            wb = b.weights(spec.replacement_symbols, spec.replacement_lengths, validate=validate)
            if wa.ndim == 1: wa = wa.unsqueeze(0)
            if wb.ndim == 1: wb = wb.unsqueeze(0)
            next_batch = torch.broadcast_shapes(weight.shape[:1], wa.shape[:1], wb.shape[:1])[0]
            # Dense full-world tensor is necessarily exponential. SIMD sum
            # over that tensor is differentiable to all orders in PyTorch.
            worlds = (weight.expand(next_batch, -1)[:, :, None, None] *
                      wa.expand(next_batch, -1)[:, None, :, None] *
                      wb.expand(next_batch, -1)[:, None, None, :]).flatten(1)
            idx = getattr(self, f'target_{i}').unsqueeze(0).expand(next_batch, -1)
            weight = worlds.new_zeros((next_batch, self.layer_sizes[i + 1])).scatter_add(1, idx, worlds)
        if not batched and batch == 1 and all(spec.pattern_symbols.ndim == 2 and spec.replacement_symbols.ndim == 2 and spec.pattern_lengths.ndim == 1 and spec.replacement_lengths.ndim == 1 for spec in rules):
            weight = weight[0]
        layer = self._layer_states[-1]
        if self.track_features:
            return ExactLaw(tuple(state[0] for state in layer), weight,
                            tuple(state[2] for state in layer), tuple(state[1] for state in layer))
        return ExactLaw(tuple(layer), weight, None, None)


class TrainableRule(nn.Module):
    """Logits for independently learned position tokens and length laws.

    Supports MUST be static. It is an explicit limitation of exact state-space
    compilation, rather than an implicit top-k/argmax/zero-probability prune.
    """
    def __init__(self, spec: RuleSupport, *, dtype=torch.float64):
        super().__init__()
        self.spec = spec
        self.pattern_logits = nn.ParameterList([nn.Parameter(torch.zeros(len(r), dtype=dtype)) for r in spec.pattern])
        self.replacement_logits = nn.ParameterList([nn.Parameter(torch.zeros(len(r), dtype=dtype)) for r in spec.replacement])
        self.pattern_length_logits = nn.Parameter(torch.zeros(len(spec.pattern)+1, dtype=dtype))
        self.replacement_length_logits = nn.Parameter(torch.zeros(len(spec.replacement)+1, dtype=dtype))
        for kind, rows in (('p', spec.pattern), ('r', spec.replacement)):
            for i, row in enumerate(rows):
                self.register_buffer(f'{kind}_indices_{i}', torch.tensor(row, dtype=torch.long), persistent=False)

    def _build(self, logits, tag, vocab, temperature):
        rows = []
        for i, l in enumerate(logits):
            channels = getattr(self, f'{tag}_indices_{i}')
            p = (l / temperature).softmax(-1)
            rows.append(l.new_zeros(vocab).scatter(0, channels, p))
        # For empty operand, return proper empty [0,vocab] on same device/dtype.
        return torch.stack(rows) if rows else self.pattern_length_logits.new_zeros((0, vocab))

    def forward(self, temperature=1., *, n=15):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        return RuleDistribution(
            self._build(self.pattern_logits, 'p', 256+n, temperature),
            (self.pattern_length_logits / temperature).softmax(-1),
            self._build(self.replacement_logits, 'r', 256+n, temperature),
            (self.replacement_length_logits / temperature).softmax(-1))


class TrainableSerial(nn.Module):
    """Exact serial operator for a chosen static input/rule support topology."""
    def __init__(self, x_supports, rules, *, n=15, track_features=True,
                 max_worlds=100_000, max_states=100_000, max_total_worlds=500_000,
                 dtype=torch.float64):
        super().__init__()
        self.engine = ExactSerialPushforward(x_supports, rules, n=n, track_features=track_features,
                    max_worlds=max_worlds, max_states=max_states, max_total_worlds=max_total_worlds)
        self.rules = nn.ModuleList([TrainableRule(r, dtype=dtype) for r in rules])
        self.n = n

    def forward(self, X, rho_x, temperature=1., validate=True):
        return self.engine(X, rho_x, [r(temperature, n=self.n) for r in self.rules], validate=validate)


def deterministic_text(x: bytes, *, device='cpu', dtype=torch.float64):
    """Return suitable empty/one-hot coordinates for a deterministic byte string."""
    x = bytes(x)
    p = torch.nn.functional.one_hot(torch.tensor(list(x), device=device, dtype=torch.long), 256).to(dtype)
    rho = p.new_zeros(len(x)+1)
    rho[-1] = 1.
    return p, rho, tuple((c,) for c in x)
