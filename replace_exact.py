"""Exact finite-support C2 extension of the corrected Nim byte Replace.

No sampling, soft matching, detached probability pruning, or straight-through
estimators. Compile a STATIC categorical support, then reuse the tensor plan.
The combinatorial compiler is exponential: this is a small-problem reference.
"""
from dataclasses import dataclass
from itertools import product
import math
import torch
from torch import Tensor, nn


from discrete_replace import discrete_replace


def smoothstep5(t: Tensor) -> Tensor:
    # Clamp is safe here: the polynomial has zero first AND second derivative
    # at both junctions. Do not replace this by rounding or a detached gate.
    z = t.clamp(0, 1)
    return z.pow(3) * (10 + z * (-15 + 6 * z))


def length_distribution(length: Tensor, capacity: int) -> Tensor:
    """C2 cardinal interpolation; length is a coordinate, not E[length]."""
    if capacity < 0:
        raise ValueError('negative capacity')
    if capacity == 0:
        return (length * 0 + 1).reshape(1)
    g = smoothstep5(length - torch.arange(capacity, device=length.device,
                                         dtype=length.dtype))
    return torch.cat(((1 - g[:1]), g[:-1] - g[1:], g[-1:]))


def _supports(rows, alphabet):
    rows = tuple(tuple(int(c) for c in row) for row in rows)
    for row in rows:
        if not row or len(set(row)) != len(row):
            raise ValueError('support rows must be nonempty and contain no duplicates')
        if any(c < 0 or c >= alphabet for c in row):
            raise ValueError('support channel outside alphabet')
    return rows


def _state_count(rows):
    count, prefix = 1, 1
    for row in rows:
        prefix *= len(row)
        count += prefix
    return count


class _Operand(nn.Module):
    def __init__(self, rows, alphabet):
        super().__init__()
        self.rows, self.alphabet, self.capacity = rows, alphabet, len(rows)
        self.states = tuple(s for k in range(len(rows) + 1)
                            for s in product(*rows[:k]))
        self.register_buffer('lengths', torch.tensor([len(s) for s in self.states]))
        # One index table per length; never omit current zero probabilities.
        self.groups = []
        for k in range(len(rows) + 1):
            states = tuple(product(*rows[:k]))
            table = torch.tensor(states, dtype=torch.long).reshape(len(states), k)
            name = f'indices_{k}'
            self.register_buffer(name, table)
            self.groups.append(name)
        mask = torch.zeros((len(rows), alphabet), dtype=torch.bool)
        for i, row in enumerate(rows):
            mask[i, list(row)] = True
        self.register_buffer('mask', mask)

    def weights(self, p, length=None, rho=None):
        if rho is None:
            rho = length_distribution(length, self.capacity)
        weights = []
        for k, name in enumerate(self.groups):
            indices = getattr(self, name)
            if k == 0:
                w = p.sum() * 0 + p.new_ones((1,))
            else:
                w = p[torch.arange(k, device=p.device)[None, :], indices].prod(-1)
            weights.append(rho[k] * w)
        return torch.cat(weights)

    def validate(self, p, length):
        if p.shape != (self.capacity, self.alphabet):
            raise ValueError(f'expected probability shape {(self.capacity, self.alphabet)}')
        if length.ndim != 0 or not p.is_floating_point():
            raise ValueError('length must be scalar and probabilities floating point')
        if p.device != self.mask.device or length.device != p.device or length.dtype != p.dtype:
            raise ValueError('move plan, probabilities and lengths to same device/dtype')
        with torch.no_grad():
            if not torch.isfinite(p).all() or not torch.isfinite(length):
                raise ValueError('nonfinite input')
            if (p < 0).any() or not torch.allclose(p.sum(-1), p.new_ones(self.capacity), atol=1e-6, rtol=1e-6):
                raise ValueError('probability rows must be nonnegative and sum to one')
            if p.masked_select(~self.mask).ne(0).any():
                raise ValueError('nonzero probability outside STATIC support')
            if not 0 <= length.item() <= self.capacity:
                raise ValueError('length coordinate outside capacity')


@dataclass
class ReplaceResult:
    probabilities: Tensor  # [output_capacity, 256], uniform padding
    length: Tensor         # expected DISCRETE output length
    presence: Tensor       # per-position existence probabilities
    outcome_weights: Tensor
    outcomes: tuple         # full joint output law, before marginal projection


class ExactReplace(nn.Module):
    """Compile once; forward evaluates the exact finite sum with autograd.

    Support rows are fixed channel indices. For A/B, channel 256+r maps to
    wildcard/capture r. Supports are part of the model, not inferred from p.
    Missing channels are fixed zero; gradients only cover the specified face.
    ALL prefix lengths 0..capacity are compiled, even for integer lengths.
    """
    def __init__(self, x_supports, a_supports, b_supports, *, n=15,
                 min_wildcard_len=0, max_output=1_048_576,
                 output_capacity=None, max_worlds=100_000, max_output_cells=4_000_000):
        super().__init__()
        if not 1 <= n <= 15:
            raise ValueError('n must be 1..15 for the supplied Nim copy opcodes')
        xs = _supports(x_supports, 256)
        aa = _supports(a_supports, 256 + n)
        bb = _supports(b_supports, 256 + n)
        if len(bb) > 64 or min_wildcard_len < 0 or max_output < len(xs):
            raise ValueError('invalid capacity/minimum; max_output must cover input capacity')
        counts = tuple(_state_count(s) for s in (xs, aa, bb))
        self.world_count = math.prod(counts)
        if self.world_count > max_worlds:
            raise ValueError(f'exact enumeration needs {self.world_count:,} worlds '
                             f'(limit {max_worlds:,}); use smaller STATIC supports/capacities')
        self.x = _Operand(xs, 256)
        self.a = _Operand(aa, 256 + n)
        self.b = _Operand(bb, 256 + n)
        self.n = n
        outcomes, lookup, ids = [], {}, []
        decode = lambda s: tuple(c if c < 256 else -(c - 256) - 1 for c in s)
        for x in self.x.states:
            for a in self.a.states:
                for b in self.b.states:
                    y = discrete_replace(x, decode(a), decode(b),
                                         min_wildcard_len=min_wildcard_len,
                                         max_output=max_output)
                    if y not in lookup:
                        lookup[y] = len(outcomes)
                        outcomes.append(y)
                    ids.append(lookup[y])
        self.outcomes = tuple(outcomes)
        needed = max(map(len, outcomes), default=0)
        self.output_capacity = needed if output_capacity is None else output_capacity
        if self.output_capacity < needed:
            raise ValueError(f'output_capacity must be at least {needed}; no silent truncation')
        if self.output_capacity * 256 > max_output_cells:
            raise ValueError('dense output exceeds max_output_cells')
        self.register_buffer('world_to_output', torch.tensor(ids, dtype=torch.long))
        self.register_buffer('output_lengths', torch.tensor([len(y) for y in outcomes], dtype=torch.long))
        sources, flat_indices, positions = [], [], []
        for i, y in enumerate(outcomes):
            for t, c in enumerate(y):
                sources.append(i)
                flat_indices.append(t * 256 + c)
                positions.append(t)
        self.register_buffer('sources', torch.tensor(sources, dtype=torch.long))
        self.register_buffer('flat_indices', torch.tensor(flat_indices, dtype=torch.long))
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.long))

    def forward(self, X, lx, A, la, B, lb, *, validate=True):
        if validate:
            for op, p, length in ((self.x, X, lx), (self.a, A, la), (self.b, B, lb)):
                op.validate(p, length)
            if not X.dtype == A.dtype == B.dtype:
                raise ValueError('all probabilities must share dtype')
        wx, wa, wb = self.x.weights(X, lx), self.a.weights(A, la), self.b.weights(B, lb)
        return self._contract(wx, wa, wb)

    def forward_distributions(self, X, rho_x, A, rho_a, B, rho_b, *, validate=True):
        """Training API: learn categorical length laws directly (no flat gates)."""
        if validate:
            for op, p, rho in ((self.x, X, rho_x), (self.a, A, rho_a), (self.b, B, rho_b)):
                op.validate(p, p.new_zeros(()))
                if rho.shape != (op.capacity + 1,) or rho.dtype != p.dtype or rho.device != p.device:
                    raise ValueError("invalid length distribution shape/device/dtype")
                with torch.no_grad():
                    if not torch.isfinite(rho).all() or (rho < 0).any() or not torch.allclose(rho.sum(), rho.new_tensor(1.), atol=1e-6, rtol=1e-6):
                        raise ValueError("invalid length distribution")
            if not X.dtype == A.dtype == B.dtype:
                raise ValueError("all probabilities must share dtype")
        return self._contract(self.x.weights(X, rho=rho_x),
                              self.a.weights(A, rho=rho_a),
                              self.b.weights(B, rho=rho_b))

    def _contract(self, wx, wa, wb):
        worlds = (wx[:, None, None] * wa[None, :, None] * wb[None, None, :]).reshape(-1)
        q = worlds.new_zeros(len(self.outcomes)).scatter_add(0, self.world_to_output, worlds)
        values = q[self.sources]
        presence = q.new_zeros(self.output_capacity).scatter_add(0, self.positions, values)
        mass = q.new_zeros(self.output_capacity * 256).scatter_add(0, self.flat_indices, values)
        # Use total mass (not hard-coded 1): this also equals the finite polynomial
        # off the probability simplex, which is useful for derivative checking.
        Y = mass.reshape(self.output_capacity, 256) + (q.sum() - presence)[:, None] / 256
        length = (q * self.output_lengths).sum()
        return ReplaceResult(Y, length, presence, q, self.outcomes)


def supported_probabilities(logits: Tensor, supports, alphabet: int) -> Tensor:
    """Equal support size per row; differentiably scatter softmax into full rows."""
    if logits.ndim != 2 or len(supports) != logits.shape[0]:
        raise ValueError('logits/support shape mismatch')
    if any(len(row) != logits.shape[1] for row in supports):
        raise ValueError('this helper requires equal support size per row')
    index = torch.tensor(supports, dtype=torch.long, device=logits.device).reshape(logits.shape)
    return logits.new_zeros((len(supports), alphabet)).scatter(1, index, logits.softmax(-1))


class LearnableSequence(nn.Module):
    """Train symbols and the entire length law, with smooth softmax logits.

    Variable support sizes are allowed; use the same supports in ExactReplace.
    The expected length is available for monitoring, but cannot substitute for
    the full length distribution without losing information.
    """
    def __init__(self, supports, alphabet, *, dtype=torch.float64):
        super().__init__()
        self.supports = _supports(supports, alphabet)
        self.alphabet = alphabet
        self.symbol_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(len(row), dtype=dtype)) for row in self.supports])
        self.length_logits = nn.Parameter(torch.zeros(len(self.supports) + 1, dtype=dtype))
        for i, row in enumerate(self.supports):
            self.register_buffer(f'channels_{i}', torch.tensor(row, dtype=torch.long))

    def forward(self, temperature=1.0):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        rows = [z.new_zeros(self.alphabet).scatter(0, getattr(self, f'channels_{i}'),
                (z / temperature).softmax(-1)) for i, z in enumerate(self.symbol_logits)]
        p = torch.stack(rows) if rows else self.length_logits.new_empty((0, self.alphabet))
        rho = (self.length_logits / temperature).softmax(-1)
        return p, rho

    def expected_length(self):
        rho = self()[1]
        return (rho * torch.arange(len(rho), device=rho.device)).sum()

    @torch.no_grad()
    def decode(self):
        """Deployment only; argmax is deliberately outside the training path."""
        k = int(self.length_logits.argmax())
        return tuple(self.supports[i][int(self.symbol_logits[i].argmax())] for i in range(k))


def outcome_nll(result: ReplaceResult, target) -> Tensor:
    """Exact sequence NLL, retaining copy/length correlations (not token CE)."""
    target = tuple(target)
    try:
        index = result.outcomes.index(target)
    except ValueError as error:
        raise ValueError('target unreachable under this static support') from error
    probability = result.outcome_weights[index]
    # No epsilon or detached clipping changes the objective. In training, use
    # interior logits and float64; a zero probability correctly has infinite NLL.
    return -probability.log()


def normalized_entropy(logits: Tensor, temperature=1.0) -> Tensor:
    """Shannon entropy / log(K), scalar logits vector. C-infinity for finite logits."""
    if logits.ndim != 1 or logits.numel() == 0 or temperature <= 0:
        raise ValueError('nonempty vector and positive temperature required')
    if logits.numel() == 1:
        return logits.sum() * 0
    logp = (logits / temperature).log_softmax(-1)
    return -(logp.exp() * logp).sum() / math.log(logits.numel())


def discreteness_penalty(sequence: LearnableSequence, temperature=1.0) -> Tensor:
    """Mean of length entropy and mean symbol-row entropy, both normalized.

    Deliberately NOT weighted by position presence: entropy alone should not
    reward deleting uncertain positions. Unused rows may become arbitrary.
    This penalty is not a preference for short rules.
    """
    length_h = normalized_entropy(sequence.length_logits, temperature)
    if not sequence.symbol_logits:
        return length_h
    symbol_h = torch.stack([normalized_entropy(z, temperature)
                            for z in sequence.symbol_logits]).mean()
    return (length_h + symbol_h) / 2


def entropy_coefficient(step: int, *, warmup=100, ramp=200, maximum=0.05) -> float:
    """C2 ramp in training time; hyperparameters, not learned quantities."""
    if warmup < 0 or ramp <= 0 or maximum < 0:
        raise ValueError('invalid schedule')
    t = max(0., min(1., (step - warmup) / ramp))
    return maximum * t**3 * (10 + t * (-15 + 6*t))
