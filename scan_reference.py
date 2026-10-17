"""Previous scan, retained only for regression and benchmark comparison."""
import torch

def reach_doubling(events):
    """Ready-state probabilities for nonoverlapping variable-span events.
    events[B,L,S]: probability of consuming d tokens in column d-1.
    Column-stochastic S-state automaton, associative prefix product in O(log L)
    dependent steps. O(B L S^3 log L) work; no per-position Python loop.
    """
    b, length, spans = events.shape
    first = torch.cat((1-events[..., 1:].sum(-1, keepdim=True), events[..., 1:]), -1)
    tail = torch.eye(spans, device=events.device, dtype=events.dtype)[:, :-1]
    tail = tail.expand(b, length, spans, spans-1)
    a = torch.cat((first[..., None], tail), -1)
    stride = 1
    while stride < length:
        a = torch.cat((a[:, :stride], a[:, stride:] @ a[:, :-stride]), 1)
        stride *= 2
    return torch.cat((torch.ones_like(events[:, :1, 0]), a[:, :-1, 0, 0]), 1)

