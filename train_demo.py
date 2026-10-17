"""Learn [a, wildcard, b] -> [capture0, capture0], including rule lengths."""
import json
from pathlib import Path
import torch
from replace_exact import ExactReplace, LearnableSequence, outcome_nll, discrete_replace, discreteness_penalty, entropy_coefficient

def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    a_supports = [[97, 98, 256]] * 3
    b_supports = [[99, 100, 256]] * 2
    A = LearnableSequence(a_supports, 257)
    B = LearnableSequence(b_supports, 257)
    # A tiny seeded symmetry break; no target-specific initialization.
    with torch.no_grad():
        for p in list(A.parameters()) + list(B.parameters()):
            p.add_(0.03 * torch.randn_like(p))
    examples = [(b'acb', b'cc'), (b'adb', b'dd'), (b'accb', b'cccc')]
    data = []
    for raw, target in examples:
        plan = ExactReplace([[c] for c in raw], a_supports, b_supports, n=1)
        X = torch.nn.functional.one_hot(torch.tensor(list(raw)), 256).double()
        rho = torch.nn.functional.one_hot(torch.tensor(len(raw)), len(raw) + 1).double()
        data.append((plan, X, rho, target))
    opt = torch.optim.Adam(list(A.parameters()) + list(B.parameters()), lr=0.08)
    history = []
    best_discrete = None
    decode = lambda s: tuple(c if c < 256 else -(c - 256) - 1 for c in s)
    for step in range(401):
        pa, ra = A()
        pb, rb = B()
        task_loss = torch.stack([outcome_nll(plan.forward_distributions(X, rho, pa, ra, pb, rb), target)
                            for plan, X, rho, target in data]).mean()
        penalty = (discreteness_penalty(A) + discreteness_penalty(B)) / 2
        coefficient = entropy_coefficient(step)
        loss = task_loss + coefficient * penalty
        # This demo measures only its training examples, not generalization.
        if step % 20 == 0:
            da, db = decode(A.decode()), decode(B.decode())
            errors = sum(discrete_replace(raw, da, db) != tuple(target) for raw, target in examples)
            candidate = {'pattern': da, 'replacement': db, 'training_errors': errors, 'step': step}
            if best_discrete is None or errors < best_discrete['training_errors']:
                best_discrete = candidate
        if step % 100 == 0:
            record = {'step': step, 'sequence_nll': float(task_loss.detach()),
                      'entropy_penalty': float(penalty.detach()), 'entropy_coefficient': coefficient,
                      'decoded_training_errors': errors,
                      'pattern_expected_length': float(A.expected_length().detach()),
                      'replacement_expected_length': float(B.expected_length().detach())}
            history.append(record)
            print(json.dumps(record))
        if step < 400:
            opt.zero_grad()
            loss.backward()
            opt.step()
    params = tuple(A.parameters()) + tuple(B.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # A real Hessian-vector product of the same smooth objective; no STE.
    vectors = tuple(torch.ones_like(p) * (i + 1) / p.numel() for i, p in enumerate(params))
    # Nonconstant directions within each categorical row avoid softmax shift invariance.
    vectors = tuple(v * torch.arange(1, v.numel() + 1, dtype=v.dtype).reshape_as(v) for v in vectors)
    directional = sum((g * v).sum() for g, v in zip(grads, vectors))
    hvp = torch.autograd.grad(directional, params)
    print(json.dumps({'hessian_vector_norm': float(torch.sqrt(sum(h.square().sum() for h in hvp)))}))
    decode = lambda s: tuple(c if c < 256 else -(c - 256) - 1 for c in s)
    a, b = decode(A.decode()), decode(B.decode())
    correct = all(discrete_replace(raw, a, b) == tuple(target) for raw, target in examples)
    print(json.dumps({'discrete_pattern': a, 'discrete_replacement': b,
                      'all_training_examples_exact': correct}))
    if not correct:
        raise RuntimeError('learning example failed')
    artifact = dict(best_discrete)
    artifact.update(min_wildcard_len=0, max_output=1048576,
                    semantics='replace_fixed.nim byte/capture subset; unmatched input is preserved',
                    selection_data='three training examples; no held-out claim')
    Path(__file__).with_name('learned_rule.json').write_text(json.dumps(artifact, indent=2) + '\n')
    return history

if __name__ == '__main__':
    main()
