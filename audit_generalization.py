#!/usr/bin/env python3
"""Paired, held-out corpus audit with frozen readouts (no fitting on audit data)."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import main as m


def load_model(path):
    data = json.loads(Path(path).read_text())
    rules = [m.Rule(r['a'], r['b']) for r in data['rules']]
    return m.Genome(rules, embedding=data['embedding'],
                    readout_weights=[float(r['weight']) for r in data['rules']])


def run(args):
    random.seed(args.seed); np.random.seed(args.seed)
    corpus = m.load_local_corpus(args.local_corpus, 1000000, args.min_chunk, args.max_chunk)
    if len(corpus) < 2:
        raise ValueError('Audit needs at least two corpus chunks; use a held-out corpus.')
    sampler = m.CorpusSampler(corpus)
    inputs, target, cases, samples = m.build_dataset(corpus, sampler, args.cases, args.samples, args.max_noise, args.max_chunk)
    result = {'seed': args.seed, 'cases': len(np.unique(cases)), 'samples': args.samples,
              'method': 'frozen-readout, identical held-out trajectories for every model', 'models': []}
    for path in args.models:
        g = load_model(path)
        if args.backend == 'mps':
            if m.torch is None or not m.torch.backends.mps.is_available():
                raise RuntimeError('MPS requires PyTorch and an Apple GPU')
            ev = m.MpsPopulationEvaluator(len(inputs), len(g.rules), max(map(len, inputs)), args.max_output, genome_batch=1, progress=False)
            ev.set_inputs(inputs)
            x, baseline, _ = ev.evaluate_population([g])[0]
            del ev
        else:
            x, baseline, _, _ = m.trajectory_features_cpu(inputs, g, args.max_output)
        # Absolutely no label-dependent fit or calibration on this corpus.
        w = np.asarray(g.readout_weights)
        active = np.flatnonzero(w)
        pred = baseline + x[:, active].astype(np.float64) @ w[active]
        scores = [m.spearman(pred[cases == c], target[cases == c]) for c in np.unique(cases)]
        result['models'].append({'path': path, 'mean_spearman': float(np.mean(scores)), 'case_scores': scores})
    if len(result['models']) == 2:
        delta = np.subtract(result['models'][1]['case_scores'], result['models'][0]['case_scores'])
        rng = np.random.default_rng(args.seed)
        boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
        result['paired_difference_second_minus_first'] = {
            'mean': float(delta.mean()), 'bootstrap_95_percent': np.quantile(boot, [.025, .975]).tolist()}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--local-corpus', required=True, help='corpus excluded from training')
    p.add_argument('--backend', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--cases', type=int, default=32)
    p.add_argument('--samples', type=int, default=36)
    p.add_argument('--max-noise', type=float, default=.35)
    p.add_argument('--min-chunk', type=int, default=96)
    p.add_argument('--max-chunk', type=int, default=4000)
    p.add_argument('--max-output', type=int, default=m.MAX_OUTPUT)
    p.add_argument('--seed', type=int, default=99173)
    p.add_argument('--output', default='')
    run(p.parse_args())
