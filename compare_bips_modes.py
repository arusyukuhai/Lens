#!/usr/bin/env python3
"""Reproducible 9-arm BIPS ablation. No result is assumed to be superior.

Optional --audit-corpus MUST be a separate held-out corpus, never included in
--train-corpus. Each arm uses the same seed, training source and generations;
for rigorous conclusions compare wall-clock-matched runs across several seeds.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-corpus', required=True)
    ap.add_argument('--audit-corpus', default='', help='entirely separate corpus for frozen-readout audit')
    ap.add_argument('--output', default='bips_ablation')
    ap.add_argument('--generations', type=int, default=40)
    ap.add_argument('--population', type=int, default=96)
    ap.add_argument('--elites', type=int, default=40)
    ap.add_argument('--rules', type=int, default=1500)
    ap.add_argument('--cases', type=int, default=3)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--cpu-threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--bips-cegar-budget', type=int, default=12)
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='additional main.py options (put last)')
    args = ap.parse_args()
    if args.population <= args.elites:
        ap.error('population must exceed elites or no new BIPS children are produced')
    if args.cases == 2:
        # Strict wall-work matching: with two cases no split can supply two
        # training originals and a third held-out original. Rebalance ALL
        # six arms, including guided-only, so comparisons stay fair.
        old_total=2*max(3,args.samples)
        args.cases=3
        args.samples=max(3,round(old_total/3))
        print(f'balanced all ablation arms to {args.cases} cases x '
              f'{args.samples} trajectories (was 2x{old_total//2})',flush=True)
    root = Path(args.output).absolute(); root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    arms = {
        'guided_only': ['--bips-share','0','--bips-confirm-budget','0'],
        'bips_unvalidated': ['--bips-relational-share','0','--bips-cegar-budget','0','--bips-confirm-budget','0'],
        'bips_fixed_keys': ['--bips-relational-share','0','--bips-causal-budget','0','--no-bips-cegar-adaptive-probes','--bips-confirm-budget','0','--no-bips-cegar-require-validation-firing'],
        'bips_causal_adaptive': ['--bips-relational-share','0','--bips-causal-budget','2','--bips-cegar-adaptive-probes','--bips-confirm-budget','0','--no-bips-cegar-require-validation-firing'],
        'bips_confirmed': ['--bips-relational-share','0','--bips-causal-budget','2','--bips-cegar-adaptive-probes','--bips-confirm-budget','2','--bips-edit-contrast-weight','0','--bips-cegar-require-validation-firing'],
        'bips_edit_aligned': ['--bips-relational-share','0','--bips-causal-budget','2','--bips-cegar-adaptive-probes','--bips-confirm-budget','2','--bips-edit-contrast-weight','.15','--bips-edit-contrast-pairs','3','--bips-cegar-require-validation-firing'],
        'relational_marker': ['--bips-relational-share','0.5','--bips-relational-goal','marker'],
        'relational_consumer': ['--bips-relational-share','0.5','--bips-relational-goal','consumer'],
        'relational_mixed': ['--bips-relational-share','0.5','--bips-relational-goal','mixed'],
    }
    results=[]
    for name, switches in arms.items():
        out=root/name;out.mkdir(exist_ok=True)
        cmd=[sys.executable,str(code/'main.py'),
             '--backend','cpu-native','--local-corpus',str(Path(args.train_corpus).absolute()),
             '--generations',str(args.generations),'--population',str(args.population),
             '--elites',str(args.elites),'--rules',str(args.rules),
             '--cases',str(args.cases),'--samples',str(args.samples),
             '--cpu-threads',str(args.cpu_threads),'--seed',str(args.seed),
             '--learning-mode','guided','--readout-mode','linear-pearson',
             '--inference-mode','ga','--bips-share','0.35',
             '--bips-cegar-budget',str(args.bips_cegar_budget),
             '--hof-size','4096','--checkpoint-every','0','--no-plot','--no-progress',
             '--save',str(out/'best.json'),'--current-save',str(out/'latest.json'),
             '--history-csv',str(out/'history.csv'),'--checkpoint',str(out/'checkpoint.npz')]
        cmd += switches + list(args.extra)
        start=time.perf_counter()
        with (out/'stdout.txt').open('w') as f:
            completed=subprocess.run(cmd,cwd=str(code),stdout=f,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-start
        report={'arm':name,'seconds':elapsed,'exit_status':completed.returncode}
        history=out/'history.csv'
        if history.exists():
            with history.open(newline='') as f:
                rows=list(csv.DictReader(f))
            if rows:
                last=rows[-1]
                for key in ('best','best_ever','inference_best_ever','bips_added',
                            'bips_prev_improved_d2','bips_prev_improved_d3','bips_prev_improved_d4',
                            'bips_causal_macros_checked','bips_cegar_adaptive_expansions',
                            'bips_confirm_accepted','bips_confirm_checked',
                            'bips_edit_contrast_pairs','bips_edit_contrast_best_gain',
                            'bips_relational_selected','bips_relational_readout_improved',
                            'bips_relational_best_margin_gain','bips_relational_consumer_hit'):
                    report[key]=last.get(key,'')
        results.append(report)
        print(name, 'exit',completed.returncode,'time',round(elapsed,2), flush=True)
    if args.audit_corpus:
        models=[str(root/name/'best.json') for name in arms
                if (root/name/'best.json').exists()]
        if models:
            audit_cmd=[sys.executable,str(code/'audit_generalization.py'),
                       '--models',*models, '--local-corpus',str(Path(args.audit_corpus).absolute()),
                       '--backend','cpu-native','--metric','pearson',
                       '--cpu-threads',str(args.cpu_threads),
                       '--output',str(root/'frozen_audit.json')]
            subprocess.run(audit_cmd,cwd=str(code),check=True,
                           stdout=(root/'audit_stdout.json').open('w'))
    (root/'summary.json').write_text(json.dumps(results,indent=2))
    print('Saved comparison:',root/'summary.json')


if __name__=='__main__':
    run()
