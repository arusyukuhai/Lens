"""Migrate v6 Refined wildcard model WEIGHTS to v7 unbounded-DP weights.

Usage: python -m lens_gpu.upgrade_weights old_v6.pt v7_initial_weights.pt

This does not preserve the optimizer or resume the previous training run.
The differentiable transducer has changed, and model retraining is necessary.
"""
import argparse
from pathlib import Path
from dataclasses import asdict
import torch
if __package__:
    from .model import Config,Lens
else:
    from model import Config,Lens


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('destination')
    a=parser.parse_args()
    ck=torch.load(a.source,map_location='cpu',weights_only=True)
    v=ck['config'].get('format_version')
    if v!=6:
        raise ValueError('only v6 -> v7 weight migration is supported')
    cfg=dict(ck['config']);cfg['format_version']=7;cfg['max_capture']=0
    cfg.setdefault('exact_max_output',1048576)
    m=Lens(Config(**cfg))
    m.load_state_dict(ck['model'],strict=True)
    res={'config':asdict(m.config),'model':m.state_dict(),
         'args':dict(ck.get('args') or {}),
         'migration_warning':'v6->v7 weights only; changed soft wildcard computation; retrain',
         'origin':str(Path(a.source))}
    torch.save(res,a.destination)
    print('Saved',a.destination,'\nWarning: optimizer state NOT migrated; retraining required.')

if __name__=='__main__':main()
