"""Predict intactness. --integer uses CPU integer serial rewrites + learned MLP."""
import argparse,json
from pathlib import Path
import torch
if __package__:
    from .model import Lens,Config,discrete_features
else:
    from model import Lens,Config,discrete_features
if __package__:
    from .train import select_device
else:
    from train import select_device

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint'); p.add_argument('input'); p.add_argument('--device',default='auto')
p.add_argument('--integer',action='store_true'); p.add_argument('--soft',action='store_true')
p.add_argument('--capacity-factor',type=float,default=2.,help='tensor capacity multiplier for growing RHS; use --integer for unbounded exact rewrites')
a=p.parse_args()
device=select_device(a.device)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
if ck['config'].get('format_version')!=7:
    raise ValueError('format_version=7 unbounded-DP checkpoint required (use upgrade_weights to migrate v6 weights)')
m=Lens(Config(**ck['config'])).to(device).eval(); m.load_state_dict(ck['model'])
raw=Path(a.input).read_bytes()
if not raw: raise ValueError('empty input')
with torch.no_grad():
    if a.integer:
        features,state=discrete_features(raw,m.export_rules())
        f=torch.tensor([features],device=device,dtype=torch.float32)
        pred=m.head(torch.log1p(m.config.feature_gain*f)).item()
    else:
        import math
        capacity=max(2,math.ceil(len(raw)*a.capacity_factor))
        tok=torch.tensor([list(raw)+[0]*(capacity-len(raw))],device=device)
        length=torch.tensor([len(raw)],device=device)
        pred=m(tok,length,temperature=ck.get('args',{}).get('temperature',1.),hard=not a.soft).item()
print(json.dumps({'prediction':pred,'bytes':len(raw),'mode':'old-Replace exact CPU rewrites + head' if a.integer or not a.soft else 'GPU unbounded-capture DP mean-field surrogate + head'}))
