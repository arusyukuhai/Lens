"""Predict intactness. --integer uses CPU integer serial rewrites + learned MLP."""
import argparse,json
from pathlib import Path
import torch
from model import Lens,Config,discrete_features
from train import select_device

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint'); p.add_argument('input'); p.add_argument('--device',default='auto')
p.add_argument('--integer',action='store_true'); p.add_argument('--soft',action='store_true')
a=p.parse_args()
device=select_device(a.device)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
m=Lens(Config(**ck['config'])).to(device).eval(); m.load_state_dict(ck['model'])
raw=Path(a.input).read_bytes()
if not raw: raise ValueError('empty input')
with torch.no_grad():
    if a.integer:
        features,state=discrete_features(raw,m.export_rules())
        f=torch.tensor([features],device=device,dtype=torch.float32)
        pred=m.head(torch.log1p(m.config.feature_gain*f)).item()
    else:
        tok=torch.tensor([list(raw)+[0]*max(0,3-len(raw))],device=device)
        length=torch.tensor([len(raw)],device=device)
        pred=m(tok,length,temperature=ck['args']['temperature'],hard=not a.soft).item()
print(json.dumps({'prediction':pred,'bytes':len(raw),'mode':'integer rewrites + float MLP' if a.integer else 'soft' if a.soft else 'hard'}))
