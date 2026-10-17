"""Reload the learned soft pipeline; evaluate full-support uncertain input."""
import argparse,json,random
from pathlib import Path
import torch
from soft_pipeline import SoftReplaceLayer,SoftReplacePipeline
from train_soft_long import families,document,encode,objective

p=argparse.ArgumentParser()
p.add_argument('results',help='soft_parametric_long_results.json with adjacent .pt')
p.add_argument('--length',type=int,default=4096)
p.add_argument('--noise',type=float,default=.1)
a=p.parse_args()
torch.set_num_threads(1)
path=Path(a.results);r=json.loads(path.read_text());cfg=r['config']
f=families();rng=random.Random(59)
for family in f:rng.shuffle(family)
model=SoftReplacePipeline([SoftReplaceLayer(family,min_capture=1,max_capture=3,
                          trainable_literals=cfg.get('learn_bytes',False),
                          match_power=cfg.get('match_power',2)) for family in f])
model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True))
stages=document(990,a.length);X,l=encode(stages[0],a.noise)
with torch.no_grad():Y,ol=model(X,l)
n=len(stages[-1]);target=torch.tensor(list(stages[-1]))
print(json.dumps(dict(input_bytes=len(stages[0]),target_bytes=n,noise=a.noise,
                     position_accuracy=float((Y[0,:n].argmax(-1)==target).float().mean()),
                     predicted_length=float(ol),loss=float(objective(Y,ol,stages[-1])))))
