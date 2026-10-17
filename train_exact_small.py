"""Small-support trainable serial Lens using EXACT joint probability pushforward.

This is not a performance substitution for the 1500-rule mean-field GPU trainer:
all reachable text+firing-history states are enumerated at compile time. It
fails explicitly when the stated budget is exceeded. Example:

python train.py --demo --steps 50 --device cpu --out exact_demo
python train.py --dataset examples/dataset.json --rules-json examples/rules.json --steps 50
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from lens_gpu.exact_pushforward import (RuleSupport, TrainableRule,
                                       ExactSerialPushforward, deterministic_text)


class SwiGLUHead(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.first = nn.Linear(n, hidden * 2)
        self.second = nn.Linear(hidden, hidden * 2)
        self.last = nn.Linear(hidden, 1)

    def forward(self, x):
        a, b = self.first(x).chunk(2, dim=-1)
        a = F.silu(a) * b
        a, b = self.second(a).chunk(2, dim=-1)
        return self.last(F.silu(a) * b).squeeze(-1)


def read_support_file(path):
    """Supports are integer channels 0..255 for bytes, 256..270 for *1..*15."""
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(raw, list) or not raw:
        raise ValueError('rules.json must be a nonempty JSON array')
    return [RuleSupport(r['pattern'], r['replacement']) for r in raw]


def load_dataset(path):
    """Text encoded as UTF8, or explicit byte numbers for exact byte control."""
    records = json.loads(Path(path).read_text(encoding='utf8'))
    if not records or not isinstance(records,list):
        raise ValueError('dataset must be nonempty JSON array')
    items=[]
    for row in records:
        s = row['text'].encode('utf8') if isinstance(row['text'],str) else bytes(row['text'])
        if not s: raise ValueError('empty documents unsupported by noise-rate feature normalisation')
        items.append((s,float(row['target'])))
    return items


def build_demo():
    rules=[RuleSupport([[ord('a'),ord('b')]],[[ord('X'),ord('Y')]]),
           RuleSupport([[ord('X')]], [[ord('Z')]])]
    data=[(x, x.count(b'a') / len(x)) for x in [b'aa',b'ab',b'ba',b'bb']]
    return rules,data


def select_device(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    return torch.device(device)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device=select_device(args.device)
    dtype=torch.float32 if device.type=='mps' else torch.float64
    rules, data = build_demo() if args.demo else (read_support_file(args.rules_json),load_dataset(args.dataset))
    # Every example is a deterministic observed string, and *all reachable*
    # probabilistic outcomes of the learned rule sequence are enumerated.
    input_plans={}
    for text,_ in data:
        if text in input_plans: continue
        _,_,support=deterministic_text(text,dtype=dtype)
        plan=ExactSerialPushforward(support,rules,track_features=True,
                 max_worlds=args.max_worlds,max_states=args.max_states,
                 max_total_worlds=args.max_total_worlds,max_output=args.max_output)
        input_plans[text]=plan.to(device)
    learned=nn.ModuleList([TrainableRule(rule,dtype=dtype) for rule in rules]).to(device)
    head=SwiGLUHead(len(rules),args.hidden).to(device,dtype)
    optim=torch.optim.AdamW(list(learned.parameters())+list(head.parameters()),lr=args.lr)
    inputs={x:deterministic_text(x,device=device,dtype=dtype)[:2] for x in input_plans}
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    history=[]
    print(json.dumps({'device':str(device),'dataset_rows':len(data),'rules':len(rules),
           'worlds_per_plan':{x.hex():list(input_plans[x].world_counts) for x in input_plans},
           'states_per_plan':{x.hex():input_plans[x].layer_sizes for x in input_plans},
           'type':'exact_joint_pushforward_NO_APPROXIMATION'},ensure_ascii=False),flush=True)
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        distributions=[rule(temperature=args.temperature) for rule in learned]
        predictions=[]
        targets=[]
        # Compilation was done once. The numerical contractions, rule
        # probabilities, nonlinear head and gradients all run on device.
        for text,y in data:
            x,rho=inputs[text]
            law=input_plans[text](x,rho,distributions,validate=False)
            predictions.append(law.expected_head(head))
            targets.append(y)
        p=torch.stack(predictions)
        labels=p.new_tensor(targets)
        loss=(p-labels).square().mean()
        loss.backward()
        optim.step()
        if step==0 or step==args.steps-1 or (step+1)%args.log_every==0:
            row={'step':step+1,'mse':float(loss.detach()),'prediction':p.detach().cpu().tolist()}
            print(json.dumps(row),flush=True)
            history.append(row)
    checkpoint={'rules':[asdict(r) for r in rules],
                'rule_weights':learned.state_dict(), 'head_weights':head.state_dict(),
                'hidden':args.hidden,'dtype':str(dtype),'args':vars(args)}
    torch.save(checkpoint,output/'last.pt')
    (output/'training.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    print('checkpoint:',output/'last.pt',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true',help='train a tiny exact end-to-end two-rule demo')
    p.add_argument('--dataset',help='JSON array of {text,target} examples')
    p.add_argument('--rules-json',help='JSON array of static categorical support sets')
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--lr',type=float,default=0.01)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--max-worlds',type=int,default=100000)
    p.add_argument('--max-total-worlds',type=int,default=500000)
    p.add_argument('--max-states',type=int,default=100000)
    p.add_argument('--max-output',type=int,default=1048576)
    p.add_argument('--out',default='exact_run')
    p.add_argument('--seed',type=int,default=23)
    args=p.parse_args()
    if not args.demo and (not args.dataset or not args.rules_json):
        p.error('Use --demo or supply BOTH --dataset and --rules-json')
    if args.steps<1 or args.hidden<1 or args.log_every<1 or args.temperature<=0 or args.lr<=0:
        p.error('invalid training arguments')
    train(args)

if __name__=='__main__':main()
