"""Measure actual serial forward/backward on selected device, with synchronization."""
import argparse,json,time
import torch
from model import Config,Lens
from train import select_device,sync
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='auto');p.add_argument('--rules',type=int,default=1500)
p.add_argument('--hidden',type=int,default=2048);p.add_argument('--length',type=int,default=128)
p.add_argument('--batch',type=int,default=2);p.add_argument('--max-capture',type=int,default=4)
p.add_argument('--checkpoint-every',type=int,default=25);p.add_argument('--iterations',type=int,default=3)
p.add_argument('--threads',type=int,default=4);p.add_argument('--output',default='benchmark.json')
a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(1729)
d=select_device(a.device);m=Lens(Config(a.rules,a.hidden,a.max_capture,a.checkpoint_every)).to(d)
m.initialize_from_training([b'abcdefabcdef0123456789'])
x=torch.randint(97,103,(a.batch,a.length),device=d);length=torch.full((a.batch,),a.length,device=d)
if d.type=='cuda':torch.cuda.reset_peak_memory_stats()
times=[]
for i in range(a.iterations+1):
    m.zero_grad(set_to_none=True);sync(d);t=time.perf_counter()
    loss=m(x,length).square().mean();loss.backward();sync(d)
    elapsed=time.perf_counter()-t
    if i:times.append(elapsed)
    print(json.dumps({'iteration':i,'warmup':i==0,'seconds':elapsed,'loss':float(loss.detach())}),flush=True)
result={'settings':vars(a),'device':str(d),'torch':torch.__version__,'seconds':times,
        'rows_per_second':a.batch/(sum(times)/len(times)),
        'first_rule_grad_norm':float(m.rules[0].outer.grad.norm()),
        'last_rule_grad_norm':float(m.rules[-1].outer.grad.norm()),
        'parameters':sum(p.numel() for p in m.parameters())}
if d.type=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
if d.type=='mps':result['allocated_bytes']=torch.mps.current_allocated_memory()
open(a.output,'w').write(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
