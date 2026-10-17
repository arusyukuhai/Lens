"""Single-size synchronized variable-length Lens benchmark."""
import argparse,json,math,time
import torch
if __package__:
    from .model import Config,Lens
else:
    from model import Config,Lens
if __package__:
    from .train import select_device,sync
else:
    from train import select_device,sync
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='auto')
p.add_argument('--rules',type=int,default=16)
p.add_argument('--hidden',type=int,default=128)
p.add_argument('--length',type=int,default=128)
p.add_argument('--capacity-factor',type=float,default=2.)
p.add_argument('--batch',type=int,default=2)
p.add_argument('--max-capture',type=int,default=0,help='ignored legacy option; no capture-length limit')
p.add_argument('--max-stars',type=int,default=3)
p.add_argument('--max-pattern',type=int,default=4)
p.add_argument('--max-replacement',type=int,default=4)
p.add_argument('--checkpoint-every',type=int,default=4)
p.add_argument('--iterations',type=int,default=3)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--output',default='benchmark.json')
a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(1729)
if a.capacity_factor<1: p.error('capacity-factor must be >=1')
d=select_device(a.device)
config=Config(rules=a.rules,hidden=a.hidden,max_capture=a.max_capture,max_stars=a.max_stars,
              max_pattern=a.max_pattern,max_replacement=a.max_replacement,
              checkpoint_every=a.checkpoint_every)
m=Lens(config).to(d)
m.initialize_from_training([b'abcdefabcdef0123456789'])
capacity=math.ceil(a.length*a.capacity_factor)
x=torch.zeros((a.batch,capacity),device=d,dtype=torch.long)
x[:,:a.length]=torch.randint(97,103,(a.batch,a.length),device=d)
lengths=torch.full((a.batch,),a.length,device=d)
if d.type=='cuda':torch.cuda.reset_peak_memory_stats()
times=[]
for i in range(a.iterations+1):
    m.zero_grad(set_to_none=True);sync(d);t=time.perf_counter()
    loss=m(x,lengths).square().mean();loss.backward();sync(d)
    elapsed=time.perf_counter()-t
    if i:times.append(elapsed)
    print(json.dumps({'iteration':i,'warmup':i==0,'seconds':elapsed,'loss':float(loss.detach())}),flush=True)
result={'settings':vars(a),'device':str(d),'torch':torch.__version__,'seconds':times,
        'rows_per_second':a.batch/(sum(times)/len(times)),
        'first_rule_grad_norm':float(m.rules[0].pattern_bytes.grad.norm()),
        'parameters':sum(p.numel() for p in m.parameters())}
if d.type=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
if d.type=='mps':result['allocated_bytes']=torch.mps.current_allocated_memory()
open(a.output,'w').write(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
