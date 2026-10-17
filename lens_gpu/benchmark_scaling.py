"""Synchronized forward + backward batch-scaling for variable-length Lens.

Example:
 python benchmark_scaling.py --device mps --rules 8 --length 128 \
     --batches 1,2,4,8 --max-pattern 3 --max-replacement 3

Do not directly compare with v2 fixed 3->1 model: the grammars differ.
"""
import argparse,gc,json,statistics,time,math
import torch
if __package__:
    from .model import Lens,Config
else:
    from model import Lens,Config
if __package__:
    from .train import select_device,sync
else:
    from train import select_device,sync

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
p.add_argument('--rules',type=int,default=8)
p.add_argument('--hidden',type=int,default=64)
p.add_argument('--length',type=int,default=128)
p.add_argument('--capacity-factor',type=float,default=2.)
p.add_argument('--batches',default='1,2,4,8')
p.add_argument('--max-capture',type=int,default=0,help='ignored legacy option; no capture-length limit')
p.add_argument('--max-stars',type=int,default=3)
p.add_argument('--max-pattern',type=int,default=4)
p.add_argument('--max-replacement',type=int,default=4)
p.add_argument('--checkpoint-every',type=int,default=4)
p.add_argument('--warmup',type=int,default=1)
p.add_argument('--iterations',type=int,default=3)
p.add_argument('--threads',type=int,default=2)
p.add_argument('--out',default='scaling_hybrid.json')
a=p.parse_args()
if min(a.rules,a.length,a.iterations,a.threads,a.max_stars,a.max_pattern,a.max_replacement)<1 or a.capacity_factor<1:
    p.error('invalid settings')
batches=[int(x) for x in a.batches.split(',')]
if any(x<1 for x in batches):p.error('batches must be positive')
torch.set_num_threads(a.threads)
d=select_device(a.device)
capacity=math.ceil(a.length*a.capacity_factor)
config=Config(rules=a.rules,hidden=a.hidden,max_capture=a.max_capture,max_stars=a.max_stars,
              max_pattern=a.max_pattern,max_replacement=a.max_replacement,
              checkpoint_every=a.checkpoint_every)
results=[]
for batch in batches:
    torch.manual_seed(1289)
    m=Lens(config).to(d)
    m.initialize_from_training([b'abcdeabcdexyz01234'])
    x=torch.zeros(batch,capacity,device=d,dtype=torch.long)
    x[:,:a.length]=torch.randint(97,103,(batch,a.length),device=d)
    lengths=torch.full((batch,),a.length,device=d,dtype=torch.float32)
    if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d)
    times=[]
    for i in range(a.warmup+a.iterations):
        m.zero_grad(set_to_none=True)
        sync(d);start=time.perf_counter()
        m(x,lengths).square().mean().backward()
        sync(d)
        if i>=a.warmup:times.append(time.perf_counter()-start)
    sec=statistics.median(times)
    item={'batch':batch,'seconds_median':round(sec,6),
          'samples_per_second':round(batch/sec,4),'measurements_seconds':times}
    if d.type=='cuda':item['peak_allocated_mib']=round(torch.cuda.max_memory_allocated(d)/2**20,2)
    if d.type=='mps' and hasattr(torch.mps,'current_allocated_memory'):
        item['allocated_mib']=round(torch.mps.current_allocated_memory()/2**20,2)
    print(json.dumps(item),flush=True)
    results.append(item)
    del m,x,lengths
    gc.collect()
    if d.type=='cuda':torch.cuda.empty_cache()
with open(a.out,'w') as f:
    json.dump({'settings':vars(a),'device':str(d),'torch':torch.__version__,
               'results':results},f,indent=2)
