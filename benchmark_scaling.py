"""Synchronized forward+backward batch scaling: original 256 model vs vectorized.

Run, e.g.:
  python benchmark_scaling.py --device cuda --rules 32 --length 512 --batches 1,2,4,8
Results include rows/s (do not compare only step times when batch sizes differ).
"""
import argparse
import gc
import json
import statistics
import time
import torch
import model
import model_baseline
from train import select_device, sync

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
p.add_argument('--rules',type=int,default=8)
p.add_argument('--hidden',type=int,default=64)
p.add_argument('--length',type=int,default=128)
p.add_argument('--batches',default='1,2,4,8')
p.add_argument('--max-capture',type=int,default=4)
p.add_argument('--checkpoint-every',type=int,default=4)
p.add_argument('--warmup',type=int,default=1)
p.add_argument('--iterations',type=int,default=3)
p.add_argument('--threads',type=int,default=2)
p.add_argument('--only-new',action='store_true')
p.add_argument('--compile-rule',action='store_true')
p.add_argument('--out',default='scaling.json')
a=p.parse_args()
if a.rules<1 or a.length<3 or a.iterations<1 or a.warmup<0 or a.threads<1: p.error('invalid arguments')
batches=[int(x) for x in a.batches.split(',')]
if any(x<1 for x in batches): p.error('batches must be positive')
if a.compile_rule and not a.only_new: p.error('--compile-rule requires --only-new')
torch.set_num_threads(a.threads)
d=select_device(a.device)
config=model.Config(a.rules,a.hidden,a.max_capture,a.checkpoint_every)
results=[]
for batch in batches:
    cases=[('optimized',model)] if a.only_new else [('baseline',model_baseline),('optimized',model)]
    for name,module in cases:
        torch.manual_seed(1289)
        m=module.Lens(config).to(d)
        if a.compile_rule: m.enable_compiled_rule()
        x=torch.randint(0,256,(batch,a.length),device=d)
        lens=torch.full((batch,),a.length,device=d,dtype=torch.float32)
        if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d)
        times=[]
        for i in range(a.warmup+a.iterations):
            m.zero_grad(set_to_none=True)
            sync(d)
            start=time.perf_counter()
            m(x,lens).square().mean().backward()
            sync(d)
            if i>=a.warmup: times.append(time.perf_counter()-start)
        med=statistics.median(times)
        item={'batch':batch,'impl':name,'seconds_median':round(med,6),
              'samples_per_second':round(batch/med,4),'measurements_seconds':times}
        if d.type=='cuda':item['peak_allocated_mib']=round(torch.cuda.max_memory_allocated(d)/1024**2,2)
        if d.type=='mps' and hasattr(torch.mps,'current_allocated_memory'):
            item['allocated_mib']=round(torch.mps.current_allocated_memory()/1024**2,2)
        print(json.dumps(item),flush=True)
        results.append(item)
        del m,x,lens
        gc.collect()
        if d.type=='cuda':torch.cuda.empty_cache()
summary=[]
for b in batches:
    old=next((r for r in results if r['batch']==b and r['impl']=='baseline'),None)
    new=next(r for r in results if r['batch']==b and r['impl']=='optimized')
    if old:summary.append({'batch':b,'throughput_speedup':round(new['samples_per_second']/old['samples_per_second'],3)})
print('speedups:',json.dumps(summary),flush=True)
with open(a.out,'w') as f:json.dump({'settings':vars(a),'device':str(d),'torch':torch.__version__,
                                      'results':results,'speedups':summary},f,indent=2)
