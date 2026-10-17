"""Numerical equivalence and synchronized scan/end-to-end timing against old code."""
import argparse,json,time,statistics
import torch
import model
from scan_reference import reach_doubling
from train import select_device,sync
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',default='scan_upgrade.json');a=p.parse_args()
d=select_device(a.device);torch.set_num_threads(1);torch.manual_seed(42)
new=model.reach_parallel
checks=[]
for length in (1,2,3,7,16,31,129):
 for spans in (1,3,6):
    dtype=torch.float32 if d.type=='mps' else torch.float64
    x=(torch.rand(2,length,spans,device=d,dtype=dtype)*(.6/spans)).requires_grad_()
    weight=torch.randn(2,length,device=d,dtype=dtype)
    vals=[]
    for fn in (reach_doubling,new):
        y=fn(x)
        # Length=1 or span=1 can yield constant results with no gradient path.
        loss=(y*weight).sum()+x.square().sum()*.01
        g=torch.autograd.grad(loss,x,create_graph=True)[0]
        h=torch.autograd.grad(g.square().sum(),x)[0]
        vals.append((y,g,h))
    errors=[]
    for v,w in zip(*vals):
        torch.testing.assert_close(v,w,rtol=2e-4 if dtype==torch.float32 else 1e-9,atol=2e-5 if dtype==torch.float32 else 1e-10)
        errors.append(float((v-w).abs().max().detach()))
    checks.append({'length':length,'spans':spans,'errors_output_grad_hvp':errors})

def timed(call,repeats=5):
    call();sync(d);times=[]
    for _ in range(repeats):
        sync(d);t=time.perf_counter();call();sync(d);times.append(time.perf_counter()-t)
    return statistics.median(times)

bench=[]
for length in (128,512,2048):
    x=(torch.rand(8,length,6,device=d)*.08).requires_grad_()
    times={}
    for name,fn in [('old',reach_doubling),('new',new)]:
        def run():
            x.grad=None;fn(x).square().mean().backward()
        times[name]=timed(run)
    bench.append({'batch':8,'length':length,'spans':6,'seconds':times,'speedup':times['old']/times['new']})
full=[]
for rules,length,hidden,cap in ((16,256,128,4),(1500,16,2048,1)):
    m=model.Lens(model.Config(rules,hidden,cap,25)).to(d)
    m.initialize_from_training([b'abcdefabcdef0123456789'])
    x=torch.randint(97,103,(2,length),device=d);lens=torch.full((2,),length,device=d)
    times={};preds=[];grads=[]
    for name,fn in [('old',reach_doubling),('new',new)]:
        model.reach_parallel=fn
        def run():
            m.zero_grad(set_to_none=True);y=m(x,lens);y.square().mean().backward();return y
        times[name]=timed(run,repeats=2)
        preds.append(run().detach());grads.append([v.grad.detach().clone() for v in m.parameters()])
    torch.testing.assert_close(*preds,rtol=2e-4,atol=2e-5)
    maximum=0.
    for g,h in zip(*grads):
        torch.testing.assert_close(g,h,rtol=5e-3,atol=2e-5)
        maximum=max(maximum,float((g-h).abs().max()))
    full.append({'rules':rules,'length':length,'batch':2,'hidden':hidden,'max_capture':cap,'seconds':times,
                 'speedup':times['old']/times['new'],'max_gradient_abs_difference':maximum})
    print(json.dumps(full[-1]),flush=True)
model.reach_parallel=new
result={'device':str(d),'torch':str(torch.__version__),'threads':1,'checks':checks,'scan_forward_backward':bench,'model_forward_backward':full}
open(a.output,'w').write(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
