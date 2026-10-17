import json,time,statistics
from pathlib import Path
import torch
from soft_pipeline import _Solve,SoftReplaceLayer,SoftReplacePipeline
from train_soft_long import families,document,encode,objective

def eager(w,s):
    rows=[]
    for i in range(w.shape[1]):
        value=s[:,i]
        for d in range(1,min(i,w.shape[2])+1):
            value=value+w[:,i-d,d-1]*rows[i-d]
        rows.append(value)
    return torch.stack(rows,1)

def run():
    torch.set_num_threads(1);torch.manual_seed(72)
    scan=[]
    for L in (256,1024,4096):
        w=(torch.rand(1,L,5)*.08).requires_grad_();s=torch.rand(1,L,requires_grad=True)
        native=_Solve.apply(w,s,False);slow=eager(w,s)
        torch.testing.assert_close(native,slow,atol=1e-6,rtol=1e-5)
        row={'length':L}
        for name,fn in [('native',lambda:_Solve.apply(w,s,False)),('torch_loop',lambda:eager(w,s))]:
            times=[]
            for _ in range(3):
                start=time.perf_counter();y=fn();torch.autograd.grad(y.square().sum(),(w,s))
                times.append(time.perf_counter()-start)
            row[name+'_forward_backward_seconds']=statistics.median(times)
        row['speedup']=row['torch_loop_forward_backward_seconds']/row['native_forward_backward_seconds']
        scan.append(row)
    model=SoftReplacePipeline([SoftReplaceLayer(f,min_capture=1,max_capture=3) for f in families()])
    for layer in model.layers:
        with torch.no_grad():layer.logits[0]=8
    pipeline=[]
    for L in (256,1024,4096):
        stages=document(170,L);X,l=encode(stages[0],.1);X.requires_grad_()
        times=[]
        for _ in range(3):
            start=time.perf_counter();Y,ol=model(X,l);loss=objective(Y,ol,stages[-1])
            torch.autograd.grad(loss,(X,*tuple(model.parameters())))
            times.append(time.perf_counter()-start)
        pipeline.append({'input_bytes':len(stages[0]),'output_capacity':Y.shape[1],
                         'three_layers_forward_backward_seconds':statistics.median(times)})
    result={'cpu_threads':1,'scan':scan,'pipeline':pipeline,
            'note':'scan speedup is only the recurrence kernel; full-layer absolute times separately measured'}
    Path(__file__).with_name('soft_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':run()
