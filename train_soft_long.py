"""Three sequential soft Replace layers; every input channel is strictly positive.

This is candidate-rule learning, not unrestricted program induction. A staged
curriculum uses known intermediate targets, followed by end-to-end training.
Held-out bytes/documents are generated with distinct seeds and not optimized.
"""
import argparse,json,time,random
from pathlib import Path
import torch
from torch.nn import functional as F
from soft_pipeline import Rule,SoftReplaceLayer,SoftReplacePipeline
from long_exact import BytePattern


def families():
    return [
        [Rule((60,-1,62),(91,-1,93)),Rule((60,-1,62),(40,-1,41)),Rule((60,-1,62),(123,-1,125))],
        [Rule((91,-1,93),(123,-1,125)),Rule((91,-1,93),(60,-1,62)),Rule((91,-1,93),(40,-1,41))],
        [Rule((123,-1,125),(33,-1,-1,63)),Rule((123,-1,125),(33,-1,63)),Rule((123,-1,125),(63,-1,-1,33))]]


def document(seed,minimum):
    rng=random.Random(seed)
    stages=[bytearray() for _ in range(4)]
    while len(stages[0])<minimum:
        outside=bytes(rng.choices(b' abcdefghijklmnopqrstuvwxyz0123456789.',k=rng.randint(5,13)))
        # Distractors contain no brackets of the next rule, but are preserved.
        payload=bytes(rng.choices(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ',k=rng.randint(1,3)))
        records=[b'<'+payload+b'>',b'['+payload+b']',b'{'+payload+b'}',b'!'+payload*2+b'?']
        for s,rec in zip(stages,records):s.extend(outside+rec)
    for s in stages:s.extend(b' trailing <unfinished')
    return [bytes(s) for s in stages]


def encode(raw,noise,dtype=torch.float32):
    hard=F.one_hot(torch.tensor(list(raw)),256).to(dtype)[None]
    # Independent nonuniform full-support uncertainty at every byte position.
    gen=torch.Generator().manual_seed(821+len(raw))
    background=torch.rand(hard.shape,generator=gen,dtype=dtype)+.1
    background=background/background.sum(-1,keepdim=True)
    return (1-noise)*hard+noise*background,torch.tensor([float(len(raw))],dtype=dtype)


def objective(Y,length,target):
    n=len(target)
    labels=torch.tensor(list(target))
    # No clipping/masking depends on a learned length. Fixed supervised target.
    ce=-(Y[0,:n].gather(1,labels[:,None]).squeeze(1)+1e-12).log().mean()
    lenloss=((length-n)/max(1,n)).square().mean()
    return ce+1000*lenloss


def discrete_apply(raw,rules):
    for rule in rules:
        raw=BytePattern(rule.pattern,min_wildcard_len=1).apply(raw,rule.replacement)
    return raw


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--length',type=int,default=1024)
    ap.add_argument('--warmup',type=int,default=40)
    ap.add_argument('--joint',type=int,default=30)
    ap.add_argument('--learn-bytes',action='store_true')
    ap.add_argument('--match-power',type=int,default=2)
    ap.add_argument('--noise',type=float,default=.001)
    ap.add_argument('--output',default='soft_long_results.json')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(73)
    family=families()
    # Shuffle each candidate list so the correct index is not shared across layers.
    rng=random.Random(59)
    for f in family:rng.shuffle(f)
    model=SoftReplacePipeline([SoftReplaceLayer(f,min_capture=1,max_capture=3,match_power=args.match_power,trainable_literals=args.learn_bytes) for f in family])
    for layer in model.layers:
        with torch.no_grad():layer.logits.add_(torch.randn_like(layer.logits)*.03)
    docs=[document(100+i,args.length) for i in range(4)]
    history=[];start=time.perf_counter()
    # Curriculum: teach each rule on uncertain intermediate tapes, then jointly fine tune.
    for j,layer in enumerate(model.layers):
        if args.learn_bytes:
            opt=torch.optim.Adam([{'params':[layer.logits]+list(layer.output_logits),'lr':.2},
                                  {'params':list(layer.pattern_logits),'lr':.02}])
        else:
            opt=torch.optim.Adam(layer.parameters(),lr=.2)
        for step in range(args.warmup):
            stages=docs[step%len(docs)]
            X,l=encode(stages[j],args.noise)
            Y,ol=layer(X,l)
            task=objective(Y,ol,stages[j+1])
            reg=0.02*max(0.,(step-args.warmup/2)/(args.warmup/2))
            loss=task+reg*layer.regularizer()
            opt.zero_grad();loss.backward();opt.step()
            if step in (0,args.warmup-1):
                record=dict(phase='curriculum',layer=j,step=step,loss=float(task.detach()),
                            probabilities=layer.logits.softmax(-1).detach().tolist())
                history.append(record);print(json.dumps(record),flush=True)
    if args.learn_bytes:
        pattern_params=[p for layer in model.layers for p in layer.pattern_logits]
        other_params=[p for layer in model.layers for p in [layer.logits]+list(layer.output_logits)]
        opt=torch.optim.Adam([{'params':other_params,'lr':.05},{'params':pattern_params,'lr':.005}])
    else:
        opt=torch.optim.Adam(model.parameters(),lr=.08)
    for step in range(args.joint):
        stages=docs[step%len(docs)]
        X,l=encode(stages[0],args.noise)
        Y,ol=model(X,l)
        task=objective(Y,ol,stages[3])
        loss=task+.02*model.regularizer()
        opt.zero_grad();loss.backward();opt.step()
        if step in (0,args.joint-1):
            record=dict(phase='end_to_end',step=step,loss=float(task.detach()))
            history.append(record);print(json.dumps(record),flush=True)
    elapsed=time.perf_counter()-start
    learned=model.decode()
    held=[]
    for length in (args.length,4096,16384):
        for seed in (800,801):
            stages=document(seed,length)
            predicted=discrete_apply(stages[0],learned)
            held.append(dict(seed=seed,input_bytes=len(stages[0]),output_bytes=len(stages[-1]),
                             exact=predicted==stages[-1]))
    soft=[]
    for soft_length,noise in [(args.length,.001),(args.length,.01),(args.length,.1),(4096,.1),(16384,.1)]:
        stages=document(990,soft_length)
        X,l=encode(stages[0],noise)
        with torch.no_grad():Y,ol=model(X,l)
        n=len(stages[-1]); labels=torch.tensor(list(stages[-1]))
        soft.append(dict(input_bytes=len(stages[0]),noise=noise,loss=float(objective(Y,ol,stages[-1])),
                         position_accuracy=float((Y[0,:n].argmax(-1)==labels).float().mean()),
                         predicted_length=float(ol),true_length=n))
    # Input gradient reaches every position through all three learned layers.
    stages=document(991,128);X,l=encode(stages[0],.01);X.requires_grad_()
    Y,ol=model(X,l);task=objective(Y,ol,stages[-1])
    gx=torch.autograd.grad(task,X)[0]
    result=dict(config=vars(args),torch_version=torch.__version__,device='cpu',threads=1,
                training_seconds=elapsed,history=history,held_out=held,soft_held_out=soft,
                input_gradient_nonzero_positions=int((gx.abs().sum(-1)>0).sum()),
                input_gradient_positions=X.shape[1],
                rules=[dict(pattern=r.pattern,replacement=r.replacement) for r in learned],
                probabilities=[l.logits.softmax(-1).detach().tolist() for l in model.layers],
                limitations=['finite rule candidate banks','capture lengths 1..3','mean-field matching and transport',
                             'curriculum uses intermediate supervision','CPU implementation','delimiter confidence sharpening is a changed soft extension',
                             'when learn_bytes: pattern delimiters use a strong logit-10 prior; replacement byte logits start uniform'])
    torch.save(model.state_dict(),Path(args.output).with_suffix('.pt'))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(training_seconds=elapsed,held_out_exact=sum(h['exact'] for h in held),
                         held_out_total=len(held),soft_held_out=soft)),flush=True)
if __name__=='__main__':main()
