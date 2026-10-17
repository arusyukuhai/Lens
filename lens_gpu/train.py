"""End-to-end serial Lens trainer. Run python train.py --help."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
if __package__:
    from .model import Lens,Config
else:
    from model import Lens,Config
if __package__:
    from .data import load_corpus,split_corpus,alphabet_of,diffusion_batch,synthetic
else:
    from data import load_corpus,split_corpus,alphabet_of,diffusion_batch,synthetic


def anneal(step,total,maximum=.02,cycles=4,warmup=.1):
    """Growing oscillatory envelope, zero at start, maximum at final update.
    u*sin^2(pi*(cycles+1/2)*u), after warmup. No temp/STE substitutions.
    """
    t=step/max(total-1,1)
    u=min(1.,max(0.,(t-warmup)/(1-warmup)))
    return maximum*u*math.sin(math.pi*(cycles+.5)*u)**2


def select_device(name):
    if name=='auto': name='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    if name=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS requested but unavailable')
    return torch.device(name)


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize()
    elif device.type=='mps': torch.mps.synchronize()


def regression_loss(pred,target,kind):
    if kind=='mse': return F.mse_loss(pred,target)
    if kind=='pseudo_huber': return (.01*(torch.sqrt(1+((pred-target)/.1).square())-1)).mean()
    a=pred-pred.mean(); b=target-target.mean()
    return 1-(a*b).sum()/torch.sqrt((a.square().sum()+1e-8)*(b.square().sum()+1e-8))


@torch.no_grad()
def evaluate(model,chunks,alphabet,args,device,hard=False):
    model.eval(); predictions=[]; targets=[]
    gen=torch.Generator().manual_seed(args.seed+99001)
    for first in range(0,len(chunks),args.batch_chunks):
        tok,lens,target=diffusion_batch(chunks[first:first+args.batch_chunks],alphabet,args.levels,gen,capacity_factor=args.capacity_factor)
        size=args.microbatch or len(tok)
        for start in range(0,len(tok),size):
            sl=slice(start,start+size)
            predictions.append(model(tok[sl].to(device),lens[sl].to(device),args.temperature,hard,args.amp).cpu())
        targets.append(target)
    p=torch.cat(predictions).double(); y=torch.cat(targets).double()
    a=p-p.mean(); b=y-y.mean(); denom=(a.square().sum()*b.square().sum()).sqrt()
    return {'mse':float((p-y).square().mean()),'mae':float((p-y).abs().mean()),
            'pearson':float((a*b).sum()/denom) if denom>0 else None,
            'constant_mean_mse':float((y-y.mean()).square().mean()),'rows':len(y)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus'); p.add_argument('--synthetic',action='store_true')
    p.add_argument('--out',default='lens_serial_run'); p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    p.add_argument('--rules',type=int,default=1500); p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--max-capture',type=int,default=0,help='DEPRECATED and ignored: captures have no length limit')
    p.add_argument('--exact-max-output',type=int,default=1048576,help='old-style integer interpreter safety limit; overflow cancels that single rule')
    p.add_argument('--max-stars',type=int,default=3,help='maximum independently captured wildcards per LHS (numbered *1,*2,...)'); p.add_argument('--checkpoint-every',type=int,default=25)
    p.add_argument('--max-pattern',type=int,default=4); p.add_argument('--max-replacement',type=int,default=4)
    p.add_argument('--capacity-factor',type=float,default=1.5,help='fixed buffer / maximum input length; >=1, increase for expanding rules')
    p.add_argument('--chunk-len',type=int,default=2000); p.add_argument('--max-chunks',type=int,default=65536)
    p.add_argument('--valid',type=int,default=8); p.add_argument('--test',type=int,default=8)
    p.add_argument('--batch-chunks',type=int,default=2); p.add_argument('--levels',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=0,help='0 (default): all batch rows simultaneously; positive values cap GPU/CPU memory'); p.add_argument('--accumulate',type=int,default=1)
    p.add_argument('--steps',type=int,default=2000); p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--log-every',type=int,default=10); p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--rule-lr',type=float,default=.01)
    p.add_argument('--reg-max',type=float,default=.02); p.add_argument('--reg-cycles',type=int,default=4)
    p.add_argument('--reg-warmup',type=float,default=.1); p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--loss',choices=['mse','pseudo_huber','pearson'],default='mse')
    p.add_argument('--amp',action='store_true',help='CUDA bfloat16 MLP only; rule math remains float32')
    p.add_argument('--compile-head',action='store_true',help='experimental CUDA/CPU head compilation; MPS eager')
    p.add_argument('--compile-rule',action='store_true',help='not validated for variable-length grammar; currently rejected')
    p.add_argument('--optimizer-kernels',choices=['auto','foreach','fused','scalar'],default='auto',
                   help='optimizer update kernel mode; auto uses CUDA fused AdamW and portable defaults elsewhere')
    p.add_argument('--threads',type=int,default=4); p.add_argument('--resume')
    p.add_argument('--hard-finetune',type=int,default=0,help='optional extra steps training MLP on exact old-Replace discrete rules')
    a=p.parse_args()
    if min(a.steps,a.batch_chunks,a.accumulate,a.eval_every,a.log_every,a.threads,a.max_pattern,a.max_replacement,a.max_stars)<1 or a.microbatch<0 or a.levels<2 or a.capacity_factor<1:
        p.error('counts must be positive, levels >=2')
    if a.max_stars>a.max_pattern or a.max_stars>15:p.error('--max-stars must be <= min(--max-pattern,15)')
    if a.exact_max_output<1:p.error('--exact-max-output must be positive')
    if a.max_capture!=0:print('[warning] --max-capture is deprecated and IGNORED; captures are unbounded',flush=True)
    if not 0<=a.reg_warmup<1 or a.reg_cycles<0 or a.reg_max<0 or a.temperature<=0: p.error('invalid annealing settings')
    if a.loss=='pearson' and ((a.microbatch and a.microbatch<a.batch_chunks*a.levels) or a.accumulate!=1):
        p.error('Pearson requires one unsplit batch: microbatch >= batch-chunks*levels and accumulate=1')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    device=select_device(a.device)
    if a.amp and (device.type!='cuda' or not torch.cuda.is_bf16_supported()): p.error('--amp needs a CUDA GPU supporting bfloat16')
    if a.compile_head and device.type=='mps': p.error('MPS uses eager mode; omit --compile-head')
    if a.compile_rule: p.error('--compile-rule is not yet validated for the variable-length grammar; omit it')
    if a.optimizer_kernels=='fused' and device.type!='cuda': p.error('fused optimizer mode requires CUDA')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if a.synthetic:
        a.corpus=str(out/'synthetic.txt'); synthetic(a.corpus,length=a.chunk_len)
    if not a.corpus: p.error('supply --corpus or --synthetic')
    chunks=load_corpus(a.corpus,a.chunk_len,a.max_chunks)
    train,val,test,manifest=split_corpus(chunks,a.seed,a.valid,a.test)
    alphabet=alphabet_of(train)
    config=Config(rules=a.rules,hidden=a.hidden,max_capture=a.max_capture,max_stars=a.max_stars,
                  checkpoint_every=a.checkpoint_every,max_pattern=a.max_pattern,
                  max_replacement=a.max_replacement,exact_max_output=a.exact_max_output)
    model=Lens(config).to(device)
    if a.compile_rule: model.enable_compiled_rule()
    gen=torch.Generator().manual_seed(a.seed+1)
    rng=random.Random(a.seed+2); order=list(range(len(train))); rng.shuffle(order); cursor=0
    optimizer_mode=('fused' if device.type=='cuda' else 'scalar') if a.optimizer_kernels=='auto' else a.optimizer_kernels
    kernel_kw={'fused':True} if optimizer_mode=='fused' else {'foreach':True} if optimizer_mode=='foreach' else {'foreach':False}
    opt=torch.optim.AdamW([{'params':model.rules.parameters(),'lr':a.rule_lr,'weight_decay':0.},
                           {'params':model.head.parameters(),'lr':a.lr,'weight_decay':.01}],**kernel_kw)
    start_step=0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['config']!=asdict(config) or ck['manifest']!=manifest: raise ValueError('resume architecture/corpus split mismatch')
        for key in ('steps','reg_max','reg_cycles','reg_warmup','temperature','loss','batch_chunks','levels','microbatch','accumulate','lr','rule_lr','capacity_factor','exact_max_output'):
            if ck['args'][key]!=getattr(a,key): raise ValueError('resume requires same '+key)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start_step=ck['step']+1
        gen.set_state(ck['data_rng']); rng.setstate(ck['order_rng']); order=ck['order']; cursor=ck['cursor']
        torch.set_rng_state(ck['torch_rng'])
    else: model.initialize_from_training(train,a.seed)
    # Compile wrapper is separate: saved head state_dict always remains portable.
    original_head=model.head
    if a.compile_head: model.head=torch.compile(model.head)
    (out/'config.json').write_text(json.dumps({'args':vars(a),'model':asdict(config),'device':str(device),'torch':torch.__version__},indent=2))
    (out/'split.json').write_text(json.dumps(manifest,indent=2))
    def next_batch():
        nonlocal cursor,order
        chosen=[]
        for _ in range(a.batch_chunks):
            if cursor==len(order): rng.shuffle(order); cursor=0
            chosen.append(train[order[cursor]]); cursor+=1
        return diffusion_batch(chosen,alphabet,a.levels,gen,pin=device.type=='cuda',capacity_factor=a.capacity_factor)
    def save(name,step):
        model.head=original_head
        torch.save({'model':model.state_dict(),'config':asdict(config),'args':vars(a),'optimizer':opt.state_dict(),
                    'step':step,'manifest':manifest,'data_rng':gen.get_state(),'order_rng':rng.getstate(),
                    'order':order,'cursor':cursor,'torch_rng':torch.get_rng_state()},out/name)
        if a.compile_head: model.head=compiled_head
    compiled_head=model.head
    print(json.dumps({'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),
                      'serial_rules':a.rules,'rows_per_update':a.batch_chunks*a.levels*a.accumulate,
                      'microbatch':a.microbatch or a.batch_chunks*a.levels,
                      'microbatch_mode':'full' if a.microbatch==0 else 'limited',
                      'compile_rule':a.compile_rule,'capacity_factor':a.capacity_factor,'max_pattern':a.max_pattern,'max_replacement':a.max_replacement,'max_stars':a.max_stars,'optimizer_mode':optimizer_mode,'train_docs':len(train),'inference_semantics':'old-Nim exact on CPU','training_semantics':'C2 unbounded-capture DP mean-field surrogate'},ensure_ascii=False),flush=True)
    begin=time.perf_counter(); history=[]
    for step in range(start_step,a.steps):
        model.train(); opt.zero_grad(set_to_none=True)
        coefficient=anneal(step,a.steps,a.reg_max,a.reg_cycles,a.reg_warmup)
        total=torch.zeros((),device=device)
        for _ in range(a.accumulate):
            tokens,lens,target=next_batch(); rows=len(tokens)
            micro=a.microbatch or rows
            for first in range(0,rows,micro):
                sl=slice(first,first+micro)
                x=tokens[sl].to(device,non_blocking=device.type=='cuda')
                l=lens[sl].to(device,non_blocking=device.type=='cuda')
                y=target[sl].to(device,non_blocking=device.type=='cuda')
                pred=model(x,l,a.temperature,amp=a.amp)
                loss=regression_loss(pred,y,a.loss)
                weight=len(y)/(rows*a.accumulate)
                (loss*weight).backward(); total+=loss.detach()*weight
        reg=model.regularizer(a.temperature)
        (coefficient*reg).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,
                                            foreach=True if device.type=='cuda' else None)
        opt.step()
        if step%a.log_every==0 or step==a.steps-1:
            sync(device)
            row={'step':step,'loss':float(total),'lambda':coefficient,'impurity':float(reg.detach()),
                 'grad_norm':float(norm),'elapsed':time.perf_counter()-begin}
            history.append(row); print(json.dumps(row),flush=True)
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if (step+1)%a.eval_every==0 or step==a.steps-1:
            metric={'step':step,'validation_soft':evaluate(model,val,alphabet,a,device),
                    'validation_hard':evaluate(model,val,alphabet,a,device,hard=True)}
            print(json.dumps(metric),flush=True)
            with (out/'eval.jsonl').open('a') as f: f.write(json.dumps(metric)+'\n')
            save('last.pt',step)
    sync(device); training_seconds=time.perf_counter()-begin
    # Final model, no test-based checkpoint selection; heldout evaluated only here.
    result={'config':asdict(config),'args':vars(a),'training_seconds':training_seconds,'history':history,
            'test_soft':evaluate(model,test,alphabet,a,device),
            'test_hard':evaluate(model,test,alphabet,a,device,True),
            'limitations':['exact mode uses old-Nim wildcard semantics and CPU integer execution','soft capture lengths have no artificial bound beyond input buffer size','soft mode uses conditional-mean delimiter positions and is not a full probabilistic pushforward','geometry count combinatorial in max_pattern / max_stars but NOT capture length','soft intermediate strings have a finite tensor/output capacity; increase --capacity-factor to avoid overflow','soft output mapping currently uses quadratic memory and greedy scan uses triangular solve; long texts are costly','CUDA and MPS unverified in this environment']}
    if a.hard_finetune:
        for param in model.rules.parameters(): param.requires_grad_(False)
        head_opt=torch.optim.AdamW(model.head.parameters(),lr=a.lr)
        for _ in range(a.hard_finetune):
            head_opt.zero_grad(set_to_none=True); model.train()
            tok,lens,y=next_batch()
            micro=a.microbatch or len(tok)
            for first in range(0,len(tok),micro):
                sl=slice(first,first+micro)
                pred=model(tok[sl].to(device),lens[sl].to(device),a.temperature,hard=True,amp=a.amp)
                (F.mse_loss(pred,y[sl].to(device))*len(y[sl])/len(y)).backward()
            head_opt.step()
        result['test_hard_after_head_finetune']=evaluate(model,test,alphabet,a,device,True)
        # Separate inference checkpoint; last.pt stays resumable continuous training.
        model.head=original_head
        torch.save({'model':model.state_dict(),'config':asdict(config),'args':vars(a)},out/'hard_finetuned.pt')
    (out/'rules.json').write_text(json.dumps(model.export_rules(),indent=2))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
