"""CPU-native outer Spearman benchmark; measures model packing+dispatch+readout.

Run on the target Mac: python benchmark_outer_native.py --models 64 --samples 200 --bytes 900
This synthetic benchmark is NOT a substitute for full training profiling.
"""
import argparse
import random
import statistics
import time
import numpy as np
import native_cpu_replacer as native
import main


def model(seed, rules, alphabet):
    rng=random.Random(seed)
    out=[]
    for _ in range(rules):
        x,y,z=(rng.randrange(alphabet) for i in range(3))
        typ=rng.randrange(12)
        if typ<=3: out.append(main.Rule([x,y],[z]))
        elif typ<=6: out.append(main.Rule([x],[y]))
        elif typ<=8: out.append(main.Rule([x,y,z],[x]))
        elif typ==9: out.append(main.Rule([x,-1,y],[y,-1]))
        elif typ==10: out.append(main.Rule([-2,x,-2],[-128,-1]))
        else: out.append(main.Rule([x,x],[x]))
    return main.Genome(out)


def measure(models, inputs, targets, case_ids, sample_ids, threads=8, batch=32):
    packed=native.prepare_inputs(inputs)
    all_old=[]
    t0=time.perf_counter()
    for g in models:
        all_old.append(native.trajectory_features_native(packed,g,2048,cpu_threads=threads,return_states=False))
    serial=time.perf_counter()-t0
    output=[]
    t0=time.perf_counter()
    for b in range(0,len(models),batch):
        gs=models[b:b+batch]
        with native.NativeInferenceContext(gs,2048,cpu_threads=threads) as ctx:
            f, baselines, stats=ctx.features(packed)
            for i in range(len(gs)):
                output.append((f[i,:,:len(gs[i].rules)],baselines[i],stats[i]))
    fused=time.perf_counter()-t0
    for ref,got in zip(all_old,output):
        for a,b in zip(ref[:3],got):np.testing.assert_array_equal(a,b)
    old_r=[];new_r=[]
    x=output
    for _ in range(2):
        t0=time.perf_counter()
        for g,(feat,b,st) in zip(models,x):
            main.score_genome_features(g,feat,b,targets,case_ids,sample_ids,.1)
        old_r.append(time.perf_counter()-t0)
        t0=time.perf_counter()
        pairs=main.case_adjacent_rank_pairs(targets,case_ids)
        for g,(feat,b,st) in zip(models,x):
            main.score_genome_features(g,feat,b,targets,case_ids,sample_ids,.1,prepared_pairs=pairs)
        new_r.append(time.perf_counter()-t0)
    print(f'outer models={len(models)} samples={len(inputs)} input_bytes={max(map(len,inputs))}, rules={len(models[0].rules)} threads={threads}')
    print(f'old C++ launch/pack={serial:.3f}s fused launch/pack={fused:.3f}s speedup={serial/max(fused,1e-9):.2f}x')
    print(f'old Python readout={statistics.median(old_r):.3f}s shared-ranked={statistics.median(new_r):.3f}s')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',type=int,default=32)
    ap.add_argument('--rules',type=int,default=1500)
    ap.add_argument('--samples',type=int,default=100)
    ap.add_argument('--bytes',type=int,default=600)
    ap.add_argument('--cpu-threads',type=int,default=8)
    ap.add_argument('--cpu-model-batch',type=int,default=32)
    args=ap.parse_args()
    rng=random.Random(24)
    models=[model(i,args.rules,128) for i in range(args.models)]
    inputs=[[rng.randrange(110) for _ in range(args.bytes)] for j in range(args.samples)]
    cases=np.arange(args.samples,dtype=np.int32)//max(1,args.samples//2)
    y=np.asarray([rng.random() for _ in inputs]);ids=np.arange(len(inputs),dtype=np.int32)
    measure(models,inputs,y,cases,ids,args.cpu_threads,args.cpu_model_batch)
