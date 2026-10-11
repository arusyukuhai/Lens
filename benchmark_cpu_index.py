"""Repeatable native CPU inference benchmark. Compare with prior ZIP on same host.

python benchmark_cpu_index.py --scenario sparse
python benchmark_cpu_index.py --scenario dense

Measurements exclude first-use compilation and native context construction; reports
creation separately. Results are synthetic, not an Apple Silicon performance claim.
"""
import argparse
import random
import statistics
import time
import numpy as np
import main
import native_cpu_replacer as native


def bench_one(scenario, count, n_rules, n_candidates, n_bytes, threads, repeats):
    rand=random.Random(1234)
    candidates=[[rand.randrange(256 if scenario!='dense' else 40) for _ in range(n_bytes)]
                for _ in range(n_candidates)]
    models=[]
    if scenario=='dense':
        bank=[main.Rule([i],[i+1]) for i in range(30)]+[
            main.Rule([4,-1,5],[-1,4]),main.Rule([6,-2,7],[-128,-1])]
        for _ in range(count):
            g=main.Genome(rand.choices(bank,k=n_rules))
            g.readout_weights=[rand.uniform(-.05,.05) for _ in g.rules]
            models.append(g)
    else:
        for _ in range(count):
            rules=[]
            for i in range(n_rules):
                v=rand.randrange(512 if scenario=='sparse' else 160)
                if i % 4==0: rules.append(main.Rule([v,rand.randrange(512)],[rand.randrange(512)]))
                else: rules.append(main.Rule([v],[rand.randrange(512)]))
            g=main.Genome(rules)
            g.readout_weights=[rand.uniform(-.05,.05) for _ in g.rules]
            models.append(g)
    input_data=native.prepare_inputs(candidates)
    t0=time.perf_counter()
    ctx=native.NativeInferenceContext(models,256,cpu_threads=threads)
    setup=time.perf_counter()-t0
    try:
        outputs=[];times=[]
        for _ in range(repeats+1):
            t=time.perf_counter(); out=ctx.score(input_data)
            dt=time.perf_counter()-t
            outputs.append(out)
            times.append(dt)
        for a in outputs[1:]:np.testing.assert_array_equal(outputs[0],a)
        print(f'{scenario} models={count} rules={n_rules} candidates={n_candidates} bytes={n_bytes} threads={threads} '
              f'setup={setup:.5f}s median={statistics.median(times[1:]):.5f}s '
              f'checksum={np.sum(out):.12g}')
    finally:ctx.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',choices=['sparse','dense','mixed'],default='sparse')
    ap.add_argument('--models',type=int,default=64)
    ap.add_argument('--rules',type=int,default=750)
    ap.add_argument('--candidates',type=int,default=28)
    ap.add_argument('--bytes',type=int,default=96)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--repeats',type=int,default=5)
    x=ap.parse_args()
    bench_one(x.scenario,x.models,x.rules,x.candidates,x.bytes,x.threads,x.repeats)
