#!/usr/bin/env python3
"""Standalone regression tests for the optimized single-pass Replacer.

Run: python test_fast.py
Does not require the previous version; checks exact CPU behavior and host-side
MPS staging. Metal compilation needs a Mac with MPS and is not tested here.
"""
import random
import numpy as np
import main as m

def check_cpu():
    rng = random.Random(1034)
    for _ in range(70):
        rules = []
        for i in range(rng.randrange(5,55)):
            a, b = rng.randrange(256), rng.randrange(256)
            pattern = rng.choice(([a], [a,b], [a,-1,b], [-1,b], [a,-1]))
            rep = rng.choice(([a], [], [-1], [-32], [a,b]))
            rules.append(m.Rule(list(pattern),list(rep)))
        genome = m.Genome(rules)
        inputs = [[rng.randrange(256) for _ in range(rng.randrange(1,90))] for j in range(4)]
        got = m.trajectory_features_cpu(inputs, genome, 200)
        ref_features=np.zeros_like(got[0]); ref_base=np.zeros_like(got[1]); ref_stats=np.zeros_like(got[2]); ref_states=[]
        for j, raw in enumerate(inputs):
            state=m.embed_inputs([raw],genome.embedding)[0]
            ref_stats[j,0]=1
            for ri,rule in enumerate(rules):
                state, matched, overflow = m.replace_once_cpu(state, rule, 200)
                if matched: ref_features[j,ri]=1; ref_stats[j,1]+=1
                if overflow:
                    ref_base[j]-=1;ref_stats[j,2]=1; break
            ref_states.append(state)
        assert np.array_equal(got[0],ref_features)
        assert np.array_equal(got[1],ref_base)
        assert np.array_equal(got[2],ref_stats)
        assert got[3]==ref_states
    print('CPU exact feature/state comparison: PASS')

def check_hof():
    rng=random.Random(1011)
    for _ in range(150):
        n=rng.randrange(1,130)
        candidate=m.Genome([m.Rule([i%256],[i%256]) for i in range(n)])
        archive=[]
        for j in range(20):
            rules=[m.Rule(list(r.pattern),list(r.replacement)) for r in candidate.rules]
            for k in range(rng.randrange(n+1)):
                rules[rng.randrange(n)].pattern=[rng.randrange(256)]
            embedding=list(candidate.embedding)
            for k in range(rng.randrange(0,60)):
                embedding[rng.randrange(256)]=rng.randrange(512)
            archive.append(m.HallOfFameEntry(m.Genome(rules,embedding=embedding)))
        t=rng.choice((0.,.015,.03,.3,.6,.92))
        scores=[m.genome_structural_distance(candidate,e.genome) for e in archive]
        want=int(np.argmin(scores)) if min(scores)<t else -1
        assert m._nearest_hof_below_threshold(candidate,archive,t)==want
    print('HoF nearest-lineage exactness: PASS')

def check_inference_cache():
    def evaluator(genomes,candidates,backend,max_output,**kwargs):
        x=np.asarray(candidates,dtype=np.float64)
        return np.asarray([x[:,::i+1].sum(axis=1) for i in range(len(genomes))])
    original=m.evaluate_inference_candidates
    try:
        m.evaluate_inference_candidates=evaluator
        random.seed(7)
        corpus=[b'abcdefghijklmnopq'*4,b'ABCDEFGH12345678'*4]
        sampler=m.CorpusSampler(corpus)
        data=m.RollingInferenceSet(corpus,sampler,2,48,.09)
        genomes=[m.Genome([m.Rule([65],[66])],fitness=.7+i*.001,
                          inference_accuracy=.15,readout_weights=[1.]) for i in range(4)]
        res=m.run_string_inference_ga(genomes,data,sampler,8,7,4,'cpu',4096)
        assert res['inference_cache_hits']>0
        assert res['model_jobs']==res['kernel_jobs']
        assert all(np.isfinite(g.inference_accuracy) for g in genomes)
    finally:
        m.evaluate_inference_candidates=original
    print('Inference GA repeated-candidate memoization: PASS')

def check_staging():
    if m.torch is None: return
    torch=m.torch
    # Test the MPS evaluator's input-prefix host staging without requiring an
    # MPS machine or compiling a Metal kernel.
    ev=m.MpsPopulationEvaluator.__new__(m.MpsPopulationEvaluator)
    ev.sample_count=20
    ev.raw_stride=24
    ev.raw_inputs=torch.zeros((20,24),dtype=torch.int32)
    ev.raw_lengths=torch.zeros(20,dtype=torch.int32)
    ev.set_inputs([[1,2,3],[1,2,3],[4,5]],refresh_anchor_statistics=True)
    assert ev.active_sample_count==3 and ev.unique_input_count==2
    assert ev._sample_alias.tolist()==[0,0,2]
    assert ev.raw_lengths[:3].tolist()==[3,0,2]
    counts=ev._raw_byte_counts.copy()
    ev.set_inputs([[100,200],[99]],refresh_anchor_statistics=False)
    assert ev.active_sample_count==2 and ev.raw_lengths[:2].tolist()==[2,1]
    assert ev.raw_inputs[0,:2].tolist()==[100,200]
    assert np.array_equal(counts,ev._raw_byte_counts)
    print('MPS input-prefix staging (host simulation): PASS')

if __name__=='__main__':
    m.self_test()
    check_cpu()
    check_hof()
    check_inference_cache()
    check_staging()
    print('ALL OPTIMIZATION TESTS: PASS')
