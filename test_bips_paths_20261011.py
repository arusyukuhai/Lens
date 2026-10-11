"""Bounded BIPS information-channel regression tests."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import main as m
import lens_bips as bp


def _fixture():
    orig = b'alpha beta gamma delta abcxyz def'
    cases = []
    for j in range(3):
        clean = bytes((v + j) % 127 if v not in (32,) else v for v in orig)
        noisy = bytearray(clean); noisy[5] = ord('Q') + j
        noisy2 = bytearray(clean); noisy2[7] = ord('J') + j
        cases.append(SimpleNamespace(inputs=[list(clean),list(noisy),list(noisy2)],
                                     target=np.array([1.0,.45,.6])))
    roll = SimpleNamespace(slots=cases)
    genome = m.Genome([m.Rule([100+i], [100+i]) for i in range(14)],
                      readout_weights=[0.0]*14)
    f = np.zeros((9,14),dtype=np.float32)
    base = np.zeros(9)
    target = np.concatenate([case.target for case in cases])
    return roll,genome,f,base,target


def test_bips_forward_backward_real_two_stage_cpu_and_embedding():
    g = m.Genome([m.Rule([1],[1]) for _ in range(3)])
    tokens = bp.available_latent_tokens(g, 3)
    assert all(v not in g.embedding for v in tokens)
    p = bp.PathProposal(1.0,b'ab',b'cd')
    r1,r2 = bp.compile_path(m,g,p,tokens)
    assert r2.pattern == [tokens[0],-1,ord('c'),ord('d')]
    assert r2.replacement == [tokens[1],-1]
    assert all(m.is_rule_nonexpanding(r) for r in (r1,r2))
    a=m.embed_inputs([b'abXYcdEND'],g.embedding)[0]
    b,match,_,_=m.replace_once_cpu(a,r1,200)
    assert match and b == [tokens[0]] + list(b'XYcdEND')
    c,match,_,_=m.replace_once_cpu(b,r2,200)
    assert match and c == [tokens[1]] + list(b'XYEND')


def test_bips_three_rules_carry_distant_information_and_no_hidden_side_effect():
    g = m.Genome([m.Rule([1],[1]) for _ in range(4)])
    g.embedding = m.initial_embedding(True)
    tokens = bp.available_latent_tokens(g, 3)
    p = bp.PathProposal(1.0,b'ab',b'ef',b'cd')
    seq = bp.compile_path(m,g,p,tokens)
    s=m.embed_inputs([b'abXYcdZef!'],g.embedding)[0]
    for rule in seq:
        assert -2 not in rule.pattern and -3 not in rule.pattern
        assert all(x >= 0 or x == -1 for x in rule.pattern + rule.replacement)
        s,matched,overflow,_=m.replace_once_cpu(s,rule,200)
        assert matched and not overflow
    assert s[0]==tokens[2]
    # Input bytes and private latent intermediary are disjoint even with LUT.
    assert set(tokens).isdisjoint(set(g.embedding))


def test_bips_training_split_and_grouped_children():
    roll,g,feat,base,y = _fixture()
    proposals=bp.mine_paths(roll,g,feat,base,y,max_proposals=24,max_sources=16)
    assert proposals, 'need one valid multi-rule witness from the training split'
    assert set(p.depth for p in proposals)=={2,3}
    first=[(p.score,p.prefix,p.middle,p.suffix) for p in proposals]
    roll.slots[-1].inputs=[list(b'ZZZ')]*3
    roll.slots[-1].target=np.array([0.,1.,0.])
    y[-3:]=roll.slots[-1].target
    new=bp.mine_paths(roll,g,feat,base,y,max_proposals=24,max_sources=16)
    assert first==[(p.score,p.prefix,p.middle,p.suffix) for p in new]
    parent_snapshot=[(r.pattern[:],r.replacement[:]) for r in g.rules]
    children=bp.make_path_offspring(m,g,proposals,feat,8)
    assert children and len(children)<=8
    for child in children:
        assert child._origin_operator.startswith('bips:d')
        assert child._mix_rows in (2,3)
        assert child.embedding==g.embedding
        changed=[i for i,(a,b) in enumerate(zip(child.rules,g.rules))
                 if a.pattern!=b.pattern or a.replacement!=b.replacement]
        assert len(changed)==child._mix_rows
        assert changed==list(range(changed[0],changed[-1]+1))
        assert all(m.is_rule_nonexpanding(child.rules[i]) for i in changed)
        assert all(-2 not in child.rules[i].pattern and -3 not in child.rules[i].pattern for i in changed)
    assert parent_snapshot==[(r.pattern,r.replacement) for r in g.rules]


def test_bips_integration_two_generations(tmp_path):
    corpus=tmp_path/'corpus.txt'
    corpus.write_bytes(b'abcdefghij klmnopqrst uvwxyz\n===SPLIT===\n'
                       b'abcdefghijklmnopqrstuvwxyz zyxw\n===SPLIT===\n'
                       b'the little brown fox jumps over a hill\n===SPLIT===\n'
                       b'control flow optimization and memory safety')
    cmd=[sys.executable, str(Path(m.__file__)), '--backend','cpu-native',
         '--local-corpus',str(corpus),'--min-chunk','8','--max-chunk','100',
         '--trajectory-len','50','--cases','3','--samples','10','--rules','14',
         '--population','16','--elites','1','--generations','2',
         '--hof-size','8','--hof-eval','0','--inference-cases','1',
         '--inference-population','4','--inference-generations','2',
         '--inference-ensemble','2','--inference-budget-mode','direct',
         '--ablation-every','0','--guided-candidates','8','--guided-source-samples','24',
         '--bips-share','0.7','--bips-depth','3','--bips-sources','20',
         '--no-progress','--no-plot',
         '--save',str(tmp_path/'checkpoint.json'),
         '--current-save',str(tmp_path/'current.json'),
         '--history-csv',str(tmp_path/'history.csv')]
    p=subprocess.run(cmd,cwd=tmp_path,text=True,capture_output=True,timeout=90)
    assert p.returncode==0,p.stderr+'\n'+p.stdout
    assert 'bips_paths=' in p.stdout
    import csv
    with open(tmp_path/'history.csv') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert int(rows[0]['bips_added']) <= int(rows[0]['bips_proposals'])
    # A validation-aware CEGAR may reject every path on a tiny synthetic
    # corpus. Rejection is not a test failure; it prevents spurious promotion.
    assert 'bips_cegar_validation_rejected' in rows[0]
    assert 'bips_prev_evaluated_d2' in rows[1]
    if int(rows[0]['bips_added']):
        assert int(rows[1]['bips_prev_evaluated_d2']) + int(rows[1]['bips_prev_evaluated_d3']) > 0
    assert 'infPlan=1x4x2/2r' in p.stdout


def test_bips_native_parity_across_two_and_three_stages():
    import native_cpu_replacer as native
    g=m.Genome([m.Rule([255],[255]) for _ in range(5)])
    for p in (bp.PathProposal(1.0,b'ab',b'ef'),
              bp.PathProposal(1.0,b'ab',b'ef',b'cd')):
        rules=bp.compile_path(m,g,p,bp.available_latent_tokens(g,3))
        child=m.clone_genome_shallow(g)
        child.rules[1:1+len(rules)]=rules
        cases=[list(x) for x in (b'abXXcdYYef!!!',b'ab cd ef',b'abcdefffff',b'garbage')]
        a=m.trajectory_features_cpu(cases,child,100,hidden_rewrite_phases=2)
        b=native.trajectory_features_native(cases,child,100,hidden_rewrite_phases=2,cpu_threads=2)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(a[2],b[2])
        assert a[3]==b[3]
