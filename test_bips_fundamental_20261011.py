"""Source-balanced BIPS learning and inference-aligned edit contrast tests."""
from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path
import numpy as np
import lens_bips_cegar as cg
import main as m


def test_confirmation_can_use_two_training_originals_not_population_holdout():
    # Source 2 is an outer evaluation case and must not be consulted.
    group=np.repeat([0,1],45)
    src=SimpleNamespace(origin_case=0)
    train,val=cg._fresh_confirmation_groups(src,[0,1,45,46],group,64)
    assert train and val
    assert len(train)>=8 and len(val)>=8
    assert all(group[k]==0 for k in train)
    assert all(group[k]==1 for k in val)
    assert not set(train+val).intersection([0,1,45,46])


def test_edit_contrast_prefers_real_repairs_and_ignores_training_oracle(monkeypatch):
    # The downstream score prefers 'A' at the corrupted position. The source
    # readout learns this preference; validation clean string only builds the
    # correct and incorrect edits. The other feature is identically zero.
    train_rows=[b'AX' if i%2 else b'BX' for i in range(20)]
    val_rows=[b'AX'] + [b'BX' for _ in range(16)]
    rows=train_rows+val_rows
    groups=np.array([0]*20+[1]*len(val_rows))
    y=np.array([1.0 if i%2 else 0.0 for i in range(20)]+[1.0]+[.8]*16)
    x=np.zeros((len(rows),2));b=np.zeros(len(rows))
    x[:20,0]=[1.0 if i%2 else 0.0 for i in range(20)]
    source=object();child=object()
    def fake_features(mod,g,samples,*args):
        fx=np.zeros((len(samples),2),dtype=float)
        if g is child:
            fx[:,0]=[float(s[0]==ord('A')) for s in samples]
        return fx,np.zeros(len(samples))
    monkeypatch.setattr(cg,'_concrete_features',fake_features)
    # The already-executed child features on the source training rows:
    child_training=np.zeros((20,2));child_training[:,0]=x[:20,0]
    out=cg._edit_contrast_on_other_original(
        m,source,child,list(range(20)),list(range(21,37)),
        rows,groups,y,x,b,child_training,np.zeros(20),128,2,1,3)
    assert out is not None
    gain,npairs=out
    assert gain>0.3 and npairs>0


def test_default_bips_case_rebalance_preserves_total_work(tmp_path):
    src=tmp_path/'corpus.txt'
    src.write_bytes((b'abc def ghi jkl mno pqr stu vwx yz\n===SPLIT===\n'
                     b'testing information transfer through rules\n===SPLIT===\n'
                     b'symbolic pattern rewrite train crossfit\n===SPLIT===\n'
                     b'a separate validation original from the code\n'))
    cmd=[sys.executable,str(Path(m.__file__)),'--backend','cpu-native',
         '--learning-mode','guided','--local-corpus',str(src),
         '--min-chunk','8','--max-chunk','200','--trajectory-len','70',
         '--rules','12','--population','9','--elites','1',
         '--generations','1','--samples','12',
         '--bips-depth','3','--bips-sources','8','--bips-cegar-budget','3',
         '--hof-size','4','--hof-eval','0','--no-inference',
         '--no-plot','--checkpoint-every','0',
         '--save',str(tmp_path/'best.json'),
         '--current-save',str(tmp_path/'latest.json'),
         '--history-csv',str(tmp_path/'history.csv')]
    completed=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert completed.returncode==0,completed.stderr+'\n'+completed.stdout
    assert 'BIPS case rebalance 2x12 -> 3x8' in completed.stdout
    assert 'on 24 trajectories' in completed.stdout
    assert 'confirmInsufficient' in completed.stdout


def test_contrast_only_reranks_confirmed_candidates(monkeypatch):
    n=90
    groups=np.array([0]*45+[1]*45)
    rows=[b'AX' if i%2 else b'BX' for i in range(n)]
    x=np.zeros((n,2));b=np.zeros(n);y=np.linspace(0,1,n)
    child=object();parent=object()
    meta=([0,1,2,3,45,46,47,48],4,8,2,None,None)
    monkeypatch.setattr(cg,'_concrete_features',
                        lambda mod,g,samples,*args:
                        (np.zeros((len(samples),2)),np.zeros(len(samples))))
    monkeypatch.setattr(cg,'_validate_concrete_candidate',lambda *args:(.2,.04,.1))
    monkeypatch.setattr(cg,'_edit_contrast_on_other_original',
                        lambda *args:(.7,3))
    common=dict(budget=1,max_probes=48,min_mse_ratio=.01,
                origin_by_child={id(child):0})
    basic=cg._confirm_macro_candidates(None,parent,[(1.,child,meta)],rows,groups,x,b,y,
                                       128,2,1,stats=cg.RefinementStatistics(),**common,
                                       edit_contrast_weight=0)
    stats=cg.RefinementStatistics()
    better=cg._confirm_macro_candidates(None,parent,[(1.,child,meta)],rows,groups,x,b,y,
                                        128,2,1,stats=stats,**common,
                                        edit_contrast_weight=.15)
    assert len(basic)==len(better)==1
    assert abs((better[0][0]-basic[0][0])-.105)<1e-12
    assert stats.edit_contrast_pairs==3 and stats.edit_contrast_positive==1
