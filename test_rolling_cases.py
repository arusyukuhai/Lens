"""Rolling training-case schedule: one change, correct cadence, resumability."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import main as m


class RollingCaseTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [f'chunk-{i:03d} ABCDEF'.encode() for i in range(32)]

    def test_one_slot_on_each_boundary(self):
        schedule = m.RollingTrainingCases(self.corpus, cases=8, seed=79, period=3)
        previous = None
        changed_generations = []
        for gen in range(29):
            rows = schedule.for_generation(gen)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(set(rows)), 8)
            if previous is not None:
                changed = [i for i,(a,b) in enumerate(zip(previous,rows)) if a != b]
                if gen % 3 == 0:
                    self.assertEqual(changed, [(gen//3-1) % 8])
                    changed_generations.append(gen)
                else:
                    self.assertEqual(changed, [])
            previous=rows
        self.assertEqual(changed_generations, [3,6,9,12,15,18,21,24,27])

    def test_one_case_and_small_corpus(self):
        for corpus, cases in [(self.corpus[:2], 6), (self.corpus[:1], 4), (self.corpus[:8],8)]:
            rolling=m.RollingTrainingCases(corpus, cases, 11, 1)
            prev=rolling.for_generation(0)
            for gen in range(1,10):
                curr=rolling.for_generation(gen)
                changed=sum(a!=b for a,b in zip(prev,curr))
                self.assertEqual(changed, 0 if len(corpus)==1 else 1)
                prev=curr

    def test_snapshot_resume_includes_rng(self):
        direct=m.RollingTrainingCases(self.corpus,8,200,3)
        for gen in range(16):
            direct.for_generation(gen)
        resumed=m.RollingTrainingCases(self.corpus,8,200,3, direct.snapshot())
        for gen in range(16,77):
            self.assertEqual(direct.for_generation(gen),resumed.for_generation(gen))
            self.assertEqual(direct.indices,resumed.indices)

    def test_invalid_corpus_rejected_on_resume(self):
        original=m.RollingTrainingCases(self.corpus, 4, 61, 3)
        state=original.snapshot()
        altered=list(self.corpus)
        altered[0]=b'changed corpus'
        with self.assertRaisesRegex(ValueError,'Checkpoint case rotation state differs'):
            m.RollingTrainingCases(altered,4,61,3,state)

    def test_legacy_checkpoint_api_unchanged(self):
        corpus=self.corpus
        gs=[m.new_genome(4,random.Random(1),corpus,m._bigram_seeds(corpus))]
        with tempfile.TemporaryDirectory() as td:
            f=str(Path(td)/'ckpt.pkl')
            m.save_checkpoint(f,4,gs,gs[0],[],random.Random(5))
            self.assertEqual(len(m.load_checkpoint(f)),6)
            self.assertEqual(len(m.load_checkpoint(f,include_rotation_state=True)),7)
            self.assertIsNone(m.load_checkpoint(f,include_rotation_state=True)[-1])

    def test_cli_uninterrupted_equals_checkpoint_restart(self):
        # Full GA test, not just independently stepping case schedules.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            corpus=root/'corpus.txt'
            corpus.write_bytes(b'===SPLIT==='.join(self.corpus))
            def run(tag,until,load=''):
                checkpoint=str(root/f'{tag}.pkl')
                args=[sys.executable,str(Path(m.__file__)),'--backend','cpu',
                      '--local-corpus',str(corpus),'--rules','7','--population','10',
                      '--cases','4','--case-rotate-every','2','--generations',str(until),
                      '--seed','123','--cpu-workers','2','--no-tqdm','--no-plot',
                      '--checkpoint',checkpoint,'--checkpoint-every','2',
                      '--save',str(root/f'{tag}.json'),
                      '--history-csv',str(root/f'{tag}.csv')]
                if load:
                    args += ['--load',load]
                subprocess.run(args,cwd=root,capture_output=True,text=True,check=True)
                return checkpoint
            run('continuous',9)
            checkpoint=run('staged',5)
            run('staged',9,checkpoint)
            full=json.loads((root/'continuous.json').read_text())
            staged=json.loads((root/'staged.json').read_text())
            self.assertEqual(full,staged)
            # Timing differs, but fitness and rotation schedule must be exact.
            import csv
            with (root/'continuous.csv').open() as f:
                rows1=list(csv.DictReader(f))
            with (root/'staged.csv').open() as f:
                rows2=list(csv.DictReader(f))
            self.assertEqual(len(rows1),9)
            self.assertEqual(len(rows2),9)
            for a,b in zip(rows1,rows2):
                for col in ('generation','best_accuracy','mean_accuracy',
                            'median_accuracy','case_rotation','rotated_case'):
                    self.assertEqual(a[col],b[col],f'gen={a["generation"]} field={col}')
            self.assertEqual([r['rotated_case'] for r in rows1],
                             ['-1','-1','0','-1','1','-1','2','-1','3'])


if __name__=='__main__':
    unittest.main()
