"""Static wiring and schedule sanity checks for Fine EVERY generation.
This is NOT a Nim build or end-to-end numeric test.
"""
from pathlib import Path
src = Path(__file__).with_name('cms.nim').read_text()
def eval_due(step,total,every):
    return step==0 or step==total-1 or (step+1)%every==0
for total in [1,2,5,12,100]:
    for every in [1,2,5,11,200]:
        due = [s for s in range(total) if eval_due(s,total,every)]
        assert due[0] == 0 and due[-1] == total-1
        if every==1: assert due == list(range(total))
assert [i for i in range(12) if eval_due(i,12,5)] == [0,4,9,11]
assert 'let doEvaluate = evaluationDue(step, cfg.steps, cfg.evalEvery)' in src
assert 'if sameGrid:\n      fineRows = rows\n      fineStageCache = stageCache' in src
assert 'let selectedAdd = rankBranches(fineRows, shortlistAdd, cfg.beamWidth,' in src
assert 'let selectedReplace = rankBranches(fineRows, shortlistReplace, cfg.beamWidth,' in src
assert 'let selectedParents = rankBranches(fineRows, parents, cfg.beamWidth,' in src
assert 'let selectedBaseline = selectedParents[0].score' in src
assert 'echo "[compare] step=", step, " scoring=fine"' in src
assert 'if doEvaluate:\n      # The evaluation schedule is independent' in src
assert 'validation = makeRows(source, cfg, codec, 0, true, fineCache,' in src
assert 'if cfg.evalPower == cfg.finePower:' in src
assert 'saveRules(cfg.outDir / "rules.tsv", current.rules)' in src
assert '" finePearson=", current.score' in src
assert '" evaluation=skipped time="' in src
assert 'selectedAdd = bestCurrentBranches(shortlistAdd, cfg.beamWidth)' not in src
assert '(if doEvaluate: "fine" else: "coarse")' not in src
print('PASS: Fine reranking each generation; periodic detailed diagnostics independent; first/last checkpoints')
