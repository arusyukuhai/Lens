"""Independent schedule/compatibility checks for the source-code changes.

This is NOT a Nim compile test and does not prove branch-selection semantics.
"""
from pathlib import Path

src = Path(__file__).with_name('cms.nim').read_text()

def eval_due(step: int, total: int, every: int) -> bool:
    return step == 0 or step == total - 1 or (step + 1) % every == 0

for count in (1, 2, 3, 5, 12, 100):
    for freq in (1, 2, 5, 11, 200):
        due = [i for i in range(count) if eval_due(i, count, freq)]
        assert due[0] == 0 and due[-1] == count - 1
        assert all(eval_due(i, count, freq) for i in due)
        if freq == 1: assert due == list(range(count))

assert [i for i in range(12) if eval_due(i, 12, 5)] == [0, 4, 9, 11]
assert 'if not evalPowerProvided: result.evalPower = result.finePower' in src
assert 'if result.evalEvery < 1:' in src
assert 'let doEvaluate = evaluationDue(step, cfg.steps, cfg.evalEvery)' in src
assert '(if doEvaluate: "fine" else: "coarse")' in src
assert 'selectedAdd = bestCurrentBranches(shortlistAdd, cfg.beamWidth)' in src
assert 'selectedReplace = bestCurrentBranches(shortlistReplace, cfg.beamWidth)' in src
assert 'selectedAdd = rankBranches(fineRows, shortlistAdd, cfg.beamWidth,' in src
assert 'selectedReplace = rankBranches(fineRows, shortlistReplace, cfg.beamWidth,' in src
assert 'saveRules(cfg.outDir / "rules.tsv", current.rules)' in src
assert '" validPearson=skipped rules="' in src
assert 'evalTrainRows = makeRows(source, cfg, codec, step, false, evalCache,' in src
assert 'var evalCache = DiffusionCache' in src
assert 'validation = makeRows(source, cfg, codec, 0, true, fineCache,' in src
assert 'profileEval.evaluateEncodeSeconds' in src
assert 'proc evaluationDue' in src
assert 'doAssert evaluationDue(t, 12, 5)' in src
print('PASS: schedule for 30 cases, evaluation flag wiring, cache isolation, sparse plots and saved rules')
