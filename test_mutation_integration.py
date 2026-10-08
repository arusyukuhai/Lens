"""Static integration regressions. Actual Nim compile and --self-test required."""
import re
from pathlib import Path
s = Path(__file__).with_name('main.nim').read_text()
trial = s.split('proc trialRule(', 1)[1].split('\nproc changeRule(', 1)[0]
assert 'diff3Mix(a.a, b.a, c.a' in trial
assert 'diff3Mix(a.b, b.b, c.b' in trial
assert 'cfg.mutationRate > 0.0' in trial
assert 'cfg.mutationRate >= 1.0' in trial
assert 'mutateRuleOnce(result, cfg, vocab)' in trial
assert 'rand(99) < 85' not in trial
assert 'rand(99) < 9' not in trial
assert 'randomRule(' not in trial
mutation = s.split('proc mutateRuleOnce(', 1)[1].split('\nproc trialRule(', 1)[0]
assert 'mutateSeq(r.a' in mutation and 'mutateSeq(r.b' in mutation
assert 'result.a == r.a' in mutation and 'result.b == r.b' in mutation
assert 'randomOther(' in mutation
assert 'mutationRate: 0.05' in s
assert 'of "mutation-rate": result.mutationRate = parseFloat(value)' in s
assert '"mutationRate": cfg.mutationRate' in s
assert 'hasKey("mutationRate")' in s
assert 'checkDifferentialMutation()' in s
assert re.search(r'proc dominates\(.*?a.rho >= b.rho and a.inference >= b.inference.*?a.rho > b.rho or a.inference > b.inference',s,re.S)
assert re.search(r'if dominates\(trial.score, current.score\):', s)
assert 'ReplacePasses = 1' in s
assert 'diffusionSteps: 100' in s
assert 'chunkAdmission": "skip-oversize-v1"' in s
print('PASS: rare per-trial DE mutation, options, checkpoint migration, Pareto gate, one-pass/100-step/chunk policy intact')
