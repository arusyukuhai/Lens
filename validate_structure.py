"""Nim source static checks. NOT a substitute for nim c or training execution."""
from pathlib import Path
from pygments.lexers import get_lexer_by_name
from pygments.token import Error, Comment, String

src=Path('main.nim').read_text(encoding='utf8')
replace=Path('replace.nim').read_text(encoding='utf8')
ridge=Path('ridge.nim').read_text(encoding='utf8')
lexer=get_lexer_by_name('nim')
tokens=list(lexer.get_tokens(src))
assert not [(t,v) for t,v in tokens if t in Error], 'Invalid Nim lexical token'
matching={'(':')','[':']','{':'}'}
stack=[]
for typ, text in tokens:
    if typ in Comment or typ in String: continue
    for ch in text:
        if ch in matching: stack.append(matching[ch])
        elif ch in matching.values():
            assert stack and stack.pop()==ch, f'Unbalanced closing {ch}'
assert not stack, f'Unclosed symbols {stack}'
checks={
    'replace API fallback': 'proc replaceSeqCompiled*' in replace and 'when compiles(replaceSeqCompiledInPlace(' in src and 'replaceSeqCompiled(state, pat, replacement)' in src,
    'diff3 operator': 'diff3Apply(donorBase, donorChanged, target)' in src,
    'rank ridge': 'fitPairwiseRankRidgeAllModel(' in src and 'proc fitPairwiseRankRidgeAllModel*' in ridge,
    'two replacement passes': 'ReplacePasses = 2' in src and '0..<ReplacePasses' in src,
    'single search width': 'SearchWidth = 1' in src and 'BeamWidth' not in src,
    'single DE population': 'proc evolveGreedy(' in src and 'for iter in 1..cfg.iterations:' in src and 'for i in 0..<population.len:' in src,
    'no global beam selection': all(s not in src for s in ['chooseBeam(', 'rankCandidates(', 'evolveBeam(', 'trainer.beams']),
    'only active rule changes': 'candidateWithEditedRule(current, ruleIndex, r, ctx, cfg)' in src and 'result.rules[i] = (if i == ruleIndex: replacement else: rule)' in src,
    'full model fitness': 'for i in 0..<m.rules.len:' in src and 'doAssert patterns.len == m.rules.len' in src and 'predictFeatures(txt, m, patterns)' in src,
    'regression for back stage cascade': 'proc checkBackwardStageSemantics()' in src and 'doAssert after == @[1.0, 1.0, 1.0]' in src,
    'strict Pareto DE acceptance': 'if dominates(trial.score, current.score):' in src and 'if better(trial.score, current.score):' not in src,
    'strict Pareto incumbent acceptance': 'if greedyAccept(result, trial):' in src and 'if dominates(proposal.score, incumbent.score):' in src,
    'incumbent acceptance regression': 'proc checkGreedyAcceptance()' in src and 'checkGreedyAcceptance()' in src,
    'fixed 100 diffusion': 'diffusionSteps: 100' in src and 'for k in 0..<cfg.diffusionSteps:' in src and 'diffusionRate' not in src,
    'correct corruption labels': 'float64(incorrect)/float64(clean.len)' in src,
    'only corrupted inference positions': 'result[p] = randomOtherWithRng(result[p], vocab, rng)' in src and 'for p in c.loci:' in src,
    'inference target only for scoring': src.count('c.clean[')==1,
    'five-point raw MA historical best': 'rhoBestMA = max(t.history[last-1].rhoBestMA, t.history[last].rhoMA)' in src,
    'version 3 single model checkpoint': 'root["version"] = %3' in src and 'root["model"] = toJsonModel(t.current.model)' in src,
    'legacy version 2 checkpoint import': 'root["beams"][0]' in src and 'if version == 3:' in src,
}
for k, ok in checks.items(): print(('PASS' if ok else 'FAIL') + ': ' + k)
assert all(checks.values()), 'At least one condition failed'
print('PASS: Nim tokenization and bracket balance (source-level checks only)')
