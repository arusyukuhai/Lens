import std/[unittest, math, sequtils]
import lens_search, lens_dependency, lens_core, cms_miner

suite "Counterfactual-guided nonlocal Regrow":
  test "positive mask rescue wins a slot while strongest drift is preserved":
    let impacted = @[
      RuleImpact(index: 2, priority: 0.92),
      RuleImpact(index: 6, priority: 0.74),
      RuleImpact(index: 9, priority: 0.35)]
    let ranked = rankedRegrowTargets(impacted, @[-0.20,0.08,0.16],
                                    @[true,true,true],2)
    check ranked.len == 2
    check ranked[0].index == 2
    check ranked[1].index == 9

  test "probe disabled falls back to drift instead of inventing rescue":
    let impacted = @[
      RuleImpact(index: 4,priority: 0.8),
      RuleImpact(index: 7,priority: 0.4)]
    let ranked = rankedRegrowTargets(impacted, @[0.0,0.0],
                                    @[false,false],2)
    check ranked.len == 2
    check ranked[0].index == 4
    check ranked[1].index == 7

  test "restoration screen finds distant-pattern firing and keeps exploration":
    let pool = @[
      PatternEntry(value: @[99],rank: 100),
      PatternEntry(value: @[88],rank: 90),
      PatternEntry(value: @[7],rank: 80),
      PatternEntry(value: @[4],rank: 70)]
    let stages = @[@[7,88],@[4],@[7,99],@[88,4],@[7],@[99]]
    let wanted = @[1.0,0.0,1.0,0.0,1.0,0.0]
    let chosen = restorationShortlist(pool,stages,wanted,2)
    check chosen.len == 2
    check chosen[0].value == @[7] # Exact firing restoration despite lower frequency
    check chosen[1].value == @[99] # Exploration retained

  test "disabled node has no effect when its literal exceeds all trained tokens":
    let dataset = trajectories(@[@[1,2,1,2],@[2,2,1]],44,1,4)
    let old = @[Rule(lhs: @[1],rhs: @[256]),
                Rule(lhs: @[256],rhs: @[257]),
                Rule(lhs: @[2],rhs: @[258])]
    var masked = copyRules(old)
    masked[1].lhs = @[0x200000]
    let actual = traceRules(masked,dataset)
    let expected = traceRules(@[old[0],old[2]],dataset)
    check actual.texts == expected.texts
    check actual.columns[1].allIt(it == 0.0)
