import std/[unittest]
import lens_core, lens_dependency, lens_search

suite "v10 dependency replay and joint counterfactual ranking":
  test "changed root never fires: common prefix still advances in BOTH chains":
    # v9 falsely reported changes in R2/R3 because revised R0 was not run.
    let baseline = @[
      Rule(lhs: @[1], rhs: @[10]),
      Rule(lhs: @[23], rhs: @[24]),
      Rule(lhs: @[10], rhs: @[11]),
      Rule(lhs: @[11], rhs: @[12])]
    var edited = copyRules(baseline)
    edited[1].lhs = @[25]
    let impacts = affectedRuleOrder(baseline, edited,
        @[@[1], @[1,1], @[7,1]], 1, 32)
    check impacts.len == 0

  test "a changed root affects a distant rule but not unrelated rules":
    let baseline = @[
      Rule(lhs: @[1], rhs: @[10]),
      Rule(lhs: @[10], rhs: @[20]),
      Rule(lhs: @[7], rhs: @[8]),
      Rule(lhs: @[20], rhs: @[30]),
      Rule(lhs: @[5], rhs: @[6])]
    var edited = copyRules(baseline)
    edited[1].rhs = @[21]
    let impacts = affectedRuleOrder(baseline, edited,
        @[@[1,7,5], @[1,5], @[1,7], @[2,7,5]], 1, 32)
    check impacts.len == 1
    check impacts[0].index == 3
    check impacts[0].firingDrift > 0.0

  test "pairwise synergy promotes affected target without losing top drift":
    let impacts = @[
      RuleImpact(index: 2, priority: 0.95),
      RuleImpact(index: 5, priority: 0.74),
      RuleImpact(index: 8, priority: 0.68)]
    let rescues = @[-0.06, 0.01, -0.03]
    let tested = @[true,true,true]
    let before = rankedRegrowTargets(impacts, rescues, tested, 2)
    check before[0].index == 2
    check before[1].index == 5
    let withPair = rankedRegrowTargets(impacts, rescues, tested, 2,
                                      @[0.0,0.0,0.12])
    check withPair[0].index == 2
    check withPair[1].index == 8

  test "disabled pair probes match original v9 ranking":
    let impacts = @[
      RuleImpact(index: 3, priority: 0.9),
      RuleImpact(index: 6, priority: 0.5)]
    let p = rankedRegrowTargets(impacts, @[0.0,0.1], @[true,true], 2)
    let q = rankedRegrowTargets(impacts, @[0.0,0.1], @[true,true], 2, @[])
    check p == q
