import std/[unittest, math, sequtils]
import lens_validation, lens_core, lens_search, lens_dependency

suite "Disjoint source validation and rollback":
  test "three pools plus protected holdout never overlap":
    let p = splitChunks(103,3,2,16,32)
    check p.trainEnd == 53
    check p.searchEnd == 69
    check p.acceptEnd == 101
    check p.total == 103
    for i in 0..<100:
      let search = rotatingIndices(p.trainEnd,p.searchEnd,2,i)
      let accept = rotatingIndices(p.searchEnd,p.acceptEnd,3,i)
      for x in search:
        check x >= p.trainEnd and x < p.searchEnd
      for x in accept:
        check x >= p.searchEnd and x < p.acceptEnd
      check search[0] != accept[0]
  test "small corpus gets at least one search and acceptance chunk":
    let p = splitChunks(5,3,0,64,64)
    check p.trainEnd == 3
    check p.searchEnd == 4
    check p.acceptEnd == 5
    expect ValueError:
      discard splitChunks(4,3,0,64,64)
  test "acceptance disallows degradation and penalizes rule bloat":
    check canAccept(0.61,2,0.60,2,0.001,0.002)
    check not canAccept(0.601,10,0.60,2,0.001,0.002)
    check not canAccept(0.59,1,0.60,2,0.001,0.002)
    check not canAccept(NaN,1,0.60,2,0.001,0.002)
  test "rollback preserves prefix and can recover shorter alternative":
    let clean = @[@[1,2,1],@[2,1,1]]
    let tr = trajectories(clean,44,1,3)
    let va = trajectories(clean,55,1,3)
    var cfg = defaultSearchConfig()
    cfg.mining.maxGram=1; cfg.mining.maxTotalGram=1
    cfg.mining.maxWildcards=0; cfg.mining.width=1024
    cfg.mining.depth=4; cfg.beam=2; cfg.regrowBeam=2
    cfg.regrowAblate=2; cfg.regrowPositions=1
    cfg.regrowDepth=1
    var st = State(rules: @[
      Rule(lhs:@[1], rhs:@[256]), Rule(lhs:@[2],rhs:@[257])],
      nextToken:258)
    st.evaluate(tr,va,cfg.ridge)
    let rebuilt=rollbackRegrow(st,clean,tr,va,cfg)
    check rebuilt.nextToken >= 258
    check rebuilt.evaluation.score + 1e-10 >= st.evaluation.score
    check st.rules.len == 2

suite "Nonlocal dependency-aware regrowth":
  test "only distant R4 changes when R1 changes; R2 and R3 are preserved":
    let before = @[
      Rule(lhs:@[1],rhs:@[256]),
      Rule(lhs:@[7],rhs:@[70]),
      Rule(lhs:@[8],rhs:@[80]),
      Rule(lhs:@[256],rhs:@[9]),
      Rule(lhs:@[70],rhs:@[71])]
    var after = copyRules(before)
    after[0].lhs = @[6]
    let order = affectedRuleOrder(before,after,
      @[@[1,7,8],@[1,8,7],@[1,7],@[2,7,8]],0,64)
    check order.len > 0
    check order[0].index == 3
    for impact in order:
      check impact.index notin [1,2,4]
  test "same firing but different capture edit is detected":
    let before = @[
      Rule(lhs:@[8],rhs:@[1,5,2]),
      Rule(lhs:@[1,-1,2],rhs:@[7,-1,8])]
    var after = copyRules(before)
    after[0].rhs = @[1,6,2]
    let order = affectedRuleOrder(before,after,@[@[8]],0,64)
    check order.len == 1
    check order[0].index == 1
    check order[0].firingDrift == 0
    check order[0].editDrift > 0
  test "identical transformation has no affected nodes":
    let before = @[Rule(lhs:@[1],rhs:@[100]), Rule(lhs:@[100],rhs:@[2])]
    let order = affectedRuleOrder(before,copyRules(before),@[@[1],@[1,1]],0,64)
    check order.len == 0

suite "Replay data isolation":
  test "running two alternative rule chains never mutates the original input rows":
    let dataset = DataSet(texts: @[@[1,2,3],@[1,3]])
    let originalRows = dataset.texts
    discard traceRules(@[Rule(lhs:@[1],rhs:@[256])],dataset)
    discard traceRules(@[Rule(lhs:@[2],rhs:@[257])],dataset)
    check dataset.texts == originalRows
    check dataset.texts == @[@[1,2,3],@[1,3]]

suite "Balanced causal-drift sampling":
  test "sample counts stay balanced across unequal case lengths":
    var cases: seq[int]
    for _ in 0..<90: cases.add(11)
    for _ in 0..<5: cases.add(12)
    for _ in 0..<15: cases.add(13)
    let sample = chooseImpactSamples(cases.len, 12, cases)
    check sample.len == 12
    var perCase = [0,0,0]
    for i in sample:
      if cases[i] == 11: inc perCase[0]
      elif cases[i] == 12: inc perCase[1]
      else: inc perCase[2]
    check perCase == [4,4,4]
  test "limited budget balances distinct cases instead of earliest noise rows":
    var cases: seq[int]
    for c in 0..<10:
      for _ in 0..<20: cases.add(c)
    let sample = chooseImpactSamples(cases.len, 5, cases)
    check sample.len == 5
    check sample.mapIt(cases[it]) == @[0,2,4,6,8]
  test "unrelated prefix insertion does not change a later rule's local edit":
    let baseline = @[
      Rule(lhs:@[1], rhs:@[1,1]),
      Rule(lhs:@[8], rhs:@[80])]
    var revised = copyRules(baseline)
    revised[0].rhs = @[1,1,1]
    let impacted = affectedRuleOrder(baseline,revised,
      @[@[1,8,7],@[1,9,8],@[1,8,9,8]],0,64)
    check impacted.len == 0
  test "nonadjacent changed captures remain detectable with case sampling":
    let baseline = @[
      Rule(lhs:@[8],rhs:@[1,5,2]),
      Rule(lhs:@[7],rhs:@[77]),
      Rule(lhs:@[1,-1,2],rhs:@[7,-1,8])]
    var revised = copyRules(baseline)
    revised[0].rhs = @[1,6,2]
    let impacted = affectedRuleOrder(baseline,revised,
      @[@[8,7],@[8],@[7,8],@[8,7]],0,4,@[0,0,1,1])
    check impacted.len == 1
    check impacted[0].index == 2
    check impacted[0].firingDrift == 0
    check impacted[0].editDrift > 0
