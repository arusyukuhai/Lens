import std/[unittest, math]
import lens_core, lens_search, cms_miner, replace, spearman_rank_ridge

suite "Lens semantics":
  test "cumulative noise changes exactly one new position at every stage":
    let clean = @[@[1,2,1,2], @[2,1]]
    let data = trajectories(clean, 17, 2)
    check data.texts.len == 2*(5+3)
    for ci, chunk in clean:
      for path in 0..<2:
        var last: seq[int]
        var level = 0
        for row, cid in data.cases:
          if cid == ci*2+path:
            var differences = 0
            for i in 0..<chunk.len:
              if data.texts[row][i] != chunk[i]: inc differences
            check differences == level
            check abs(data.target[row]-float64(level)/float64(chunk.len)) < 1e-12
            if last.len > 0:
              var changes = 0
              for i in 0..<chunk.len:
                if data.texts[row][i] != last[i]: inc changes
              check changes == 1
            last = data.texts[row]
            inc level
    check clean == @[@[1,2,1,2], @[2,1]]

  test "existence and replacement both require nonempty captures":
    let p = compileSeqPattern(@[1,-1,2], 1)
    check not containsSeqCompiled(@[1,2], p)
    check containsSeqCompiled(@[1,9,2], p)
    check replaceSeqStrict(@[1,9,2], p, @[256,-1,257]) == @[256,9,257]
    check containsSeqCompiled(@[1,2], compileSeqPattern(@[1,-1,2]))

  test "existence counts once, not number of occurrences":
    check presence(@[@[1,1,1], @[2], @[1]], @[1]) == @[1.0,0.0,1.0]

  test "per-rule phi includes predecessors and successors, excludes self":
    let a = @[0.0,0,1,1]
    let b = @[0.0,1,0,1]
    let means = allRulePhi(@[a,a,b])
    check means == @[0.5,0.5,0.0]
    check allRulePhi(@[a,a,a]) == @[1.0,1.0,1.0]
    check absPhi(a, @[1.0,1,0,0]) == 1.0
    check absPhi(a, @[1.0,1,1,1]) == 0.0

  test "primal rank ridge agrees with supplied dual implementation":
    let data = DataSet(texts: @[@[1],@[2],@[3],@[4],@[5]],
      target: @[0.0,0.25,0.5,0.75,1.0], cases: @[0,0,0,0,0])
    let cols = @[@[1.0,1,0,0,0], @[0.0,1,1,0,0], @[1.0,1,1,1,1]]
    var rows = newSeq[seq[float64]](5)
    for i in 0..<5:
      for col in cols: rows[i].add(col[i])
    let primal = fitPrimal(cols, data, 0.01)
    let dual = fitPairwiseRankRidgeAll(rows, newSeq[float64](5), data.target, data.cases, 0.01)
    for i in 0..<primal.len: check abs(primal[i]-dual[i]) < 1e-7

  test "physical beam bound and deterministic rotation of swap branches":
    let lhs = @[1,-1,2,-1,3,-1,4,-1,5]
    let variants = replacementVariants(lhs, 256)
    check variants.len == 7
    var states: seq[State]
    for rhs in variants:
      states.add(State(rules: @[Rule(lhs: lhs, rhs: rhs)], nextToken: 261))
    check selectBeam(states, 1).len == 1
    check selectBeam(states, 2).len == 2
    check selectBeam(states, 4).len == 4
    check selectBeam(states, 20).len == 7
    check selectBeam(states, 2)[0].rules[0].rhs == variants[0]
    # Every swap remains discoverable across iterations without keeping all
    # seven full states alive within the same beam.
    for i in 0..<6:
      let selected = selectBeam(states, 2, i)
      check selected[1].rules[0].rhs == variants[i+1]
    let p = compileSeqPattern(@[1,-1,2,-1,3], 1)
    let rhs = replacementVariants(@[1,-1,2,-1,3],256)
    check replaceSeqStrict(@[1,10,2,20,3],p,rhs[0]) == @[256,10,257,20,258]
    check replaceSeqStrict(@[1,10,2,20,3],p,rhs[1]) == @[256,20,257,10,258]

  test "beam width is a state cap even with multiple LHS families":
    var states: seq[State]
    for lhsToken in 1..4:
      for rhsToken in 10..12:
        states.add(State(rules: @[Rule(lhs: @[lhsToken], rhs: @[rhsToken])],
          evaluation: Evaluation(score: float64(lhsToken)/4.0,
            means: @[0.0]), novelty: 0.0))
    let selected = selectBeam(states, 4)
    check selected.len == 4
    var bestFound = false
    for state in selected:
      if state.rules[0].lhs == @[4]: bestFound = true
    check bestFound
    # Legacy checkpoint can contain 12 states, but the next expansion must
    # never use more than four of them as parents.
    var cfg = defaultSearchConfig()
    cfg.beam = 4
    cfg.lookahead = 0
    check addBeam(states, @[], DataSet(), DataSet(), cfg).len == 4

  test "editing an upstream rule changes downstream firing on full replay":
    let data = DataSet(texts: @[@[1,2]])
    let old = @[Rule(lhs: @[1],rhs: @[256]), Rule(lhs: @[256,2],rhs: @[257])]
    var edited = copyRules(old)
    edited[0].lhs = @[3]
    check traceRules(old,data).columns == @[@[1.0],@[1.0]]
    check traceRules(edited,data).columns == @[@[0.0],@[0.0]]
    check old[0].lhs == @[1]

  test "miner combines chunks but never crosses chunk boundaries":
    var cfg = defaultMiningConfig()
    cfg.maxGram = 2; cfg.maxTotalGram = 2; cfg.maxWildcards = 0
    cfg.width = 1024; cfg.depth = 7
    let mined = minePatterns(@[@[1],@[2],@[1]], cfg)
    var oneCount = 0'u32
    for e in mined:
      check e.value != @[1,2]
      if e.value == @[1]: oneCount = e.rank
    check oneCount >= 2

  test "transaction repair commits improvements and leaves reject input unchanged":
    let data = DataSet(texts: @[@[1,2],@[1],@[2],@[]],
      target: @[0.0,1.0/3,2.0/3,1.0],cases: @[0,0,0,0])
    let clean = @[@[1,2],@[1],@[2]]
    var cfg = defaultSearchConfig()
    cfg.mining.maxGram = 1; cfg.mining.maxTotalGram = 1; cfg.mining.maxWildcards = 0
    cfg.mining.width = 1024; cfg.mining.depth = 4
    cfg.beam = 2; cfg.repairPasses = 1
    var original = State(rules: @[
      Rule(lhs: @[1],rhs: @[1]),Rule(lhs: @[1],rhs: @[1])], nextToken: 256)
    original.evaluate(data,data,cfg.ridge)
    check original.evaluation.means == @[1.0,1.0]
    let improved = repair(original,clean,data,data,cfg)
    check improved.evaluation.score > original.evaluation.score
    check original.rules[0].lhs == @[1]
    check original.rules[1].lhs == @[1]
    var unbeatable = original
    unbeatable.evaluation.score = 1.0
    let rejected = repair(unbeatable,clean,data,data,cfg)
    check rejected.rules == unbeatable.rules
    check rejected.evaluation.score == 1.0
    var atThreshold = original
    atThreshold.evaluation.means = @[0.5,0.5]
    check repair(atThreshold,clean,data,data,cfg).rules == original.rules

  test "repair shortlist includes frequency leaders and dispersed candidates":
    var mined: seq[PatternEntry]
    for i in 0..<100:
      mined.add(PatternEntry(value: @[100+i], rank: uint32(100-i)))
    let shortlist = repairCandidates(mined, @[999], @[256], 12)
    check shortlist.len == 12
    for i in 0..<6:
      check shortlist[i].value == mined[i].value
    check shortlist[^1].value[0] >= 190
    check repairCandidates(mined, @[999], @[256], 0).len == 100
    check repairCandidates(mined, @[999], @[256], 200).len == 100
    check repairCandidates(mined, @[100], @[256], 0).len == 99

  test "replaying unchanged prefix and edited suffix preserves full trace":
    let data = DataSet(texts: @[@[1,2], @[1,9,2], @[3,2], @[2,1,2]])
    let rules = @[
      Rule(lhs: @[1], rhs: @[256]),
      Rule(lhs: @[256,2], rhs: @[257]),
      Rule(lhs: @[257], rhs: @[258])]
    let pre = traceRules(rules[0..<1], data)
    var rowCopy = newSeq[seq[int]](pre.texts.len)
    for i, row in pre.texts: rowCopy[i] = row
    let tail = traceRules(rules[1..<rules.len], DataSet(texts: rowCopy))
    let full = traceRules(rules, data)
    check pre.columns & tail.columns == full.columns
    check tail.texts == full.texts

  test "log is natural log and exact perfection is infinity":
    check abs(logScore(0.5)-ln(2.0)) < 1e-12
    check logScore(1.0) == Inf
