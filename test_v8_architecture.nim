## v8 tests. Run: nim c -r -d:release --opt:speed test_v8_architecture.nim
import std/[unittest, sets, math]
import lens_core, lens_search, lens_validation, cms_miner, batch_presence

suite "v8 residual-supervised proposals and source-consensus":
  test "screen is deterministic, bounded, no duplicate entries":
    var data: DataSet
    var columns = newSeq[seq[uint64]](50)
    for c in 0..<50: columns[c] = @[0'u64]
    var parentPrediction: seq[float64]
    for row in 0..<10:
      data.texts.add(@[row])
      data.cases.add(if row < 5: 0 else: 1)
      data.target.add(float64(row mod 5)/4.0)
      parentPrediction.add(0.0)
    for c in 0..<50:
      for row in 0..<10:
        if (row + c) mod 3 == 0:
          columns[c][0] = columns[c][0] or (1'u64 shl row)
    let selected = residualScreen(columns,10,parentPrediction,data,8,4)
    check selected.len == 12
    check residualScreen(columns,10,parentPrediction,data,8,4) == selected
    var seen = initHashSet[int]()
    for id in selected:
      check id >= 0 and id < 50
      check id notin seen
      seen.incl(id)
    check residualScreen(columns,10,parentPrediction,data,0,4).len == 50

  test "source consensus averages noise paths before counting wins":
    # Rho of the new single feature: positive on source 0, negative on source 1.
    # Source 0 is duplicated across two paths; its agreement is NOT counted
    # as two independent source wins.
    var data: DataSet
    for source in 0..<2:
      for path in 0..<2:
        for step in 0..<3:
          let hasToken = (if source == 0: step > 0 else: step < 2)
          data.texts.add(@[(if hasToken: 9 else: 1)])
          data.target.add(float64(step)/2.0)
          data.cases.add(source*2+path)
    let root = State(rules: @[], evaluation: Evaluation(weights: @[]))
    let child = State(rules: @[Rule(lhs:@[9],rhs:@[10])],
                      evaluation: Evaluation(weights: @[1.0]))
    check abs(sourceWinFraction(child,root,data,2) - 0.5) < 1e-8
    check abs(sourceWinFraction(root,root,data,2)) < 1e-8

  test "selected-only exact SEARCH matching agrees with all-candidate matching":
    let patterns = @[
      @[1], @[1,-1,3], @[2,-1,3], @[4,-1,1], @[3], @[1,-1,2,-1,3]]
    let inputs = @[
      @[1,2,3], @[1,8,9,3], @[4,5,1,2,3], @[2,1,3],
      @[9,9,9], @[1,4,2,7,3], @[1,3,2,3]]
    let allPresence = batchPresence(compileBatch(patterns),inputs)
    let selectedIds = @[5,1,3]
    var shortlist: seq[seq[int]]
    for id in selectedIds: shortlist.add(patterns[id])
    let shortPresence = batchPresence(compileBatch(shortlist),inputs)
    check shortPresence.rows == allPresence.rows
    for localId,id in selectedIds:
      check shortPresence.columns[localId] == allPresence.columns[id]

  test "sampled gap mining can be disabled for exhaustive equivalence":
    var cfg = defaultMiningConfig()
    cfg.maxGram = 2
    cfg.maxTotalGram = 3
    cfg.maxWildcards = 2
    cfg.topPerShape = 10000
    cfg.width = 65536
    cfg.startStride = 1
    let chunks = @[@[1,2,3,4,5,6,7,8,9,10,11,12]]
    cfg.gapSamples = 0
    let full = minePatterns(chunks,cfg)
    cfg.gapSamples = 4
    let sampled = minePatterns(chunks,cfg)
    check sampled.len <= full.len
    var all = initHashSet[seq[int]]()
    for entry in full: all.incl(entry.value)
    for entry in sampled: check entry.value in all
