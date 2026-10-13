import std/[unittest, random, sequtils, math]
import cms, lens_core, lens_search, batch_presence

suite "v6: generalization and exact speedups":
  test "partition shuffle is reproducible and preserves every eligible span":
    var spans: seq[ChunkSpan]
    for i in 0..<100:
      spans.add((first: int64(i*9), past: int64(i*9+8)))
    let a = shuffledCorpusSpans(spans, 42)
    let b = shuffledCorpusSpans(spans, 42)
    let c = shuffledCorpusSpans(spans, 43)
    check a == b
    check a != c
    check a != spans
    check spans[0].first == 0 # Source was not mutated.
    var used = newSeq[bool](spans.len)
    for span in a:
      let idx = int(span.first div 9)
      check not used[idx]
      used[idx] = true
    check used.allIt(it)

  test "dispersion penalty is zero for consistent performance":
    var data: DataSet
    data.target = @[0.0, 0.33, 0.66, 1.0, 0.0, 0.33, 0.66, 1.0]
    data.cases = @[0,0,0,0,1,1,1,1]
    let ranks = prepareSpearman(data)
    let p = data.target
    let consistent = caseSpearmanStability(ranks,p,0.35)
    check abs(consistent.mean-1.0) < 1e-10
    check abs(consistent.merit-1.0) < 1e-10
    let unstable = @[0.0,0.33,0.66,1.0,1.0,0.66,0.33,0.0]
    let legacy = scoreSpearman(ranks,unstable)
    let robust = caseSpearmanStability(ranks,unstable,0.35)
    check abs(legacy-robust.mean) < 1e-12
    check robust.merit < robust.mean

  test "first-literal indexed batch matcher remains exact with rare anchors":
    var rng = initRand(763)
    let patterns: seq[seq[int]] = @[
      @[1], @[1,-1,2], @[2,-1,3], @[3,3,-1,1],
      @[999], @[999,-1,1], @[777,-1,777], @[1,-1,1,-1,1],
      @[4,-1,4,-1,4,-1,4]]
    var data: seq[seq[int]]
    for i in 0..<50:
      var row: seq[int]
      for j in 0..<rand(rng,50):
        row.add(rand(rng,5))
      data.add(row)
    let batched = batchPresence(compileBatch(patterns),data)
    for c,p in patterns:
      check toColumn(batched.columns[c],batched.rows) == presence(data,p)

suite "v6: stable corruption process":
  test "shared symbols limit every changed code point on unseen inputs":
    let clean: seq[seq[int]] = @[@[1,2,3,4,5,6]]
    let rows = trajectories(clean,44,1,6,alphabetOverride= @[7,8,9])
    check rows.texts.len == 7
    for row in rows.texts:
      check row.len == 6
      for i,token in row:
        check token == clean[0][i] or token in @[7,8,9]
