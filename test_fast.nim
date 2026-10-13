import std/[unittest, random, math, sequtils]
import lens_core, batch_presence, nsga2_pareto, spearman_rank_ridge

suite "Optimized algorithms vs reference":
  test "batched Aho existence matches original matcher, including absent anchors and overlaps":
    var rng = initRand(42)
    for round in 0..<20:
      var texts: seq[seq[int]]
      for row in 0..<15:
        var text: seq[int]
        for i in 0..<rand(rng,40): text.add(rand(rng,4))
        texts.add(text)
      var patterns: seq[seq[int]] = @[@[1],@[1,-1,1],@[1,1],@[1,1,-1,1,1]]
      for c in 0..<150:
        var pattern: seq[int]
        for k in 0..rand(rng,4):
          if k > 0: pattern.add(-1)
          for t in 0..rand(rng,3): pattern.add(rand(rng,5))
        patterns.add(pattern)
      let batch = batchPresence(compileBatch(patterns),texts)
      for c,pattern in patterns:
        check toColumn(batch.columns[c],batch.rows) == presence(texts,pattern)

  test "block solve agrees with full rank ridge including duplicate and constant features":
    var rng = initRand(700)
    for p in [0,1,4,12,32]:
      let rows = 160
      var data = DataSet(texts: newSeq[seq[int]](rows))
      for i in 0..<rows:
        data.target.add(float64(i mod 40)/39.0)
        data.cases.add(i div 40)
      var columns: seq[seq[float64]]
      for c in 0..<p:
        var column: seq[float64]
        for i in 0..<rows: column.add(float64(rand(rng,1)))
        columns.add(column)
      if p > 3:
        columns[1] = columns[0]
        columns[2] = newSeqWith(rows,1.0)
      for ridge in [0.00001,0.01,1.0]:
        let cache = prepareRankAppend(columns,data,ridge)
        for trial in 0..<8:
          var column = newSeq[float64](rows)
          for i in 0..<rows: column[i] = float64(rand(rng,1))
          if trial == 0 and p > 0: column = columns[0]
          if trial == 1: column = newSeqWith(rows,1.0)
          let reference = fitPrimal(columns & @[column],data,ridge)
          let fast = fitAppend(cache,column)
          for c in 0..<fast.len: check abs(reference[c]-fast[c]) < 1e-7

  test "cached Spearman matches reference across ties, cases and constants":
    var rng = initRand(71)
    var data: DataSet
    for i in 0..<320:
      data.target.add(float64(rand(rng,12)))
      data.cases.add(rand(rng,4))
    let cache = prepareSpearman(data)
    for trial in 0..<100:
      var pred: seq[float64]
      for i in 0..<320: pred.add(float64(rand(rng,9)))
      if trial == 0: pred = newSeq[float64](320)
      check abs(scoreSpearman(cache,pred)-meanCaseSpearman(pred,data.target,data.cases)) < 1e-12

  test "Fenwick Pareto ranks, crowding and survivor order match quadratic NSGA-II":
    var rng = initRand(81)
    for trial in 0..<100:
      var scores: seq[seq[float64]]
      for i in 0..<100:
        scores.add(@[float64(rand(rng,9)),float64(rand(rng,9))])
      for senses in [@[odMinimize,odMinimize],@[odMaximize,odMinimize]]:
        let reference = nsga2SortGeneral(scores,senses)
        let fast = nsga2Sort(scores,senses)
        check fast.rank == reference.rank
        check fast.crowding == reference.crowding
        check fast.order == reference.order

  test "packed phi matches scalar phi, including partial uint64 tail":
    var rng = initRand(34)
    for rows in [0,1,63,64,65,135]:
      for trial in 0..<40:
        var a,b: seq[float64]
        for i in 0..<rows:
          a.add(float64(rand(rng,1)))
          b.add(float64(rand(rng,1)))
        let pa = packColumn(a)
        let pb = packColumn(b)
        check abs(packedPhi(pa,pb,bitCount(pa),bitCount(pb),rows)-absPhi(a,b)) < 1e-12
