## All-pairs centered rank regression tests for accuracy-oriented v8 branch.
## nim c -r -d:release --opt:speed test_dense_rank.nim
import std/[unittest, math, random]
import lens_core, lens_search

suite "exact all-pairs centered regression":
  test "rank-centered objective exactly equals all-pair difference objective":
    var data: DataSet
    data.target = @[0.0,0.25,0.5,0.75,1.0,0.0,0.5,1.0]
    data.cases = @[0,0,0,0,0,1,1,1]
    let column = @[0.4,0.1,0.8,0.5,0.7,0.9,0.2,0.3]
    let g = trainingRankGraph(data,0)
    check g.centered
    check g.hi.len == 8
    check g.caseCount == 2
    let design = designRankColumn(g,column)
    var centeredLoss = 0.0
    for i in 0..<design.len:
      let delta = design[i]-g.margin[i]
      centeredLoss += g.caseWeight[i]*delta*delta
    var brute = 0.0
    for ids in g.caseGroups:
      let m = ids.len
      var y: seq[float64]
      for i,row in ids:
        discard i
        y.add(data.target[row])
      # All target levels are distinct, and the target is linear within case.
      for i in 0..<m:
        for j in (i+1)..<m:
          let dx = column[ids[j]]-column[ids[i]]
          let dy = y[j]-y[i]
          brute += (dx-dy)*(dx-dy)/float64(m*m)
    check abs(centeredLoss-brute) < 1e-10

  test "fast append equals complete solve in dense objective":
    var rng = initRand(1729)
    var d: DataSet
    var a,b,c: seq[float64]
    for caseId,size in @[(7,7),(31,17),(92,6)]:
      for j in 0..<size:
        d.cases.add(caseId)
        d.target.add(float64(j)/float64(size-1))
        a.add(float64(rand(rng,0..1)))
        b.add(float64(rand(rng,0..1)))
        c.add(float64(rand(rng,0..1)))
    for lam in [0.001,0.01,0.1,1.0]:
      let cache = prepareRankAppend(@[a,b],d,lam,true,0)
      let fast = fitAppend(cache,c)
      let full = fitPrimal(@[a,b,c],d,lam,true,0)
      check fast.len == full.len
      for i in 0..<full.len:
        check abs(full[i]-fast[i]) < 1e-8

  test "allpairs is the default and sampled mode remains selectable":
    let cfg = defaultSearchConfig()
    check cfg.rankScales == 0
    check cfg.readoutRidgeProbes == 3
    let data = DataSet(target: @[0.0,0.5,1.0],cases: @[1,1,1])
    check trainingRankGraph(data,0).centered
    check not trainingRankGraph(data,4).centered
