## v7 regression parity / source balance / readout isolation
## Run: nim c -r -d:release --opt:speed test_v7_multiscale.nim
import std/[unittest, math, random]
import lens_core, spearman_rank_ridge, lens_validation

suite "v7 multiscale rank regression":
  test "edge weights give each source corruption trajectory unit total weight":
    let labels = @[0.0,0.25,0.5,0.75,1.0,
                   0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]
    let ids = @[0,0,0,0,0,1,1,1,1,1,1,1,1]
    let oldGraph = caseAdjacentRankPairs(labels,ids)
    let graph = caseAdjacentRankPairs(labels,ids,4)
    check graph.caseCount == 2
    check oldGraph.hi.len == 12
    check graph.hi.len > oldGraph.hi.len
    var total: array[2,float64]
    for e in 0..<graph.hi.len:
      let src = ids[graph.hi[e]]
      check ids[graph.lo[e]] == src
      total[src] += graph.caseWeight[e]
    check abs(total[0]-1.0) < 1e-12
    check abs(total[1]-1.0) < 1e-12

  test "cached one-column fit equals complete weighted multiscale regression":
    var rng = initRand(20261013)
    var data: DataSet
    var first, second, extra: seq[float64]
    for caseId, size in @[7,31,11]:
      for j in 0..<size:
        data.target.add(float64(j)/float64(size-1))
        data.cases.add(caseId)
        first.add(float64(rand(rng,0..1)))
        second.add(float64(rand(rng,0..1)))
        extra.add(float64(rand(rng,0..1)))
    let columns = @[first,second]
    for scales in [1,2,4]:
      for balanced in [false,true]:
        let cached = prepareRankAppend(columns,data,0.05,balanced,scales)
        let actual = fitAppend(cached,extra)
        let expected = fitPrimal(columns & @[extra],data,0.05,balanced,scales)
        check actual.len == expected.len
        for i in 0..<actual.len:
          check abs(actual[i]-expected[i]) < 1e-6

  test "readout proposal does not mutate saved weights or rules":
    let r = Rule(lhs: @[1,2], rhs: @[1,3])
    let champion = State(rules: @[r],
                         evaluation: Evaluation(weights: @[2.0]))
    let trained = State(rules: @[r],
                        evaluation: Evaluation(weights: @[6.0]))
    let blended = blendReadout(champion,trained,0.25)
    check abs(blended.evaluation.weights[0]-3.0) < 1e-12
    check champion.evaluation.weights == @[2.0]
    check trained.evaluation.weights == @[6.0]
    expect(ValueError): discard blendReadout(champion,trained,0.0)
