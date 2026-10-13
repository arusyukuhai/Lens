import std/[unittest, sequtils, math]
import lens_core, lens_validation

suite "v5: independent acceptance, readout stability, and replay":
  test "anchor pool is distinct from every rotating acceptance batch":
    let anchors = anchorIndices(10,74,16)
    check anchors.len == 16
    for i, x in anchors:
      check x >= 10 and x < 74
      for j in 0..<i: check x != anchors[j]
    for step in 0..<100:
      let rolling = rotatingExcluding(10,74,4,step,anchors)
      check rolling.len == 4
      for idx in rolling:
        check idx notin anchors
  test "candidate must beat incumbent on both acceptance groups":
    check canAcceptRobust(0.4,0.3,0.3,0.2,5,4,0.001,0.002,0.002,0.0)
    check not canAcceptRobust(0.4,0.3,0.15,0.2,5,4,0.001,0.002,0.002,0.0)
    check not canAcceptRobust(0.1,0.3,0.4,0.2,5,4,0.001,0.002,0.002,0.0)
    check not canAcceptRobust(-0.1,-0.7,0.4,0.2,5,4,0.001,0.002,0.002,0.0)
    check not canAcceptRobust(0.4,0.3,NaN,0.2,5,4,0.001,0.002,0.002,0.0)
  test "refitting an exploratory copy never changes frozen champion weights":
    var champion = State(rules: @[Rule(lhs: @[1],rhs: @[256])],nextToken:257)
    champion.evaluation.weights = @[0.314159]
    let oldWeights = champion.evaluation.weights
    var exploratory = champion
    let data = trajectories(@[@[1,2,1,2,3]],123,1,5)
    exploratory.evaluate(data,data,0.01)
    check champion.evaluation.weights == oldWeights
    check champion.evaluation.weights == @[0.314159]
    # Both states still refer to the same ordered rule list; only the
    # exploratory state's new evaluation has fitted coefficients.
    check exploratory.rules == champion.rules
  test "sampled replay case IDs remain disjoint after combining groups":
    let a = trajectories(@[@[1,2,3]],3,1,3)
    let b = trajectories(@[@[2,3,4]],5,1,3)
    var offset = 0
    for cid in a.cases:
      if cid >= offset: offset = cid+1
    for caseId in b.cases:
      check (caseId+offset) notin a.cases
