## nim c -r -d:release --opt:speed test_v9_pearson_ols.nim
import std/[unittest, math]
import pearson_ols, lens_core, lens_validation

proc predicted(columns: seq[seq[float64]], fit: OlsFit, rows: int): seq[float64] =
  predict(columns, fit.weights, rows, fit.intercept)

suite "Pearson + unregularized, intercept OLS":
  test "simple binary indicator and nonzero intercept":
    let columns = @[@[0.0, 0.0, 1.0, 1.0]]
    let y = @[0.15,0.15,0.85,0.85]
    let c = @[0,0,0,0]
    let model = fitOLS(columns,y,c)
    check abs(model.intercept-0.15) < 1e-9
    check abs(model.weights[0]-0.70) < 1e-9
    check abs(pearson(predicted(columns,model,4), y)-1.0) < 1e-9

  test "empty rule set gives mean-noise intercept":
    let model = fitOLS(@[],@[0.0,0.25,0.75,1.0],@[0,0,0,0])
    check model.weights.len == 0
    check abs(model.intercept-0.5) < 1e-12

  test "identical and constant binary columns remain stable, no ridge":
    let columns = @[
      @[0.0,0.0,1.0,1.0,0.0,1.0],
      @[0.0,0.0,1.0,1.0,0.0,1.0],
      @[1.0,1.0,1.0,1.0,1.0,1.0]]
    let y = @[0.1,0.2,0.7,0.8,0.0,0.9]
    let cases = @[0,0,0,1,1,1]
    let model = fitOLS(columns,y,cases)
    for w in model.weights: check classify(w) notin {fcNan,fcInf,fcNegInf}
    check abs(model.weights[1]) < 1e-8 or abs(model.weights[0]) < 1e-8
    let extra = columns[0]
    let appended = fitOLSAppend(prepareOLSAppend(columns,y,cases),extra)
    let pa = predicted(columns & @[extra],appended,y.len)
    let pf = predicted(columns & @[extra],fitOLS(columns & @[extra],y,cases),y.len)
    for i in 0..<y.len: check abs(pa[i]-pf[i]) < 1e-8

  test "one-column fast update equals whole OLS refit":
    var columns = newSeq[seq[float64]](4)
    var extra: seq[float64]
    var target: seq[float64]
    var cases: seq[int]
    for i in 0..<120:
      let a = (if i mod 2 == 0: 1.0 else: 0.0)
      let b = (if i mod 3 == 0: 1.0 else: 0.0)
      let c = (if i mod 5 == 0: 1.0 else: 0.0)
      let d = (if i mod 7 == 0: 1.0 else: 0.0)
      columns[0].add(a)
      columns[1].add(b)
      columns[2].add(c)
      columns[3].add(d)
      extra.add(if i mod 11 < 4: 1.0 else: 0.0)
      target.add(0.12+0.2*a+0.15*b-0.1*c+0.3*extra[^1])
      cases.add(if i < 90: i div 30 else: 3)
    for balance in [true,false]:
      let fast = fitOLSAppend(prepareOLSAppend(columns,target,cases,balance),extra)
      let full = fitOLS(columns & @[extra],target,cases,balance)
      check abs(fast.intercept-full.intercept) < 1e-7
      for i in 0..<fast.weights.len:
        check abs(fast.weights[i]-full.weights[i]) < 1e-7

  test "Pearson metrics are computed by independent corruption path":
    let data = DataSet(
      texts: @[@[1],@[2],@[3],@[4],@[5],@[6]],
      target: @[0.0,0.5,1.0,0.0,0.5,1.0],
      cases: @[0,0,0,1,1,1])
    let cache = preparePearson(data)
    check abs(scorePearson(cache,@[0.0,1.0,2.0,0.0,2.0,4.0])-1.0) < 1e-10
    check abs(scorePearson(cache,@[2.0,1.0,0.0,2.0,1.0,0.0])+1.0) < 1e-10
    check scorePearson(cache,@[1.0,1.0,1.0,1.0,1.0,1.0]) == 0.0

  test "blended readout interpolates intercept and weights":
    let rule = Rule(lhs:@[1],rhs:@[2])
    let champion = State(rules:@[rule],evaluation:Evaluation(weights:@[1.0],intercept:0.1))
    let fitted = State(rules:@[rule],evaluation:Evaluation(weights:@[3.0],intercept:0.5))
    let blended = blendReadout(champion,fitted,0.5)
    check abs(blended.evaluation.weights[0]-2.0) < 1e-12
    check abs(blended.evaluation.intercept-0.3) < 1e-12
    check abs(champion.evaluation.intercept-0.1) < 1e-12
