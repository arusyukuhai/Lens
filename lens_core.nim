## Training/evaluation engine. Rules are ordered, applied once each, left to right.
import std/[math, sequtils, random, algorithm, sets, times]
import replace, spearman_rank_ridge

type
  Rule* = object
    lhs*, rhs*: seq[int]
  DataSet* = object
    texts*: seq[seq[int]]
    target*: seq[float64]  # fraction of positions replaced with different symbols
    cases*: seq[int]
  Trace* = object
    columns*: seq[seq[float64]]
    texts*: seq[seq[int]]
  Evaluation* = object
    train*, score*, searchMerit*: float64
    weights*: seq[float64]
    means*: seq[float64]
  State* = object
    rules*: seq[Rule]
    nextToken*: int
    evaluation*: Evaluation
    novelty*: float64

proc copyInts*(xs: seq[int]): seq[int] =
  result = newSeq[int](xs.len)
  for i, x in xs: result[i] = x

proc copyRules*(rules: seq[Rule]): seq[Rule] =
  for rule in rules:
    result.add(Rule(lhs: copyInts(rule.lhs), rhs: copyInts(rule.rhs)))

proc wildcardCount*(pattern: seq[int]): int =
  for t in pattern:
    if t < 0: inc result

proc trajectories*(chunks: seq[seq[int]], seed: int, paths = 1,
                   steps = 0, progress = false, progressLabel = "",
                   alphabetOverride: seq[int] = @[]): DataSet =
  ## Cumulative corruption: mutate each position ONCE, always to a different token.
  ## steps=0 records every single-character mutation; otherwise explicitly subsample.
  var alphabet: seq[int]
  if alphabetOverride.len > 0:
    alphabet = alphabetOverride
  else:
    var alphabetSet = initHashSet[int]()
    for chunk in chunks:
      for t in chunk: alphabetSet.incl(t)
    alphabet = toSeq(alphabetSet)
    alphabet.sort()
  if alphabet.len < 2:
    alphabet.add(if alphabet.len == 0 or alphabet[0] != 97: 97 else: 98)
  var rng = initRand(seed)
  var nextReport = epochTime() + 2.0
  for ci, chunk in chunks:
    for path in 0..<paths:
      var text = copyInts(chunk)
      var order = toSeq(0..<chunk.len)
      shuffle(rng, order)
      result.texts.add(copyInts(text))
      result.target.add(0.0)
      result.cases.add(ci * paths + path)
      let count = if steps <= 0: chunk.len else: min(steps, chunk.len)
      var nextLevel = 1
      for k, pos in order:
        # Large unbounded noise trajectories can spend many seconds copying
        # strings before the normal [iteration] log appears.
        if progress and (k mod 512 == 0) and epochTime() >= nextReport:
          stderr.writeLine("[trajectory progress] ", progressLabel,
            " chunk=", ci+1, "/", chunks.len,
            " path=", path+1, "/", paths,
            " stage=", k+1, "/", chunk.len,
            " rows=", result.texts.len)
          nextReport = epochTime() + 2.0
        var token = alphabet[rand(rng, alphabet.high)]
        while token == chunk[pos]: token = alphabet[rand(rng, alphabet.high)]
        text[pos] = token
        if (k + 1) * count >= nextLevel * chunk.len:
          result.texts.add(copyInts(text))
          result.target.add(float64(k + 1) / float64(chunk.len))
          result.cases.add(ci * paths + path)
          inc nextLevel

proc traceRules*(rules: seq[Rule], data: DataSet): Trace =
  # Never assign into the source DataSet's outer row sequence: a candidate
  # replay must not contaminate another candidate's training inputs.
  result.texts = newSeq[seq[int]](data.texts.len)
  for i, row in data.texts: result.texts[i] = row
  for rule in rules:
    let pat = compileSeqPattern(rule.lhs, 1)
    var column = newSeq[float64](data.texts.len)
    for row in 0..<result.texts.len:
      if containsSeqCompiled(result.texts[row], pat):
        column[row] = 1.0
        result.texts[row] = replaceSeqStrict(result.texts[row], pat, rule.rhs)
    result.columns.add(column)

proc transform*(rules: seq[Rule], chunks: seq[seq[int]]): seq[seq[int]] =
  result = traceRules(rules, DataSet(texts: chunks)).texts

proc presence*(texts: seq[seq[int]], lhs: seq[int]): seq[float64] =
  let pat = compileSeqPattern(lhs, 1)
  result = newSeq[float64](texts.len)
  for row, text in texts:
    if containsSeqCompiled(text, pat): result[row] = 1.0

proc absPhi*(a, b: seq[float64]): float64 =
  if a.len != b.len: raise newException(ValueError, "phi row mismatch")
  var sa, sb, sab: float64
  for i in 0..<a.len:
    sa += a[i]
    sb += b[i]
    sab += a[i] * b[i]
  let n = float64(a.len)
  let denominator = sqrt(sa * (n-sa) * sb * (n-sb))
  # Constant presence has no covariance/information; undefined phi is treated as 0.
  if denominator <= 0: return 0.0
  min(1.0, abs((n*sab - sa*sb) / denominator))

proc meanPhi*(column: seq[float64], previous: seq[seq[float64]]): float64 =
  if previous.len == 0: return 0.0
  for other in previous: result += absPhi(column, other)
  result /= float64(previous.len)

proc allRulePhi*(columns: seq[seq[float64]]): seq[float64] =
  ## Each row includes ALL other rules, both before and after it; diagonal excluded.
  result = newSeq[float64](columns.len)
  if columns.len <= 1: return
  for i in 0..<columns.len:
    for j in 0..<columns.len:
      if i != j: result[i] += absPhi(columns[i], columns[j])
    result[i] /= float64(columns.len - 1)

proc predict*(columns: seq[seq[float64]], weights: seq[float64], rows: int): seq[float64] =
  result = newSeq[float64](rows)
  for c, column in columns:
    for row, x in column: result[row] += weights[c]*x

proc fitPrimal*(columns: seq[seq[float64]], data: DataSet, ridge: float64,
                balanceCases = false, rankScales = 1): seq[float64] =
  ## Primal form of supplied adjacent-pair rank ridge, no feature selection cap.
  ## Equivalent scaled objective to fitPairwiseRankRidgeAll(... baseline=0).
  ## O(E*P^2 + P^3), avoiding the original E-by-E dual solve when E >> P.
  let p = columns.len
  result = newSeq[float64](p)
  if p == 0: return
  let graph = caseAdjacentRankPairs(data.target, data.cases, rankScales)
  if graph.hi.len == 0: return
  var gram = newSeq[float64](p*p)
  var rhs = newSeq[float64](p)
  var delta = newSeq[float64](p)
  for e in 0..<graph.hi.len:
    let edgeWeight = (if balanceCases: graph.caseWeight[e] else: 1.0)
    for c in 0..<p:
      delta[c] = columns[c][graph.hi[e]] - columns[c][graph.lo[e]]
    for c in 0..<p:
      rhs[c] += edgeWeight*delta[c]*graph.margin[e]
      for d in 0..c: gram[c*p+d] += edgeWeight*delta[c]*delta[d]
  let effectiveEdges = (if balanceCases: graph.caseCount else: graph.hi.len)
  for c in 0..<p:
    let ss = gram[c*p+c]
    for d in 0..<c: gram[d*p+c] = gram[c*p+d]
    gram[c*p+c] += (if ss > 0: ridge * ss / float64(effectiveEdges) else: 1.0)
  # In-place Cholesky; lambda > 0 and inactive columns diagonal=1 guarantee SPD.
  for i in 0..<p:
    for j in 0..i:
      var v = gram[i*p+j]
      for k in 0..<j: v -= gram[i*p+k]*gram[j*p+k]
      if i == j:
        if v <= 0: raise newException(ValueError, "rank ridge lost positive definiteness")
        gram[i*p+i] = sqrt(v)
      else: gram[i*p+j] = v / gram[j*p+j]
  for i in 0..<p:
    var v = rhs[i]
    for j in 0..<i: v -= gram[i*p+j]*result[j]
    result[i] = v / gram[i*p+i]
  for i in countdown(p-1, 0):
    var v = result[i]
    for j in i+1..<p: v -= gram[j*p+i]*result[j]
    result[i] = max(-WeightClip, min(WeightClip, v / gram[i*p+i]))

type
  SpearmanCase = object
    ids: seq[int]
    centeredTarget: seq[float64]
    variance: float64
  SpearmanCache* = object
    groups: seq[SpearmanCase]

proc prepareSpearman*(data: DataSet): SpearmanCache =
  var order = toSeq(0..<data.cases.len)
  order.sort(proc(a,b: int): int =
    let c = cmp(data.cases[a],data.cases[b])
    if c != 0: c else: cmp(a,b))
  var start = 0
  while start < order.len:
    var stop = start+1
    while stop < order.len and data.cases[order[stop]] == data.cases[order[start]]: inc stop
    if stop-start >= 2:
      var group: SpearmanCase
      group.ids = order[start..<stop]
      var values: seq[float64]
      for id in group.ids: values.add(data.target[id])
      group.centeredTarget = averageRanks(values)
      let mean = float64(values.len+1)*0.5
      for x in group.centeredTarget.mitems:
        x -= mean
        group.variance += x*x
      result.groups.add(group)
    start = stop

proc caseSpearmanStability*(cache: SpearmanCache, prediction: seq[float64],
                            penalty: float64): tuple[mean, merit: float64] =
  if cache.groups.len == 0: return (mean: 0.0, merit: 0.0)
  var sumScore, sumSq: float64
  for group in cache.groups:
    var values = newSeq[float64](group.ids.len)
    for i,id in group.ids: values[i] = prediction[id]
    let ranks = averageRanks(values)
    let mean = float64(ranks.len+1)*0.5
    var numerator, variance: float64
    for i,x in ranks:
      let centered = x-mean
      numerator += centered*group.centeredTarget[i]
      variance += centered*centered
    let denominator = sqrt(variance*group.variance)
    let rho = (if denominator > Tiny: numerator/denominator else: 0.0)
    sumScore += rho
    sumSq += rho*rho
  let mean = sumScore/float64(cache.groups.len)
  let variance = max(0.0, sumSq/float64(cache.groups.len) - mean*mean)
  result = (mean: mean, merit: mean - penalty*sqrt(variance))

proc scoreSpearman*(cache: SpearmanCache, prediction: seq[float64]): float64 =
  caseSpearmanStability(cache, prediction, 0.0).mean


proc scoreColumns*(trainColumns, selectColumns: seq[seq[float64]],
                   train, selection: DataSet, ridge: float64,
                   computePhi = true, stabilityPenalty = 0.0,
                   balanceCases = false, rankScales = 1): Evaluation =
  result.weights = fitPrimal(trainColumns, train, ridge, balanceCases, rankScales)
  result.train = meanCaseSpearman(predict(trainColumns, result.weights, train.texts.len),
                                 train.target, train.cases)
  let selectPred = predict(selectColumns, result.weights, selection.texts.len)
  let stable = caseSpearmanStability(prepareSpearman(selection), selectPred, stabilityPenalty)
  result.score = stable.mean
  result.searchMerit = stable.merit
  if computePhi: result.means = allRulePhi(trainColumns)

proc evaluate*(state: var State, train, selection: DataSet, ridge: float64,
               stabilityPenalty = 0.0, balanceCases = false, rankScales = 1) =
  let tr = traceRules(state.rules, train)
  let va = traceRules(state.rules, selection)
  state.evaluation = scoreColumns(tr.columns, va.columns, train, selection,
                                   ridge, stabilityPenalty = stabilityPenalty,
                                   balanceCases = balanceCases, rankScales = rankScales)

proc replacementVariants*(lhs: seq[int], firstToken: int): seq[seq[int]] =
  ## One fresh marker per literal segment, preserve captures, then every single pair swap.
  let k = wildcardCount(lhs)
  var base = @[firstToken]
  for w in 0..<k:
    base.add(-w-1)
    base.add(firstToken+w+1)
  result.add(base)
  for i in 0..<k:
    for j in i+1..<k:
      var variant = copyInts(base)
      swap(variant[2*i+1], variant[2*j+1])
      result.add(variant)

proc logScore*(rho: float64): float64 =
  ## Mathematical value (Infinity at rho=1). Plotting clamps separately, transparently.
  if rho >= 1.0: Inf else: -ln(1.0-rho)

# Shared rank-ridge factorization for one-column candidate additions.
type
  RankAppendCache* = object
    graph: RankPairGraph
    differences: seq[seq[float64]]
    lower, baseWeights: seq[float64]
    p: int
    ridge: float64
    balanceCases: bool
    effectiveEdges: int
    edgeScale: seq[float64]

proc solveLowerFactor(lower, rhs: seq[float64], p: int): seq[float64] =
  result = newSeq[float64](p)
  for i in 0..<p:
    var v = rhs[i]
    for j in 0..<i: v -= lower[i*p+j]*result[j]
    result[i] = v/lower[i*p+i]
  for i in countdown(p-1,0):
    var v = result[i]
    for j in i+1..<p: v -= lower[j*p+i]*result[j]
    result[i] = v/lower[i*p+i]

proc prepareRankAppend*(columns: seq[seq[float64]], data: DataSet,
                        ridge: float64, balanceCases = false,
                        rankScales = 1): RankAppendCache =
  result.p = columns.len
  result.ridge = ridge
  result.balanceCases = balanceCases
  result.graph = caseAdjacentRankPairs(data.target, data.cases, rankScales)
  result.effectiveEdges = (if balanceCases: result.graph.caseCount else: result.graph.hi.len)
  let p = result.p
  let edges = result.graph.hi.len
  result.edgeScale = newSeq[float64](edges)
  for e in 0..<edges:
    result.edgeScale[e] = (if balanceCases: sqrt(result.graph.caseWeight[e]) else: 1.0)
  result.lower = newSeq[float64](p*p)
  result.differences = newSeq[seq[float64]](p)
  var rhs = newSeq[float64](p)
  for c in 0..<p:
    result.differences[c] = newSeq[float64](edges)
    for e in 0..<edges:
      let scale = result.edgeScale[e]
      result.differences[c][e] =
        (columns[c][result.graph.hi[e]] - columns[c][result.graph.lo[e]])*scale
      rhs[c] += result.differences[c][e]*result.graph.margin[e]*scale
    for d in 0..c:
      var dot = 0.0
      for e in 0..<edges: dot += result.differences[c][e]*result.differences[d][e]
      result.lower[c*p+d] = dot
    let ss = result.lower[c*p+c]
    result.lower[c*p+c] += (if ss > 0: ridge*ss/float64(result.effectiveEdges) else: 1.0)
  for i in 0..<p:
    for j in 0..i:
      var v = result.lower[i*p+j]
      for k in 0..<j: v -= result.lower[i*p+k]*result.lower[j*p+k]
      if i == j:
        if v <= 0: raise newException(ValueError, "rank append cache is not positive definite")
        result.lower[i*p+i] = sqrt(v)
      else: result.lower[i*p+j] = v/result.lower[j*p+j]
  result.baseWeights = solveLowerFactor(result.lower, rhs, p)

proc fitAppend*(cache: RankAppendCache, column: seq[float64]): seq[float64] =
  ## Block Cholesky append: same regularized system, O(E*P + P^2) per candidate.
  let p = cache.p
  let edges = cache.graph.hi.len
  result = newSeq[float64](p+1)
  if edges == 0: return
  var cross = newSeq[float64](p)
  var ss, rhs: float64
  for e in 0..<edges:
    let scale = cache.edgeScale[e]
    let d = (column[cache.graph.hi[e]] - column[cache.graph.lo[e]])*scale
    ss += d*d
    rhs += d*cache.graph.margin[e]*scale
    if d != 0:
      for c in 0..<p: cross[c] += cache.differences[c][e]*d
  var diagonal = ss + (if ss > 0: cache.ridge*ss/float64(cache.effectiveEdges) else: 1.0)
  # y=L^-1 cross, z=A^-1 cross. Use block Cholesky for numerical stability.
  var y = newSeq[float64](p)
  for i in 0..<p:
    var v = cross[i]
    for j in 0..<i: v -= cache.lower[i*p+j]*y[j]
    y[i] = v/cache.lower[i*p+i]
    diagonal -= y[i]*y[i]
  if diagonal <= 0 or classify(diagonal) in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "unstable rank append Schur complement")
  var z = newSeq[float64](p)
  for i in countdown(p-1,0):
    var v = y[i]
    for j in i+1..<p: v -= cache.lower[j*p+i]*z[j]
    z[i] = v/cache.lower[i*p+i]
  for i in 0..<p: rhs -= cross[i]*cache.baseWeights[i]
  let extra = rhs/diagonal
  result[p] = max(-WeightClip,min(WeightClip,extra))
  for i in 0..<p:
    result[i] = max(-WeightClip,min(WeightClip,cache.baseWeights[i]-z[i]*extra))

proc predictAppend*(columns: seq[seq[float64]], column, weights: seq[float64]): seq[float64] =
  result = predict(columns, weights, column.len)
  for i,v in column: result[i] += weights[^1]*v
