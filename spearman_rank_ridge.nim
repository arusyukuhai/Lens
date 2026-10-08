## spearman_rank_ridge.nim
##
## Standalone Nim port of the rank-aware readout used by the supplied Python
## Replacer/Lens trainer.
##
## Features:
##   * averageRanks: average-rank tie handling, ranks 1..N
##   * spearman: Pearson correlation of average ranks
##   * fitDualRidge: exact dual ridge with per-feature invRMS^2 scaling
##   * caseAdjacentRankPairs: O(m) adjacent rank constraints per case
##   * fitPairwiseRankRidgeAll: sparse pairwise rank-ridge over all trajectories
##   * predictLinear / meanCaseSpearman convenience helpers
##
## Matrix convention:
##   Public APIs use `seq[seq[float64]]` for X and `seq[float64]` for vectors.
##   No caller-side flattening or rows/cols bookkeeping is required.
##
## No external linear-algebra package is required.

import std/[algorithm, math]

const
  DefaultRankMaxFeatures* = 256
  Tiny* = 1.0e-12
  WeightClip* = 1.0e6


type
  RankPairGraph* = object
    hi*: seq[int]
    lo*: seq[int]
    margin*: seq[float64]

  RankRidgeModel* = object
    weights*: seq[float64]
    selectedFeatures*: seq[int]

  IndexedValue = object
    value: float64
    index: int


proc floatOrder(a, b: float64): int {.inline.} =
  ## Total-ish order matching NumPy's useful behavior for finite values and
  ## placing NaNs last. Equal finite values compare equal so tie averaging works.
  let aNan = classify(a) == fcNan
  let bNan = classify(b) == fcNan
  if aNan:
    if bNan: return 0
    return 1
  if bNan:
    return -1
  if a < b: return -1
  if a > b: return 1
  0


proc stableSortedIndices(values: openArray[float64]): seq[int] =
  ## Deterministic stable ordering: value first, original index as tie breaker.
  var a = newSeq[IndexedValue](values.len)
  for i in 0 ..< values.len:
    a[i] = IndexedValue(value: values[i], index: i)
  a.sort(proc(x, y: IndexedValue): int =
    let c = floatOrder(x.value, y.value)
    if c != 0: c else: cmp(x.index, y.index)
  )
  result = newSeq[int](a.len)
  for i in 0 ..< a.len:
    result[i] = a[i].index


proc averageRanks*(x: seq[float64]): seq[float64] =
  ## Average ranks for ties, 1..N, matching scipy/NumPy-style Spearman ranking.
  ## Example: [10, 20, 20, 40] -> [1, 2.5, 2.5, 4].
  let n = x.len
  result = newSeq[float64](n)
  if n == 0:
    return

  let order = stableSortedIndices(x)
  var start = 0
  while start < n:
    var stop = start + 1
    # Python source uses != for tie boundaries.  Treat NaN as never tied,
    # while ordinary equal finite/infinite values share an average rank.
    while stop < n:
      let a = x[order[start]]
      let b = x[order[stop]]
      if classify(a) == fcNan or classify(b) == fcNan or a != b:
        break
      inc stop

    # Ranks are one-based. For [start, stop), average rank is
    # ((start+1) + stop) / 2.
    let rank = (float64(start + 1) + float64(stop)) * 0.5
    for k in start ..< stop:
      result[order[k]] = rank
    start = stop


proc spearman*(x, y: seq[float64]): float64 =
  ## Spearman rho = Pearson correlation of average ranks.
  ## Returns 0 for mismatched lengths, N<2, or zero rank variance.
  if x.len < 2 or x.len != y.len:
    return 0.0

  let rx = averageRanks(x)
  let ry = averageRanks(y)
  let n = rx.len

  var meanX = 0.0
  var meanY = 0.0
  for i in 0 ..< n:
    meanX += rx[i]
    meanY += ry[i]
  meanX /= float64(n)
  meanY /= float64(n)

  var num = 0.0
  var ssX = 0.0
  var ssY = 0.0
  for i in 0 ..< n:
    let dx = rx[i] - meanX
    let dy = ry[i] - meanY
    num += dx * dy
    ssX += dx * dx
    ssY += dy * dy

  let den = sqrt(ssX * ssY)
  if den <= Tiny:
    return 0.0
  result = num / den


proc choleskySolve(a: var seq[float64], b: openArray[float64], n: int,
                   x: var seq[float64]): bool =
  ## In-place Cholesky factorization of symmetric positive-definite A.
  ## Lower triangle becomes L where A = L*L^T, then solves for x.
  if n <= 0 or a.len != n*n or b.len != n:
    return false

  for i in 0 ..< n:
    for j in 0 .. i:
      var s = a[i*n + j]
      var k = 0
      while k < j:
        s -= a[i*n + k] * a[j*n + k]
        inc k
      if i == j:
        if not (s > 0.0) or classify(s) in {fcNan, fcInf, fcNegInf}:
          return false
        a[i*n + i] = sqrt(s)
      else:
        let d = a[j*n + j]
        if abs(d) <= Tiny:
          return false
        a[i*n + j] = s / d

    # Upper triangle is no longer needed.
    for j in (i + 1) ..< n:
      a[i*n + j] = 0.0

  var z = newSeq[float64](n)
  for i in 0 ..< n:
    var s = b[i]
    var j = 0
    while j < i:
      s -= a[i*n + j] * z[j]
      inc j
    let d = a[i*n + i]
    if abs(d) <= Tiny:
      return false
    z[i] = s / d

  x.setLen(n)
  var ii = n
  while ii > 0:
    dec ii
    var s = z[ii]
    var j = ii + 1
    while j < n:
      s -= a[j*n + ii] * x[j]
      inc j
    let d = a[ii*n + ii]
    if abs(d) <= Tiny:
      return false
    x[ii] = s / d

  true


proc gaussianSolve(aIn: openArray[float64], bIn: openArray[float64], n: int,
                   x: var seq[float64]): bool =
  ## Generic partial-pivot fallback. The Python source falls back to lstsq;
  ## for the normally-positive ridge lambda this path should almost never run.
  if n <= 0 or aIn.len != n*n or bIn.len != n:
    return false
  var a = newSeq[float64](aIn.len)
  for i in 0 ..< aIn.len:
    a[i] = aIn[i]
  var b = newSeq[float64](bIn.len)
  for i in 0 ..< bIn.len:
    b[i] = bIn[i]

  for col in 0 ..< n:
    var pivot = col
    var best = abs(a[col*n + col])
    for r in (col + 1) ..< n:
      let v = abs(a[r*n + col])
      if v > best:
        best = v
        pivot = r
    if best <= Tiny or classify(best) == fcNan:
      return false

    if pivot != col:
      for c in col ..< n:
        swap(a[col*n + c], a[pivot*n + c])
      swap(b[col], b[pivot])

    let diag = a[col*n + col]
    for r in (col + 1) ..< n:
      let f = a[r*n + col] / diag
      if f == 0.0:
        continue
      a[r*n + col] = 0.0
      for c in (col + 1) ..< n:
        a[r*n + c] -= f * a[col*n + c]
      b[r] -= f * b[col]

  x.setLen(n)
  var ii = n
  while ii > 0:
    dec ii
    var s = b[ii]
    for c in (ii + 1) ..< n:
      s -= a[ii*n + c] * x[c]
    let d = a[ii*n + ii]
    if abs(d) <= Tiny:
      return false
    x[ii] = s / d
  true


proc matrixShape(x: seq[seq[float64]], who: string): tuple[rows, cols: int] =
  ## Validate rectangular 2-D input without copying it.
  result.rows = x.len
  result.cols = (if x.len == 0: 0 else: x[0].len)
  for r in 0 ..< result.rows:
    if x[r].len != result.cols:
      raise newException(ValueError, who & ": ragged matrix")


proc fitDualRidgeFlat(x: openArray[float64], rows, cols: int,
                      y: openArray[float64], lam: float64): seq[float64] =
  ## Private flat-matrix kernel used for the selected pairwise design matrix.
  if rows < 0 or cols < 0 or x.len != rows*cols or y.len != rows:
    raise newException(ValueError, "fitDualRidgeFlat: invalid matrix/vector dimensions")

  result = newSeq[float64](cols)
  if rows < 2 or cols == 0:
    return

  var ss = newSeq[float64](cols)
  for r in 0 ..< rows:
    let off = r * cols
    for c in 0 ..< cols:
      let v = x[off + c]
      ss[c] += v * v

  var active: seq[int] = @[]
  for c in 0 ..< cols:
    if ss[c] > Tiny:
      active.add(c)
  if active.len == 0:
    return

  var scale = newSeq[float64](active.len)
  for k, c in active:
    scale[k] = float64(rows) / ss[c]

  var kMat = newSeq[float64](rows * rows)
  for i in 0 ..< rows:
    let oi = i * cols
    for j in 0 .. i:
      let oj = j * cols
      var acc = 0.0
      for k, c in active:
        acc += x[oi + c] * scale[k] * x[oj + c]
      kMat[i*rows + j] = acc
      kMat[j*rows + i] = acc
  for i in 0 ..< rows:
    kMat[i*rows + i] += lam

  var alpha: seq[float64] = @[]
  var chol = kMat
  if not choleskySolve(chol, y, rows, alpha):
    if not gaussianSolve(kMat, y, rows, alpha):
      return

  for k, c in active:
    var dot = 0.0
    for r in 0 ..< rows:
      dot += x[r*cols + c] * alpha[r]
    var w = scale[k] * dot
    if w > WeightClip: w = WeightClip
    elif w < -WeightClip: w = -WeightClip
    result[c] = w


proc fitDualRidge*(x: seq[seq[float64]],
                   y: seq[float64],
                   lam: float64): seq[float64] =
  ## Exact dual ridge from the Python source, with a natural 2-D API.
  ##
  ## `x` is a `seq[seq[float64]]`; no caller-side flattening is required.
  let (rows, cols) = matrixShape(x, "fitDualRidge")
  if y.len != rows:
    raise newException(ValueError, "fitDualRidge: X/y row mismatch")

  result = newSeq[float64](cols)
  if rows < 2 or cols == 0:
    return

  var ss = newSeq[float64](cols)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      let v = float64(x[r][c])
      ss[c] += v * v

  var active: seq[int] = @[]
  for c in 0 ..< cols:
    if ss[c] > Tiny:
      active.add(c)
  if active.len == 0:
    return

  var scale = newSeq[float64](active.len)
  for k, c in active:
    scale[k] = float64(rows) / ss[c]

  var kMat = newSeq[float64](rows * rows)
  for i in 0 ..< rows:
    for j in 0 .. i:
      var acc = 0.0
      for k, c in active:
        acc += float64(x[i][c]) * scale[k] * float64(x[j][c])
      kMat[i*rows + j] = acc
      kMat[j*rows + i] = acc
  for i in 0 ..< rows:
    kMat[i*rows + i] += lam

  var alpha: seq[float64] = @[]
  var chol = kMat
  if not choleskySolve(chol, y, rows, alpha):
    if not gaussianSolve(kMat, y, rows, alpha):
      return

  for k, c in active:
    var dot = 0.0
    for r in 0 ..< rows:
      dot += float64(x[r][c]) * alpha[r]
    var w = scale[k] * dot
    if w > WeightClip: w = WeightClip
    elif w < -WeightClip: w = -WeightClip
    result[c] = w


proc caseAdjacentRankPairs*(cleanliness: seq[float64],
                            caseIds: seq[int]): RankPairGraph =
  ## Build the same sparse case-local adjacent rank graph as the Python code.
  ## For a case with m rows, all m rows participate but only m-1 edges are built.
  if cleanliness.len != caseIds.len:
    raise newException(ValueError, "caseAdjacentRankPairs: length mismatch")
  let n = cleanliness.len
  if n < 2:
    return

  # Match np.unique(case_ids): process case IDs in ascending order.
  var rowOrder = newSeq[int](n)
  for i in 0 ..< n:
    rowOrder[i] = i
  rowOrder.sort(proc(a, b: int): int =
    let c = cmp(caseIds[a], caseIds[b])
    if c != 0: c else: cmp(a, b)
  )

  var groupStart = 0
  while groupStart < n:
    let cid = caseIds[rowOrder[groupStart]]
    var groupStop = groupStart + 1
    while groupStop < n and caseIds[rowOrder[groupStop]] == cid:
      inc groupStop
    let m = groupStop - groupStart

    if m >= 2:
      var ids = newSeq[int](m)
      var vals = newSeq[float64](m)
      for j in 0 ..< m:
        ids[j] = rowOrder[groupStart + j]
        vals[j] = cleanliness[ids[j]]

      let ranks = averageRanks(vals)
      let rankOrder = stableSortedIndices(ranks)
      let den = max(1.0, float64(m - 1))

      for j in 0 ..< (m - 1):
        let loLocal = rankOrder[j]
        let hiLocal = rankOrder[j + 1]
        let loId = ids[loLocal]
        let hiId = ids[hiLocal]
        let loRank = (ranks[loLocal] - 1.0) / den
        let hiRank = (ranks[hiLocal] - 1.0) / den
        result.lo.add(loId)
        result.hi.add(hiId)
        result.margin.add(hiRank - loRank)

    groupStart = groupStop


proc pairDiffF32(x: seq[seq[float64]], hi, lo, c: int): float64 {.inline.} =
  ## Python computes pair_x in float32, then promotes to float64 for fitting.
  ## Preserve that rounding point for close parity with the source implementation.
  float64(float32(x[hi][c]) - float32(x[lo][c]))


proc fitPairwiseRankRidgeAllModel*(
    x: seq[seq[float64]],
    baseline, cleanliness: seq[float64],
    caseIds: seq[int],
    ridgeLambda: float64,
    maxFeatures: int = DefaultRankMaxFeatures
  ): RankRidgeModel =
  ## Sparse pairwise Rank Ridge from all trajectories.
  ##
  ## Public X is a natural 2-D matrix; no caller-side rows/cols bookkeeping or
  ## flattening is needed. The only flattened matrix created internally contains
  ## the already-selected rank-pair features.
  let (rows, cols) = matrixShape(x, "fitPairwiseRankRidgeAll")
  if baseline.len != rows or cleanliness.len != rows or caseIds.len != rows:
    raise newException(ValueError, "fitPairwiseRankRidgeAll: row-vector length mismatch")

  result.weights = newSeq[float64](cols)
  if rows == 0 or cols == 0:
    return

  let graph = caseAdjacentRankPairs(cleanliness, caseIds)
  let pairCount = graph.hi.len
  if pairCount == 0:
    return


  var pairY = newSeq[float64](pairCount)
  for e in 0 ..< pairCount:
    let hi = graph.hi[e]
    let lo = graph.lo[e]
    pairY[e] = graph.margin[e] - (baseline[hi] - baseline[lo])

  var ss = newSeq[float64](cols)
  var cov = newSeq[float64](cols)
  for e in 0 ..< pairCount:
    let hi = graph.hi[e]
    let lo = graph.lo[e]
    let py = pairY[e]
    for c in 0 ..< cols:
      let d = pairDiffF32(x, hi, lo, c)
      ss[c] += d*d
      cov[c] += d*py

  var usable: seq[int] = @[]
  for c in 0 ..< cols:
    if ss[c] > Tiny:
      usable.add(c)
  if usable.len == 0:
    return

  let cap = max(1, maxFeatures)
  if usable.len > cap:
    usable.sort(proc(a, b: int): int =
      let ra = abs(cov[a]) / sqrt(ss[a])
      let rb = abs(cov[b]) / sqrt(ss[b])
      if ra > rb: return -1
      if ra < rb: return 1
      cmp(a, b)
    )
    usable.setLen(cap)

  result.selectedFeatures = usable

  let q = usable.len
  var pairX = newSeq[float64](pairCount * q)
  for e in 0 ..< pairCount:
    let hi = graph.hi[e]
    let lo = graph.lo[e]
    let off = e * q
    for j, c in usable:
      pairX[off + j] = pairDiffF32(x, hi, lo, c)

  let selectedW = fitDualRidgeFlat(pairX, pairCount, q, pairY, ridgeLambda)
  for j, c in usable:
    result.weights[c] = selectedW[j]


proc fitPairwiseRankRidgeAll*(
    x: seq[seq[float64]],
    baseline, cleanliness: seq[float64],
    caseIds: seq[int],
    ridgeLambda: float64,
    maxFeatures: int = DefaultRankMaxFeatures
  ): seq[float64] =
  fitPairwiseRankRidgeAllModel(
    x, baseline, cleanliness, caseIds, ridgeLambda, maxFeatures
  ).weights


proc predictLinear*(x: seq[seq[float64]],
                    baseline, weights: seq[float64]): seq[float64] =
  ## pred = baseline + X*w.
  let (rows, cols) = matrixShape(x, "predictLinear")
  if baseline.len != rows or weights.len != cols:
    raise newException(ValueError, "predictLinear: dimension mismatch")
  result = newSeq[float64](rows)
  for r in 0 ..< rows:
    var acc = baseline[r]
    for c in 0 ..< cols:
      let w = weights[c]
      if w != 0.0:
        acc += float64(x[r][c]) * w
    result[r] = acc


proc meanCaseSpearman*(pred, target: seq[float64],
                       caseIds: seq[int]): float64 =
  ## Mean Spearman across cases, using every row in each case.
  ## Mirrors pairwise-all scoring (not the legacy holdout split).
  if pred.len != target.len or pred.len != caseIds.len:
    raise newException(ValueError, "meanCaseSpearman: length mismatch")
  let n = pred.len
  if n == 0:
    return 0.0

  var rowOrder = newSeq[int](n)
  for i in 0 ..< n:
    rowOrder[i] = i
  rowOrder.sort(proc(a, b: int): int =
    let c = cmp(caseIds[a], caseIds[b])
    if c != 0: c else: cmp(a, b)
  )

  var total = 0.0
  var count = 0
  var start = 0
  while start < n:
    let cid = caseIds[rowOrder[start]]
    var stop = start + 1
    while stop < n and caseIds[rowOrder[stop]] == cid:
      inc stop
    let m = stop - start
    if m >= 2:
      var a = newSeq[float64](m)
      var b = newSeq[float64](m)
      for j in 0 ..< m:
        let id = rowOrder[start + j]
        a[j] = pred[id]
        b[j] = target[id]
      total += spearman(a, b)
      inc count
    start = stop

  if count == 0: 0.0 else: total / float64(count)


proc fitAndScorePairwiseRankRidge*(
    x: seq[seq[float64]],
    baseline, cleanliness: seq[float64],
    caseIds: seq[int],
    ridgeLambda: float64,
    maxFeatures: int = DefaultRankMaxFeatures
  ): tuple[model: RankRidgeModel, prediction: seq[float64], score: float64] =
  ## End-to-end helper corresponding to the source's pairwise-all readout path.
  result.model = fitPairwiseRankRidgeAllModel(
    x, baseline, cleanliness, caseIds, ridgeLambda, maxFeatures
  )
  result.prediction = predictLinear(x, baseline, result.model.weights)
  result.score = meanCaseSpearman(result.prediction, cleanliness, caseIds)
  if classify(result.score) in {fcNan, fcInf, fcNegInf}:
    result.score = -1.0


when isMainModule:
  # Spearman smoke test from the Python source.
  doAssert abs(spearman(@[1.0, 2.0, 3.0], @[3.0, 2.0, 1.0]) + 1.0) < 1.0e-9

  # Ties use average ranks.
  let rr = averageRanks(@[10.0, 20.0, 20.0, 40.0])
  doAssert rr.len == 4
  doAssert abs(rr[0] - 1.0) < 1.0e-12
  doAssert abs(rr[1] - 2.5) < 1.0e-12
  doAssert abs(rr[2] - 2.5) < 1.0e-12
  doAssert abs(rr[3] - 4.0) < 1.0e-12

  # Same dual-ridge smoke matrix as the supplied Python source.
  let xx = @[
    @[0.0, 0.0],
    @[1.0, 0.0],
    @[2.0, 1.0],
    @[3.0, 1.0]
  ]
  let yy = @[1.0, 0.7, 0.4, 0.1]
  let w = fitDualRidge(xx, yy, 4.0)
  doAssert w.len == 2
  doAssert abs(w[0] - 0.0540229885057471) < 1.0e-9
  doAssert abs(w[1] - 0.0574712643678161) < 1.0e-9

  # Tiny rank-ridge sanity case: two cases, monotone target.
  let x2 = @[
    @[0.0, 0.0],
    @[1.0, 0.0],
    @[2.0, 1.0],
    @[0.0, 0.0],
    @[1.0, 1.0],
    @[2.0, 2.0]
  ]
  let base2 = @[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  let clean2 = @[0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
  let cases2 = @[0, 0, 0, 1, 1, 1]
  let fitted = fitAndScorePairwiseRankRidge(x2, base2, clean2, cases2, 4.0)
  doAssert fitted.model.weights.len == 2
  doAssert fitted.prediction.len == 6
  doAssert abs(fitted.model.weights[0] - 0.1923076923076923) < 1.0e-9
  doAssert abs(fitted.model.weights[1] - 0.1538461538461539) < 1.0e-9
  doAssert abs(fitted.score - 1.0) < 1.0e-9

  echo "spearman_rank_ridge: self-test passed"
