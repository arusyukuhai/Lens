## Ordinary least squares on binary rule-presence features.
## Unpenalized intercept, per-case balanced optional row weights, and
## rank-revealing pivoted Cholesky for dependent/constant indicator columns.
## This is NOT ridge; dependent columns receive coefficient zero.
import std/[math, tables]

type
  OlsFit* = object
    weights*: seq[float64]
    intercept*: float64
  OlsAppendCache* = object
    p*: int
    base*: OlsFit
    rowWeights: seq[float64]
    meanY: float64
    means: seq[float64]
    target: seq[float64]
    baseColumns: seq[seq[float64]]
    active: seq[int]
    lower: seq[float64]
    activeBeta: seq[float64]
    tolerance: float64

proc balancedWeights(target: seq[float64], cases: seq[int], balance: bool): seq[float64] =
  if target.len != cases.len: raise newException(ValueError,"OLS target/case dimension mismatch")
  result = newSeq[float64](target.len)
  if target.len == 0: return
  if not balance:
    for i in 0..<result.len: result[i] = 1.0/float64(result.len)
  else:
    var count = initCountTable[int]()
    for c in cases: count.inc(c)
    let total = float64(count.len)
    for i,c in cases: result[i] = 1.0 / (total * float64(count[c]))

proc solveCholesky(lower: seq[float64], rhs: seq[float64]): seq[float64] =
  let p = rhs.len
  result = newSeq[float64](p)
  for i in 0..<p:
    var v = rhs[i]
    for j in 0..<i: v -= lower[i*p+j]*result[j]
    result[i] = v/lower[i*p+i]
  for i in countdown(p-1,0):
    var v = result[i]
    for j in i+1..<p: v -= lower[j*p+i]*result[j]
    result[i] = v/lower[i*p+i]

proc prepareOLSAppend*(columns: seq[seq[float64]], target: seq[float64],
                       cases: seq[int], balance = false,
                       tolerance = 1.0e-10): OlsAppendCache =
  ## Small P-by-P covariance matrix. Total row mass is one.
  if tolerance <= 0.0 or tolerance >= 0.1: raise newException(ValueError,"Invalid OLS tolerance")
  result.p = columns.len
  result.tolerance = tolerance
  result.target = target
  result.rowWeights = balancedWeights(target,cases,balance)
  let n = target.len
  let p = columns.len
  for col in columns:
    if col.len != n: raise newException(ValueError,"OLS column dimension mismatch")
  result.baseColumns = columns
  result.means = newSeq[float64](p)
  result.base.weights = newSeq[float64](p)
  if n == 0: return
  for i in 0..<n:
    let w = result.rowWeights[i]
    result.meanY += w*target[i]
    for j in 0..<p: result.means[j] += w*columns[j][i]
  result.base.intercept = result.meanY
  if p == 0: return
  var gram = newSeq[float64](p*p)
  var rhs = newSeq[float64](p)
  for i in 0..<n:
    let wi = result.rowWeights[i]
    let dy = target[i]-result.meanY
    for j in 0..<p:
      let dj = columns[j][i]-result.means[j]
      rhs[j] += wi*dj*dy
      for k in 0..j:
        gram[j*p+k] += wi*dj*(columns[k][i]-result.means[k])
  for j in 0..<p:
    for k in 0..<j: gram[k*p+j] = gram[j*p+k]

  # Pivoted Cholesky on the correlation matrix identifies linearly dependent
  # columns without adding a ridge penalty to the regression objective.
  var normGram = newSeq[float64](p*p)
  var scales = newSeq[float64](p)
  var residual = newSeq[float64](p)
  for j in 0..<p:
    scales[j] = sqrt(max(0.0,gram[j*p+j]))
    residual[j] = (if scales[j] > 1.0e-14: 1.0 else: 0.0)
  for j in 0..<p:
    for k in 0..<p:
      if scales[j] > 1.0e-14 and scales[k] > 1.0e-14:
        normGram[j*p+k] = gram[j*p+k] / (scales[j]*scales[k])
  var choleskyRows = newSeq[seq[float64]](p)
  while result.active.len < p:
    var pivot = -1
    var top = tolerance
    for j in 0..<p:
      if residual[j] > top:
        top = residual[j]
        pivot = j
    if pivot < 0: break
    let previousDepth = result.active.len
    result.active.add(pivot)
    let d = sqrt(top)
    let pivotFactors = choleskyRows[pivot]
    var updates = newSeq[float64](p)
    for j in 0..<p:
      var v = normGram[j*p+pivot]
      for k in 0..<previousDepth:
        v -= choleskyRows[j][k]*pivotFactors[k]
      updates[j] = v/d
    for j in 0..<p:
      choleskyRows[j].add(updates[j])
      residual[j] = max(0.0,residual[j]-updates[j]*updates[j])
    residual[pivot] = -1.0
  let q = result.active.len
  if q == 0: return
  result.lower = newSeq[float64](q*q)
  var activeRhs = newSeq[float64](q)
  for j in 0..<q:
    activeRhs[j] = rhs[result.active[j]]
    for k in 0..j:
      var v = gram[result.active[j]*p+result.active[k]]
      for t in 0..<k: v -= result.lower[j*q+t]*result.lower[k*q+t]
      if j == k:
        if v <= 0: raise newException(ValueError,"OLS independent basis lost positive definiteness")
        result.lower[j*q+j] = sqrt(v)
      else:
        result.lower[j*q+k] = v / result.lower[k*q+k]
  result.activeBeta = solveCholesky(result.lower,activeRhs)
  for j,id in result.active: result.base.weights[id] = result.activeBeta[j]
  for j in 0..<p: result.base.intercept -= result.means[j]*result.base.weights[j]

proc fitOLS*(columns: seq[seq[float64]], target: seq[float64], cases: seq[int],
             balance = false, tolerance = 1.0e-10): OlsFit =
  prepareOLSAppend(columns,target,cases,balance,tolerance).base

proc fitOLSAppend*(cache: OlsAppendCache, column: seq[float64]): OlsFit =
  ## Frisch-Waugh-Lovell one-column update. O(rows*rank + rank^2) per
  ## candidate, with no Search/Acceptance leakage. Singular candidates
  ## keep the original prediction unchanged.
  let n = cache.target.len
  if column.len != n: raise newException(ValueError,"OLS append column dimension mismatch")
  result.weights = newSeq[float64](cache.p+1)
  for j,v in cache.base.weights: result.weights[j] = v
  result.intercept = cache.base.intercept
  if n == 0: return
  var mx = 0.0
  for i in 0..<n: mx += cache.rowWeights[i]*column[i]
  let q = cache.active.len
  var cross = newSeq[float64](q)
  var variance = 0.0
  var covariance = 0.0
  for i in 0..<n:
    let w = cache.rowWeights[i]
    let dx = column[i]-mx
    variance += w*dx*dx
    covariance += w*dx*(cache.target[i]-cache.meanY)
    for j,id in cache.active:
      cross[j] += w*dx*(cache.baseColumns[id][i]-cache.means[id])
  if variance < 1.0e-14: return
  let projection = solveCholesky(cache.lower,cross)
  var conditionalVar = variance
  var conditionalCov = covariance
  for j in 0..<q:
    conditionalVar -= cross[j]*projection[j]
    conditionalCov -= cross[j]*cache.activeBeta[j]
  if conditionalVar <= cache.tolerance*variance: return
  let extra = conditionalCov/conditionalVar
  if classify(extra) in {fcNan,fcInf,fcNegInf}: return
  result.weights[cache.p] = extra
  for j,id in cache.active: result.weights[id] -= projection[j]*extra
  result.intercept = cache.meanY-mx*extra
  for j in 0..<cache.p: result.intercept -= result.weights[j]*cache.means[j]

proc pearson*(a,b: openArray[float64]): float64 =
  if a.len != b.len or a.len < 2: return 0.0
  var ma = 0.0
  var mb = 0.0
  for i in 0..<a.len:
    ma += a[i]
    mb += b[i]
  ma /= float64(a.len)
  mb /= float64(b.len)
  var va = 0.0
  var vb = 0.0
  var cov = 0.0
  for i in 0..<a.len:
    let da = a[i]-ma
    let db = b[i]-mb
    va += da*da
    vb += db*db
    cov += da*db
  if va <= 1.0e-24 or vb <= 1.0e-24: return 0.0
  max(-1.0,min(1.0,cov/sqrt(va*vb)))
