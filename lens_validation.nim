## Disjoint source-chunk partitions and independent model acceptance.
## Source chunks, not individual corrupted rows, are the split unit.
import std/[math, tables]
import lens_core, spearman_rank_ridge

type
  ChunkPartition* = object
    trainEnd*, searchEnd*, acceptEnd*, total*: int

proc splitChunks*(total, window, holdout, requestedSearchPool,
                  requestedAcceptPool: int): ChunkPartition =
  if window < 1 or holdout < 0 or requestedSearchPool < 1 or requestedAcceptPool < 1:
    raise newException(ValueError, "Invalid chunk split parameters")
  let available = total - holdout
  if available < window + 2:
    raise newException(ValueError,
      "Need at least window+2 eligible non-heldout source chunks for disjoint train/search/accept. " &
      "Lower --window or --holdout, or supply more chunks.")
  # Always keep at least one chunk in each validation pool, even for small corpora.
  let a = min(requestedAcceptPool, max(1, (available-window-1) div 2))
  let s = min(requestedSearchPool, available-window-a)
  result = ChunkPartition(trainEnd: available-a-s, searchEnd: available-a,
                          acceptEnd: available, total: total)
  if result.trainEnd < window or result.searchEnd == result.trainEnd or
     result.acceptEnd == result.searchEnd:
    raise newException(ValueError, "Cannot create disjoint source-chunk splits")

proc rotatingIndices*(start, finish, number, iteration: int): seq[int] =
  if finish <= start or number < 1:
    raise newException(ValueError, "Empty validation pool")
  let n = finish-start
  for i in 0..<min(number,n):
    result.add(start + ((iteration mod n)*min(number,n) + i) mod n)

proc outOfSampleRho*(state: State, validation: DataSet): float64 =
  let tr = traceRules(state.rules, validation)
  result = meanCaseSpearman(predict(tr.columns, state.evaluation.weights,
                                    validation.texts.len), validation.target, validation.cases)

proc acceptanceMerit*(rho: float64, rules: int, penalty: float64): float64 =
  rho - float64(rules)*penalty

proc canAccept*(candidateRho: float64, candidateRules: int,
                baselineRho: float64, baselineRules: int,
                penalty, margin: float64): bool =
  if classify(candidateRho) in {fcNan, fcInf, fcNegInf} or
      classify(baselineRho) in {fcNan, fcInf, fcNegInf}: return false
  acceptanceMerit(candidateRho, candidateRules, penalty) >
    acceptanceMerit(baselineRho, baselineRules, penalty) + margin


proc anchorIndices*(start, finish, count: int): seq[int] =
  ## Evenly spaced, disjoint source indices in the acceptance reservoir.
  let n = finish - start
  if n <= 1 or count <= 0: return
  let m = min(count, n - 1) # Leave a source for rotating acceptance.
  for j in 0..<m:
    result.add(start + ((2*j+1)*n) div (2*m))

proc rotatingExcluding*(start, finish, number, iteration: int,
                        excluded: seq[int]): seq[int] =
  ## Use the same deterministic time schedule, skipping fixed anchors.
  var eligible: seq[int]
  for i in start..<finish:
    if i notin excluded: eligible.add(i)
  if eligible.len == 0: return
  let take = min(max(0,number),eligible.len)
  for j in 0..<take:
    result.add(eligible[((iteration mod eligible.len)*take+j) mod eligible.len])

proc canAcceptRobust*(candidateRot, baselineRot, candidateAnchor, baselineAnchor: float64,
                      candidateRules, baselineRules: int,
                      penalty, margin, anchorMargin, floorRho: float64): bool =
  ## Pairwise test: candidates must beat the SAME frozen incumbent on both
  ## independent groups; an overall negative correlation is not 'progress'.
  if classify(candidateAnchor) in {fcNan, fcInf, fcNegInf} or
      classify(baselineAnchor) in {fcNan, fcInf, fcNegInf}: return false
  if candidateRot <= floorRho or candidateAnchor <= floorRho: return false
  result = canAccept(candidateRot,candidateRules,baselineRot,baselineRules,
                     penalty,margin) and
           canAccept(candidateAnchor,candidateRules,baselineAnchor,baselineRules,
                     penalty,anchorMargin)


proc sourceWinFraction*(candidate, incumbent: State, data: DataSet,
                        paths: int, minimumGain = 0.0): float64 =
  ## Paired, per-source improvement test. Corruptions of one original chunk
  ## belong to the SAME source and must not count as independent wins.
  ## Does not consult protected held-out; use only rotating acceptance/anchors.
  if paths < 1 or data.texts.len != data.cases.len or
      data.texts.len != data.target.len:
    raise newException(ValueError, "sourceWinFraction: invalid data")
  let newer = traceRules(candidate.rules, data)
  let older = traceRules(incumbent.rules, data)
  let yp = predict(newer.columns,candidate.evaluation.weights,data.texts.len)
  let bp = predict(older.columns,incumbent.evaluation.weights,data.texts.len)
  var grouped = initOrderedTable[int,seq[int]]()
  for id,cid in data.cases:
    grouped.mgetOrPut(cid,@[]).add(id)
  type Accumulator = tuple[sum: float64, count: int]
  var sources = initOrderedTable[int,Accumulator]()
  for cid,ids in grouped:
    if ids.len < 2: continue
    var x = newSeq[float64](ids.len)
    var z = newSeq[float64](ids.len)
    var y = newSeq[float64](ids.len)
    for j,id in ids:
      x[j] = yp[id]
      z[j] = bp[id]
      y[j] = data.target[id]
    let gain = spearman(x,y) - spearman(z,y)
    let source = cid div paths
    var stat = sources.getOrDefault(source,(sum: 0.0,count: 0))
    stat.sum += gain
    inc stat.count
    sources[source] = stat
  if sources.len == 0: return 0.0
  var wins = 0
  for _,stat in sources:
    if stat.count > 0 and stat.sum/float64(stat.count) > minimumGain:
      inc wins
  float64(wins)/float64(sources.len)

proc blendReadout*(champion, fitted: State, alpha: float64): State =
  ## Propose a new readout without mutating the frozen champion. A separate
  ## out-of-source acceptance test MUST gate the returned proposal.
  if alpha <= 0 or alpha > 1 or
      classify(alpha) in {fcNan, fcInf, fcNegInf} or
      champion.rules != fitted.rules or
      champion.evaluation.weights.len != champion.rules.len or
      fitted.evaluation.weights.len != champion.rules.len:
    raise newException(ValueError, "Invalid readout refresh proposal")
  result = champion
  result.evaluation.weights = newSeq[float64](champion.rules.len)
  for i in 0..<result.evaluation.weights.len:
    result.evaluation.weights[i] =
      (1.0-alpha)*champion.evaluation.weights[i] +
      alpha*fitted.evaluation.weights[i]
