## Lens count-feature greedy/beam symbolic learner.
## Uses the optimized matcher/replacer in replace.nim and the bounded CMS miner.
import std/[os, strutils, sequtils, tables, hashes, algorithm, math, times]
import replace, fast_count_min_sketch, token_codec

type
  Config = object
    corpus, outDir, resumeFile: string
    chunks, trainChunks, validChunks, chunkLen, noiseSteps: int
    steps, maxRules, minNgram, maxGram, maxTotalGram, maxWildcards: int
    maxGap, mineStarts, mineBudget, mineTracked, candidates, beamWidth, swapDepth: int
    minGain, ridgeLambda: float64
    screenRows, screenCandidates, plotEvery: int # 0 disables respective work
    cmsWidth, cmsDepth: int
    useFingerprint: bool # exact no-false-negative match rejection
    usePrefixCache: bool # reuse mining and exact counts among equal prefixes
    usePrefixStateCache: bool # reuse pre-swap rewritten rows when ranking branches
    useDeltaSuffix: bool # skip downstream rewrites on rows with no new-rule matches
    tokenSeed: int
    tokenMapFile: string

  Rule = object
    pat: seq[int]
    rhs: seq[int]

  Row = object
    text: seq[int]
    target: float64

  Encoded = object
    cols: seq[seq[float64]] # each old rule's number of non-overlapping matches
    finalTexts: seq[seq[int]]
    y: seq[float64]

  Fit = object
    q: seq[seq[float64]] # orthonormal column basis, including intercept
    beta: seq[float64]
    residual: seq[float64]
    removalLoss: seq[float64]
    sse, sst, pearson: float64
    rank: int
    # Ridge-specific sufficient statistics, based on centered, variance-normalized
    # counts (the intercept is deliberately excluded from the penalty).
    ridgeLambda: float64
    featureCols: seq[seq[float64]]
    featureMeans, featureScales: seq[float64]
    standardizedGram, inversePenalizedGram: seq[float64]
    standardizedXty, standardizedBeta: seq[float64]
    targetMean: float64

  PatternScore = object
    pat: seq[int]
    gain: float64
    frequency: int

  TrackedPattern = object
    pat: seq[int]
    estimate: uint32 # CMS estimate at the last update; monotone lower bound

  MineState = object
    text: seq[int]
    cfg: Config
    sketch: CountMinSketch
    tracked: seq[TrackedPattern] # min-heap keyed by the CMS estimate
    positions: Table[seq[int], int] # pattern -> index in tracked heap
    offers, limit: int
    gapProductLimit: float64
    gapRootLimit: int

  Branch = object
    rules: seq[Rule]
    description: string
    score: float64 # Current training-window fitted Pearson, refreshed by rankBranches
    editedAt: int # index of the rule most recently added/replaced, or -1

  MoveKind = enum
    moveNone, moveAdd, moveReplace

  DiffusionCache = object
    byChunk: Table[int, seq[Row]]
    recentlyUsed: seq[int]
    capacity: int

proc readInt(s: string, fallback: int): int =
  try: result = parseInt(s)
  except ValueError: result = fallback

proc readFloat(s: string, fallback: float64): float64 =
  try: result = parseFloat(s)
  except ValueError: result = fallback

proc parseConfig(): Config =
  result = Config(corpus: "github-code-dataset.txt", outDir: "lens_run",
    chunks: 65536, trainChunks: 3, validChunks: 2, chunkLen: 1500,
    noiseSteps: 0, steps: 100, maxRules: 24, minNgram: 1,
    maxGram: 4, maxTotalGram: 8, maxWildcards: 3, maxGap: 16,
    mineStarts: 12, mineBudget: 180000, mineTracked: 25000,
    candidates: 384, beamWidth: 4, swapDepth: 3, minGain: 1e-9,
    ridgeLambda: 0.1, screenRows: 32, screenCandidates: 96,
    useFingerprint: true, usePrefixCache: true, usePrefixStateCache: true,
    useDeltaSuffix: false, plotEvery: 1,
    cmsWidth: 65536, cmsDepth: 4,
    tokenSeed: 1729, tokenMapFile: "")
  if not fileExists(result.corpus) and fileExists("github-code.txt"):
    result.corpus = "github-code.txt"
  for arg in commandLineParams():
    if arg == "--help" or arg == "-h":
      echo "Lens Ridge/OLS count-rule training"
      echo "--self-test | --corpus=FILE --out=DIR --resume=FILE --chunks=N --train=N --valid=N"
      echo "--chunk-len=N --noise-steps=N (0=one step per character) --steps=N"
      echo "--max-rules=N --min-ngram=N --max-gram=N --max-total-gram=N"
      echo "--max-wildcards=N --max-gap=N --mine-starts=N --mine-budget=N"
      echo "--mine-tracked=N --candidates=N --beam=N --swap-depth=N"
      echo "--ridge-lambda=0.1 (standardized predictors; 0=legacy OLS)"
      echo "--screen-rows=32 --screen-candidates=96 (0 disables; approximate screen)"
      echo "--fingerprint=1 (0 disables; exact presence-based matcher skips)"
      echo "--prefix-cache=1 (0 disables; compare old per-parent mining/counting)"
      echo "--prefix-state-cache=1 (0 disables; compare full re-encoding for swaps)"
      echo "--delta-suffix=1 (0 disables; compare regular prefix-state cache)"
      echo "--plot-every=1 (5 renders every 5 steps; 0 disables rendering)"
      echo "--cms-width=65536 --cms-depth=4 (power-of-two width; depth=2/3 faster but noisier)"
      echo "Build with -d:lensProfile to profile mining, screening and exact scoring"
      echo "--token-seed=N --token-map=FILE (re-use mapping across runs)"
      quit(0)
    if not arg.startsWith("--") or not arg.contains('='): continue
    let parts = arg[2..^1].split('=', 1)
    let key = parts[0]
    let val = parts[1]
    case key
    of "corpus": result.corpus = val
    of "out": result.outDir = val
    of "resume": result.resumeFile = val
    of "chunks": result.chunks = readInt(val, result.chunks)
    of "train": result.trainChunks = readInt(val, result.trainChunks)
    of "valid": result.validChunks = readInt(val, result.validChunks)
    of "chunk-len": result.chunkLen = readInt(val, result.chunkLen)
    of "noise-steps": result.noiseSteps = readInt(val, result.noiseSteps)
    of "steps": result.steps = readInt(val, result.steps)
    of "max-rules": result.maxRules = readInt(val, result.maxRules)
    of "min-ngram": result.minNgram = readInt(val, result.minNgram)
    of "max-gram": result.maxGram = readInt(val, result.maxGram)
    of "max-total-gram": result.maxTotalGram = readInt(val, result.maxTotalGram)
    of "max-wildcards": result.maxWildcards = readInt(val, result.maxWildcards)
    of "max-gap": result.maxGap = readInt(val, result.maxGap)
    of "mine-starts": result.mineStarts = readInt(val, result.mineStarts)
    of "mine-budget": result.mineBudget = readInt(val, result.mineBudget)
    of "mine-tracked": result.mineTracked = readInt(val, result.mineTracked)
    of "candidates": result.candidates = readInt(val, result.candidates)
    of "beam": result.beamWidth = readInt(val, result.beamWidth)
    of "swap-depth": result.swapDepth = readInt(val, result.swapDepth)
    of "ridge-lambda": result.ridgeLambda = readFloat(val, result.ridgeLambda)
    of "screen-rows": result.screenRows = readInt(val, result.screenRows)
    of "screen-candidates": result.screenCandidates = readInt(val, result.screenCandidates)
    of "fingerprint": result.useFingerprint = readInt(val, 1) != 0
    of "prefix-cache": result.usePrefixCache = readInt(val, 1) != 0
    of "prefix-state-cache": result.usePrefixStateCache = readInt(val, 1) != 0
    of "delta-suffix": result.useDeltaSuffix = readInt(val, 1) != 0
    of "plot-every": result.plotEvery = readInt(val, result.plotEvery)
    of "cms-width": result.cmsWidth = readInt(val, result.cmsWidth)
    of "cms-depth": result.cmsDepth = readInt(val, result.cmsDepth)
    of "lookahead-budget", "lookahead-candidates", "beam-expand", "beam-tolerance":
      stderr.writeLine("[deprecated] ", key,
        " is ignored; the beam now uses direct current-window Pearson only")
    of "token-seed": result.tokenSeed = readInt(val, result.tokenSeed)
    of "token-map": result.tokenMapFile = val
    else: stderr.writeLine("Unknown option: ", arg)
  result.chunkLen = max(8, result.chunkLen)
  result.trainChunks = max(1, result.trainChunks)
  result.validChunks = max(0, result.validChunks)
  result.steps = max(1, result.steps)
  result.maxRules = max(1, result.maxRules)
  result.minNgram = max(1, result.minNgram)
  result.maxGram = max(result.minNgram, result.maxGram)
  result.maxTotalGram = max(result.minNgram, result.maxTotalGram)
  result.maxWildcards = max(0, min(15, result.maxWildcards))
  result.maxGap = max(1, result.maxGap)
  result.mineStarts = max(1, result.mineStarts)
  result.mineBudget = max(100, result.mineBudget)
  result.mineTracked = max(100, result.mineTracked)
  result.candidates = max(1, result.candidates)
  result.beamWidth = max(1, result.beamWidth)
  result.swapDepth = max(1, result.swapDepth)
  result.screenRows = max(0, result.screenRows)
  result.screenCandidates = max(0, result.screenCandidates)
  result.plotEvery = max(0, result.plotEvery)
  if result.cmsWidth <= 0 or (result.cmsWidth and (result.cmsWidth - 1)) != 0 or
      result.cmsWidth > 1048576 or result.cmsDepth < 1 or result.cmsDepth > 8:
    quit("Invalid CMS dimensions: width must be a power of two <= 2^20; depth 1..8", 1)
  if classify(result.ridgeLambda) in {fcNan, fcInf, fcNegInf} or result.ridgeLambda < 0.0:
    quit("--ridge-lambda must be finite and nonnegative", 1)

proc loadCorpus(cfg: Config): seq[seq[int]] =
  if not fileExists(cfg.corpus):
    quit("Corpus missing: " & cfg.corpus, 1)
  let f = open(cfg.corpus, fmRead)
  defer: f.close()
  var current: seq[int] = @[]
  while not f.endOfFile and result.len < cfg.chunks:
    let line = f.readLine()
    if line == "===SPLIT===":
      if current.len > 0:
        result.add(current)
      current = @[]
    else:
      if current.len < cfg.chunkLen:
        for ch in line:
          if current.len >= cfg.chunkLen: break
          current.add(ord(ch))
        if current.len < cfg.chunkLen: current.add(10)
  if current.len > 0 and result.len < cfg.chunks:
    result.add(current)
  if result.len == 0: quit("No nonempty corpus chunks", 1)

proc randomNext(seed: var uint64): uint64 {.inline.} =
  seed = seed xor (seed shl 13)
  seed = seed xor (seed shr 7)
  seed = seed xor (seed shl 17)
  result = seed

proc appendDiffusion(rows: var seq[Row], original: seq[int], chunkId: int,
                     cfg: Config, codec: TokenCodec) =
  if original.len < 2: return
  var rng = (uint64(chunkId + 1) * 0x9E3779B97F4A7C15'u64) xor 0xD1B54A32D192ED03'u64
  var indices = newSeq[int](original.len)
  for i in 0..<indices.len: indices[i] = i
  for i in countdown(indices.len - 1, 1):
    let j = int(randomNext(rng) mod uint64(i + 1))
    swap(indices[i], indices[j])
  var data = newSeq[int](original.len)
  for i, v in original: data[i] = v
  let stages = (if cfg.noiseSteps <= 0: original.len else: min(original.len, cfg.noiseSteps))
  var previous = 0
  for stage in 0..stages:
    let changed = stage * original.len div stages
    while previous < changed:
      let pos = indices[previous]
      # Diffuse the frequency-encoded text BEFORE applying ANY rules.
      # Noise is an existing, corpus-observed alphabet token (not OOV),
      # drawn by frequency and different from the original token.
      data[pos] = noiseToken(codec, chunkId, pos, original[pos])
      inc previous
    var snapshot = newSeq[int](data.len)
    for i, v in data: snapshot[i] = v
    rows.add(Row(text: snapshot, target: 1.0 - float64(changed) / float64(original.len)))

proc rowsForChunk(corpus: seq[seq[int]], cfg: Config, codec: TokenCodec,
                  chunkId: int, cache: var DiffusionCache): seq[Row] =
  if cache.byChunk.hasKey(chunkId):
    # LRU cache: neighboring rotations reuse all but one chunk.
    for i in 0..<cache.recentlyUsed.len:
      if cache.recentlyUsed[i] == chunkId:
        cache.recentlyUsed.delete(i)
        break
    cache.recentlyUsed.add(chunkId)
    return cache.byChunk[chunkId]
  appendDiffusion(result, corpus[chunkId], chunkId, cfg, codec)
  if cache.recentlyUsed.len >= cache.capacity:
    let evicted = cache.recentlyUsed[0]
    cache.recentlyUsed.delete(0)
    cache.byChunk.del(evicted)
  cache.recentlyUsed.add(chunkId)
  cache.byChunk[chunkId] = result

proc makeRows(corpus: seq[seq[int]], cfg: Config, codec: TokenCodec,
              step: int, validation: bool, cache: var DiffusionCache): seq[Row] =
  let validCount = min(cfg.validChunks, max(0, corpus.len - 1))
  let trainPool = max(1, corpus.len - validCount)
  if validation:
    for i in trainPool..<corpus.len:
      let chunkRows = rowsForChunk(corpus, cfg, codec, i, cache)
      result.add(chunkRows)
  else:
    for offset in 0..<cfg.trainChunks:
      let idx = (step + offset) mod trainPool
      let chunkRows = rowsForChunk(corpus, cfg, codec, idx, cache)
      result.add(chunkRows)

when defined(lensProfile):
  type EvaluationCounters = object
    evaluateCalls, rankCalls, rankVariants, rankCacheCalls: int
    rankPrefixBuilds, rankPrefixHits, rankPrefixEvictions: int
    rankLastCountOnly: int
    evaluateEncodeSeconds, evaluateRidgeSeconds: float64
    rankTotalSeconds, rankPrefixBuildSeconds: float64
    rankSuffixSeconds, rankRidgeSeconds: float64
    deltaCalls, deltaFallbacks, deltaRowsScanned: int
    deltaMatchedRows, deltaReusedRows, deltaBaselineBuilds: int
    deltaPatternCacheHits: int
    deltaSeconds, deltaRidgeSeconds: float64
  var profileEval: EvaluationCounters

proc encode(rows: seq[Row], rules: seq[Rule]): Encoded =
  let n = rows.len
  result.y = newSeq[float64](n)
  result.finalTexts = newSeq[seq[int]](n)
  result.cols = newSeq[seq[float64]](rules.len)
  var programs = newSeq[SeqReplaceProgram](rules.len)
  for j in 0..<rules.len:
    programs[j] = compileSeqReplaceProgram(rules[j].pat, rules[j].rhs, 1)
    result.cols[j] = newSeq[float64](n)
  for i in 0..<n:
    result.y[i] = rows[i].target
    var state = rows[i].text
    for j in 0..<rules.len:
      result.cols[j][i] = float64(applyAndCount(state, programs[j]))
    result.finalTexts[i] = state

proc dot(a, b: seq[float64]): float64 {.inline.} =
  for i in 0..<a.len: result += a[i] * b[i]

proc ols(cols: seq[seq[float64]], y: seq[float64],
         computeDeletion = true): Fit =
  ## Two-pass modified Gram-Schmidt, O(n*p*p), no ridge penalty.
  ## Singular columns are skipped, and deletion scoring falls back to
  ## exact reduced fits when full-rank Gram inverse does not exist.
  let n = y.len
  let p = cols.len + 1 # the intercept is never deleted
  result.beta = newSeq[float64](p)
  result.residual = newSeq[float64](n)
  result.removalLoss = newSeq[float64](cols.len)
  if n == 0: return
  var rmat = newSeq[float64](p * p)
  var active: seq[int] = @[]
  var qty: seq[float64] = @[]
  var mean = 0.0
  for v in y: mean += v
  mean /= float64(n)
  for v in y: result.sst += (v - mean) * (v - mean)
  for j in 0..<p:
    var v = newSeq[float64](n)
    if j == 0:
      for i in 0..<n: v[i] = 1.0
    else:
      for i in 0..<n: v[i] = cols[j-1][i]
    let originalNorm = sqrt(dot(v, v))
    for passIndex in 0..1:
      for k in 0..<active.len:
        let projection = dot(result.q[k], v)
        rmat[active[k] * p + j] += projection
        for i in 0..<n: v[i] -= projection * result.q[k][i]
    let norm = sqrt(max(0.0, dot(v, v)))
    if norm <= 1e-10 * max(1.0, originalNorm):
      continue
    rmat[j * p + j] = norm
    for i in 0..<n: v[i] /= norm
    active.add(j)
    result.q.add(v)
    qty.add(dot(v, y))
  result.rank = active.len
  for i in countdown(active.len - 1, 0):
    let j = active[i]
    var value = qty[i]
    for k in i+1..<active.len:
      value -= rmat[j * p + active[k]] * result.beta[active[k]]
    result.beta[j] = value / rmat[j * p + j]
  for i in 0..<n:
    result.residual[i] = y[i]
    for k in 0..<result.q.len:
      result.residual[i] -= result.q[k][i] * qty[k]
    result.sse += result.residual[i] * result.residual[i]
  if result.sst > 1e-24:
    result.pearson = sqrt(max(0.0, min(1.0, 1.0 - result.sse/result.sst)))

  if computeDeletion and cols.len > 0:
    if active.len == p:
      # C_jj = ||row_j(R^-1)||^2.
      var invR = newSeq[float64](p * p)
      for col in 0..<p:
        for i in countdown(p - 1, 0):
          var value = (if i == col: 1.0 else: 0.0)
          for k in i+1..<p:
            value -= rmat[i * p + k] * invR[k * p + col]
          invR[i * p + col] = value / rmat[i * p + i]
      for j in 1..<p:
        var cjj = 0.0
        for k in 0..<p:
          cjj += invR[j * p + k] * invR[j * p + k]
        if cjj > 1e-24:
          result.removalLoss[j-1] = result.beta[j] * result.beta[j] / cjj
    else:
      # Exact fallback for collinear predictors; expensive but safe.
      for j in 0..<cols.len:
        var reduced: seq[seq[float64]] = @[]
        for k in 0..<cols.len:
          if k != j: reduced.add(cols[k])
        let reducedFit = ols(reduced, y, false)
        result.removalLoss[j] = max(0.0, reducedFit.sse - result.sse)

# Ridge regression with free intercept and per-feature standardization.
# Minimize ||y - intercept - sum_j x_j*beta_j||^2 + n*lambda*sum_j(theta_j)^2,
# with theta_j=beta_j*stddev(x_j). This avoids count-frequency-dependent
# regularization and gives a positive-definite solve for lambda > 0.
proc choleskySolve(l: seq[float64], n: int,
                   rhs: openArray[float64]): seq[float64] =
  var tmp = newSeq[float64](n)
  result = newSeq[float64](n)
  for i in 0..<n:
    var v = rhs[i]
    for k in 0..<i:
      v -= l[i*n+k] * tmp[k]
    tmp[i] = v / l[i*n+i]
  for i in countdown(n-1, 0):
    var v = tmp[i]
    for k in i+1..<n:
      v -= l[k*n+i] * result[k]
    result[i] = v / l[i*n+i]

proc standardizedPearson(theta: seq[float64],
                         crossY, gram: seq[float64], sst: float64): float64 =
  let m = theta.len
  if m == 0 or sst <= 1e-24: return 0.0
  var cov, normSq = 0.0
  for j in 0..<m:
    cov += theta[j] * crossY[j]
    var dotRow = 0.0
    for k in 0..<m:
      dotRow += gram[j*m+k] * theta[k]
    normSq += theta[j] * dotRow
  if normSq <= 1e-24: return 0.0
  result = max(-1.0, min(1.0, cov / sqrt(sst * normSq)))

proc ridge(cols: seq[seq[float64]], y: seq[float64],
           ridgeLambda: float64, computeDeletion = true): Fit =
  ## lambda is relative to N because each column has variance one.
  ## lambda = 0 retains exactly the previous OLS behavior.
  if ridgeLambda == 0.0:
    return ols(cols, y, computeDeletion)
  let n = y.len
  let m = cols.len
  result.beta = newSeq[float64](m+1)
  result.residual = newSeq[float64](n)
  result.removalLoss = newSeq[float64](m)
  result.ridgeLambda = ridgeLambda
  result.featureCols = cols
  result.featureMeans = newSeq[float64](m)
  result.featureScales = newSeq[float64](m)
  result.standardizedGram = newSeq[float64](m*m)
  result.standardizedXty = newSeq[float64](m)
  result.standardizedBeta = newSeq[float64](m)
  if computeDeletion:
    result.inversePenalizedGram = newSeq[float64](m*m)
  if n == 0: return
  let nf = float64(n)
  for v in y: result.targetMean += v
  result.targetMean /= nf
  result.beta[0] = result.targetMean
  for v in y:
    result.sst += (v-result.targetMean)*(v-result.targetMean)
  for j in 0..<m:
    var mean = 0.0
    for v in cols[j]: mean += v
    mean /= nf
    result.featureMeans[j] = mean
    var variance = 0.0
    for v in cols[j]:
      variance += (v-mean)*(v-mean)
    let std = sqrt(max(0.0, variance/nf))
    # A constant column contains no new information; Ridge handles it safely.
    if std > 1e-12 * max(1.0, abs(mean)):
      result.featureScales[j] = std
    else:
      result.featureScales[j] = 1.0
    var xy = 0.0
    for i in 0..<n:
      xy += (cols[j][i]-mean)*(y[i]-result.targetMean)
    result.standardizedXty[j] = xy/result.featureScales[j]
  for j in 0..<m:
    for k in 0..j:
      var cross = 0.0
      for i in 0..<n:
        cross += (cols[j][i]-result.featureMeans[j]) *
          (cols[k][i]-result.featureMeans[k])
      cross /= result.featureScales[j]*result.featureScales[k]
      result.standardizedGram[j*m+k] = cross
      result.standardizedGram[k*m+j] = cross
  if m > 0:
    var penalized = newSeq[float64](m*m)
    for k in 0..<m*m: penalized[k] = result.standardizedGram[k]
    for j in 0..<m: penalized[j*m+j] += nf*ridgeLambda
    var l = newSeq[float64](m*m)
    for i in 0..<m:
      for j in 0..i:
        var value = penalized[i*m+j]
        for k in 0..<j:
          value -= l[i*m+k]*l[j*m+k]
        if i == j:
          if value <= 0.0:
            raise newException(ValueError, "Ridge Cholesky failed: invalid Gram")
          l[i*m+j] = sqrt(value)
        else:
          l[i*m+j] = value/l[j*m+j]
    result.standardizedBeta = choleskySolve(l, m, result.standardizedXty)
    if computeDeletion:
      for col in 0..<m:
        var rhs = newSeq[float64](m)
        rhs[col] = 1.0
        let solution = choleskySolve(l, m, rhs)
        for j in 0..<m:
          result.inversePenalizedGram[j*m+col] = solution[j]
    for j in 0..<m:
      result.beta[j+1] = result.standardizedBeta[j] /
        result.featureScales[j]
      result.beta[0] -= result.beta[j+1]*result.featureMeans[j]
  result.rank = m + 1
  for i in 0..<n:
    var pred = result.beta[0]
    for j in 0..<m:
      pred += result.beta[j+1] * cols[j][i]
    result.residual[i] = y[i] - pred
    result.sse += result.residual[i]*result.residual[i]
  result.pearson = standardizedPearson(result.standardizedBeta,
    result.standardizedXty, result.standardizedGram, result.sst)
  if computeDeletion:
    # Removing j and refitting the other Ridge weights via one Schur update:
    # theta'_k = theta_k - C[k,j]/C[j,j]*theta_j (k != j).
    # Unlike OLS, beta_j^2/C_jj measures PENALIZED-OBJECTIVE loss,
    # not Pearson loss. Rank deletions by true Pearson change instead.
    for j in 0..<m:
      let denom = result.inversePenalizedGram[j*m+j]
      if denom <= 1e-24: continue
      var reduced = newSeq[float64](m)
      for k in 0..<m:
        if k != j:
          reduced[k] = result.standardizedBeta[k] -
            result.inversePenalizedGram[k*m+j]/denom *
              result.standardizedBeta[j]
      let reducedPearson = standardizedPearson(reduced,
        result.standardizedXty, result.standardizedGram, result.sst)
      result.removalLoss[j] = result.pearson - reducedPearson

proc ridgeAppendedPearson(z: seq[float64], fit: Fit): float64 =
  ## Exact *Pearson* of the Ridge solution after appending z, calculated
  ## through a one-column block inverse. No complete refit needed.
  let n = z.len
  let m = fit.standardizedBeta.len
  if n == 0 or fit.sst <= 1e-24: return fit.pearson
  let nf = float64(n)
  var meanZ = 0.0
  for value in z: meanZ += value
  meanZ /= nf
  var ss = 0.0
  for value in z: ss += (value-meanZ)*(value-meanZ)
  let scale = sqrt(max(0.0, ss/nf))
  if scale <= 1e-12*max(1.0, abs(meanZ)):
    return fit.pearson
  let zz = ss/(scale*scale)
  var residualCross = 0.0
  for i in 0..<n:
    residualCross += (z[i]-meanZ)*fit.residual[i]
  residualCross /= scale
  var cross = newSeq[float64](m)
  for j in 0..<m:
    for i in 0..<n:
      cross[j] += (z[i]-meanZ) *
        (fit.featureCols[j][i]-fit.featureMeans[j])
    cross[j] /= scale*fit.featureScales[j]
  var projection = newSeq[float64](m)
  for j in 0..<m:
    for k in 0..<m:
      projection[j] += fit.inversePenalizedGram[j*m+k]*cross[k]
  # g = z_std^T(y_center - X_std*theta) is the Ridge residual cross-product.
  let g = residualCross
  var denom = zz + nf*fit.ridgeLambda
  for j in 0..<m: denom -= cross[j]*projection[j]
  if denom <= 1e-15: return fit.pearson
  let alpha = g/denom
  var newTheta = newSeq[float64](m)
  for j in 0..<m:
    newTheta[j] = fit.standardizedBeta[j] - projection[j]*alpha
  var cov = alpha*(g + dot(cross, fit.standardizedBeta))
  for j in 0..<m: cov += newTheta[j]*fit.standardizedXty[j]
  var predictionNorm = alpha*alpha*zz
  for j in 0..<m:
    predictionNorm += 2.0*alpha*newTheta[j]*cross[j]
    var row = 0.0
    for k in 0..<m:
      row += fit.standardizedGram[j*m+k]*newTheta[k]
    predictionNorm += newTheta[j]*row
  if predictionNorm <= 1e-24: return 0.0
  result = max(-1.0, min(1.0, cov/sqrt(fit.sst*predictionNorm)))

proc candidateGain(z: seq[float64], fit: Fit): float64 =
  if fit.ridgeLambda > 0.0:
    return ridgeAppendedPearson(z, fit) - fit.pearson
  var g = dot(z, fit.residual)
  var s = dot(z, z)
  for q in fit.q:
    let proj = dot(q, z)
    s -= proj * proj
  if s <= 1e-9 * max(1.0, dot(z,z)):
    return 0.0
  result = max(0.0, g * g / s)

proc patternString(p: seq[int]): string =
  result = p.mapIt($it).join(",")

proc parsePattern(s: string): seq[int] =
  for part in s.split(','):
    if part.strip().len > 0: result.add(parseInt(part.strip()))

proc saveRules(path: string, rules: seq[Rule]) =
  var outp = "# Lens rule file: pattern TAB replacement\n"
  for r in rules:
    outp.add(patternString(r.pat) & "\t" & patternString(r.rhs) & "\n")
  writeFile(path, outp)

proc wildcardCount(pat: openArray[int]): int {.inline.} =
  for value in pat:
    if value < 0: inc result

proc captureIndex(opcode: int): int {.inline.} =
  ## Replacement opcodes: plain, sort, reverse, inverse_sort.
  if opcode >= -15 and opcode <= -1: return -opcode - 1
  if opcode >= -31 and opcode <= -16: return -opcode - 16
  if opcode >= -47 and opcode <= -32: return -opcode - 32
  if opcode >= -63 and opcode <= -48: return -opcode - 48
  return -1

proc isAlternatingReplacement(pat: seq[int], rhs: seq[int]): bool =
  ## The RHS MUST be [fresh0, capture0, fresh1, ..., captureK-1, freshK].
  ## A transformed capture (-16/-32/-48 etc.) occupies a capture slot.
  let count = wildcardCount(pat)
  if rhs.len != count * 2 + 1: return false
  for i, value in rhs:
    if i mod 2 == 0:
      if value < RuleBase: return false
      # Distinct literal markers for A, B, C, ... within the same RHS.
      var previous = 0
      while previous < i:
        if rhs[previous] == value: return false
        previous += 2
    else:
      let capture = captureIndex(value)
      if capture < 0 or capture >= count: return false
  result = true

proc appendFreshMarker(rhs: var seq[int], codec: TokenCodec,
                       pat: seq[int], frequency, gapIndex: int) =
  ## Reuse the same deterministic hash scheme with a different salt per gap.
  ## Resolve extremely rare within-rule collisions deterministically.
  var salt = gapIndex
  while true:
    let token = freshRuleToken(codec, pat, frequency, salt)
    var exists = false
    for old in rhs:
      if old == token:
        exists = true
        break
    if not exists:
      rhs.add(token)
      return
    inc salt

proc alternatingRhs(pat: seq[int], frequency: int, codec: TokenCodec,
                    captureOps: openArray[int]): seq[int] =
  ## New hashed token at the beginning, BETWEEN captures and at the end.
  appendFreshMarker(result, codec, pat, frequency, 0)
  for i, opcode in captureOps:
    result.add(opcode)
    appendFreshMarker(result, codec, pat, frequency, i + 1)
  doAssert isAlternatingReplacement(pat, result)

proc loadRules(path: string, codec: TokenCodec): seq[Rule] =
  if not fileExists(path): quit("Resume file missing: " & path, 1)
  var upgraded = 0
  for line in lines(path):
    if line.len == 0 or line[0] == '#': continue
    let fields = line.split('\t')
    if fields.len != 2: continue
    let pat = parsePattern(fields[0])
    var rhs = parsePattern(fields[1])
    if not isAlternatingReplacement(pat, rhs):
      let count = wildcardCount(pat)
      # Upgrade old [A,-1,-2] to [A,-1,B,-2,C], retaining A.
      if rhs.len != count + 1 or rhs.len == 0 or rhs[0] < RuleBase:
        raise newException(ValueError,
          "Unsupported resume RHS; expected alternating fresh/capture tokens: " & line)
      let oldRhs = rhs
      rhs = @[oldRhs[0]]
      # The original marker carries a log2-frequency bucket. Recover a
      # representative frequency so newly inserted markers use that bucket.
      let bucket = min(20, max(0, (oldRhs[0] - RuleBase) div 16_000_000))
      let representativeFrequency = 1 shl bucket
      for i in 0..<count:
        rhs.add(oldRhs[i + 1])
        appendFreshMarker(rhs, codec, pat, representativeFrequency, i + 1)
      if not isAlternatingReplacement(pat, rhs):
        raise newException(ValueError, "Invalid legacy rule: " & line)
      inc upgraded
    result.add(Rule(pat: pat, rhs: rhs))
  if upgraded > 0:
    stderr.writeLine("[resume] migrated ", upgraded,
      " legacy RHS rules to alternating fresh/capture form; " &
      "this changes their previous rewrite behavior")

proc generateReplacementVariants(pat: seq[int], frequency: int,
                                 codec: TokenCodec): seq[Rule] =
  ## Every replacement is strictly A -1 B -2 C ... (fresh tokens at all
  ## boundaries). Single wildcard variants preserve both surrounding tokens.
  ## The first marker uses the historical salt=0 for stable identity.
  let k = wildcardCount(pat)
  var plain = newSeq[int](k)
  for i in 0..<k: plain[i] = -i - 1
  result.add(Rule(pat: pat,
    rhs: alternatingRhs(pat, frequency, codec, plain)))
  if k == 1:
    for code in [-16, -32, -48]:
      result.add(Rule(pat: pat,
        rhs: alternatingRhs(pat, frequency, codec, @[code])))
  elif k >= 2:
    for a in 0..<k:
      for b in a+1..<k:
        # Clone the capture slots before swapping, not the literal markers.
        var ops = newSeq[int](k)
        for i in 0..<k: ops[i] = plain[i]
        swap(ops[a], ops[b])
        result.add(Rule(pat: pat,
          rhs: alternatingRhs(pat, frequency, codec, ops)))

proc productLimit(n, gaps: int): float64 =
  if gaps < 2: return Inf
  var fact = 1.0
  for i in 2..gaps-1: fact *= float64(i)
  let logn = ln(float64(max(n, 2)))
  fact * float64(n) / pow(logn, float64(gaps - 1))

proc exchangeHeap(m: var MineState, a, b: int) {.inline.} =
  ## Keep Table indices correct when the priority queue exchanges positions.
  swap(m.tracked[a], m.tracked[b])
  m.positions[m.tracked[a].pat] = a
  m.positions[m.tracked[b].pat] = b

proc siftUp(m: var MineState, start: int) =
  var i = start
  while i > 0:
    let parent = (i - 1) div 2
    if m.tracked[parent].estimate <= m.tracked[i].estimate: break
    m.exchangeHeap(parent, i)
    i = parent

proc siftDown(m: var MineState, start: int) =
  var i = start
  while 2 * i + 1 < m.tracked.len:
    var child = 2 * i + 1
    if child + 1 < m.tracked.len and
       m.tracked[child + 1].estimate < m.tracked[child].estimate:
      inc child
    if m.tracked[i].estimate <= m.tracked[child].estimate: break
    m.exchangeHeap(i, child)
    i = child

proc refreshedMinimum(m: var MineState): uint32 =
  ## CMS collisions can increase an item's estimate without that pattern
  ## occurring again. Refresh the minimum lazily before an eviction.
  while m.tracked.len > 0:
    let observed = m.sketch.estimate(m.tracked[0].pat)
    if observed <= m.tracked[0].estimate: break
    m.tracked[0].estimate = observed
    m.siftDown(0)
  result = m.tracked[0].estimate

proc offer(m: var MineState, pat: seq[int]) =
  if m.offers >= m.limit: return
  inc m.offers
  # Update AND read the CMS estimate in one hashing/counter pass.
  let estimate = m.sketch.addAndEstimate(pat)
  # Once full, a key that cannot beat the cached minimum cannot alter
  # the heap. Avoid hashing every low-frequency key for Table lookup.
  # An already tracked key with estimate <= minimum is unchanged, too.
  if m.tracked.len >= m.cfg.mineTracked and
      estimate <= m.tracked[0].estimate:
    return
  let idx = m.positions.getOrDefault(pat, -1)
  if idx >= 0:
    m.tracked[idx].estimate = estimate
    m.siftDown(idx)
    return
  if m.tracked.len < m.cfg.mineTracked:
    # The recursive miner reuses and mutates pat; freeze the stored key.
    var snapshot = newSeq[int](pat.len)
    for i, v in pat: snapshot[i] = v
    m.tracked.add(TrackedPattern(pat: snapshot, estimate: estimate))
    m.positions[snapshot] = m.tracked.len - 1
    m.siftUp(m.tracked.len - 1)
    return
  if m.tracked.len == 0 or estimate <= m.tracked[0].estimate: return
  # An untracked pattern may be discovered late. Admit it as soon as its
  # whole-stream CMS count beats the least frequent retained pattern.
  if estimate <= m.refreshedMinimum(): return
  m.positions.del(m.tracked[0].pat)
  var snapshot = newSeq[int](pat.len)
  for i, v in pat: snapshot[i] = v
  m.tracked[0] = TrackedPattern(pat: snapshot, estimate: estimate)
  m.positions[snapshot] = 0
  m.siftDown(0)

proc mineSuffix(m: var MineState, lens: seq[int], part, previousEnd: int,
                product: float64, allShort: bool, pat: var seq[int]) =
  if m.offers >= m.limit: return
  if part == lens.len:
    m.offer(pat)
    return
  let required = lens[part]
  var remaining = 0
  for j in part+1..<lens.len: remaining += lens[j] + 1
  let allowed = min(m.cfg.maxGap, m.text.len - previousEnd - required - remaining)
  if allowed < 1: return
  let saved = pat.len
  pat.add(-1)
  for gap in 1..allowed:
    if m.offers >= m.limit: break
    let prod2 = product * float64(gap)
    let short2 = allShort and gap <= m.gapRootLimit
    if lens.len >= 3 and prod2 > m.gapProductLimit and not short2:
      continue
    let start = previousEnd + gap
    for j in 0..<required: pat.add(m.text[start+j])
    mineSuffix(m, lens, part+1, start+required, prod2, short2, pat)
    pat.setLen(saved+1)
  pat.setLen(saved)

proc mineShape(m: var MineState, lens: seq[int]) =
  if lens.len == 0 or m.text.len == 0: return
  var needed = 0
  for x in lens: needed += x
  needed += lens.len - 1
  if needed > m.text.len: return
  m.gapProductLimit = productLimit(m.text.len, lens.len-1)
  m.gapRootLimit = int(floor(pow(float64(m.text.len),
                                1.0 / float64(max(1, lens.len-1)))))
  let maxStart = m.text.len - needed
  let stride = max(1, (maxStart+1) div m.cfg.mineStarts)
  var start = 0
  var countStarts = 0
  while start <= maxStart and countStarts < m.cfg.mineStarts and m.offers < m.limit:
    var pattern: seq[int] = @[]
    for i in 0..<lens[0]: pattern.add(m.text[start+i])
    mineSuffix(m, lens, 1, start + lens[0], 1.0, true, pattern)
    start += stride
    inc countStarts

proc shapeList(cfg: Config, cur: var seq[int], total: int,
               dest: var seq[seq[int]]) =
  if cur.len > 0:
    var snapshot = newSeq[int](cur.len)
    for i, v in cur: snapshot[i] = v
    dest.add(snapshot)
  if cur.len >= cfg.maxWildcards + 1: return
  for gram in cfg.minNgram..cfg.maxGram:
    if total+gram > cfg.maxTotalGram: break
    cur.add(gram)
    shapeList(cfg, cur, total+gram, dest)
    cur.setLen(cur.len - 1)

proc mine(text: seq[int], cfg: Config,
          shapes: seq[seq[int]]): seq[PatternScore] =
  if shapes.len == 0: return
  var m = MineState(text: text, cfg: cfg,
    sketch: initCountMinSketch(cfg.cmsWidth, cfg.cmsDepth),
    positions: initTable[seq[int], int]())
  let perShape = max(1, cfg.mineBudget div shapes.len)
  for s in shapes:
    m.limit = m.offers + perShape
    mineShape(m, s)
  # Refresh estimates before final ranking; candidates are subsequently
  # counted EXACTLY over all diffusion rows by bestPatterns.
  for item in m.tracked:
    result.add(PatternScore(pat: item.pat, gain: 0.0,
      frequency: int(m.sketch.estimate(item.pat))))
  result.sort(proc(a,b: PatternScore): int = cmp(b.frequency, a.frequency))
  if result.len > cfg.candidates:
    result.setLen(cfg.candidates)

# 512-bit per-row token-presence filter. Unlike the approximate candidate
# screen, this ONLY skips rows which definitely cannot contain the pattern.
# A false-positive bloom hit costs a normal matcher call; false negatives
# are impossible for any pattern with at least one literal token.
type TokenFingerprint = array[8, uint64]

proc tokenFingerprintBit(token: int): int {.inline.} =
  # Nonnegative tokens only; deterministic multiplicative mixing.
  int((uint64(token) * 0x9E3779B97F4A7C15'u64) shr 55)

proc fingerprintText(text: openArray[int]): TokenFingerprint =
  for value in text:
    if value < 0: continue
    let bit = tokenFingerprintBit(value)
    result[bit shr 6] = result[bit shr 6] or (1'u64 shl (bit and 63))

proc fingerprintRows(texts: seq[seq[int]]): seq[TokenFingerprint] =
  result = newSeq[TokenFingerprint](texts.len)
  for i in 0..<texts.len:
    result[i] = fingerprintText(texts[i])

proc mayContainPattern(fp: TokenFingerprint, pat: openArray[int]): bool {.inline.} =
  for value in pat:
    if value < 0: continue # wildcards are not literal requirements
    let bit = tokenFingerprintBit(value)
    if (fp[bit shr 6] and (1'u64 shl (bit and 63))) == 0'u64:
      return false
  true

proc countFeatures(texts: seq[seq[int]], pat: seq[int],
                   fingerprints: seq[TokenFingerprint] = @[]): seq[float64] =
  let compiled = compileSeqPattern(pat, 1)
  result = newSeq[float64](texts.len)
  for i in 0..<texts.len:
    if fingerprints.len == texts.len and
        not mayContainPattern(fingerprints[i], pat):
      continue
    result[i] = float64(countSeqCompiled(texts[i], compiled))

proc miningSampleIndices(n: int, limit = 8): seq[int] =
  ## Sample the ENTIRE rotated diffusion window, not just the untouched
  ## first snapshot of its first chunk. Spread samples deterministically.
  if n <= 0 or limit <= 0: return
  let k = min(n, limit)
  for j in 0..<k:
    let index = (if k == 1: 0 else: j * (n - 1) div (k - 1))
    if result.len == 0 or index != result[^1]: result.add(index)

proc mineWindow(texts: seq[seq[int]], cfg: Config): seq[PatternScore] =
  ## Share the existing search budget among representative noise stages
  ## across ALL the training chunks. Full count vectors are still measured
  ## across every diffusion row in bestPatterns below.
  let indices = miningSampleIndices(texts.len)
  if indices.len == 0: return
  var local = cfg
  local.mineBudget = max(1, cfg.mineBudget div indices.len)
  local.mineTracked = max(1, cfg.mineTracked div indices.len)
  local.candidates = max(1, (cfg.candidates + indices.len - 1) div indices.len)
  var seen = initTable[seq[int], bool]()
  # Every sampled row shares exactly the same set of pattern shapes.
  # Generate it once per window instead of rebuilding the recursive tree.
  var shapes: seq[seq[int]] = @[]
  var cur: seq[int] = @[]
  shapeList(cfg, cur, 0, shapes)
  for rowIndex in indices:
    for candidate in mine(texts[rowIndex], local, shapes):
      if not seen.hasKeyOrPut(candidate.pat, true):
        result.add(candidate)

# Two-stage candidate screening. A sparse sample estimates how strongly
# candidate firing counts correlate with the current residual. This is a
# cheap HEURISTIC, not the exact Ridge marginal gain: retaining some CMS
# frequency leaders reduces the risk of discarding useful rare/indirect keys.
# Every survivor is still evaluated with exact full-window counts and Ridge.
proc screenPatterns(texts: seq[seq[int]], fit: Fit, cfg: Config,
                    candidates: seq[PatternScore],
                    fingerprints: seq[TokenFingerprint] = @[]): seq[PatternScore] =
  if cfg.screenRows <= 0 or cfg.screenCandidates <= 0 or
      candidates.len <= cfg.screenCandidates or
      fit.residual.len != texts.len:
    return candidates
  let indices = miningSampleIndices(texts.len, cfg.screenRows)
  if indices.len < 3: return candidates
  let n = float64(indices.len)
  var sumResidual = 0.0
  for i in indices:
    sumResidual += fit.residual[i]

  type ScreenScore = tuple[index: int, proxy: float64]
  var ranked: seq[ScreenScore] = @[]
  for j, candidate in candidates:
    let pattern = compileSeqPattern(candidate.pat, 1)
    var sumZ, sumZZ, sumZR = 0.0
    for i in indices:
      let z = (if fingerprints.len == texts.len and
                  not mayContainPattern(fingerprints[i], candidate.pat):
                 0.0
               else: float64(countSeqCompiled(texts[i], pattern)))
      sumZ += z
      sumZZ += z*z
      sumZR += z*fit.residual[i]
    let centeredZ = max(0.0, sumZZ - sumZ*sumZ/n)
    let centeredCross = sumZR - sumZ*sumResidual/n
    # Larger |partial-residual covariance| / sigma(z) is preferred.
    let proxy = (if centeredZ > 1e-12:
                   abs(centeredCross) / sqrt(centeredZ)
                 else: 0.0)
    ranked.add((j, proxy))
  ranked.sort(proc(a, b: ScreenScore): int =
    let c = cmp(b.proxy, a.proxy)
    if c != 0: c else: cmp(candidates[b.index].frequency,
                          candidates[a.index].frequency))

  let target = min(cfg.screenCandidates, candidates.len)
  let reserveFreq = max(1, target div 8)
  var selected = newSeq[bool](candidates.len)
  var taken = 0
  for i in 0..<min(target-reserveFreq, ranked.len):
    selected[ranked[i].index] = true
    inc taken

  # Protect part of the shortlist by approximate CMS frequency, regardless
  # of the residual proxy. The final feature counts never use CMS values.
  var byFreq = ranked
  byFreq.sort(proc(a, b: ScreenScore): int =
    let c = cmp(candidates[b.index].frequency,
                candidates[a.index].frequency)
    if c != 0: c else: cmp(b.proxy, a.proxy))
  for entry in byFreq:
    if taken >= target: break
    if not selected[entry.index]:
      selected[entry.index] = true
      inc taken
  for j in 0..<candidates.len:
    if selected[j]: result.add(candidates[j])

# Reuse mining, row fingerprints and EXACT count vectors whenever different
# beam parents request candidate scoring at the same prefix of the SAME
# rotated training window. The cache is thrown away between steps.
# Store only 192 full count vectors per prefix to bound memory use.
const MaxCachedFeatureVectorsPerPrefix = 192
# Limits apply to stored token *elements*, not the borrowed row.text references.
# At most about 48MB of materialized prefix text (plus small metadata).
const MaxCachedPrefixStateTokens = 6_000_000
const MaxCachedPrefixStates = 2

type
  PrefixState = ref object
    encoded: Encoded
    tokenElements: int
  StageEntry = ref object
    mined: seq[PatternScore]
    fingerprints: seq[TokenFingerprint]
    vectors: Table[seq[int], seq[float64]]
  StageCache = ref object
    stages: Table[string, StageEntry]
    reusedMining, reusedCounts: int
    prefixStates: Table[string, PrefixState]
    prefixOrder: seq[string]
    prefixTokenElements: int

proc stageSignature(rules: seq[Rule], stop: int): string =
  ## Prefix rules completely determine their rewritten diffusion rows,
  ## within one step. The integers are serialized without hash collisions.
  for j in 0..<min(stop, rules.len):
    result.add($rules[j].pat.len & ":" & patternString(rules[j].pat) & "/")
    result.add($rules[j].rhs.len & ":" & patternString(rules[j].rhs) & ";")

proc bestPatterns(texts: seq[seq[int]], fit: Fit, cfg: Config,
                  exclude: seq[Rule], count: int,
                  insertionMode = false,
                  cache: StageCache = nil, stageKey: string = ""): seq[PatternScore] =
  if texts.len == 0: return
  when defined(lensProfile):
    let startTime = epochTime()
  var entry: StageEntry = nil
  var mined: seq[PatternScore]
  if cache != nil:
    if cache.stages.hasKey(stageKey):
      entry = cache.stages[stageKey]
      inc cache.reusedMining
    else:
      entry = StageEntry(
        mined: mineWindow(texts, cfg),
        fingerprints: (if cfg.useFingerprint: fingerprintRows(texts)
                       else: newSeq[TokenFingerprint](0)),
        vectors: initTable[seq[int], seq[float64]]())
      cache.stages[stageKey] = entry
    mined = entry.mined
  else:
    mined = mineWindow(texts, cfg)
  when defined(lensProfile):
    let mineTime = epochTime()
  # Do not spend approximate-screen slots on existing-rule patterns.
  var eligible: seq[PatternScore] = @[]
  for candidate in mined:
    var banned = false
    for r in exclude:
      if candidate.pat == r.pat:
        banned = true
        break
    if not banned: eligible.add(candidate)
  let fingerprints = (if entry != nil: entry.fingerprints
                      elif cfg.useFingerprint: fingerprintRows(texts)
                      else: newSeq[TokenFingerprint](0))
  let candidates = screenPatterns(texts, fit, cfg, eligible, fingerprints)
  when defined(lensProfile):
    let screenTime = epochTime()
  var scored: seq[PatternScore] = @[]
  # Appended-rule Pearson is exact among surviving candidates;
  # mid-chain scores remain heuristic.
  # At mid-chain insertion the candidate also changes downstream count
  # columns, so the candidate ranking is a HEURISTIC, not a rejection test.
  for originalCandidate in candidates:
    var candidate = originalCandidate
    var counts: seq[float64]
    if entry != nil and entry.vectors.hasKey(candidate.pat):
      counts = entry.vectors[candidate.pat]
      inc cache.reusedCounts
    else:
      counts = countFeatures(texts, candidate.pat, fingerprints)
      if entry != nil and entry.vectors.len < MaxCachedFeatureVectorsPerPrefix:
        entry.vectors[candidate.pat] = counts
    var actualOccurrences = 0
    for value in counts: actualOccurrences += int(value)
    if actualOccurrences <= 0: continue
    candidate.frequency = actualOccurrences
    candidate.gain = candidateGain(counts, fit)
    if insertionMode or candidate.gain > cfg.minGain:
      scored.add(candidate)
  when defined(lensProfile):
    var totalTokens = 0
    var maxTokens = 0
    for t in texts:
      totalTokens += t.len
      maxTokens = max(maxTokens, t.len)
    stderr.writeLine("[profile] mine=", mineTime-startTime,
      "s screen=", screenTime-mineTime,
      "s exact=", epochTime()-screenTime,
      "s mined=", mined.len, " eligible=", eligible.len,
      " exact_candidates=", candidates.len,
      " tokens=", totalTokens, " max_row=", maxTokens,
      " prefix_cache=", (if entry != nil: "on" else: "off"),
      " reused_mine=", (if cache != nil: cache.reusedMining else: 0),
      " reused_vectors=", (if cache != nil: cache.reusedCounts else: 0))
  scored.sort(proc(a,b: PatternScore): int =
    let order = cmp(b.gain, a.gain)
    if order != 0: order else: cmp(b.frequency, a.frequency))
  if not insertionMode:
    for i in 0..<min(count, scored.len): result.add(scored[i])
    return
  # For replacement, diversify the limited full-pipeline evaluation budget:
  # include promising FWL candidates plus frequently firing candidates
  # even when their immediate gain is zero. Never claim exhaustive search.
  let gainSlots = max(1, count div 2)
  for i in 0..<min(gainSlots, scored.len): result.add(scored[i])
  var remaining: seq[PatternScore] = @[]
  for i in result.len..<scored.len: remaining.add(scored[i])
  remaining.sort(proc(a,b: PatternScore): int =
    let order = cmp(b.frequency, a.frequency)
    if order != 0: order else: cmp(b.gain, a.gain))
  for i in 0..<min(count - result.len, remaining.len):
    result.add(remaining[i])

proc prefixTexts(rows: seq[Row], rules: seq[Rule], stop: int): seq[seq[int]] =
  ## Each row starts as (alphabet encoded -> diffused) and each rule sees
  ## the previous rule's rewritten output, NEVER the original row again.
  let count = max(0, min(stop, rules.len))
  var programs = newSeq[SeqReplaceProgram](count)
  for j in 0..<count:
    programs[j] = compileSeqReplaceProgram(rules[j].pat, rules[j].rhs, 1)
  result = newSeq[seq[int]](rows.len)
  for i, row in rows:
    var state = row.text
    for j in 0..<count:
      discard applyAndCount(state, programs[j])
    result[i] = state

proc evaluate(rows: seq[Row], rules: seq[Rule],
              ridgeLambda: float64, removal = false): tuple[enc: Encoded, fit: Fit] =
  when defined(lensProfile):
    inc profileEval.evaluateCalls
    let start = epochTime()
  result.enc = encode(rows, rules)
  when defined(lensProfile):
    let encodedAt = epochTime()
    profileEval.evaluateEncodeSeconds += encodedAt - start
  result.fit = ridge(result.enc.cols, result.enc.y, ridgeLambda, removal)
  when defined(lensProfile):
    profileEval.evaluateRidgeSeconds += epochTime() - encodedAt

proc alteredIndex(reference: Encoded, current: Encoded,
                  excluded: seq[bool], fit: Fit): int =
  var maxChange = -1.0
  for j in 0..<min(reference.cols.len, current.cols.len):
    if excluded[j]: continue
    var difference = 0.0
    for i in 0..<reference.y.len:
      difference += abs(reference.cols[j][i] - current.cols[j][i])
    if difference > maxChange:
      maxChange = difference
      result = j
  if maxChange > 0.0: return
  # No changed downstream counts: take the next least costly deletion.
  var loss = Inf
  result = -1
  for j in 0..<excluded.len:
    if not excluded[j] and fit.removalLoss[j] < loss:
      loss = fit.removalLoss[j]
      result = j

proc bestInsertion(rows: seq[Row], existing: seq[Rule], at: int,
                   cfg: Config, codec: TokenCodec, keep: int,
                   cache: StageCache = nil): seq[Branch] =
  ## Mine exactly at the insertion point, AFTER earlier rules and BEFORE
  ## later ones. FWL is a prefilter; full rewritten OLS judges acceptance.
  let baseline = evaluate(rows, existing, cfg.ridgeLambda, true)
  let stageTexts = prefixTexts(rows, existing, at)
  let winners = bestPatterns(stageTexts, baseline.fit, cfg, existing,
                             max(1, keep), true, cache,
                             stageSignature(existing, at))
  for candidate in winners:
    # A replacement at the END of the sequence cannot modify downstream
    # feature columns. The block Ridge/OLS append formula gives its exact
    # Pearson, independent of the replacement RHS (plain/sort/reverse/etc.).
    var tailPearson = 0.0
    if at == existing.len:
      if baseline.fit.sst <= 1e-24:
        tailPearson = baseline.fit.pearson
      elif baseline.fit.ridgeLambda > 0.0:
        tailPearson = max(-1.0, min(1.0,
          baseline.fit.pearson + candidate.gain))
      else:
        tailPearson = sqrt(max(0.0, min(1.0,
          1.0 - max(0.0, baseline.fit.sse-candidate.gain)/baseline.fit.sst)))
    for variant in generateReplacementVariants(candidate.pat, candidate.frequency, codec):
      var rules: seq[Rule] = @[]
      for r in existing: rules.add(r)
      rules.insert(variant, at)
      result.add(Branch(rules: rules, score: tailPearson,
        description: "replace " & $at & " with " & patternString(candidate.pat) &
          " => " & patternString(variant.rhs), editedAt: at))

proc sharedPrefixLength(branches: seq[Branch]): int =
  ## Exact rule identity includes BOTH LHS and RHS, since a prefix output
  ## depends on all earlier replacement results.
  if branches.len == 0: return 0
  result = branches[0].rules.len
  for b in branches:
    result = min(result, b.rules.len)
  for j in 0..<result:
    let rule = branches[0].rules[j]
    for k in 1..<branches.len:
      if branches[k].rules[j].pat != rule.pat or
          branches[k].rules[j].rhs != rule.rhs:
        return j

proc encodeSuffixForRanking(prefix: Encoded, rules: seq[Rule],
                            prefixLength: int): Encoded =
  ## All rule-count columns preceding the first difference are borrowed,
  ## read-only. Recompute only suffix columns on the same pre-diffused rows.
  ## The final RHS never affects any count column, so COUNT the last rule
  ## without materializing its rewritten output. This is exact, not a proxy.
  let n = prefix.y.len
  result.y = prefix.y
  result.cols = newSeq[seq[float64]](rules.len)
  for j in 0..<prefixLength:
    result.cols[j] = prefix.cols[j]
  if prefixLength >= rules.len: return
  var programs = newSeq[SeqReplaceProgram](max(0, rules.len-prefixLength-1))
  for offset in 0..<programs.len:
    let r = rules[prefixLength + offset]
    programs[offset] = compileSeqReplaceProgram(r.pat, r.rhs, 1)
    result.cols[prefixLength + offset] = newSeq[float64](n)
  let last = rules.len - 1
  let lastPattern = compileSeqPattern(rules[last].pat, 1)
  # If the last rewrite *could* exceed the replacement engine's output cap,
  # execute it normally to preserve the same overflow errors as full encode.
  # Otherwise its output is dead and only its exact count is required.
  let safeInputLength = MAX_REPLACE_OUTPUT div max(2, rules[last].rhs.len + 1)
  var lastProgram = compileSeqReplaceProgram(
    rules[last].pat, rules[last].rhs, 1)
  result.cols[last] = newSeq[float64](n)
  for i in 0..<n:
    # applyAndCount never mutates the tokens of its input buffer: it builds
    # a fresh output and reassigns state. The cached prefix is immutable.
    var state = prefix.finalTexts[i]
    for offset in 0..<programs.len:
      result.cols[prefixLength + offset][i] =
        float64(applyAndCount(state, programs[offset]))
    if state.len <= safeInputLength:
      result.cols[last][i] = float64(countSeqCompiled(state, lastPattern))
    else:
      result.cols[last][i] = float64(applyAndCount(state, lastProgram))

proc cachedRankingPrefix(rows: seq[Row], branches: seq[Branch],
                         prefixLength: int, cache: StageCache): Encoded =
  let key = stageSignature(branches[0].rules, prefixLength)
  if cache != nil and cache.prefixStates.hasKey(key):
    when defined(lensProfile):
      inc profileEval.rankPrefixHits
    return cache.prefixStates[key].encoded
  when defined(lensProfile):
    let start = epochTime()
    inc profileEval.rankPrefixBuilds
  var common: seq[Rule] = @[]
  for j in 0..<prefixLength:
    common.add(branches[0].rules[j])
  result = encode(rows, common)
  when defined(lensProfile):
    profileEval.rankPrefixBuildSeconds += epochTime()-start
  # Prefix 0 is just references to input rows. Larger prefixes may own
  # millions of rewritten tokens, so use a strict step-local LRU budget.
  if cache != nil:
    var storedTokens = 0
    if prefixLength > 0:
      for row in result.finalTexts:
        storedTokens += row.len
    if storedTokens <= MaxCachedPrefixStateTokens:
      while cache.prefixOrder.len >= MaxCachedPrefixStates or
          (cache.prefixOrder.len > 0 and
           cache.prefixTokenElements + storedTokens > MaxCachedPrefixStateTokens):
        let oldKey = cache.prefixOrder[0]
        cache.prefixOrder.delete(0)
        cache.prefixTokenElements -= cache.prefixStates[oldKey].tokenElements
        cache.prefixStates.del(oldKey)
        when defined(lensProfile):
          inc profileEval.rankPrefixEvictions
      cache.prefixStates[key] = PrefixState(
        encoded: result, tokenElements: storedTokens)
      cache.prefixOrder.add(key)
      cache.prefixTokenElements += storedTokens

# In a group of exchange siblings only the rule at editedAt changes.
# If the new rule does not match one particular pre-rule row, its input to
# every later rule is IDENTICAL to the input with that rule omitted. Cache
# that omitted-rule suffix once per row, and reuse all downstream counts.
# Unlike a prefix-only cache, this avoids the costly suffix traversal itself.

type DeltaSuffixWork = object
  middlePrograms: seq[SeqReplaceProgram] # suffix except last, shared by siblings
  lastPattern: SeqPattern
  lastProgram: SeqReplaceProgram
  lastSafeInputLength: int
  baselineReady: seq[bool]
  baselineCols: seq[seq[float64]] # tail-local columns; allocated on demand

proc identicalExchangeSuffix(branches: seq[Branch], at: int): bool =
  if branches.len <= 1 or at < 0: return false
  let reference = branches[0].rules
  if at >= reference.len - 1: return false # no downstream work to avoid
  for b in branches:
    if b.rules.len != reference.len or b.editedAt != at:
      return false
    for j in at+1..<reference.len:
      if b.rules[j].pat != reference[j].pat or
          b.rules[j].rhs != reference[j].rhs:
        return false
  result = true

proc initDeltaSuffixWork(rules: seq[Rule], at, n: int): DeltaSuffixWork =
  let last = rules.len - 1
  for j in at+1..<last:
    result.middlePrograms.add(compileSeqReplaceProgram(
      rules[j].pat, rules[j].rhs, 1))
  result.lastPattern = compileSeqPattern(rules[last].pat, 1)
  result.lastProgram = compileSeqReplaceProgram(
    rules[last].pat, rules[last].rhs, 1)
  result.lastSafeInputLength = MAX_REPLACE_OUTPUT div
    max(2, rules[last].rhs.len + 1)
  result.baselineReady = newSeq[bool](n)
  result.baselineCols = newSeq[seq[float64]](rules.len - at - 1)
  for j in 0..<result.baselineCols.len:
    result.baselineCols[j] = newSeq[float64](n)

proc runDeltaTail(state: var seq[int], cols: var seq[seq[float64]],
                  colStart, row: int, work: var DeltaSuffixWork) =
  for offset in 0..<work.middlePrograms.len:
    cols[colStart + offset][row] =
      float64(applyAndCount(state, work.middlePrograms[offset]))
  let last = colStart + work.middlePrograms.len
  if state.len <= work.lastSafeInputLength:
    cols[last][row] = float64(countSeqCompiled(state, work.lastPattern))
  else:
    cols[last][row] = float64(applyAndCount(state, work.lastProgram))

proc rankDeltaExchange(prefix: Encoded, branches: seq[Branch],
                       at: int, ridgeLambda: float64): seq[Branch] =
  ## Count each distinct candidate LHS once, even when RHS differs through
  ## plain/sort/reverse/inverse_sort. A cache miss in downstream counts is
  ## computed lazily from the pre-rule text, with the candidate OMITTED.
  ## No sampled/approximate features enter the Ridge regression.
  let n = prefix.y.len
  let totalRules = branches[0].rules.len
  var work = initDeltaSuffixWork(branches[0].rules, at, n)
  var occurrenceCache = initTable[seq[int], seq[int]]()
  for b in branches:
    var counts: seq[int]
    if occurrenceCache.hasKey(b.rules[at].pat):
      counts = occurrenceCache[b.rules[at].pat]
      when defined(lensProfile):
        inc profileEval.deltaPatternCacheHits
    else:
      let pattern = compileSeqPattern(b.rules[at].pat, 1)
      counts = newSeq[int](n)
      for i in 0..<n:
        counts[i] = countSeqCompiled(prefix.finalTexts[i], pattern)
      occurrenceCache[b.rules[at].pat] = counts
      when defined(lensProfile):
        profileEval.deltaRowsScanned += n
    var encoded: Encoded
    encoded.y = prefix.y
    encoded.cols = newSeq[seq[float64]](totalRules)
    for j in 0..<at:
      encoded.cols[j] = prefix.cols[j]
    for j in at..<totalRules:
      encoded.cols[j] = newSeq[float64](n)
    var program = compileSeqReplaceProgram(b.rules[at].pat,
                                           b.rules[at].rhs, 1)
    for i in 0..<n:
      let hits = counts[i]
      encoded.cols[at][i] = float64(hits)
      if hits == 0:
        if not work.baselineReady[i]:
          var state = prefix.finalTexts[i]
          runDeltaTail(state, work.baselineCols, 0, i, work)
          work.baselineReady[i] = true
          when defined(lensProfile):
            inc profileEval.deltaBaselineBuilds
        for j in 0..<work.baselineCols.len:
          encoded.cols[at + 1 + j][i] = work.baselineCols[j][i]
        when defined(lensProfile):
          inc profileEval.deltaReusedRows
      else:
        var state = prefix.finalTexts[i]
        let observed = applyAndCount(state, program)
        when defined(lensDebug):
          doAssert observed == hits # independent count/rewrite equivalence
        discard observed
        runDeltaTail(state, encoded.cols, at + 1, i, work)
        when defined(lensProfile):
          inc profileEval.deltaMatchedRows
    var scored = b
    when defined(lensProfile):
      let startRidge = epochTime()
    scored.score = ridge(encoded.cols, encoded.y, ridgeLambda, false).pearson
    when defined(lensProfile):
      let ridgeElapsed = epochTime() - startRidge
      profileEval.deltaRidgeSeconds += ridgeElapsed
      profileEval.rankRidgeSeconds += ridgeElapsed
    result.add(scored)

proc rankBranches(rows: seq[Row], branches: seq[Branch], keep: int,
                  ridgeLambda: float64, stateCache: StageCache = nil,
                  enableStateCache = false,
                  enableDeltaSuffix = true): seq[Branch] =
  ## The pre-swap prefix is the same for all sibling candidates. Cache its
  ## entire rewritten row state (and earlier exact count columns), then
  ## apply ONLY the changed suffix. No score approximation is introduced.
  when defined(lensProfile):
    let rankStart = epochTime()
    inc profileEval.rankCalls
    profileEval.rankVariants += branches.len
  var ranking: seq[Branch] = @[]
  if enableStateCache and branches.len > 1:
    when defined(lensProfile):
      inc profileEval.rankCacheCalls
    let prefixLength = sharedPrefixLength(branches)
    let prefix = cachedRankingPrefix(rows, branches, prefixLength, stateCache)
    if enableDeltaSuffix and identicalExchangeSuffix(branches, prefixLength):
      when defined(lensProfile):
        let deltaStart = epochTime()
        let oldDeltaRidge = profileEval.deltaRidgeSeconds
        inc profileEval.deltaCalls
      ranking = rankDeltaExchange(prefix, branches, prefixLength, ridgeLambda)
      when defined(lensProfile):
        let deltaElapsed = epochTime() - deltaStart
        profileEval.deltaSeconds += deltaElapsed
        # Track suffix time separately from Ridge for apples-to-apples logs.
        profileEval.rankSuffixSeconds += max(0.0,
          deltaElapsed - profileEval.deltaRidgeSeconds + oldDeltaRidge)
    else:
      when defined(lensProfile):
        if enableDeltaSuffix: inc profileEval.deltaFallbacks
      for b in branches:
        var scored = b
        when defined(lensProfile):
          let start = epochTime()
        let enc = encodeSuffixForRanking(prefix, b.rules, prefixLength)
        when defined(lensProfile):
          let afterEncoding = epochTime()
          profileEval.rankSuffixSeconds += afterEncoding-start
          if b.rules.len > prefixLength:
            inc profileEval.rankLastCountOnly
        scored.score = ridge(enc.cols, enc.y, ridgeLambda, false).pearson
        when defined(lensProfile):
          profileEval.rankRidgeSeconds += epochTime()-afterEncoding
        ranking.add(scored)
  else:
    for b in branches:
      var scored = b
      scored.score = evaluate(rows, b.rules, ridgeLambda).fit.pearson
      ranking.add(scored)
  ranking.sort(proc(a, b: Branch): int = cmp(b.score, a.score))
  for i in 0..<min(keep, ranking.len):
    result.add(ranking[i])
  when defined(lensProfile):
    profileEval.rankTotalSeconds += epochTime()-rankStart

proc operatorName(branch: Branch): string =
  if branch.editedAt < 0 or branch.editedAt >= branch.rules.len:
    return "none"
  let rhs = branch.rules[branch.editedAt].rhs
  if rhs.len == 3:
    case rhs[1]
    of -1: return "plain"
    of -16: return "sort"
    of -32: return "reverse"
    of -48: return "inverse_sort"
    else: return "capture"
  return "multi_or_literal"

proc bestCurrentBranches(branches: seq[Branch], keep: int): seq[Branch] =
  ## Children of attemptAddition/attemptExchange already have exact scores.
  ## Avoid evaluating their whole replacement pipelines a second time.
  var ordered = branches
  ordered.sort(proc(a, b: Branch): int = cmp(b.score, a.score))
  for i in 0..<min(keep, ordered.len):
    result.add(ordered[i])

proc chooseMove(basePearson, addPearson, replacePearson,
                minGain: float64): MoveKind =
  ## Compare BOTH operations on the same current training rows. Favor adding
  ## in a tie; replacing must strictly beat the best available addition.
  ## -Inf denotes an unavailable proposal (e.g. at maxRules).
  let threshold = basePearson + minGain
  if replacePearson > threshold and replacePearson > addPearson:
    return moveReplace
  if addPearson > threshold:
    return moveAdd
  return moveNone

proc attemptExchange(rows: seq[Row], base: Branch, cfg: Config,
                     codec: TokenCodec, baseEnc: Encoded, baseFit: Fit,
                     cache: StageCache = nil): seq[Branch] =
  if base.rules.len == 0: return
  var weakest = -1
  var leastLoss = Inf
  for i, v in baseFit.removalLoss:
    if v < leastLoss:
      leastLoss = v
      weakest = i
  if weakest < 0: return
  var trial = base
  var blocked = newSeq[bool](base.rules.len)
  var nextIndex = weakest
  for depth in 0..<min(cfg.swapDepth, base.rules.len):
    if nextIndex < 0 or blocked[nextIndex]: break
    blocked[nextIndex] = true
    var without: seq[Rule] = @[]
    for r in trial.rules: without.add(r)
    without.delete(nextIndex)
    let options = bestInsertion(rows, without, nextIndex, cfg, codec,
                                cfg.beamWidth,
                                (if cfg.usePrefixCache: cache else: nil))
    if options.len == 0: break
    let ranked = (if nextIndex == without.len:
                    # Score was EXACTLY calculated by bestInsertion using
                    # Ridge/OLS marginal addition; no repeated encode/refit.
                    bestCurrentBranches(options, cfg.beamWidth)
                  else:
                    rankBranches(rows, options, cfg.beamWidth, cfg.ridgeLambda,
                                 cache, cfg.usePrefixStateCache,
                                 cfg.useDeltaSuffix))
    if ranked.len == 0: break
    trial = ranked[0]
    if trial.score > baseFit.pearson + cfg.minGain:
      # Alternatives at the same depth remain alive until next rotation.
      # The scores were already computed in rankBranches.
      for b in ranked:
        if b.score > baseFit.pearson + cfg.minGain:
          result.add(b)
      return
    let fitTrial = evaluate(rows, trial.rules, cfg.ridgeLambda, true)
    nextIndex = alteredIndex(baseEnc, fitTrial.enc, blocked, fitTrial.fit)
  # No strict improvement => reject the whole speculative chain.

proc attemptAddition(rows: seq[Row], base: Branch, cfg: Config,
                     codec: TokenCodec, enc: Encoded, fit: Fit,
                     cache: StageCache = nil): seq[Branch] =
  if base.rules.len >= cfg.maxRules or fit.sst <= 1e-24: return
  let best = bestPatterns(enc.finalTexts, fit, cfg, base.rules,
                          cfg.beamWidth, false,
                          (if cfg.usePrefixCache: cache else: nil),
                          stageSignature(base.rules, base.rules.len))
  for c in best:
    # Adding a LAST rule affects only its own count column on this window.
    # Its RHS has no influence until later rules are appended. Reuse the
    # exact appended-fit Pearson across all variants; OLS gain measures
    # delta-SSE, Ridge gain directly measures delta-Pearson.
    var newPearson: float64
    if fit.ridgeLambda > 0.0:
      newPearson = max(-1.0, min(1.0, fit.pearson + c.gain))
    else:
      newPearson = sqrt(max(0.0, min(1.0,
        1.0 - max(0.0, fit.sse - c.gain) / fit.sst)))
    if newPearson <= fit.pearson + cfg.minGain: continue
    for r in generateReplacementVariants(c.pat, c.frequency, codec):
      var rules: seq[Rule] = @[]
      for old in base.rules: rules.add(old)
      rules.add(r)
      result.add(Branch(rules: rules, score: newPearson,
        description: "add " & patternString(c.pat) & " => " & patternString(r.rhs),
        editedAt: base.rules.len))
  result = bestCurrentBranches(result, cfg.beamWidth)

proc predictPearson(rows: seq[Row], rules: seq[Rule], beta: seq[float64]): float64 =
  if rows.len == 0 or beta.len != rules.len+1: return 0.0
  let enc = encode(rows, rules)
  var meanY, meanP = 0.0
  var predictions = newSeq[float64](rows.len)
  for i in 0..<rows.len:
    var prediction = beta[0]
    for j in 0..<rules.len:
      prediction += beta[j+1] * enc.cols[j][i]
    predictions[i] = prediction
    meanY += enc.y[i]
    meanP += prediction
  meanY /= float64(rows.len)
  meanP /= float64(rows.len)
  var numerator, dy, dp = 0.0
  for i in 0..<rows.len:
    let a = enc.y[i] - meanY
    let b = predictions[i] - meanP
    numerator += a*b
    dy += a*a
    dp += b*b
  if dy > 1e-20 and dp > 1e-20:
    result = max(-1.0, min(1.0, numerator / sqrt(dy*dp)))

proc plotMeasure(r: float64): float64 =
  -ln(max(1e-15, 1.0 - r))

# Fixed-width, trailing moving average of the *raw Pearson* coefficients.
# Maintain the sum in O(1), rather than rereading training.dat every step.
# For the first 14 steps use all observations available so far.
const MovingAverageWidth = 15

type PearsonMovingAverage = object
  values: array[MovingAverageWidth, float64]
  nextIndex, count: int
  total: float64

proc addPearson(m: var PearsonMovingAverage, r: float64): float64 =
  if m.count == MovingAverageWidth:
    m.total -= m.values[m.nextIndex]
  else:
    inc m.count
  m.values[m.nextIndex] = r
  m.total += r
  m.nextIndex = (m.nextIndex + 1) mod MovingAverageWidth
  result = m.total / float64(m.count)

proc writePlot(dir: string, ridgeLambda: float64) =
  let script = """
set terminal pngcairo size 1450,820 enhanced font 'Arial,13'
set output 'training.png'
set datafile separator whitespace
set border lc rgb '#454545' lw 1.2
set tics textcolor rgb '#333333'
set grid xtics ytics lc rgb '#dddddd' lw 1
set key top left opaque box
set xlabel 'Step'
set ylabel '-ln(1 - Pearson)'
set title 'Lens | Incremental OLS count-rule search'
set style line 1 lc rgb '#a6bee9' lw 1.3 pt 7 ps 0.30
set style line 2 lc rgb '#a2d9bf' lw 1.3 pt 5 ps 0.30
set style line 3 lc rgb '#3265c9' lw 3.0
set style line 4 lc rgb '#189567' lw 3.0
plot 'training.dat' using 1:2 with linespoints ls 1 title 'Train (raw)', \
     'training.dat' using 1:3 with linespoints ls 2 title 'Held-out (raw)', \
     'training.dat' using 1:8 with lines ls 3 title 'Train (Pearson MA15)', \
     'training.dat' using 1:9 with lines ls 4 title 'Held-out (Pearson MA15)'
"""
  let regressionName = (if ridgeLambda > 0.0:
    "Ridge count-rule search" else: "OLS count-rule search")
  writeFile(dir / "training.gnuplot",
    script.replace("Incremental OLS count-rule search", regressionName))
  if findExe("gnuplot").len > 0:
    let previous = getCurrentDir()
    try:
      setCurrentDir(dir)
      discard execShellCmd("gnuplot training.gnuplot")
    finally:
      setCurrentDir(previous)

proc selfTest() =
  # Candidate mining must include later noise stages and other chunks.
  doAssert miningSampleIndices(1) == @[0]
  doAssert miningSampleIndices(2) == @[0, 1]
  doAssert miningSampleIndices(20) == @[0, 2, 5, 8, 10, 13, 16, 19]
  # This second-row token never occurs in the first sample. The former
  # mine(texts[0]) implementation could not propose it at all.
  var sampleCfg = Config(minNgram: 1, maxGram: 1, maxTotalGram: 1,
    maxWildcards: 0, maxGap: 1, mineStarts: 8, mineBudget: 200,
    mineTracked: 200, candidates: 16,
    cmsWidth: 1024, cmsDepth: 4)
  let sampleRows = @[@[111, 111, 111], @[222, 222, 222]]
  let sampled = mineWindow(sampleRows, sampleCfg)
  var sawSecond = false
  for c in sampled:
    if c.pat == @[222]: sawSecond = true
  doAssert sawSecond
  # Reusing exactly the same prefix must not change the selected candidates,
  # exact match-count vectors, or their incremental Ridge/OLS scores.
  let cacheFit = ols(@[], @[0.0, 1.0], false)
  let baselineWinners = bestPatterns(sampleRows, cacheFit, sampleCfg, @[], 2, true)
  let stageCacheTest = StageCache(stages: initTable[string, StageEntry]())
  let cachedWinners = bestPatterns(sampleRows, cacheFit, sampleCfg, @[],
                                  2, true, stageCacheTest, "selftest-prefix")
  let cachedAgain = bestPatterns(sampleRows, cacheFit, sampleCfg, @[],
                                2, true, stageCacheTest, "selftest-prefix")
  doAssert cachedWinners.len == baselineWinners.len
  doAssert cachedAgain.len == baselineWinners.len
  for i in 0..<baselineWinners.len:
    doAssert cachedWinners[i].pat == baselineWinners[i].pat
    doAssert cachedAgain[i].pat == baselineWinners[i].pat
    doAssert abs(cachedAgain[i].gain - baselineWinners[i].gain) < 1e-12
    doAssert cachedAgain[i].frequency == baselineWinners[i].frequency
  doAssert stageCacheTest.reusedMining == 1
  doAssert stageCacheTest.reusedCounts > 0
  # Fingerprint cannot reject any actual match, including wildcards.
  let fingerprintTexts = @[@[11, 12, 13], @[11, 22, 33], @[11, 11]]
  let fingerprints = fingerprintRows(fingerprintTexts)
  for pat in [@[11], @[22], @[12, -1, 13], @[11, -1, 33], @[99]]:
    doAssert countFeatures(fingerprintTexts, pat) ==
      countFeatures(fingerprintTexts, pat, fingerprints)
  doAssert not mayContainPattern(fingerprints[0], @[99999])
  doAssert mayContainPattern(fingerprints[0], @[12, -1, 13])
  # Screening is identity when disabled or when the shortlist is small.
  let screenCfg = Config(screenRows: 3, screenCandidates: 2)
  let screenTexts = @[@[11], @[22], @[22], @[33]]
  let screenFit = ols(@[], @[0.0, 1.0, 1.0, 0.0], false)
  let screenInput = @[
    PatternScore(pat: @[11], frequency: 5),
    PatternScore(pat: @[22], frequency: 7),
    PatternScore(pat: @[33], frequency: 6)]
  let kept = screenPatterns(screenTexts, screenFit, screenCfg, screenInput)
  doAssert kept.len == 2
  var foundCorrelated = false
  for c in kept:
    if c.pat == @[22]: foundCorrelated = true
  doAssert foundCorrelated
  var exactOnly = screenCfg
  exactOnly.screenRows = 0
  doAssert screenPatterns(screenTexts, screenFit, exactOnly,
    screenInput).len == screenInput.len
  # Full tracking capacity must NOT lock in the first patterns seen.
  # CMS counts *all* offers; after enough repetitions, a late heavy hitter
  # must displace a previously tracked low-frequency key.
  var smallMiner = MineState(cfg: Config(mineTracked: 2),
    limit: 200, sketch: initCountMinSketch(4096, 4),
    positions: initTable[seq[int], int]())
  smallMiner.offer(@[10])
  smallMiner.offer(@[20])
  doAssert smallMiner.tracked.len == 2
  doAssert not smallMiner.positions.hasKey(@[30])
  smallMiner.offer(@[30]) # tie (1:1) does not evict
  doAssert not smallMiner.positions.hasKey(@[30])
  smallMiner.offer(@[30]) # CMS now estimates 2, so it displaces a 1
  doAssert smallMiner.positions.hasKey(@[30])
  doAssert smallMiner.tracked.len == 2
  doAssert smallMiner.positions.len == 2
  for i in 0..<30: smallMiner.offer(@[40])
  doAssert smallMiner.positions.hasKey(@[40])
  doAssert smallMiner.sketch.estimate(@[40]) == 30'u32
  for i, item in smallMiner.tracked:
    doAssert smallMiner.positions[item.pat] == i
  # Heap roots always represent a minimum of cached CMS estimates.
  for item in smallMiner.tracked:
    doAssert smallMiner.tracked[0].estimate <= item.estimate
  # Repeated updates of tracked patterns must not corrupt heap indices.
  for i in 0..<10: smallMiner.offer(@[30])
  for i, item in smallMiner.tracked:
    doAssert smallMiner.positions[item.pat] == i
  # Replacing a mid-chain rule may improve downstream features even when
  # its own match count is constant and FWL insertion gain is exactly zero.
  let constantRows = @[@[333], @[333], @[333], @[333]]
  let target = @[0.0, 1.0, 0.0, 1.0]
  let constantFit = ols(@[], target, false)
  doAssert bestPatterns(constantRows, constantFit, sampleCfg,
    @[], 1).len == 0
  let replacementPicks = bestPatterns(constantRows, constantFit,
    sampleCfg, @[], 1, true)
  doAssert replacementPicks.len == 1
  doAssert replacementPicks[0].pat == @[333]
  doAssert abs(replacementPicks[0].gain) < 1e-12
  # The smoothing is applied to raw Pearson, BEFORE the nonlinear -ln(1-rho).
  var moving: PearsonMovingAverage
  doAssert abs(addPearson(moving, 0.2) - 0.2) < 1e-12
  let averaged = addPearson(moving, 0.4)
  doAssert abs(averaged - 0.3) < 1e-12
  doAssert abs(plotMeasure(averaged) - (-ln(0.7))) < 1e-12
  doAssert abs(plotMeasure(averaged) -
    (plotMeasure(0.2) + plotMeasure(0.4)) / 2.0) > 1e-4
  var full: PearsonMovingAverage
  for i in 1..MovingAverageWidth:
    discard addPearson(full, float64(i) / 100.0)
  doAssert abs(full.total - 1.2) < 1e-12 # (1 + ... + 15) / 100
  doAssert abs(addPearson(full, 0.16) - 0.09) < 1e-12
  # Move policy: add wins ties, replace only wins by strictly higher Pearson.
  doAssert chooseMove(0.70, 0.78, 0.79, 1e-9) == moveReplace
  doAssert chooseMove(0.70, 0.79, 0.78, 1e-9) == moveAdd
  doAssert chooseMove(0.70, 0.79, 0.79, 1e-9) == moveAdd
  doAssert chooseMove(0.70, 0.70, 0.70, 1e-9) == moveNone
  doAssert chooseMove(0.70, -Inf, 0.71, 1e-9) == moveReplace
  doAssert chooseMove(0.70, 0.71, -Inf, 1e-9) == moveAdd
  doAssert chooseMove(0.70, -Inf, -Inf, 1e-9) == moveNone
  let y = @[0.0, 1.0, 3.0, 5.0, 8.0, 13.0]
  let a = @[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
  let b = @[0.0, 0.0, 1.0, 2.0, 3.0, 1.0]
  let z = @[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
  let current = ols(@[a, b], y)
  let deletedA = ols(@[b], y, false)
  let deletedB = ols(@[a], y, false)
  doAssert abs(current.removalLoss[0] - (deletedA.sse - current.sse)) < 1e-7
  doAssert abs(current.removalLoss[1] - (deletedB.sse - current.sse)) < 1e-7
  let added = ols(@[a, b, z], y, false)
  doAssert abs(candidateGain(z, current) - (current.sse - added.sse)) < 1e-7
  let fwPearson = sqrt(max(0.0, min(1.0,
    1.0 - max(0.0, current.sse - candidateGain(z, current)) / current.sst)))
  doAssert abs(fwPearson - added.pearson) < 1e-10
  # Ridge tests: regularized intercept-free penalty, correlations NOT sqrt(R²),
  # fast appended-rule Pearson and deletion scores equal complete Ridge refits.
  for lam in [0.01, 0.1, 3.0]:
    let cur = ridge(@[a, b], y, lam)
    let appended = ridge(@[a, b, z], y, lam, false)
    let removedA = ridge(@[b], y, lam, false)
    let removedB = ridge(@[a], y, lam, false)
    doAssert abs(cur.pearson + candidateGain(z, cur) -
      appended.pearson) < 1e-9
    doAssert abs(cur.removalLoss[0] - (cur.pearson-removedA.pearson)) < 1e-9
    doAssert abs(cur.removalLoss[1] - (cur.pearson-removedB.pearson)) < 1e-9
    doAssert abs(cur.beta[0] - (cur.targetMean -
      cur.beta[1]*cur.featureMeans[0] -
      cur.beta[2]*cur.featureMeans[1])) < 1e-9
  let ridgeConstant = ridge(@[newSeqWith(y.len, 1.0)], y, 0.1)
  doAssert abs(ridgeConstant.beta[0] - (0.0+1.0+3.0+5.0+8.0+13.0)/float64(y.len)) < 1e-9
  doAssert abs(ridgeConstant.beta[1]) < 1e-10
  doAssert candidateGain(newSeqWith(y.len, 1.0), ridgeConstant) == 0.0
  let wildcard = compileSeqPattern(@[65, -1, 66], 1)
  var txt = @[65, 49, 50, 66, 65, 51, 52, 66]
  let reference = countSeqCompiled(txt, wildcard)
  let fusedCount = replaceSeqCompiledCountInPlace(txt, wildcard,
                                                   @[65, -32, 66])
  doAssert reference == fusedCount
  doAssert fusedCount == 2
  doAssert txt == @[65, 50, 49, 66, 65, 52, 51, 66]
  var compiledLiteral = compileSeqReplaceProgram(@[65, 66], @[65, 66], 1)
  var text2 = @[65, 66, 65, 66, 65]
  let c = applyAndCount(text2, compiledLiteral)
  doAssert c == 2
  doAssert text2 == @[65, 66, 65, 66, 65]
  # Alphabet mapping uses corpus frequency, not ASCII order.
  let raw = @[@[65, 65, 65, 65, 66, 66, 67]]
  let codec = makeTokenCodec(raw, 1729'u64)
  doAssert codec.alphabet[65] < codec.alphabet[66]
  doAssert codec.alphabet[66] < codec.alphabet[67]
  let hashed = encodeAlphabet(raw, codec)
  doAssert hashed[0][0] == codec.alphabet[65]
  doAssert hashed[0][0] != 65
  let fresh = freshRuleToken(codec, @[codec.alphabet[65]], 4)
  doAssert fresh >= RuleBase
  doAssert fresh != codec.alphabet[65]
  let variants = generateReplacementVariants(
    @[codec.alphabet[65], -1, codec.alphabet[66]], 4, codec)
  doAssert variants.len == 4
  for variant in variants:
    doAssert variant.rhs[0] == freshRuleToken(codec, variant.pat, 4)
    doAssert isAlternatingReplacement(variant.pat, variant.rhs)
    doAssert variant.rhs.len == 3
    doAssert variant.rhs[0] >= RuleBase
    doAssert variant.rhs[2] >= RuleBase
    doAssert variant.rhs[0] != variant.rhs[2]
  doAssert variants[0].rhs[1] == -1
  doAssert variants[1].rhs[1] == -16
  doAssert variants[2].rhs[1] == -32
  doAssert variants[3].rhs[1] == -48
  # All four single-capture RHSs MUST rewrite differently on a nontrivial
  # capture. Their immediate count columns are identical, so a one-step OLS
  # cannot distinguish them; the next count feature CAN distinguish them.
  let va = codec.alphabet[65]
  let vb = codec.alphabet[66]
  let span = @[va, 300, 100, 200, vb]
  let expectedSpans = @[
    @[300, 100, 200], @[100, 200, 300],
    @[200, 100, 300], @[2, 0, 1]]
  for opIndex, variant in variants:
    var input = span
    var compiledOp = compileSeqReplaceProgram(variant.pat, variant.rhs, 1)
    doAssert applyAndCount(input, compiledOp) == 1
    doAssert input == @[variant.rhs[0]] & expectedSpans[opIndex] &
      @[variant.rhs[2]]
  let evalRows = @[
    Row(text: @[va, 300, 100, 200, vb], target: 1.0),
    Row(text: @[va, 100, 300, 200, vb], target: 0.7),
    Row(text: @[va, 200, 100, 300, vb], target: 0.3),
    Row(text: @[va, 100, 200, 300, vb], target: 0.0)]
  let baseEnc = encode(evalRows, @[variants[0]])
  let baseFit = ols(baseEnc.cols, baseEnc.y)
  let nextPattern = @[variants[0].rhs[0], 200, 100]
  let plainCounts = countFeatures(baseEnc.finalTexts, nextPattern)
  let reverseCounts = countFeatures(
    encode(evalRows, @[variants[2]]).finalTexts, nextPattern)
  doAssert plainCounts != reverseCounts
  doAssert candidateGain(plainCounts, baseFit) >
    candidateGain(reverseCounts, baseFit) + 1e-9
  for variant in variants:
    doAssert abs(evaluate(evalRows, @[variant], 0.0).fit.pearson -
      baseFit.pearson) < 1e-10
  var variantBranches: seq[Branch] = @[]
  for variant in variants:
    variantBranches.add(Branch(rules: @[variant], editedAt: 0))
  # Tail replacement can use incremental Ridge directly because its RHS
  # cannot affect any downstream feature. Compare with an entire Ridge refit.
  var tailCfg = sampleCfg
  tailCfg.ridgeLambda = 0.1
  let tailRows = @[
    Row(text: @[111], target: 0.0),
    Row(text: @[111, 222], target: 1.0),
    Row(text: @[111, 222, 222], target: 2.0),
    Row(text: @[111, 222, 222, 222], target: 3.0)]
  let tailTrials = bestInsertion(tailRows, @[], 0, tailCfg, codec, 2)
  doAssert tailTrials.len > 0
  for branch in tailTrials:
    let exactPearson = evaluate(tailRows, branch.rules, 0.1).fit.pearson
    doAssert abs(branch.score - exactPearson) < 1e-9
  # Sibling RHSs share the same rule-free prefix. Both cached ranking
  # and legacy full encoding MUST fit to indistinguishable Pearson.
  let siblingRankCache = StageCache(
    stages: initTable[string, StageEntry](),
    prefixStates: initTable[string, PrefixState]())
  let fastFourOLS = rankBranches(evalRows, variantBranches, 4, 0.0,
                                siblingRankCache, true)
  let fastFourRidge = rankBranches(evalRows, variantBranches, 4, 0.1,
                                  siblingRankCache, true)
  let slowFourRidge = rankBranches(evalRows, variantBranches, 4, 0.1)
  doAssert fastFourOLS.len == 4 and fastFourRidge.len == 4
  doAssert siblingRankCache.prefixStates.len == 1
  for i in 0..<4:
    doAssert abs(fastFourRidge[i].score-slowFourRidge[i].score) < 1e-10
    doAssert abs(fastFourOLS[i].score-baseFit.pearson) < 1e-10
  # Test an actual shared nonempty prefix and multiple alternative
  # intermediate rewrites feeding the same downstream rule.
  let prefixRule = Rule(pat: @[va], rhs: @[1001])
  let afterRule = Rule(pat: @[1001, -1, vb], rhs: @[2001, -1, 2002])
  let lastRule = Rule(pat: @[2001], rhs: @[3001])
  let changedRule = Rule(pat: @[1001, -1, vb], rhs: @[2001, -32, 2002])
  let midBranches = @[
    Branch(rules: @[prefixRule, afterRule, lastRule], editedAt: 1),
    Branch(rules: @[prefixRule, changedRule, lastRule], editedAt: 1)]
  let midCache = StageCache(stages: initTable[string, StageEntry](),
    prefixStates: initTable[string, PrefixState]())
  for lam in [0.0, 0.1, 1.0]:
    let slowRank = rankBranches(evalRows, midBranches, 2, lam)
    let quickRank = rankBranches(evalRows, midBranches, 2, lam,
                                 midCache, true)
    doAssert slowRank.len == quickRank.len
    for i in 0..<slowRank.len:
      doAssert slowRank[i].description == quickRank[i].description
      doAssert abs(slowRank[i].score-quickRank[i].score) < 1e-9
  doAssert midCache.prefixStates.len == 1
  # Delta suffix must agree with full encoding for mixed firing/non-firing
  # rows, and must correctly reuse following-rule counts on non-firing rows.
  let mixedRows = @[
    Row(text: @[va, 100, 200, vb], target: 0.0),
    Row(text: @[va, 2001, 200, vb], target: 0.2),
    Row(text: @[va, 300, 100, 200, vb], target: 0.4),
    Row(text: @[va, 100, 200], target: 0.6),
    Row(text: @[va, 2001, 300, vb], target: 0.8),
    Row(text: @[va, 100, 300, 200, vb], target: 1.0)]
  let suffix1 = Rule(pat: @[2001], rhs: @[3001])
  let suffix2 = Rule(pat: @[3001], rhs: @[4001])
  let altAbsent = Rule(pat: @[999998, -1, 999999],
                       rhs: @[9001, -1, 9002])
  let variedBranches = @[
    Branch(rules: @[prefixRule, afterRule, suffix1, suffix2],
           editedAt: 1, description: "plain"),
    Branch(rules: @[prefixRule, changedRule, suffix1, suffix2],
           editedAt: 1, description: "reverse"),
    Branch(rules: @[prefixRule, altAbsent, suffix1, suffix2],
           editedAt: 1, description: "absent")]
  for lam in [0.0, 0.1, 1.0]:
    let slow = rankBranches(mixedRows, variedBranches, 3, lam)
    let cached = rankBranches(mixedRows, variedBranches, 3, lam,
                              nil, true, false)
    let delta = rankBranches(mixedRows, variedBranches, 3, lam,
                             nil, true, true)
    doAssert delta.len == slow.len and delta.len == cached.len
    for i in 0..<slow.len:
      doAssert delta[i].description == slow[i].description
      doAssert abs(delta[i].score - slow[i].score) < 1e-9
      doAssert abs(delta[i].score - cached[i].score) < 1e-9
  # A pure RHS variation (identical LHS) shares its match vector, but may
  # still produce distinct later counts; both must be evaluated exactly.
  for lam in [0.0, 0.1]:
    let direct = rankBranches(evalRows, midBranches, 2, lam)
    let delta = rankBranches(evalRows, midBranches, 2, lam,
                             nil, true, true)
    for i in 0..<direct.len:
      doAssert abs(direct[i].score - delta[i].score) < 1e-9
  let fourBeams = rankBranches(evalRows, variantBranches, 4, 0.0)
  doAssert fourBeams.len == 4
  var seenOps: seq[int] = @[]
  for branch in fourBeams:
    doAssert abs(branch.score - baseFit.pearson) < 1e-10
    seenOps.add(branch.rules[0].rhs[1])
  seenOps.sort()
  doAssert seenOps == @[-48, -32, -16, -1]
  let noWildcard = generateReplacementVariants(@[codec.alphabet[65]], 4, codec)
  doAssert noWildcard.len == 1
  doAssert noWildcard[0].rhs.len == 1
  doAssert isAlternatingReplacement(noWildcard[0].pat, noWildcard[0].rhs)
  let pattern2 = @[codec.alphabet[65], -1, codec.alphabet[66],
                    -1, codec.alphabet[67]]
  let variants2 = generateReplacementVariants(pattern2, 9, codec)
  doAssert variants2.len == 2
  for variant in variants2:
    doAssert isAlternatingReplacement(variant.pat, variant.rhs)
    doAssert variant.rhs.len == 5
    doAssert variant.rhs[0] != variant.rhs[2]
    doAssert variant.rhs[2] != variant.rhs[4]
  doAssert variants2[0].rhs[1] == -1 and variants2[0].rhs[3] == -2
  doAssert variants2[1].rhs[1] == -2 and variants2[1].rhs[3] == -1
  # The compiled replacement engine must emit the interior AND trailing marker.
  var twoCaptureText = @[codec.alphabet[65], codec.alphabet[68],
                         codec.alphabet[66], codec.alphabet[69],
                         codec.alphabet[67]]
  let plain2 = variants2[0]
  var twoProgram = compileSeqReplaceProgram(plain2.pat, plain2.rhs, 1)
  doAssert applyAndCount(twoCaptureText, twoProgram) == 1
  doAssert twoCaptureText == @[plain2.rhs[0], codec.alphabet[68],
                               plain2.rhs[2], codec.alphabet[69],
                               plain2.rhs[4]]
  let pattern3 = @[codec.alphabet[65], -1, codec.alphabet[66],
                   -1, codec.alphabet[67], -1, codec.alphabet[68]]
  let variants3 = generateReplacementVariants(pattern3, 9, codec)
  doAssert variants3.len == 4 # identity plus each pair swap
  for variant in variants3:
    doAssert isAlternatingReplacement(variant.pat, variant.rhs)
    doAssert variant.rhs.len == 7
  # The final diffusion snapshot contains only corpus-observed alphabet
  # tokens; each corrupted position differs from its original value.
  # The initial snapshot is unchanged frequency-encoded input.
  var tinyCfg = Config(noiseSteps: 0)
  var diffused: seq[Row] = @[]
  appendDiffusion(diffused, hashed[0], 0, tinyCfg, codec)
  doAssert diffused.len == hashed[0].len + 1
  doAssert diffused[0].text == hashed[0]
  for stage, row in diffused:
    var changedPositions = 0
    for i, value in row.text:
      if value != hashed[0][i]:
        inc changedPositions
    doAssert changedPositions == stage
  for i, value in diffused[^1].text:
    doAssert value != hashed[0][i]
    var wasObserved = false
    for byteValue in 0..<256:
      if codec.frequencies[byteValue] > 0 and value == codec.alphabet[byteValue]:
        wasObserved = true
        break
    doAssert wasObserved
  # Corruptions are deterministic for a fixed seed/chunk/position.
  var repeatedDiffusion: seq[Row] = @[]
  appendDiffusion(repeatedDiffusion, hashed[0], 0, tinyCfg, codec)
  doAssert repeatedDiffusion[^1].text == diffused[^1].text
  # Even if one character dominates, sampling excludes it in bounded time.
  let skewed = makeTokenCodec(@[newSeqWith(200, 65) & @[66]], 11'u64)
  for position in 0..<200:
    doAssert noiseToken(skewed, 0, position, skewed.alphabet[65]) ==
      skewed.alphabet[66]
  # A single-symbol training corpus cannot be corrupted into a different
  # corpus-observed character. Fail explicitly rather than fabricating OOV.
  let degenerate = makeTokenCodec(@[@[65, 65]], 23'u64)
  var failedAsExpected = false
  try:
    discard noiseToken(degenerate, 0, 0, degenerate.alphabet[65])
  except ValueError:
    failedAsExpected = true
  doAssert failedAsExpected
  # Rule 2 MUST receive the output of rule 1 (never the pre-rule row).
  let chainRules = @[
    Rule(pat: @[codec.alphabet[65]], rhs: @[fresh]),
    Rule(pat: @[fresh], rhs: @[fresh + 1])]
  let chained = encode(@[Row(text: hashed[0], target: 1.0)], chainRules)
  doAssert chained.cols[0][0] == 4.0
  doAssert chained.cols[1][0] == 4.0
  doAssert chained.finalTexts[0][0] == fresh + 1
  # Captures may still be transformed via sort/reverse/inverse_sort.
  echo "Numerical Ridge/OLS, width-4 beam, add/replace policy, alternating RHS, tokenization, diffusion and sequential-rewrite self-tests passed"

proc main() =
  if "--self-test" in commandLineParams():
    selfTest()
    return
  let cfg = parseConfig()
  createDir(cfg.outDir)
  let rawSource = loadCorpus(cfg)
  var codec: TokenCodec
  if cfg.tokenMapFile.len > 0:
    codec = loadTokenMap(cfg.tokenMapFile)
  elif cfg.resumeFile.len > 0:
    # Resumed rules already contain hashed tokens; their exact mapping is
    # required. Never silently rebuild it from a rotated/changed corpus.
    let originalMap = parentDir(cfg.resumeFile) / "token_map.tsv"
    codec = loadTokenMap(originalMap)
  else:
    codec = makeTokenCodec(rawSource, uint64(cfg.tokenSeed))
  let source = encodeAlphabet(rawSource, codec)
  saveTokenMap(cfg.outDir / "token_map.tsv", codec)
  stderr.writeLine("[tokenization] frequency-weighted hashed bytes=256 seed=", codec.seed,
    " map=", cfg.outDir / "token_map.tsv",
    " noise=observed-frequency-weighted-different-byte",
    " noise-before-rules=true sequential-rewrite=true")
  stderr.writeLine("[regression] type=", (if cfg.ridgeLambda > 0.0: "Ridge" else: "OLS"),
    " lambda=", cfg.ridgeLambda, " standardized=", cfg.ridgeLambda > 0.0,
    " intercept_penalty=false")
  stderr.writeLine("[corpus] chunks=", source.len, " chunkLen<=", cfg.chunkLen,
    " train=", cfg.trainChunks, " valid=", cfg.validChunks,
    " noiseSteps=", cfg.noiseSteps, " (0=all)")
  var baseRules: seq[Rule] = @[]
  if cfg.resumeFile.len > 0:
    baseRules = loadRules(cfg.resumeFile, codec)
    if baseRules.len > cfg.maxRules: quit("Resume exceeds max-rules", 1)
  var beams = @[Branch(rules: baseRules, description: "initial", editedAt: -1)]
  var cache = DiffusionCache(byChunk: initTable[int, seq[Row]](),
                            capacity: max(2, cfg.trainChunks + cfg.validChunks + 1))
  let validation = makeRows(source, cfg, codec, 0, true, cache)
  let plotPath = cfg.outDir / "training.dat"
  writeFile(plotPath, "# step train_nlog1mpearson heldout_nlog1mpearson rules train_pearson heldout_pearson seconds train_ma15_nlog1mpearson heldout_ma15_nlog1mpearson\n")
  var trainMA, heldoutMA: PearsonMovingAverage
  for step in 0..<cfg.steps:
    let tick = epochTime()
    let rows = makeRows(source, cfg, codec, step, false, cache)
    when defined(lensProfile):
      let rowsReadyTime = epochTime()
    if rows.len < 3: quit("Training rows insufficient", 1)
    when defined(lensProfile):
      profileEval = EvaluationCounters()
    # All caches live exactly one rotated training window. This matters:
    # raw rows and their noise schedule change when the window rotates.
    let stageCache = (if cfg.usePrefixCache or cfg.usePrefixStateCache:
                        StageCache(stages: initTable[string, StageEntry](),
                          prefixStates: initTable[string, PrefixState]())
                      else: nil)
    # Every retained RHS is fitted on THIS rotated window and expanded.
    # No lookahead or speculative score is used for pruning.
    let parents = rankBranches(rows, beams, cfg.beamWidth, cfg.ridgeLambda,
                               stageCache, cfg.usePrefixStateCache,
                               cfg.useDeltaSuffix)
    when defined(lensProfile):
      let parentsReadyTime = epochTime()
    if parents.len == 0: break
    let incumbent = parents[0]
    var additions: seq[Branch] = @[]
    var exchanges: seq[Branch] = @[]
    for parentIndex in 0..<parents.len:
      let parent = parents[parentIndex]
      let state = evaluate(rows, parent.rules, cfg.ridgeLambda, true)
      if parent.rules.len < cfg.maxRules:
        for b in attemptAddition(rows, parent, cfg, codec,
                                 state.enc, state.fit, stageCache):
          additions.add(b)
      if parent.rules.len > 0:
        for b in attemptExchange(rows, parent, cfg, codec,
                                 state.enc, state.fit, stageCache):
          exchanges.add(b)
    when defined(lensProfile):
      let expansionReadyTime = epochTime()
      if stageCache != nil:
        stderr.writeLine("[profile-cache] prefixes=", stageCache.stages.len,
          " reused_mining=", stageCache.reusedMining,
          " reused_exact_vectors=", stageCache.reusedCounts,
          " state_entries=", stageCache.prefixStates.len,
          " state_tokens=", stageCache.prefixTokenElements)
      else:
        stderr.writeLine("[profile-cache] disabled")
      stderr.writeLine("[profile-rank] calls=", profileEval.rankCalls,
        " branches=", profileEval.rankVariants,
        " state_cache_calls=", profileEval.rankCacheCalls,
        " builds=", profileEval.rankPrefixBuilds,
        " hits=", profileEval.rankPrefixHits,
        " evictions=", profileEval.rankPrefixEvictions,
        " last_count_only=", profileEval.rankLastCountOnly,
        " total=", profileEval.rankTotalSeconds, "s",
        " prefix_build=", profileEval.rankPrefixBuildSeconds, "s",
        " suffix_encode=", profileEval.rankSuffixSeconds, "s",
        " ridge=", profileEval.rankRidgeSeconds, "s")
      stderr.writeLine("[profile-delta] calls=", profileEval.deltaCalls,
        " fallback=", profileEval.deltaFallbacks,
        " scanned_rows=", profileEval.deltaRowsScanned,
        " pattern_reuses=", profileEval.deltaPatternCacheHits,
        " matched_rows=", profileEval.deltaMatchedRows,
        " skipped_suffix_rows=", profileEval.deltaReusedRows,
        " baseline_built_rows=", profileEval.deltaBaselineBuilds,
        " total=", profileEval.deltaSeconds, "s",
        " ridge=", profileEval.deltaRidgeSeconds, "s")
      stderr.writeLine("[profile-evaluate] calls=", profileEval.evaluateCalls,
        " encode=", profileEval.evaluateEncodeSeconds,
        "s ridge=", profileEval.evaluateRidgeSeconds, "s")
    additions = bestCurrentBranches(additions, cfg.beamWidth)
    exchanges = bestCurrentBranches(exchanges, cfg.beamWidth)
    let addScore = (if additions.len > 0: additions[0].score else: -Inf)
    let replaceScore = (if exchanges.len > 0: exchanges[0].score else: -Inf)
    let move = chooseMove(incumbent.score, addScore, replaceScore, cfg.minGain)
    echo "[compare] step=", step, " baseline=", incumbent.score,
      " add=", (if additions.len > 0: $addScore else: "n/a"),
      " replace=", (if exchanges.len > 0: $replaceScore else: "n/a"),
      " chosen=", move, " beamParents=", parents.len
    case move
    of moveAdd:
      beams = additions
    of moveReplace:
      beams = exchanges
    of moveNone:
      beams = parents
    # Record the best actual current-window fitted model, NOT the forecast.
    let current = beams[0]
    let initial = evaluate(rows, current.rules, cfg.ridgeLambda, true)
    let valPearson = predictPearson(validation, current.rules, initial.fit.beta)
    let elapsed = epochTime() - tick
    let trainMetric = plotMeasure(initial.fit.pearson)
    let valMetric = plotMeasure(valPearson)
    let trainSmoothed = plotMeasure(trainMA.addPearson(initial.fit.pearson))
    let valSmoothed = plotMeasure(heldoutMA.addPearson(valPearson))
    let logLine = $step & " " & $trainMetric & " " & $valMetric & " " &
      $current.rules.len & " " & $initial.fit.pearson & " " &
      $valPearson & " " & $elapsed & " " &
      $trainSmoothed & " " & $valSmoothed & "\n"
    let stream = open(plotPath, fmAppend)
    stream.write(logLine)
    stream.close()
    saveRules(cfg.outDir / "rules.tsv", current.rules)
    if cfg.plotEvery > 0 and
        ((step + 1) mod cfg.plotEvery == 0 or step == cfg.steps - 1):
      writePlot(cfg.outDir, cfg.ridgeLambda)
    when defined(lensProfile):
      stderr.writeLine("[profile-step] rows=", rowsReadyTime-tick,
        "s rank=", parentsReadyTime-rowsReadyTime,
        "s expand=", expansionReadyTime-parentsReadyTime,
        "s selection/evaluation/plot=", epochTime()-expansionReadyTime, "s")
    echo "[step] ", step, " trainPearson=", initial.fit.pearson,
      " validPearson=", valPearson, " -ln(1-rho)=", trainMetric,
      " rules=", current.rules.len, " branch=", current.description,
      " operator=", operatorName(current),
      " time=", elapsed, "s"

when isMainModule:
  main()
