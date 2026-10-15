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
    minGain: float64
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
    score: float64 # Current training-window OLS Pearson, refreshed by rankBranches
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
    maxGram: 3, maxTotalGram: 6, maxWildcards: 3, maxGap: 16,
    mineStarts: 12, mineBudget: 180000, mineTracked: 25000,
    candidates: 384, beamWidth: 4, swapDepth: 3, minGain: 1e-9,
    tokenSeed: 1729, tokenMapFile: "")
  if not fileExists(result.corpus) and fileExists("github-code.txt"):
    result.corpus = "github-code.txt"
  for arg in commandLineParams():
    if arg == "--help" or arg == "-h":
      echo "Lens incremental OLS training"
      echo "--self-test | --corpus=FILE --out=DIR --resume=FILE --chunks=N --train=N --valid=N"
      echo "--chunk-len=N --noise-steps=N (0=one step per character) --steps=N"
      echo "--max-rules=N --min-ngram=N --max-gram=N --max-total-gram=N"
      echo "--max-wildcards=N --max-gap=N --mine-starts=N --mine-budget=N"
      echo "--mine-tracked=N --candidates=N --beam=N --swap-depth=N"
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

proc candidateGain(z: seq[float64], fit: Fit): float64 =
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
  if m.positions.hasKey(pat):
    let idx = m.positions[pat]
    # A tracked pattern can only gain estimated frequency.
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

proc mine(text: seq[int], cfg: Config): seq[PatternScore] =
  var shapes: seq[seq[int]] = @[]
  var cur: seq[int] = @[]
  shapeList(cfg, cur, 0, shapes)
  if shapes.len == 0: return
  var m = MineState(text: text, cfg: cfg,
    sketch: initCountMinSketch(65536, 4),
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

proc countFeatures(texts: seq[seq[int]], pat: seq[int]): seq[float64] =
  let compiled = compileSeqPattern(pat, 1)
  result = newSeq[float64](texts.len)
  for i in 0..<texts.len:
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
  for rowIndex in indices:
    for candidate in mine(texts[rowIndex], local):
      if not seen.hasKey(candidate.pat):
        seen[candidate.pat] = true
        result.add(candidate)

proc bestPatterns(texts: seq[seq[int]], fit: Fit, cfg: Config,
                  exclude: seq[Rule], count: int,
                  insertionMode = false): seq[PatternScore] =
  if texts.len == 0: return
  let candidates = mineWindow(texts, cfg)
  var scored: seq[PatternScore] = @[]
  # A feature's FWL gain is exact ONLY when appending at the end.
  # At mid-chain insertion the candidate also changes downstream count
  # columns, so gain is a HEURISTIC, not a valid rejection test.
  for originalCandidate in candidates:
    var candidate = originalCandidate
    var banned = false
    for r in exclude:
      if candidate.pat == r.pat:
        banned = true
        break
    if banned: continue
    let counts = countFeatures(texts, candidate.pat)
    var actualOccurrences = 0
    for value in counts: actualOccurrences += int(value)
    if actualOccurrences <= 0: continue
    candidate.frequency = actualOccurrences
    candidate.gain = candidateGain(counts, fit)
    if insertionMode or candidate.gain > cfg.minGain:
      scored.add(candidate)
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
              removal = false): tuple[enc: Encoded, fit: Fit] =
  result.enc = encode(rows, rules)
  result.fit = ols(result.enc.cols, result.enc.y, removal)

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
                   cfg: Config, codec: TokenCodec, keep: int): seq[Branch] =
  ## Mine exactly at the insertion point, AFTER earlier rules and BEFORE
  ## later ones. FWL is a prefilter; full rewritten OLS judges acceptance.
  let baseline = evaluate(rows, existing)
  let stageTexts = prefixTexts(rows, existing, at)
  let winners = bestPatterns(stageTexts, baseline.fit, cfg, existing,
                             max(1, keep), true)
  for candidate in winners:
    for variant in generateReplacementVariants(candidate.pat, candidate.frequency, codec):
      var rules: seq[Rule] = @[]
      for r in existing: rules.add(r)
      rules.insert(variant, at)
      result.add(Branch(rules: rules,
        description: "replace " & $at & " with " & patternString(candidate.pat) &
          " => " & patternString(variant.rhs), editedAt: at))

proc rankBranches(rows: seq[Row], branches: seq[Branch], keep: int): seq[Branch] =
  ## Simple beam: fit ONLY the present diffusion window. Equal-scoring
  ## RHS variants are retained independently and compete on the next step.
  var ranking: seq[Branch] = @[]
  for b in branches:
    var scored = b
    scored.score = evaluate(rows, b.rules).fit.pearson
    ranking.add(scored)
  ranking.sort(proc(a, b: Branch): int = cmp(b.score, a.score))
  for i in 0..<min(keep, ranking.len):
    result.add(ranking[i])

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
                     codec: TokenCodec, baseEnc: Encoded, baseFit: Fit): seq[Branch] =
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
    let options = bestInsertion(rows, without, nextIndex, cfg, codec, cfg.beamWidth)
    if options.len == 0: break
    let ranked = rankBranches(rows, options, cfg.beamWidth)
    if ranked.len == 0: break
    trial = ranked[0]
    if trial.score > baseFit.pearson + cfg.minGain:
      # Alternatives at the same depth remain alive until next rotation.
      # The scores were already computed in rankBranches.
      for b in ranked:
        if b.score > baseFit.pearson + cfg.minGain:
          result.add(b)
      return
    let fitTrial = evaluate(rows, trial.rules, true)
    nextIndex = alteredIndex(baseEnc, fitTrial.enc, blocked, fitTrial.fit)
  # No strict improvement => reject the whole speculative chain.

proc attemptAddition(rows: seq[Row], base: Branch, cfg: Config,
                     codec: TokenCodec, enc: Encoded, fit: Fit): seq[Branch] =
  if base.rules.len >= cfg.maxRules or fit.sst <= 1e-24: return
  let best = bestPatterns(enc.finalTexts, fit, cfg, base.rules, cfg.beamWidth)
  for c in best:
    # Adding a LAST rule affects only its own count column on this window.
    # Its RHS has no influence until later rules are appended. FWL therefore
    # computes the exact immediate OLS score for ALL RHS variants at once.
    let newSse = max(0.0, fit.sse - c.gain)
    let newPearson = sqrt(max(0.0, min(1.0, 1.0 - newSse / fit.sst)))
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

proc writePlot(dir: string) =
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
  writeFile(dir / "training.gnuplot", script)
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
    mineTracked: 200, candidates: 16)
  let sampleRows = @[@[111, 111, 111], @[222, 222, 222]]
  let sampled = mineWindow(sampleRows, sampleCfg)
  var sawSecond = false
  for c in sampled:
    if c.pat == @[222]: sawSecond = true
  doAssert sawSecond
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
    doAssert abs(evaluate(evalRows, @[variant]).fit.pearson -
      baseFit.pearson) < 1e-10
  var variantBranches: seq[Branch] = @[]
  for variant in variants:
    variantBranches.add(Branch(rules: @[variant], editedAt: 0))
  let fourBeams = rankBranches(evalRows, variantBranches, 4)
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
  echo "Numerical OLS, width-4 beam, add/replace policy, alternating RHS, tokenization, diffusion and sequential-rewrite self-tests passed"

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
    if rows.len < 3: quit("Training rows insufficient", 1)
    # Every retained RHS is fitted on THIS rotated window and expanded.
    # No lookahead or speculative score is used for pruning.
    let parents = rankBranches(rows, beams, cfg.beamWidth)
    if parents.len == 0: break
    let incumbent = parents[0]
    var additions: seq[Branch] = @[]
    var exchanges: seq[Branch] = @[]
    for parentIndex in 0..<parents.len:
      let parent = parents[parentIndex]
      let state = evaluate(rows, parent.rules, true)
      if parent.rules.len < cfg.maxRules:
        for b in attemptAddition(rows, parent, cfg, codec, state.enc, state.fit):
          additions.add(b)
      if parent.rules.len > 0:
        for b in attemptExchange(rows, parent, cfg, codec, state.enc, state.fit):
          exchanges.add(b)
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
    let initial = evaluate(rows, current.rules, true)
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
    writePlot(cfg.outDir)
    echo "[step] ", step, " trainPearson=", initial.fit.pearson,
      " validPearson=", valPearson, " -ln(1-rho)=", trainMetric,
      " rules=", current.rules.len, " branch=", current.description,
      " operator=", operatorName(current),
      " time=", elapsed, "s"

when isMainModule:
  main()
