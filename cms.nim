## Lens GA: fixed-length genomes of ordered rewrite rules.
## Bytes are literal 0..255; intermediate symbols are (256 + rule index).
## No Coarse/Fine screening, no token hashing, no rule mining or beam search.
import std/[os, strutils, math, algorithm, times, sequtils, tables]
import replace

type
  Config = object
    corpus, outDir, resume, virusResume: string
    chunks, chunkLen, trainChunks, validChunks, noiseSteps: int
    rules, generations, population, elites, archiveSize, evalEvery, plotEvery, maWindow: int
    rotateEvery, rotateStride, anchorChunks: int
    cacheEntries, cacheStride, cacheMb, blockCacheMb, blockCacheEntries: int
    diffusionSampling: string
    ridgeLambda: float64
    # Search accelerators: paired common-random-numbers screening,
    # multi-fidelity evaluation, small-scale refinement, residual guidance,
    # and learned/statistical linkage crossover.
    multifidelity, pairedSelection, localSearch, residualSearch, linkageCross: bool
    candidateFactor, stage2Chunks, stage3Chunks, stage1Noise, stage2Noise,
      stage3Noise, finalists, localTrials, residualCandidates,
      linkageSize: int
    virusEnabled: bool
    virusPopulation, virusTrials, virusPartnerSamples, virusInject: int
    seed: uint64
  Gene = object
    pat: seq[int]
    mode: int                   # 0 capture; 1 sort; 2 reverse; 3 inverse sort
  Genome = object
    genes: seq[Gene]
    score: float64 # score on CURRENT rotating training batch only
  ArchiveEntry = object
    genome: Genome
    referenceScore: float64 # score on FIXED reference rows for comparable history
  Row = object
    data: seq[int]
    target: float64
    sourceLen: int
    chunkId: int
  Model = object
    beta: seq[float64]
    intercept, trainPearson, looPearson: float64
  VOp = enum
    vConst, vFireRate, vMeanFire, vTargetCorr, vAbsTargetCorr,
    vMeanRedundancy, vMaxRedundancy, vContribution, vPosition,
    vPairCorr, vAbsPairCorr, vDistance,
    vAdd, vSub, vMul, vNeg, vAbs, vMin, vMax
  VNode = object
    op: VOp
    a, b: int
    constant: float64
  Virus = object
    targetDag, partnerDag, gateDag: seq[VNode]
    actionTrue, actionFalse, ngramLen: int
    threshold: float64
  VirusRecord = object
    virus: Virus
    trials: int
    advantage, improvement: float64
  ViralContext = object
    firing, meanFire, targetCorr, meanRedundancy, maxRedundancy,
      contribution: seq[float64]
    norm: seq[seq[float64]]
    probeStates: seq[seq[int]]   # state before rule i, last = final
  VirusTrial = object
    genome: Genome
    score, advantage: float64
    parentId: int
  Proposal = object
    genome, parent: Genome
    coarseDelta, mediumDelta, refinedDelta: float64
    originKind: string
    parentId: int
  SearchStats = object
    proposals, coarseEvaluations, mediumEvaluations,
      precisionEvaluations, accepted, localWins,
      residualWins, linkageWins, viralWins: int
    timeSeconds: float64
  Rng = object
    state: uint64
  FeatureCheckpoint = object
    ruleEnd: int                 # states are AFTER rules 0..<ruleEnd
    states: seq[seq[int]]       # immutable row states
  FeatureTrace = object
    genes: seq[Gene]            # exact genome identity; no hash collisions
    features: seq[seq[float64]] # all per-rule firing features
    checkpoints: seq[FeatureCheckpoint]
    bytes: int64
  BlockMemo = object
    ruleStart, ruleEnd: int
    genes: seq[Gene]          # immutable exact gene block
    input, output: seq[int]  # exact row state before/after the block
    counts: seq[int]         # raw firing counts (not length-normalized)
    bytes: int64
  FeatureCache = object
    entries: seq[FeatureTrace]  # least recently used first
    maxEntries, stride: int
    maxBytes, usedBytes: int64
    hits, exactHits, misses: int64
    savedRuleRows: int64        # number of rule x row applications avoided
    deadMarkerRuleRows: int64       # extra scans skipped by impossible-marker test
    blockEntries: Table[uint64, BlockMemo]   # candidate hash -> exact-verified entry
    blockKeys: seq[uint64]       # bounded FIFO eviction log
    blockKeyHead, maxBlockEntries: int
    maxBlockBytes, blockUsedBytes: int64
    blockHits, blockMisses, blockSavedRuleRows: int64

proc rnd(r: var Rng): uint64 {.inline.} =
  r.state = r.state xor (r.state shl 13)
  r.state = r.state xor (r.state shr 7)
  r.state = r.state xor (r.state shl 17)
  r.state

proc uniform(r: var Rng): float64 {.inline.} =
  float64(r.rnd() shr 11) * (1.0 / 9007199254740992.0)

proc randInt(r: var Rng, limit: int): int {.inline.} =
  if limit <= 1: return 0
  int(r.rnd() mod uint64(limit))

proc probability(r: var Rng, p: float64): bool {.inline.} =
  r.uniform() < p

proc parseIntOption(v: string, fallback: int): int =
  try: parseInt(v)
  except ValueError: fallback
proc parseFloatOption(v: string, fallback: float64): float64 =
  try: parseFloat(v)
  except ValueError: fallback

proc config(): Config =
  result = Config(corpus: "github-code.txt", outDir: "lens_ga_run",
    chunks: 65536, chunkLen: 2000, trainChunks: 2, validChunks: 2,
    noiseSteps: 16, rules: 1500, generations: 100000,
    population: 12, elites: 2, archiveSize: 8, evalEvery: 1,
    plotEvery: 1, maWindow: 75, rotateEvery: 1, rotateStride: 1, anchorChunks: 4,
    cacheEntries: 8, cacheStride: 64, cacheMb: 128,
    blockCacheMb: 128, blockCacheEntries: 8192, diffusionSampling: "random",
    ridgeLambda: 0.1,
    multifidelity: true, pairedSelection: true,
    localSearch: true, residualSearch: true, linkageCross: true,
    candidateFactor: 2, stage2Chunks: 4, stage3Chunks: 8,
    stage1Noise: 4, stage2Noise: 8, stage3Noise: 12,
    finalists: 3, localTrials: 3, residualCandidates: 12,
    linkageSize: 16,
    virusEnabled: true, virusPopulation: 10,
    virusTrials: 3, virusPartnerSamples: 24, virusInject: 2,
    seed: 1729'u64)
  for arg in commandLineParams():
    if arg in ["-h", "--help"]:
      echo "Lens fixed-rule GA (raw byte alphabet, sequential rewrite)"
      echo "--corpus=FILE --out=DIR --resume=FILE --chunks=N (0=all) --chunk-len=N"
      echo "--train=N --valid=N --noise-steps=N --rules=1500 --generations=N"
      echo "--population=12 --elites=2 --archive=8 --ridge-lambda=0.1"
      echo "--eval-every=1 --plot-every=1 --ma-window=10 (evaluated points)"
      echo "--cache-entries=8 --cache-stride=64 --cache-mb=128 (0=disable prefix)"
      echo "--block-cache-mb=128 --block-cache-entries=8192 (0=disable interval cache)"
      echo "--diffusion-sampling=random|uniform (random: unique interior steps, fixed endpoints)"
      echo "--bench-cache (compare cached/uncached equal-feature work)"
      echo "--bench-block-cache (compare identical work with divergent rule prefixes)"
      echo "--rotate-every=1 --rotate-stride=1 --anchor=4 (fixed reference, not held-out)"
      echo "--multifidelity=1 --paired=1 --candidate-factor=2 --finalists=3"
      echo "--stage2-chunks=4 --stage3-chunks=8 --stage1-noise=4"
      echo "--stage2-noise=8 --stage3-noise=12 (number of interior steps)"
      echo "--local-search=1 --local-trials=3 --residual-search=1"
      echo "--residual-candidates=12 --linkage-cross=1 --linkage-size=16"
      echo "--virus=1 (0=ordinary GA) --virus-population=10 --virus-trials=3"
      echo "--virus-partners=24 --virus-inject=2 --virus-resume=FILE --seed=1729 --self-test"
      echo "variation: 32 classical single-rule edits, 8 host crossovers, diff3"
      quit(0)
    if not arg.startsWith("--") or not arg.contains('='): continue
    let kv = arg[2..^1].split('=', 1)
    let v = kv[1]
    case kv[0]
    of "corpus": result.corpus = v
    of "out": result.outDir = v
    of "resume": result.resume = v
    of "virus-resume": result.virusResume = v
    of "chunks": result.chunks = parseIntOption(v, result.chunks)
    of "chunk-len": result.chunkLen = parseIntOption(v, result.chunkLen)
    of "train": result.trainChunks = parseIntOption(v, result.trainChunks)
    of "valid": result.validChunks = parseIntOption(v, result.validChunks)
    of "noise-steps": result.noiseSteps = parseIntOption(v, result.noiseSteps)
    of "rules": result.rules = parseIntOption(v, result.rules)
    of "generations", "steps": result.generations = parseIntOption(v, result.generations)
    of "population": result.population = parseIntOption(v, result.population)
    of "elites": result.elites = parseIntOption(v, result.elites)
    of "archive": result.archiveSize = parseIntOption(v, result.archiveSize)
    of "anchor": result.anchorChunks = parseIntOption(v, result.anchorChunks)
    of "rotate-every": result.rotateEvery = parseIntOption(v, result.rotateEvery)
    of "rotate-stride": result.rotateStride = parseIntOption(v, result.rotateStride)
    of "ridge-lambda": result.ridgeLambda = parseFloatOption(v, result.ridgeLambda)
    of "multifidelity": result.multifidelity = parseIntOption(v, 1) != 0
    of "paired": result.pairedSelection = parseIntOption(v, 1) != 0
    of "local-search": result.localSearch = parseIntOption(v, 1) != 0
    of "residual-search": result.residualSearch = parseIntOption(v, 1) != 0
    of "linkage-cross": result.linkageCross = parseIntOption(v, 1) != 0
    of "candidate-factor": result.candidateFactor = parseIntOption(v, result.candidateFactor)
    of "stage2-chunks": result.stage2Chunks = parseIntOption(v, result.stage2Chunks)
    of "stage3-chunks": result.stage3Chunks = parseIntOption(v, result.stage3Chunks)
    of "stage1-noise": result.stage1Noise = parseIntOption(v, result.stage1Noise)
    of "stage2-noise": result.stage2Noise = parseIntOption(v, result.stage2Noise)
    of "stage3-noise": result.stage3Noise = parseIntOption(v, result.stage3Noise)
    of "finalists": result.finalists = parseIntOption(v, result.finalists)
    of "local-trials": result.localTrials = parseIntOption(v, result.localTrials)
    of "residual-candidates": result.residualCandidates = parseIntOption(v, result.residualCandidates)
    of "linkage-size": result.linkageSize = parseIntOption(v, result.linkageSize)
    of "eval-every": result.evalEvery = parseIntOption(v, result.evalEvery)
    of "plot-every": result.plotEvery = parseIntOption(v, result.plotEvery)
    of "ma-window": result.maWindow = parseIntOption(v, result.maWindow)
    of "cache-entries": result.cacheEntries = parseIntOption(v, result.cacheEntries)
    of "cache-stride": result.cacheStride = parseIntOption(v, result.cacheStride)
    of "cache-mb": result.cacheMb = parseIntOption(v, result.cacheMb)
    of "block-cache-mb": result.blockCacheMb = parseIntOption(v, result.blockCacheMb)
    of "block-cache-entries": result.blockCacheEntries = parseIntOption(v, result.blockCacheEntries)
    of "diffusion-sampling": result.diffusionSampling = v
    of "virus": result.virusEnabled = parseIntOption(v, 1) != 0
    of "virus-population": result.virusPopulation = parseIntOption(v, result.virusPopulation)
    of "virus-trials": result.virusTrials = parseIntOption(v, result.virusTrials)
    of "virus-partners": result.virusPartnerSamples = parseIntOption(v, result.virusPartnerSamples)
    of "virus-inject": result.virusInject = parseIntOption(v, result.virusInject)
    of "seed": result.seed = uint64(max(1, parseIntOption(v, int(result.seed))))
    else: quit("Unknown option: --" & kv[0], 1)
  if result.rules < 1 or result.rules > 100_000 or result.population < 3 or
      result.elites < 1 or result.elites >= result.population or
      result.archiveSize < 1 or result.generations < 1 or
      result.trainChunks < 1 or result.validChunks < 1 or result.chunks < 0 or
      result.rotateEvery < 1 or result.rotateStride < 1 or result.anchorChunks < 1 or
      result.chunkLen < 2 or result.noiseSteps < 2 or result.evalEvery < 1 or
      result.cacheEntries < 0 or result.cacheEntries > 256 or
      result.cacheStride < 1 or result.cacheMb < 0 or result.cacheMb > 8192 or
      result.blockCacheMb < 0 or result.blockCacheMb > 8192 or
      result.blockCacheEntries < 0 or result.blockCacheEntries > 100000 or
      result.diffusionSampling notin ["random", "uniform"] or
      result.plotEvery < 0 or result.maWindow < 1 or result.virusPopulation < 3 or
      result.virusTrials < 1 or result.virusPartnerSamples < 1 or
      result.virusInject < 0 or result.virusInject >= result.population or
      result.candidateFactor notin 1..8 or result.stage2Chunks < 1 or
      result.stage3Chunks < 2 or result.stage1Noise < 2 or
      result.stage2Noise < 2 or result.stage3Noise < 2 or
      result.finalists < 1 or result.localTrials notin 0..64 or
      result.residualCandidates notin 1..256 or
      result.linkageSize notin 1..128 or
      result.ridgeLambda <= 0.0 or
      classify(result.ridgeLambda) in {fcNan, fcInf, fcNegInf}:
    quit("Invalid GA settings; ridge-lambda must be positive", 1)
  result.archiveSize = min(result.archiveSize, result.population)

proc loadCorpus(cfg: Config): seq[seq[int]] =
  if not fileExists(cfg.corpus): quit("Corpus missing: " & cfg.corpus, 1)
  let f = open(cfg.corpus, fmRead)
  defer: f.close()
  var current: seq[int] = @[]
  while not f.endOfFile and (cfg.chunks == 0 or result.len < cfg.chunks):
    let line = f.readLine()
    if line == "===SPLIT===":
      if current.len >= 2: result.add(current)
      current = @[]
    elif current.len < cfg.chunkLen:
      for ch in line:
        if current.len >= cfg.chunkLen: break
        current.add(ord(ch))
      if current.len < cfg.chunkLen: current.add(10)
  if current.len >= 2 and (cfg.chunks == 0 or result.len < cfg.chunks): result.add(current)
  if result.len < cfg.trainChunks + cfg.validChunks:
    quit("Not enough corpus chunks for training + held-out validation", 1)

proc observedAlphabet(source: seq[seq[int]]): seq[int] =
  var seen: array[256, bool]
  for chunk in source:
    for t in chunk:
      if not seen[t]:
        result.add(t)
        seen[t] = true
  if result.len < 2: quit("Diffusion requires at least two observed byte values", 1)

proc diffusionLevels(length, noiseSteps: int, r: var Rng,
                     sampling: string): seq[int] =
  ## Include intact and fully corrupted endpoints. Uniformly sample the
  ## interior WITHOUT replacement; the row targets remain exact fractions.
  result = @[0]
  if sampling == "uniform":
    var last = 0
    for stage in 1..<noiseSteps:
      let level = stage * length div noiseSteps
      if level > last and level < length:
        result.add(level)
        last = level
  else:
    let interiorCount = min(noiseSteps - 1, length - 1)
    var candidates = newSeq[int](length - 1)
    for i in 0..<candidates.len: candidates[i] = i + 1
    # Partial Fisher-Yates: unbiased distinct positions.
    for i in 0..<interiorCount:
      let j = i + r.randInt(candidates.len - i)
      swap(candidates[i], candidates[j])
      result.add(candidates[i])
    result.sort()
  result.add(length)

proc makeRows(source: seq[seq[int]], indices: openArray[int],
              noiseSteps: int, seed: uint64, observed: seq[int],
              sampling: string = "random"): seq[Row] =
  ## Stable per-chunk diffusion regardless of the CURRENT rotating batch.
  ## Global source index is used, never the batch-local index.
  for chunkId in indices:
    doAssert chunkId in 0..<source.len
    let original = source[chunkId]
    var r = Rng(state: seed xor (uint64(chunkId + 1) * 0x9E3779B97F4A7C15'u64))
    if r.state == 0: r.state = 0xA5A5A5A5'u64
    var positions = newSeq[int](original.len)
    for j in 0..<positions.len: positions[j] = j
    for j in countdown(positions.len - 1, 1):
      let k = r.randInt(j + 1)
      swap(positions[j], positions[k])
    var state = newSeq[int](original.len)
    for pos, tok in original: state[pos] = tok
    # Draw the entire corruption path before sampling observation steps.
    # Thus changing --stage*-noise does NOT alter any corruption value at a
    # fixed noise count; paired comparisons always share the same path.
    var replacements = newSeq[int](original.len)
    for t in 0..<original.len:
      let prior = original[positions[t]]
      var replacement = observed[r.randInt(observed.len - 1)]
      if replacement == prior: replacement = observed[^1]
      if replacement == prior: replacement = observed[0]
      replacements[t] = replacement
    let levels = diffusionLevels(original.len, noiseSteps, r, sampling)
    var last = 0
    for changed in levels:
      for t in last..<changed:
        state[positions[t]] = replacements[t]
      var snapshot = newSeq[int](state.len)
      for pos, tok in state: snapshot[pos] = tok
      result.add(Row(data: snapshot, target: 1.0 - float64(changed) / float64(original.len),
                     sourceLen: original.len, chunkId: chunkId))
      last = changed

proc shuffledTrainingIndices(trainingChunkCount: int, r: var Rng): seq[int] =
  ## One seeded permutation: visits every non-held-out source chunk before wrap.
  result = newSeq[int](trainingChunkCount)
  for i in 0..<trainingChunkCount: result[i] = i
  for i in countdown(trainingChunkCount - 1, 1):
    let j = r.randInt(i + 1)
    swap(result[i], result[j])

proc trainingIndices(order: seq[int], generation, count,
                     rotateEvery, rotateStride: int): seq[int] =
  ## A cyclic sliding window, advanced once per rotateEvery generations.
  ## For default stride=1 all training chunks get visited, including wraparound.
  doAssert order.len > 0 and count > 0
  let step = (generation div rotateEvery) mod order.len
  let start = (step * (rotateStride mod order.len)) mod order.len
  result = newSeq[int](count)
  for j in 0..<count:
    result[j] = order[(start + j) mod order.len]

proc referenceIndices(order: seq[int], count: int): seq[int] =
  ## Fixed stratified sample from the permutation, not held-out validation.
  let k = min(count, order.len)
  for i in 0..<k:
    result.add(order[i * order.len div k])

proc indexString(indices: openArray[int]): string =
  for j, id in indices:
    if j > 0: result.add(',')
    result.add($id)

proc copyGene(g: Gene): Gene =
  result.mode = g.mode
  result.pat = newSeq[int](g.pat.len)
  for i, x in g.pat: result.pat[i] = x

proc copyGenome(g: Genome): Genome =
  result.score = g.score
  result.genes = newSeq[Gene](g.genes.len)
  for i, x in g.genes: result.genes[i] = copyGene(x)

proc geneEqual(a, b: Gene): bool =
  a.mode == b.mode and a.pat == b.pat

proc genomeEqual(a, b: Genome): bool =
  if a.genes.len != b.genes.len: return false
  for i in 0..<a.genes.len:
    if not geneEqual(a.genes[i], b.genes[i]): return false
  true

proc validGene(g: Gene, index: int): bool =
  if g.pat.len == 0 or g.pat.len > 12 or g.mode notin 0..3: return false
  if g.pat[0] < 0 or g.pat[^1] < 0: return false
  var wild = 0
  for x in g.pat:
    if x < 0:
      inc wild
      if wild > 2 or x != -wild: return false
    elif x >= 256 + index:
      return false  # cannot depend on later, not-yet-created markers
  true

proc canonicalRhs(g: Gene, index: int): seq[int] =
  result = @[256 + index]
  var wild = 0
  for x in g.pat:
    if x < 0:
      let base = (case g.mode
        of 0: -1
        of 1: -16
        of 2: -32
        else: -48)
      result.add(base - wild)
      inc wild

proc randomLiteral(source: seq[seq[int]], r: var Rng, index: int): int =
  if index > 0 and r.probability(0.22):
    return 256 + r.randInt(index)
  let chunk = source[r.randInt(source.len)]
  chunk[r.randInt(chunk.len)]

proc newGene(source: seq[seq[int]], r: var Rng, index: int): Gene =
  let chunk = source[r.randInt(source.len)]
  let n = (if r.probability(0.40): 1
           elif r.probability(0.55): 2
           else: 3 + r.randInt(4))
  let start = r.randInt(max(1, chunk.len - n + 1))
  for j in 0..<min(n, chunk.len - start):
    result.pat.add(chunk[start + j])
  if index > 0 and r.probability(0.20):
    let where = r.randInt(result.pat.len)
    result.pat[where] = 256 + r.randInt(index)
  if result.pat.len >= 2 and r.probability(0.22):
    result.pat.insert(-1, 1 + r.randInt(result.pat.len - 1))
  result.mode = (if result.pat.anyIt(it < 0): r.randInt(4) else: 0)
  doAssert validGene(result, index)

proc normalizeWildcards(g: var Gene) =
  var n = 0
  for i in 0..<g.pat.len:
    if g.pat[i] < 0:
      inc n
      g.pat[i] = -n

# A shared, evolvable edit vocabulary. Every operation changes ONE rule;
# the caller decides how many distinct loci to touch (log-uniform K).
const
  RuleMaxLen = 12
  MutationOps = 32
  MutationNames: array[MutationOps, string] = [
    "regenerate", "substitute-one", "insert-one", "delete-one",
    "swap-two", "reverse-substring", "rotate-left", "rotate-right",
    "duplicate-one", "duplicate-substring", "delete-substring",
    "replace-donor-substring", "insert-donor-substring",
    "append-donor-suffix", "prepend-donor-prefix", "one-point-splice",
    "two-point-splice", "interleave-donor", "swap-halves",
    "swap-disjoint-blocks", "copy-donor", "insert-wildcard",
    "delete-wildcard", "move-wildcard", "change-capture-mode",
    "substitute-donor-position", "adjacent-copy", "rotate-substring",
    "introduce-prior-marker", "insert-corpus-ngram", "truncate-pattern",
    "mirror-substring"
  ]


proc shallowSegment(p: seq[int], start, count: int): seq[int] =
  # Never index outside the sequence, and do not retain mutable shared buffers.
  let a = max(0, min(p.len, start))
  let b = max(a, min(p.len, a+max(0,count)))
  for k in a..<b: result.add(p[k])

proc appendLimited(dst: var seq[int], src: seq[int]) =
  for x in src:
    if dst.len >= RuleMaxLen: break
    dst.add(x)

proc spliceGene(a, b: Gene, index: int, r: var Rng): Gene =
  ## Two-parent, per-gene cross: valid index-aligned marker vocabulary.
  result = copyGene(a)
  if a.pat.len == 0 or b.pat.len == 0: return
  var joined: seq[int] = @[]
  let cutA = r.randInt(a.pat.len+1)
  let cutB = r.randInt(b.pat.len+1)
  appendLimited(joined, shallowSegment(a.pat,0,cutA))
  appendLimited(joined, shallowSegment(b.pat,cutB,b.pat.len-cutB))
  if joined.len > 0:
    result.pat = joined
    normalizeWildcards(result)
    result.mode = (if r.probability(0.5): a.mode else: b.mode)
    if not result.pat.anyIt(it < 0): result.mode = 0
  if not validGene(result,index):
    result = copyGene((if r.probability(0.5): a else: b))

proc editGene(parent, donor: Gene, source: seq[seq[int]],
              r: var Rng, index, requestedOp: int,
              motif: seq[int] = @[]): Gene =
  ## Classic variation operators shared by GA and programmable viruses.
  ## Each attempt validates marker dependencies, wildcard labels, endpoints,
  ## capture mode and pattern length before accepting a change.
  for attempt in 0..<7:
    result = copyGene(parent)
    let op = (if attempt == 0: requestedOp mod MutationOps
              else: r.randInt(MutationOps))
    let n = result.pat.len
    let dn = donor.pat.len
    case op
    of 0:
      result = newGene(source,r,index)
    of 1: # Substitute exactly one token
      result.pat[r.randInt(n)] = randomLiteral(source,r,index)
    of 2: # Insert exactly one token, at any position
      if n < RuleMaxLen: result.pat.insert(randomLiteral(source,r,index),r.randInt(n+1))
    of 3: # Delete exactly one token
      if n > 1: result.pat.delete(r.randInt(n))
    of 4: # Swap arbitrary tokens
      if n > 1:
        let firstPos = r.randInt(n)
        let secondPos = r.randInt(n)
        swap(result.pat[firstPos], result.pat[secondPos])
    of 5: # Reverse a substring
      if n > 1:
        let a = r.randInt(n-1)
        let b = a+1+r.randInt(n-a-1)
        for j in 0..<((b-a+1) div 2):
          swap(result.pat[a+j],result.pat[b-j])
    of 6: # Circular left shift of whole pattern
      if n > 1:
        let first = result.pat[0]
        result.pat.delete(0)
        result.pat.add(first)
    of 7: # Circular right shift of whole pattern
      if n > 1:
        let last = result.pat[^1]
        result.pat.delete(n-1)
        result.pat.insert(last,0)
    of 8: # Duplicate a single symbol
      if n < RuleMaxLen:
        let a = r.randInt(n)
        result.pat.insert(result.pat[a],a)
    of 9: # Duplicate a contiguous region
      if n < RuleMaxLen:
        let a = r.randInt(n)
        let size = 1+r.randInt(min(n-a,RuleMaxLen-n))
        let seg = shallowSegment(result.pat,a,size)
        var rebuilt: seq[int] = @[]
        appendLimited(rebuilt,shallowSegment(result.pat,0,a+size))
        appendLimited(rebuilt,seg)
        appendLimited(rebuilt,shallowSegment(result.pat,a+size,n-a-size))
        result.pat = rebuilt
    of 10: # Remove contiguous region
      if n > 1:
        let a = r.randInt(n)
        let size = 1+r.randInt(min(n-a,n-1))
        for j in 0..<size: result.pat.delete(a)
    of 11: # Replace a region with a donor region
      if dn > 0:
        let a = r.randInt(n)
        let count = 1+r.randInt(n-a)
        let b = r.randInt(dn)
        var rebuilt: seq[int] = @[]
        appendLimited(rebuilt,shallowSegment(result.pat,0,a))
        appendLimited(rebuilt,shallowSegment(donor.pat,b,1+r.randInt(dn-b)))
        appendLimited(rebuilt,shallowSegment(result.pat,a+count,n-a-count))
        result.pat = rebuilt
    of 12: # Insert a donor region
      if dn > 0 and n < RuleMaxLen:
        let a = r.randInt(n+1)
        let b = r.randInt(dn)
        let size = 1+r.randInt(min(dn-b,RuleMaxLen-n))
        var rebuilt: seq[int] = @[]
        appendLimited(rebuilt,shallowSegment(result.pat,0,a))
        appendLimited(rebuilt,shallowSegment(donor.pat,b,size))
        appendLimited(rebuilt,shallowSegment(result.pat,a,n-a))
        result.pat = rebuilt
    of 13: # Concatenate a donor suffix
      let b = r.randInt(dn)
      appendLimited(result.pat,shallowSegment(donor.pat,b,dn-b))
    of 14: # Concatenate a donor prefix
      var rebuilt = shallowSegment(donor.pat,0,1+r.randInt(dn))
      appendLimited(rebuilt,result.pat)
      result.pat = shallowSegment(rebuilt,0,RuleMaxLen)
    of 15: # Classical one-point crossover at gene level
      result = spliceGene(parent,donor,index,r)
    of 16: # Classical two-point splice
      let a = r.randInt(n)
      let b = a+1+r.randInt(n-a)
      let c = r.randInt(dn)
      var rebuilt: seq[int] = @[]
      appendLimited(rebuilt,shallowSegment(result.pat,0,a))
      appendLimited(rebuilt,shallowSegment(donor.pat,c,1+r.randInt(dn-c)))
      appendLimited(rebuilt,shallowSegment(result.pat,b,n-b))
      result.pat = rebuilt
    of 17: # Interleave symbols from two patterns
      var rebuilt: seq[int] = @[]
      let how = max(n,dn)
      for j in 0..<how:
        if j < n and rebuilt.len < RuleMaxLen: rebuilt.add(result.pat[j])
        if j < dn and rebuilt.len < RuleMaxLen: rebuilt.add(donor.pat[j])
      result.pat = rebuilt
    of 18: # Swap left and right halves
      if n > 1:
        let cut = 1+r.randInt(n-1)
        result.pat = shallowSegment(parent.pat,cut,n-cut) & shallowSegment(parent.pat,0,cut)
    of 19: # Swap two non-overlapping one-token blocks with intact context
      if n >= 3:
        let a = r.randInt(n-2)
        let b = a+1+r.randInt(n-a-1)
        result.pat = shallowSegment(parent.pat,0,a) &
          shallowSegment(parent.pat,b,1) & shallowSegment(parent.pat,a+1,b-a-1) &
          shallowSegment(parent.pat,a,1) & shallowSegment(parent.pat,b+1,n-b-1)
    of 20: # Copy entire donor rule
      result = copyGene(donor)
    of 21: # Introduce capture wildcard at internal position
      if n >= 2 and n < RuleMaxLen:
        result.pat.insert(-1,1+r.randInt(n-1))
    of 22: # Remove a wildcard, replacing with a literal
      for j in 0..<n:
        if result.pat[j] < 0:
          result.pat[j] = randomLiteral(source,r,index)
          break
    of 23: # Move a wildcard to a new internal position
      var wildAt = -1
      for j in 0..<n:
        if result.pat[j] < 0: wildAt = j; break
      if wildAt >= 0 and n >= 3:
        result.pat.delete(wildAt)
        result.pat.insert(-1,1+r.randInt(n-2))
    of 24: # Change sort/reverse/inverse-sort capture behavior
      if result.pat.anyIt(it < 0): result.mode = (result.mode+1+r.randInt(3)) mod 4
    of 25: # Replace a token from corresponding donor position
      let j = r.randInt(n)
      result.pat[j] = donor.pat[j mod dn]
    of 26: # Replace with adjacent symbol
      if n > 1:
        let j = r.randInt(n)
        let neighbor = (if j == n-1: j-1 else: j+1)
        result.pat[j] = result.pat[neighbor]
    of 27: # Rotate a bounded substring
      if n >= 3:
        let a = r.randInt(n-2)
        let b = a+2+r.randInt(n-a-2)
        let first = result.pat[a]
        result.pat.delete(a)
        result.pat.insert(first,b)
    of 28: # Introduce an earlier rule's intermediate symbol
      if index > 0:
        result.pat[r.randInt(n)] = 256+r.randInt(index)
    of 29: # Insert a corpus n-gram, or context motif
      var fragment = motif
      if fragment.len == 0:
        let chunk = source[r.randInt(source.len)]
        let a = r.randInt(chunk.len)
        fragment = shallowSegment(chunk,a,1+r.randInt(min(4,chunk.len-a)))
      let a = r.randInt(n+1)
      var rebuilt: seq[int] = @[]
      appendLimited(rebuilt,shallowSegment(result.pat,0,a))
      appendLimited(rebuilt,fragment)
      appendLimited(rebuilt,shallowSegment(result.pat,a,n-a))
      result.pat = rebuilt
    of 30: # Drop one end (substring contraction)
      if n > 1:
        if r.probability(0.5): result.pat.delete(0)
        else: result.pat.delete(n-1)
    of 31: # Mirror substring / palindromic extension
      if n < RuleMaxLen:
        let a = r.randInt(n)
        let size = 1+r.randInt(min(n-a,RuleMaxLen-n))
        for j in countdown(a+size-1,a): result.pat.add(parent.pat[j])
    else: discard
    if result.pat.len > RuleMaxLen:
      result.pat = shallowSegment(result.pat,0,RuleMaxLen)
    normalizeWildcards(result)
    if not result.pat.anyIt(it < 0): result.mode = 0
    if validGene(result,index) and not geneEqual(result,parent): return
  # Final guaranteed nontrivial, valid fallback on even degenerate corpora.
  result = newGene(source,r,index)
  if geneEqual(result,parent):
    result = copyGene(parent)
    result.pat[0] = (result.pat[0]+1) mod 256
  doAssert validGene(result,index) and not geneEqual(result,parent)

proc mutateGene(parent: Gene, source: seq[seq[int]],
                r: var Rng, index: int): Gene =
  let donor = newGene(source,r,index)
  editGene(parent,donor,source,r,index,r.randInt(MutationOps))

proc mutationCount(r: var Rng, length: int): int =
  ## EXACT requested distribution: floor(exp(U(0, log(LEN)))).
  ## U is sampled once per offspring; LEN = number of rules.
  if length <= 1: return max(0, length)
  result = min(length, max(1, int(floor(exp(r.uniform() * ln(float64(length)))))))

proc mutateExactly(g: var Genome, source: seq[seq[int]],
                   r: var Rng, requested: int) =
  ## Exact, non-repeating mutation count; also serves as paired viral control.
  let length = g.genes.len
  let total = min(length, max(0, requested))
  var order = newSeq[int](length)
  for i in 0..<length: order[i] = i
  for k in 0..<total:
    let j = k + r.randInt(length - k)
    swap(order[k], order[j])
    let at = order[k]
    # Ordinary GA now also has horizontal rule-to-rule transduction:
    # earlier donor rules may be used as edit material at this locus.
    let donor = (if at > 0 and r.probability(0.65):
                   copyGene(g.genes[r.randInt(at)])
                 else: newGene(source,r,at))
    let current = copyGene(g.genes[at])
    g.genes[at] = editGene(current,donor,source,r,at,r.randInt(MutationOps))
  g.score = -Inf

proc simultaneousMutation(g: var Genome, source: seq[seq[int]],
                          r: var Rng): int =
  ## K = floor(exp(U(0, log(LEN)))) and K unique genes.
  result = mutationCount(r, g.genes.len)
  mutateExactly(g, source, r, result)

proc twoParentCrossover(mother, father: Genome, r: var Rng): Genome =
  ## Multi-scale crossing: block, runs, uniform, k-point, alternating,
  ## biased, gene-level one-point and gene-level two-point recombination.
  ## Parent identifiers are deliberately distinct from numeric splice bounds.
  let ruleCount = mother.genes.len
  doAssert ruleCount > 0 and father.genes.len == ruleCount
  result.genes = newSeq[Gene](ruleCount)
  let methodId = r.randInt(8)
  let segmentStart = r.randInt(ruleCount)
  let segmentEnd = segmentStart + r.randInt(ruleCount - segmentStart)
  var pickFather = r.probability(0.5)
  let switchChance = 1.0 / sqrt(float64(ruleCount))
  var cutPositions: seq[int] = @[]
  if methodId == 3:
    for j in 0..<3:
      cutPositions.add(r.randInt(ruleCount))
    cutPositions.sort()
  let blockSize = 1 + r.randInt(max(1, int(sqrt(float64(ruleCount)))))
  let fatherProbability = 0.10 + 0.80 * r.uniform()
  for locus in 0..<ruleCount:
    var takeFather: bool
    case methodId
    of 0:
      takeFather = locus >= segmentStart and locus <= segmentEnd
    of 1:
      if locus > 0 and r.probability(switchChance):
        pickFather = not pickFather
      takeFather = pickFather
    of 2:
      takeFather = r.probability(0.5)
    of 3:
      if locus in cutPositions:
        pickFather = not pickFather
      takeFather = pickFather
    of 4:
      takeFather = ((locus div blockSize) mod 2) == 1
    of 5:
      takeFather = r.probability(fatherProbability)
    of 6, 7:
      if r.probability(0.42):
        if methodId == 6:
          result.genes[locus] = spliceGene(mother.genes[locus], father.genes[locus], locus, r)
        else:
          result.genes[locus] = spliceGene(father.genes[locus], mother.genes[locus], locus, r)
      else:
        result.genes[locus] = copyGene(
          (if r.probability(0.5): mother.genes[locus] else: father.genes[locus]))
      continue
    else:
      takeFather = false
    if takeFather:
      result.genes[locus] = copyGene(father.genes[locus])
    else:
      result.genes[locus] = copyGene(mother.genes[locus])
  result.score = -Inf

proc diff3Crossover(base, left, right: Genome): Genome =
  ## Per-locus true three-way merge for an index-aligned rule genome.
  ## Non-conflicting edits from either branch are accepted; conflicts
  ## revert to the base, rather than randomly choosing a conflicting parent.
  result.genes = newSeq[Gene](base.genes.len)
  for i in 0..<base.genes.len:
    let a = base.genes[i]
    let b = left.genes[i]
    let c = right.genes[i]
    if geneEqual(b, c):
      result.genes[i] = copyGene(b)
    elif geneEqual(a, b):
      result.genes[i] = copyGene(c)
    elif geneEqual(a, c):
      result.genes[i] = copyGene(b)
    else:
      result.genes[i] = copyGene(a)
  result.score = -Inf

proc ridgeSolve(matrix: seq[float64], n: int,
                rhs: seq[float64]): tuple[alpha, inverseDiagonal: seq[float64]] =
  ## SPD solve via Cholesky (ridge guarantees positive definiteness).
  var lower = newSeq[float64](n * n)
  for i in 0..<n:
    for j in 0..i:
      var v = matrix[i*n + j]
      for k in 0..<j: v -= lower[i*n+k] * lower[j*n+k]
      if i == j:
        lower[i*n+j] = sqrt(max(v, 1e-12))
      else:
        lower[i*n+j] = v / lower[j*n+j]
  var tmp = newSeq[float64](n)
  result.alpha = newSeq[float64](n)
  result.inverseDiagonal = newSeq[float64](n)
  for i in 0..<n:
    var v = rhs[i]
    for j in 0..<i: v -= lower[i*n+j] * tmp[j]
    tmp[i] = v / lower[i*n+i]
  for i in countdown(n-1, 0):
    var v = tmp[i]
    for j in i+1..<n: v -= lower[j*n+i] * result.alpha[j]
    result.alpha[i] = v / lower[i*n+i]
  # Diagonal of inverse matrix from forward solutions L^-1 e_i.
  for i in 0..<n:
    var col = newSeq[float64](n)
    for k in i..<n:
      var v = (if k == i: 1.0 else: 0.0)
      for j in i..<k: v -= lower[k*n+j] * col[j]
      col[k] = v / lower[k*n+k]
      result.inverseDiagonal[i] += col[k]*col[k]


proc ridgeAlpha(matrix: seq[float64], n: int,
                rhs: seq[float64]): seq[float64] =
  ## Group CV needs one solve, not the entire inverse diagonal of LOO.
  var lower = newSeq[float64](n*n)
  for i in 0..<n:
    for j in 0..i:
      var value = matrix[i*n+j]
      for k in 0..<j: value -= lower[i*n+k]*lower[j*n+k]
      if i == j: lower[i*n+j] = sqrt(max(value,1e-12))
      else: lower[i*n+j] = value/lower[j*n+j]
  var tmp = newSeq[float64](n)
  result = newSeq[float64](n)
  for i in 0..<n:
    var value = rhs[i]
    for j in 0..<i: value -= lower[i*n+j]*tmp[j]
    tmp[i] = value/lower[i*n+i]
  for i in countdown(n-1,0):
    var value = tmp[i]
    for j in i+1..<n: value -= lower[j*n+i]*result[j]
    result[i] = value/lower[i*n+i]

proc pearson(a, b: seq[float64]): float64 =
  if a.len < 2 or b.len != a.len: return 0.0
  var ma, mb = 0.0
  for i in 0..<a.len:
    ma += a[i]
    mb += b[i]
  ma /= float64(a.len)
  mb /= float64(a.len)
  var xy, xx, yy = 0.0
  for i in 0..<a.len:
    let x = a[i] - ma
    let y = b[i] - mb
    xy += x*y
    xx += x*x
    yy += y*y
  if xx <= 1e-25 or yy <= 1e-25: return 0.0
  max(-1.0, min(1.0, xy / sqrt(xx*yy)))

# Exact shared-prefix memoization. A checkpoint stores the post-rewrite state
# for every diffusion row. Later genomes reuse it only if ALL preceding genes
# are identical AND the dataset matches the cache's assigned row cohort.
# Cache lifetime is controlled by the caller: rotating rows clear on rotation;
# fixed-reference and held-out cohorts have separate, persistent caches.
proc newFeatureCache(entries, stride, mb: int,
                     blockMb: int = 0, blockLimit: int = 8192): FeatureCache =
  result.maxEntries = entries
  result.stride = max(1, stride)
  result.maxBytes = int64(mb) * 1024'i64 * 1024'i64
  result.maxBlockBytes = int64(blockMb) * 1024'i64 * 1024'i64
  result.maxBlockEntries = blockLimit
  result.blockEntries = initTable[uint64, BlockMemo]()

proc clearFeatureCache(cache: var FeatureCache) =
  cache.entries.setLen(0)
  cache.usedBytes = 0
  cache.blockEntries.clear()
  cache.blockKeys.setLen(0)
  cache.blockKeyHead = 0
  cache.blockUsedBytes = 0

proc commonPrefix(a, b: seq[Gene]): int =
  while result < min(a.len, b.len) and geneEqual(a[result], b[result]):
    inc result

proc traceSize(t: FeatureTrace): int64 =
  for g in t.genes:
    result += int64(g.pat.len * sizeof(int) + sizeof(int))
  for column in t.features:
    result += int64(column.len * sizeof(float64))
  for cp in t.checkpoints:
    for state in cp.states:
      result += int64(state.len * sizeof(int))
  result += int64(t.checkpoints.len * 64 + t.features.len * 24)

proc storeTrace(cache: var FeatureCache, trace: FeatureTrace) =
  if cache.maxEntries == 0 or cache.maxBytes == 0: return
  if trace.bytes > cache.maxBytes: return
  while cache.entries.len >= cache.maxEntries or
      cache.usedBytes + trace.bytes > cache.maxBytes:
    if cache.entries.len == 0: break
    cache.usedBytes -= cache.entries[0].bytes
    cache.entries.delete(0)
  cache.usedBytes += trace.bytes
  cache.entries.add(trace)

# Interval memoization is valid only if BOTH the exact program segment and
# its exact INPUT state match. This permits suffix reuse after non-prefix
# edits which happen not to alter particular rows. The hash is ONLY an index:
# collisions are verified against full rules and full input before reuse.
proc mixBlockHash(h: var uint64, value: int) {.inline.} =
  h = (h xor cast[uint64](int64(value))) * 0x100000001b3'u64

proc blockGeneHash(genes: seq[Gene], first, after: int): uint64 =
  result = 0xcbf29ce484222325'u64
  mixBlockHash(result, first)
  mixBlockHash(result, after)
  for j in first..<after:
    mixBlockHash(result, genes[j].mode)
    mixBlockHash(result, genes[j].pat.len)
    for tok in genes[j].pat: mixBlockHash(result, tok)

proc blockRowHash(geneHash: uint64, input: seq[int]): uint64 =
  result = geneHash
  mixBlockHash(result, input.len)
  for tok in input: mixBlockHash(result, tok)

proc sameBlockGenes(genomeGenes: seq[Gene], first: int,
                    saved: seq[Gene]): bool =
  if first + saved.len > genomeGenes.len: return false
  for k, gene in saved:
    if not geneEqual(genomeGenes[first + k], gene): return false
  true

proc blockSizeBytes(m: BlockMemo): int64 =
  result = int64((m.input.len + m.output.len + m.counts.len) * sizeof(int) + 128)
  for gene in m.genes:
    result += int64(gene.pat.len * sizeof(int) + 24)

proc lookupBlock(cache: var FeatureCache, key: uint64,
                 genomeGenes: seq[Gene], first, after: int,
                 input: seq[int], output: var BlockMemo): bool =
  if not cache.blockEntries.hasKey(key):
    inc cache.blockMisses
    return false
  let entry = cache.blockEntries[key]
  if entry.ruleStart == first and entry.ruleEnd == after and
      entry.input == input and sameBlockGenes(genomeGenes, first, entry.genes):
    output = entry
    inc cache.blockHits
    cache.blockSavedRuleRows += int64(after - first)
    return true
  # Hash collision is safely treated as a miss; never reuse the wrong trace.
  inc cache.blockMisses
  false

proc storeBlock(cache: var FeatureCache, key: uint64, memo: BlockMemo) =
  if cache.maxBlockBytes <= 0 or cache.maxBlockEntries <= 0: return
  if memo.bytes > cache.maxBlockBytes or cache.blockEntries.hasKey(key): return
  while cache.blockEntries.len >= cache.maxBlockEntries or
      cache.blockUsedBytes + memo.bytes > cache.maxBlockBytes:
    if cache.blockKeyHead >= cache.blockKeys.len: return
    let oldest = cache.blockKeys[cache.blockKeyHead]
    inc cache.blockKeyHead
    if cache.blockEntries.hasKey(oldest):
      cache.blockUsedBytes -= cache.blockEntries[oldest].bytes
      cache.blockEntries.del(oldest)
  cache.blockEntries[key] = memo
  cache.blockKeys.add(key)
  cache.blockUsedBytes += memo.bytes
  # Compact FIFO metadata periodically; no unbounded eviction log.
  if cache.blockKeyHead > 4096 and cache.blockKeyHead * 2 > cache.blockKeys.len:
    if cache.blockKeyHead >= cache.blockKeys.len:
      cache.blockKeys.setLen(0)
    else:
      cache.blockKeys = cache.blockKeys[cache.blockKeyHead..^1]
    cache.blockKeyHead = 0

proc getFeatures(rows: seq[Row], g: Genome,
                 cache: var FeatureCache): seq[seq[float64]] =
  ## Cache only exact common prefixes, never independent suffixes: an edit at
  ## locus i can change EVERY subsequent rewrite state and firing count.
  let n = rows.len
  let p = g.genes.len
  let enabled = cache.maxEntries > 0 and cache.maxBytes > 0
  var reuseEntry = -1
  var checkpointIndex = -1
  var startRule = 0
  if enabled:
    for e in 0..<cache.entries.len:
      let prefix = commonPrefix(g.genes, cache.entries[e].genes)
      if prefix == p:
        inc cache.hits
        inc cache.exactHits
        cache.savedRuleRows += int64(p) * int64(n)
        let trace = cache.entries[e]
        cache.entries.delete(e)
        cache.entries.add(trace) # LRU refresh
        return trace.features
      # Use the most advanced VALID checkpoint, not simply the longest
      # prefix (two traces may have different checkpoint histories).
      if prefix > startRule:
        for k, cp in cache.entries[e].checkpoints:
          if cp.ruleEnd <= prefix and cp.ruleEnd > startRule:
            reuseEntry = e
            checkpointIndex = k
            startRule = cp.ruleEnd
  var states = newSeq[seq[int]](n)
  result = newSeq[seq[float64]](p)
  if reuseEntry >= 0:
    inc cache.hits
    cache.savedRuleRows += int64(startRule) * int64(n)
    let trace = cache.entries[reuseEntry]
    for j in 0..<startRule: result[j] = trace.features[j]
    # applyAndCount never edits the input buffer in place; it replaces the
    # seq reference after generating a new buffer. Stored states remain frozen.
    for i in 0..<n:
      states[i] = trace.checkpoints[checkpointIndex].states[i]
    cache.entries.delete(reuseEntry)
    cache.entries.add(trace)
  else:
    inc cache.misses
    for i, row in rows: states[i] = row.data

  var produced = newSeq[bool](p)
  # A marker 256+j can appear only if rule j fired on some prior row;
  # source rows contain only byte values 0..255. Restore this fact from
  # saved feature columns when replaying a cached prefix.
  for j in 0..<startRule:
    for v in result[j]:
      if v > 0.0:
        produced[j] = true
        break
  var checkpoints: seq[FeatureCheckpoint] = @[]
  # If a prefix was reused, also share its checkpoint snapshots with the new
  # trace. Only immutable snapshots are shared; mutation happens on new output.
  if reuseEntry >= 0 and enabled:
    let previous = cache.entries[^1]
    for cp in previous.checkpoints:
      if cp.ruleEnd <= startRule: checkpoints.add(cp)
  if cache.maxBlockBytes > 0 and cache.maxBlockEntries > 0:
    # Work one checkpoint-sized interval at a time. Memoize independently
    # PER ROW; divergent earlier rules can still converge on identical inputs.
    var first = startRule
    while first < p:
      let after = min(p, ((first div cache.stride) + 1) * cache.stride)
      let geneHash = blockGeneHash(g.genes, first, after)
      for j in first..<after: result[j] = newSeq[float64](n)
      var programs = newSeq[SeqReplaceProgram](after - first)
      var compiled = newSeq[bool](after - first)
      var ruleCopy: seq[Gene] = @[]
      for i in 0..<n:
        let prior = states[i]
        let key = blockRowHash(geneHash, prior)
        var memo: BlockMemo
        if lookupBlock(cache, key, g.genes, first, after, prior, memo):
          for j in first..<after:
            result[j][i] = float64(memo.counts[j-first]) / float64(rows[i].sourceLen)
            if memo.counts[j-first] > 0: produced[j] = true
          states[i] = memo.output
          continue
        var counts = newSeq[int](after-first)
        for j in first..<after:
          var impossible = false
          for tok in g.genes[j].pat:
            # All intermediate markers originate in earlier rules of the
            # SAME row. Test only that row's prior firing counts.
            if tok >= 256:
              let producer = tok - 256
              if producer >= j:
                impossible = true
                break
              if producer < first:
                if result[producer][i] == 0.0:
                  impossible = true
                  break
              elif counts[producer-first] == 0:
                impossible = true
                break
          if impossible:
            inc cache.deadMarkerRuleRows
          else:
            let local = j - first
            if not compiled[local]:
              programs[local] = compileSeqReplaceProgram(
                g.genes[j].pat, canonicalRhs(g.genes[j], j), 1)
              compiled[local] = true
            counts[local] = applyAndCount(states[i], programs[local])
          result[j][i] = float64(counts[j-first]) / float64(rows[i].sourceLen)
          if counts[j-first] > 0: produced[j] = true
        if ruleCopy.len == 0:
          ruleCopy = newSeq[Gene](after-first)
          for j in first..<after: ruleCopy[j-first] = copyGene(g.genes[j])
        memo = BlockMemo(ruleStart: first, ruleEnd: after,
                         genes: ruleCopy, input: prior, output: states[i],
                         counts: counts)
        memo.bytes = blockSizeBytes(memo)
        storeBlock(cache, key, memo)
      if enabled and after mod cache.stride == 0 and after < p:
        var snap = FeatureCheckpoint(ruleEnd: after)
        snap.states = newSeq[seq[int]](n)
        for i in 0..<n: snap.states[i] = states[i]
        checkpoints.add(snap)
      first = after
  else:
    # Identical historical path when --block-cache-mb=0.
    for j in startRule..<p:
      let gene = g.genes[j]
      result[j] = newSeq[float64](n)
      var impossible = false
      for tok in gene.pat:
        if tok >= 256 and not produced[tok - 256]:
          impossible = true
          break
      if impossible:
        cache.deadMarkerRuleRows += int64(n)
      else:
        var prog = compileSeqReplaceProgram(gene.pat, canonicalRhs(gene, j), 1)
        for i in 0..<n:
          let count = applyAndCount(states[i], prog)
          if count > 0: produced[j] = true
          result[j][i] = float64(count) / float64(rows[i].sourceLen)
      if enabled and (j+1) mod cache.stride == 0 and j+1 < p:
        var snap = FeatureCheckpoint(ruleEnd: j+1)
        snap.states = newSeq[seq[int]](n)
        for i in 0..<n: snap.states[i] = states[i]
        checkpoints.add(snap)
  if enabled:
    var trace: FeatureTrace
    trace.genes = newSeq[Gene](p)
    for j in 0..<p: trace.genes[j] = copyGene(g.genes[j])
    trace.features = result
    trace.checkpoints = move(checkpoints)
    trace.bytes = traceSize(trace)
    storeTrace(cache, trace)

proc getFeatures(rows: seq[Row], g: Genome): seq[seq[float64]] =
  # The uncached wrapper is used by baseline self-tests and optional callers.
  var disabled = newFeatureCache(0, 64, 0)
  result = getFeatures(rows, g, disabled)

proc getFeaturesReference(rows: seq[Row], g: Genome): seq[seq[float64]] =
  ## Unoptimized serial implementation used only by self-tests. This must
  ## remain independent of both prefix caching and marker-based pruning.
  let n = rows.len
  var states = newSeq[seq[int]](n)
  for i, row in rows: states[i] = row.data
  result = newSeq[seq[float64]](g.genes.len)
  for j, gene in g.genes:
    var program = compileSeqReplaceProgram(gene.pat, canonicalRhs(gene, j), 1)
    result[j] = newSeq[float64](n)
    for i in 0..<n:
      let count = applyAndCount(states[i], program)
      result[j][i] = float64(count) / float64(rows[i].sourceLen)

proc fit(rows: seq[Row], genome: Genome, ridgeLambda: float64,
         cache: var FeatureCache, fullModel: bool = true): Model =
  let n = rows.len
  let features = getFeatures(rows, genome, cache)
  let p = features.len
  if n < 3:
    result.beta = newSeq[float64](p)
    return
  var y = newSeq[float64](n)
  var meanY = 0.0
  for i, row in rows:
    y[i] = row.target
    meanY += y[i]
  meanY /= float64(n)
  var means = newSeq[float64](p)
  var scales = newSeq[float64](p)
  var standardized = newSeq[seq[float64]](p)
  var kernel = newSeq[float64](n * n)
  for j in 0..<p:
    var mean = 0.0
    for x in features[j]: mean += x
    mean /= float64(n)
    means[j] = mean
    var sq = 0.0
    for x in features[j]: sq += (x - mean)*(x - mean)
    if sq < 1e-25: continue
    let invStd = sqrt(float64(n) / sq)
    scales[j] = invStd
    standardized[j] = newSeq[float64](n)
    for i in 0..<n: standardized[j][i] = (features[j][i] - mean)*invStd
    for i in 0..<n:
      for k in 0..i:
        kernel[i*n + k] += standardized[j][i] * standardized[j][k]
  for i in 0..<n:
    kernel[i*n+i] += ridgeLambda
    for j in i+1..<n:
      kernel[i*n+j] = kernel[j*n+i]
  var centeredY = newSeq[float64](n)
  for i in 0..<n: centeredY[i] = y[i] - meanY
  let solution = ridgeSolve(kernel, n, centeredY)
  let alpha = solution.alpha
  var looPred = newSeq[float64](n)
  for i in 0..<n:
    # PRESS identity for a centered-kernel Ridge model with free intercept.
    let denominator = max(1e-9, ridgeLambda * solution.inverseDiagonal[i] - 1.0/float64(n))
    looPred[i] = y[i] - ridgeLambda * alpha[i] / denominator
  result.looPearson = pearson(looPred, y)
  if not fullModel: return  # only ranking needed: skip betas and predictions
  result.beta = newSeq[float64](p)
  result.intercept = meanY
  for j in 0..<p:
    if scales[j] == 0.0: continue
    var weight = 0.0
    for i in 0..<n: weight += standardized[j][i]*alpha[i]
    result.beta[j] = weight*scales[j]
    result.intercept -= result.beta[j]*means[j]
  var predictions = newSeq[float64](n)
  for i in 0..<n:
    predictions[i] = result.intercept
    for j in 0..<p: predictions[i] += result.beta[j]*features[j][i]
  result.trainPearson = pearson(predictions, y)

proc fit(rows: seq[Row], genome: Genome, ridgeLambda: float64): Model =
  var disabled = newFeatureCache(0, 64, 0)
  result = fit(rows, genome, ridgeLambda, disabled)

proc validate(rows: seq[Row], g: Genome, model: Model,
              cache: var FeatureCache): float64 =
  if rows.len < 3: return 0.0
  let features = getFeatures(rows, g, cache)
  var predictions = newSeq[float64](rows.len)
  var y = newSeq[float64](rows.len)
  for i, row in rows:
    y[i] = row.target
    predictions[i] = model.intercept
    for j, col in features:
      predictions[i] += model.beta[j]*col[i]
  pearson(predictions, y)

proc validate(rows: seq[Row], g: Genome, model: Model): float64 =
  var disabled = newFeatureCache(0, 64, 0)
  result = validate(rows, g, model, disabled)


# ---- Chunk-held-out model selection --------------------------------------
proc groupCvScores(rows: seq[Row], genome: Genome, ridgeLambda: float64,
                   cache: var FeatureCache): seq[float64] =
  ## Each corpus chunk is entirely held out, including all its diffusion rows.
  ## Unlike row-wise PRESS, no observations of that original text are in
  ## the regression training set for its predictions.
  if rows.len < 4: return @[]
  let columns = getFeatures(rows, genome, cache)
  var chunks: seq[int] = @[]
  for row in rows:
    if row.chunkId notin chunks: chunks.add(row.chunkId)
  if chunks.len < 2: return @[]
  for heldout in chunks:
    var trainIdx, testIdx: seq[int] = @[]
    for i, row in rows:
      if row.chunkId == heldout: testIdx.add(i)
      else: trainIdx.add(i)
    if trainIdx.len < 3 or testIdx.len < 2: continue
    let n = trainIdx.len
    var meanY = 0.0
    for i in trainIdx: meanY += rows[i].target
    meanY /= float64(n)
    var centeredY = newSeq[float64](n)
    for i, id in trainIdx: centeredY[i] = rows[id].target - meanY
    var means = newSeq[float64](columns.len)
    var scales = newSeq[float64](columns.len)
    var gram = newSeq[float64](n*n)
    for j, feature in columns:
      var avg = 0.0
      for id in trainIdx: avg += feature[id]
      avg /= float64(n)
      means[j] = avg
      var variance = 0.0
      for id in trainIdx:
        let d = feature[id]-avg
        variance += d*d
      if variance < 1e-25: continue
      scales[j] = sqrt(float64(n)/variance)
      for i in 0..<n:
        let a = (feature[trainIdx[i]]-avg)*scales[j]
        for k in 0..i:
          gram[i*n+k] += a*(feature[trainIdx[k]]-avg)*scales[j]
    for i in 0..<n:
      gram[i*n+i] += ridgeLambda
      for k in i+1..<n: gram[i*n+k] = gram[k*n+i]
    let alpha = ridgeAlpha(gram,n,centeredY)
    var preds = newSeq[float64](testIdx.len)
    var targets = newSeq[float64](testIdx.len)
    for i,id in testIdx:
      preds[i] = meanY
      targets[i] = rows[id].target
    for j, feature in columns:
      if scales[j] == 0.0: continue
      var w = 0.0
      for i, id in trainIdx:
        w += (feature[id]-means[j])*scales[j]*alpha[i]
      w *= scales[j]
      for i,id in testIdx:
        preds[i] += (feature[id]-means[j])*w
    result.add(pearson(preds, targets))

proc average(values: seq[float64]): float64 =
  if values.len == 0: return -1.0
  for value in values: result += value
  result /= float64(values.len)

proc pairedImprovement(child, parent: seq[float64]): float64 =
  ## Conservative paired mean delta; uncertainty penalty discourages
  ## promotion on a single lucky text. Not a formal CI for adaptive trials.
  let n = min(child.len, parent.len)
  if n == 0: return -Inf
  var total = 0.0
  for i in 0..<n: total += child[i]-parent[i]
  let meanDelta = total/float64(n)
  if n == 1: return meanDelta
  var spread = 0.0
  for i in 0..<n:
    let d = child[i]-parent[i]-meanDelta
    spread += d*d
  let stderr = sqrt(spread / (float64(n-1)*float64(n)))
  meanDelta - 0.5*stderr

proc samplingRows(source: seq[seq[int]], order: seq[int], gen: int,
                  cfg: Config, alphabet: seq[int], count, steps: int): seq[Row] =
  ## A cohort begins with the current rotating train window, then extends
  ## forward through the shuffled training pool without repeats.
  let full = trainingIndices(order, gen, min(order.len, max(cfg.trainChunks,count)),
                             cfg.rotateEvery, cfg.rotateStride)
  result = makeRows(source, full, steps, cfg.seed, alphabet, cfg.diffusionSampling)

proc ruleText(p: seq[int]): string =
  for i, x in p:
    if i > 0: result.add(',')
    result.add($x)

proc parseRuleText(s: string): seq[int] =
  for part in s.split(','):
    result.add(parseInt(part))

proc saveGenome(path: string, genome: Genome) =
  var contents = "# Lens GA v1 raw-byte rules; pattern TAB replacement; no hashed token map\n"
  for i, gene in genome.genes:
    contents.add(ruleText(gene.pat) & "\t" & ruleText(canonicalRhs(gene, i)) & "\n")
  writeFile(path, contents)

proc loadGenome(path: string, ruleCount: int): Genome =
  if not fileExists(path): quit("Resume file not found: " & path, 1)
  for line in lines(path):
    if line.len == 0 or line[0] == '#': continue
    let parts = line.split('\t')
    if parts.len != 2: quit("Malformed rule file: " & path, 1)
    let pattern = parseRuleText(parts[0])
    let rhs = parseRuleText(parts[1])
    let index = result.genes.len
    if rhs.len < 1 or rhs[0] != 256 + index:
      quit("Resume must be produced by Lens GA (old hashed rules are incompatible)", 1)
    var wild = 0
    for x in pattern:
      if x < 0: inc wild
    if rhs.len != wild + 1:
      quit("Malformed RHS at rule " & $index, 1)
    var mode = 0
    if wild > 0:
      case rhs[1]
      of -1: mode = 0
      of -16: mode = 1
      of -32: mode = 2
      of -48: mode = 3
      else: quit("Unsupported capture opcode in rule " & $index, 1)
    let gene = Gene(pat: pattern, mode: mode)
    if not validGene(gene, index) or canonicalRhs(gene, index) != rhs:
      quit("Invalid GA rule at " & $index, 1)
    result.genes.add(gene)
  if result.genes.len != ruleCount:
    quit("Resume rule count differs from --rules", 1)
  result.score = -Inf

proc metric(r: float64): float64 =
  -ln(max(1e-15, 1.0 - max(-1.0, min(1.0, r))))

proc writeMovingAverages(dir: string, window: int) =
  ## The trailing mean uses the last `window` EVALUATED points, not
  ## necessarily `window` generations when --eval-every > 1.
  ## Rebuild from training.dat so the raw series is never overwritten.
  let sourcePath = dir / "training.dat"
  let targetPath = dir / "training_ma.dat"
  var outFile = open(targetPath, fmWrite)
  defer: outFile.close()
  outFile.writeLine("# generation train_ma heldout_ma rotating_loo_ma fixed_reference_ma")
  let columns = [1, 2, 7, 9] # zero-based input columns (1:2, 1:3, 1:8, 1:10)
  var windows: array[4, seq[float64]]
  var totals: array[4, float64]
  for line in lines(sourcePath):
    if line.len == 0 or line[0] == '#': continue
    let fields = line.splitWhitespace()
    if fields.len < 10: continue
    var values: array[4, float64]
    var valid = true
    for j in 0..<4:
      try:
        values[j] = parseFloat(fields[columns[j]])
      except ValueError:
        valid = false
        break
      if classify(values[j]) in {fcNan, fcInf, fcNegInf}:
        valid = false
        break
    if not valid: continue
    var row = fields[0]
    for j in 0..<4:
      windows[j].add(values[j])
      totals[j] += values[j]
      if windows[j].len > window:
        totals[j] -= windows[j][0]
        windows[j].delete(0)
      row.add(" " & $(totals[j] / float64(windows[j].len)))
    outFile.writeLine(row)

proc plot(dir: string, maWindow: int) =
  writeMovingAverages(dir, maWindow)
  let script = """
set terminal pngcairo size 1550,820 enhanced font 'Arial,12'
set output 'training.png'
set xlabel 'Generation'
set ylabel '-ln(1 - Pearson)'
set title 'Lens | GA curves with trailing MA (%WINDOW% evaluated points)'
set grid
set key outside right top
set style line 1 lc rgb '#4477AA'
set style line 2 lc rgb '#CC6677'
set style line 3 lc rgb '#228833'
set style line 4 lc rgb '#AA7744'
plot 'training.dat' using 1:2 with lines ls 1 dt 3 lw 1 title 'Train raw', \
     'training_ma.dat' using 1:2 with lines ls 1 lw 2.8 title 'Train MA', \
     'training.dat' using 1:3 with lines ls 2 dt 3 lw 1 title 'Held-out raw', \
     'training_ma.dat' using 1:3 with lines ls 2 lw 2.8 title 'Held-out MA', \
     'training.dat' using 1:8 with lines ls 3 dt 3 lw 1 title 'Rotating LOO raw', \
     'training_ma.dat' using 1:4 with lines ls 3 lw 2.8 title 'Rotating LOO MA', \
     'training.dat' using 1:10 with lines ls 4 dt 3 lw 1 title 'Fixed reference raw', \
     'training_ma.dat' using 1:5 with lines ls 4 lw 2.8 title 'Fixed reference MA'
""".replace("%WINDOW%", $maWindow)
  writeFile(dir / "training.gnuplot", script)
  if findExe("gnuplot").len > 0:
    let previous = getCurrentDir()
    try:
      setCurrentDir(dir)
      discard execShellCmd("gnuplot training.gnuplot")
    finally:
      setCurrentDir(previous)

proc insertArchive(archive: var seq[ArchiveEntry], candidate: Genome,
                   fixedScore: float64, keep: int) =
  ## Never compare fitnesses from DIFFERENT rotating batches.
  ## History is ranked exclusively on one stable training-only reference set.
  if classify(fixedScore) in {fcNan, fcInf, fcNegInf}: return
  for entry in archive:
    if genomeEqual(entry.genome, candidate): return
  var archived = copyGenome(candidate)
  archived.score = -Inf # stale rotating-batch score must not be reused
  archive.add(ArchiveEntry(genome: archived, referenceScore: fixedScore))
  archive.sort(proc(a, b: ArchiveEntry): int = cmp(b.referenceScore, a.referenceScore))
  if archive.len > keep: archive.setLen(keep)

proc selectParent(population: seq[Genome], r: var Rng): int =
  ## Tournament selection within the CURRENT generation / training batch.
  result = r.randInt(population.len)
  for j in 0..<2:
    let k = r.randInt(population.len)
    if population[k].score > population[result].score: result = k



# --- Programmable Viral Evolution (PVE) -----------------------------------
# Three typed, acyclic scalar DAGs determine target rule ranking, partner
# ranking, and a conditional edit action. Their graph topology, operators,
# and edit actions themselves coevolve. Host GA stays intact.
const
  VirusNodes = 8
  VirusActions = 7 + MutationOps + 5

proc randomVNode(r: var Rng, at: int): VNode =
  let terminal = at == 0 or r.probability(0.62)
  if terminal:
    result.op = VOp(r.randInt(ord(vDistance) + 1))
  else:
    result.op = VOp(ord(vAdd) + r.randInt(ord(vMax)-ord(vAdd)+1))
  result.a = r.randInt(max(1, at))
  result.b = r.randInt(max(1, at))
  result.constant = 2.0*r.uniform() - 1.0

proc randomDag(r: var Rng): seq[VNode] =
  result = newSeq[VNode](VirusNodes)
  for i in 0..<VirusNodes: result[i] = randomVNode(r, i)

proc randomVirus(r: var Rng): Virus =
  result.targetDag = randomDag(r)
  result.partnerDag = randomDag(r)
  result.gateDag = randomDag(r)
  result.actionTrue = r.randInt(VirusActions)
  result.actionFalse = r.randInt(VirusActions)
  result.ngramLen = 2 + r.randInt(4)
  result.threshold = r.uniform()*1.6 - 0.8

proc copyVirus(v: Virus): Virus =
  result = v
  result.targetDag = newSeq[VNode](v.targetDag.len)
  result.partnerDag = newSeq[VNode](v.partnerDag.len)
  result.gateDag = newSeq[VNode](v.gateDag.len)
  for i, n in v.targetDag: result.targetDag[i] = n
  for i, n in v.partnerDag: result.partnerDag[i] = n
  for i, n in v.gateDag: result.gateDag[i] = n

proc mutateVirus(v: var Virus, r: var Rng) =
  ## Point, edge, and operator mutation; preserve DAG (edges only backward).
  let changes = 1 + r.randInt(4)
  for iter in 0..<changes:
    let which = r.randInt(6)
    case which
    of 0, 1, 2:
      let at = r.randInt(VirusNodes)
      var node = randomVNode(r, at)
      if which == 0: v.targetDag[at] = node
      elif which == 1: v.partnerDag[at] = node
      else: v.gateDag[at] = node
    of 3: v.actionTrue = r.randInt(VirusActions)
    of 4: v.actionFalse = r.randInt(VirusActions)
    else:
      if r.probability(0.5): v.ngramLen = 2 + r.randInt(4)
      else: v.threshold = r.uniform()*1.6 - 0.8

proc crossoverVirus(a, b: Virus, r: var Rng): Virus =
  ## Same-width DAG nodes retain valid backward references when swapped.
  result = copyVirus(a)
  for i in 0..<VirusNodes:
    if r.probability(0.5): result.targetDag[i] = b.targetDag[i]
    if r.probability(0.5): result.partnerDag[i] = b.partnerDag[i]
    if r.probability(0.5): result.gateDag[i] = b.gateDag[i]
  if r.probability(0.5): result.actionTrue = b.actionTrue
  if r.probability(0.5): result.actionFalse = b.actionFalse
  if r.probability(0.5): result.ngramLen = b.ngramLen
  if r.probability(0.5): result.threshold = b.threshold

proc pairCorrelation(ctx: ViralContext, i, j: int): float64 =
  if i < 0 or j < 0 or i >= ctx.norm.len or j >= ctx.norm.len or
      ctx.norm[i].len == 0 or ctx.norm[j].len == 0: return 0.0
  var sum = 0.0
  for k in 0..<ctx.norm[i].len:
    sum += ctx.norm[i][k] * ctx.norm[j][k]
  max(-1.0, min(1.0, sum))

proc buildViralContext(rows: seq[Row], g: Genome, m: Model,
                       cache: var FeatureCache): ViralContext =
  ## One context built per generation for the CURRENT best host. Redundancy
  ## estimates sample 12 peers per rule, not all 1,124,250 pairs.
  let x = getFeatures(rows, g, cache)
  let n = rows.len
  let p = x.len
  result.firing = newSeq[float64](p)
  result.meanFire = newSeq[float64](p)
  result.targetCorr = newSeq[float64](p)
  result.meanRedundancy = newSeq[float64](p)
  result.maxRedundancy = newSeq[float64](p)
  result.contribution = newSeq[float64](p)
  result.norm = newSeq[seq[float64]](p)
  var y = newSeq[float64](n)
  var meanY = 0.0
  for k, row in rows:
    y[k] = row.target
    meanY += y[k]
  meanY /= float64(n)
  var ySq = 0.0
  for k in 0..<n:
    y[k] -= meanY
    ySq += y[k]*y[k]
  if ySq > 1e-24:
    let scaleY = 1.0/sqrt(ySq)
    for k in 0..<n: y[k] *= scaleY
  for j in 0..<p:
    var avg = 0.0
    var hits = 0
    for value in x[j]:
      avg += value
      if value > 0.0: inc hits
    avg /= float64(n)
    result.firing[j] = float64(hits) / float64(n)
    result.meanFire[j] = avg
    var sq = 0.0
    for value in x[j]: sq += (value-avg)*(value-avg)
    if sq > 1e-25:
      result.norm[j] = newSeq[float64](n)
      let scale = 1.0/sqrt(sq)
      var dot = 0.0
      for k in 0..<n:
        result.norm[j][k] = (x[j][k]-avg)*scale
        dot += result.norm[j][k]*y[k]
      result.targetCorr[j] = dot
      # A *proxy*, not a leave-one-rule-out or causal contribution.
      result.contribution[j] = tanh(m.beta[j]*sqrt(sq/float64(n))*dot)
  if p > 1:
    for i in 0..<p:
      let samples = min(12, p-1)
      var total = 0.0
      for k in 0..<samples:
        let offset = 1 + ((k*137 + i*19) mod (p-1))
        let other = (i+offset) mod p
        let value = abs(pairCorrelation(result, i, other))
        total += value
        result.maxRedundancy[i] = max(result.maxRedundancy[i], value)
      result.meanRedundancy[i] = total/float64(samples)
  # One fixed representative diffusion row. Keep states BEFORE every rule;
  # state i+1 is also AFTER rule i, final state is index p.
  result.probeStates = newSeq[seq[int]](p+1)
  result.probeStates[0] = rows[n div 2].data
  for i, gene in g.genes:
    var state = result.probeStates[i]
    var program = compileSeqReplaceProgram(gene.pat, canonicalRhs(gene, i), 1)
    discard applyAndCount(state, program)
    result.probeStates[i+1] = state

proc buildViralContext(rows: seq[Row], g: Genome, m: Model): ViralContext =
  var disabled = newFeatureCache(0, 64, 0)
  result = buildViralContext(rows, g, m, disabled)

proc evalDag(nodes: seq[VNode], ctx: ViralContext, i, j: int): float64 =
  var v: array[VirusNodes, float64]
  let nRules = ctx.firing.len
  for at, node in nodes:
    let a = v[node.a]
    let b = v[node.b]
    var value = 0.0
    case node.op
    of vConst: value = node.constant
    of vFireRate: value = ctx.firing[i]
    of vMeanFire: value = ctx.meanFire[i]
    of vTargetCorr: value = ctx.targetCorr[i]
    of vAbsTargetCorr: value = abs(ctx.targetCorr[i])
    of vMeanRedundancy: value = ctx.meanRedundancy[i]
    of vMaxRedundancy: value = ctx.maxRedundancy[i]
    of vContribution: value = ctx.contribution[i]
    of vPosition: value = float64(i)/float64(max(1,nRules))
    of vPairCorr: value = pairCorrelation(ctx, i, j)
    of vAbsPairCorr: value = abs(pairCorrelation(ctx, i, j))
    of vDistance: value = (if j < 0: 0.0 else: float64(abs(i-j))/float64(max(1,nRules)))
    of vAdd: value = a+b
    of vSub: value = a-b
    of vMul: value = a*b
    of vNeg: value = -a
    of vAbs: value = abs(a)
    of vMin: value = min(a,b)
    of vMax: value = max(a,b)
    if classify(value) in {fcNan, fcInf, fcNegInf}: value = 0.0
    v[at] = max(-2.0, min(2.0, value))
  result = v[nodes.len-1]

proc viralPartner(virus: Virus, ctx: ViralContext, at, samples: int,
                  r: var Rng): int =
  ## Only earlier rules can be copied into at without future-marker leaks.
  if at <= 0: return 0
  result = r.randInt(at)
  var top = evalDag(virus.partnerDag, ctx, result, at)
  for k in 0..<samples:
    let candidate = (if k == 0: at-1 else: r.randInt(at))
    let v = evalDag(virus.partnerDag, ctx, candidate, at)
    if v > top:
      top = v
      result = candidate

proc mostFrequentNgram(ctx: ViralContext, at, sourceKind,
                       requestedLen: int): seq[int] =
  var state: seq[int]
  case sourceKind
  of 0: state = ctx.probeStates[at]   # before
  of 1: state = ctx.probeStates[at+1] # after
  else: state = ctx.probeStates[^1]  # final
  let n = min(12, max(1, requestedLen))
  if state.len < n: return @[]
  var counts = initTable[string,int]()
  var bestCount = 0
  var best = ""
  for start in 0..(state.len-n):
    var valid = true
    var key = ""
    for j in 0..<n:
      let token = state[start+j]
      if token < 0 or token >= 256+at:
        valid = false
        break
      key.add($token)
      key.add(',')
    if not valid: continue
    let next = counts.getOrDefault(key) + 1
    counts[key] = next
    if next > bestCount:
      bestCount = next
      best = key
  if best.len == 0: return @[]
  for strToken in best.split(','):
    if strToken.len > 0: result.add(parseInt(strToken))

proc viralEdit(virus: Virus, ctx: ViralContext, host: Genome,
               at, partner, action: int, source: seq[seq[int]],
               r: var Rng): Gene =
  result = copyGene(host.genes[at])
  let donor = host.genes[partner]
  case action
  of 0, 1, 2: # full-pattern n-gram transduction
    let motif = mostFrequentNgram(ctx,at,action,virus.ngramLen)
    if motif.len > 0: result = Gene(pat: motif, mode: 0)
  of 3: result = newGene(source,r,at)
  of 4: # copy earlier rule
    if partner < at: result = copyGene(donor)
  of 5: # in-gene splice from earlier rule
    if partner < at: result = spliceGene(host.genes[at],donor,at,r)
  of 6: result = mutateGene(host.genes[at],source,r,at)
  else:
    if action in 7..<(7+MutationOps):
      result = editGene(host.genes[at],donor,source,r,at,action-7)
    else:
      # N-gram-guided insertion/replacement/prepend/append/copy.
      let style = action-(7+MutationOps)
      let motif = mostFrequentNgram(ctx,at,style mod 3,virus.ngramLen)
      if motif.len > 0:
        let pos = r.randInt(host.genes[at].pat.len+1)
        var p: seq[int] = @[]
        case style
        of 0: # insert motif into target rule
          appendLimited(p,shallowSegment(host.genes[at].pat,0,pos))
          appendLimited(p,motif)
          appendLimited(p,shallowSegment(host.genes[at].pat,pos,RuleMaxLen))
        of 1: # replace segment with motif
          appendLimited(p,shallowSegment(host.genes[at].pat,0,pos))
          appendLimited(p,motif)
        of 2: # motif + rule prefix
          appendLimited(p,motif)
          appendLimited(p,host.genes[at].pat)
        of 3: # rule + motif
          appendLimited(p,host.genes[at].pat)
          appendLimited(p,motif)
        else: # motif + donor rule, bounded
          appendLimited(p,motif)
          appendLimited(p,donor.pat)
        result.pat = p
        normalizeWildcards(result)
        if not result.pat.anyIt(it < 0): result.mode = 0
  if not validGene(result,at) or geneEqual(result,host.genes[at]):
    result = mutateGene(host.genes[at],source,r,at)
  doAssert validGene(result,at) and not geneEqual(result,host.genes[at])

proc infect(virus: Virus, host: Genome, ctx: ViralContext,
            source: seq[seq[int]], partnerSamples, edits: int,
            r: var Rng): Genome =
  ## Select exactly K distinct loci, K drawn by caller from log-uniform law.
  result = copyGenome(host)
  var priority: seq[tuple[key: float64, locus: int]] = @[]
  for i in 0..<host.genes.len:
    priority.add((evalDag(virus.targetDag, ctx, i, -1) + 0.00001*r.uniform(), i))
  priority.sort(proc(a,b: tuple[key: float64, locus: int]): int = cmp(b.key,a.key))
  for j in 0..<min(edits, host.genes.len):
    let at = priority[j].locus
    let partner = viralPartner(virus, ctx, at, partnerSamples, r)
    let gate = evalDag(virus.gateDag, ctx, at, partner)
    let action = (if gate > virus.threshold: virus.actionTrue else: virus.actionFalse)
    result.genes[at] = viralEdit(virus, ctx, host, at, partner, action, source, r)
  result.score = -Inf

proc virusMerit(item: VirusRecord): float64 =
  ## A small optimism term lets never-tested viruses receive infections.
  if item.trials == 0: return 0.05
  item.advantage + 0.02/sqrt(float64(item.trials))

proc virusTournament(items: seq[VirusRecord], r: var Rng): int =
  result = r.randInt(items.len)
  for attempt in 0..<2:
    let i = r.randInt(items.len)
    if virusMerit(items[i]) > virusMerit(items[result]): result = i

proc evolveViruses(viruses: seq[VirusRecord], r: var Rng): seq[VirusRecord] =
  var ranked = viruses
  ranked.sort(proc(a,b: VirusRecord): int = cmp(virusMerit(b),virusMerit(a)))
  for i in 0..<min(2, ranked.len): result.add(ranked[i])
  while result.len < viruses.len:
    let a = virusTournament(viruses, r)
    var child: Virus
    if r.probability(0.6):
      child = crossoverVirus(viruses[a].virus,
                             viruses[virusTournament(viruses,r)].virus, r)
    else: child = copyVirus(viruses[a].virus)
    mutateVirus(child, r)
    result.add(VirusRecord(virus: child))

proc viralDescription(v: Virus): string =
  result = "# Programmable Viral Evolution v1\n"
  result.add("# Scorers are 8-node backwards-referencing DAGs; node 7 is output.\n")
  result.add("# Actions: 0 before-ngram, 1 after-ngram, 2 final-ngram, " &
             "3 original/new, 4 copy earlier rule, 5 splice, 6 normal mutation\n")
  result.add("true_action=" & $v.actionTrue & " false_action=" &
             $v.actionFalse & " ngram_len=" & $v.ngramLen &
             " gate_threshold=" & $v.threshold & "\n")
  # Nim's `for index, value in array` produces (index, value), not
  # tuple unpacking: iterate over the three DAGs and label by index.
  for dagIndex, dag in [v.targetDag, v.partnerDag, v.gateDag]:
    let label = ["target", "partner", "gate"][dagIndex]
    result.add(label & ":\n")
    for i, node in dag:
      result.add("  " & $i & " " & $node.op & " a=" & $node.a &
                 " b=" & $node.b & " const=" & $node.constant & "\n")


proc serializeVirus(v: Virus): string =
  ## Machine-readable, no dependency on JSON packages or enum spellings.
  result = $v.actionTrue & "\t" & $v.actionFalse & "\t" &
           $v.ngramLen & "\t" & $v.threshold & "\n"
  for dag in [v.targetDag, v.partnerDag, v.gateDag]:
    for node in dag:
      result.add($ord(node.op) & "\t" & $node.a & "\t" &
                 $node.b & "\t" & $node.constant & "\n")

proc loadVirus(path: string): Virus =
  if not fileExists(path): quit("Virus file not found: " & path, 1)
  var rows: seq[string] = @[]
  for line in lines(path):
    if line.strip.len > 0 and not line.startsWith("#"): rows.add(line)
  if rows.len != 1 + 3*VirusNodes:
    quit("Malformed virus genotype (expected 25 rows): " & path, 1)
  let header = rows[0].split('\t')
  if header.len != 4: quit("Malformed virus header", 1)
  result.actionTrue = parseInt(header[0])
  result.actionFalse = parseInt(header[1])
  result.ngramLen = parseInt(header[2])
  result.threshold = parseFloat(header[3])
  if result.actionTrue notin 0..<VirusActions or
      result.actionFalse notin 0..<VirusActions or result.ngramLen notin 2..5 or
      classify(result.threshold) in {fcNan, fcInf, fcNegInf}:
    quit("Invalid viral action or threshold", 1)
  var dags = [newSeq[VNode](VirusNodes), newSeq[VNode](VirusNodes),
              newSeq[VNode](VirusNodes)]
  for t in 0..<3:
    for i in 0..<VirusNodes:
      let cols = rows[1+t*VirusNodes+i].split('\t')
      if cols.len != 4: quit("Malformed viral node", 1)
      let kind = parseInt(cols[0])
      let a = parseInt(cols[1])
      let b = parseInt(cols[2])
      let c = parseFloat(cols[3])
      if kind notin 0..ord(vMax) or a notin 0..<max(1,i) or
          b notin 0..<max(1,i) or
          (i == 0 and kind >= ord(vAdd)) or
          classify(c) in {fcNan, fcInf, fcNegInf}:
        quit("Invalid viral DAG node", 1)
      dags[t][i] = VNode(op: VOp(kind), a: a, b: b, constant: c)
  result.targetDag = dags[0]
  result.partnerDag = dags[1]
  result.gateDag = dags[2]

proc runViralTrials(viruses: var seq[VirusRecord], population: seq[Genome],
                    source: seq[seq[int]], rows: seq[Row], cfg: Config,
                    r: var Rng, gen: int, cache: var FeatureCache): seq[VirusTrial] =
  ## Same host, same rotating rows, same K for virus and ordinary mutation.
  ## One diagnostic cache for the strongest host, deliberately not validation.
  let parent = population[gen mod min(3, population.len)]
  let parentModel = fit(rows, parent, cfg.ridgeLambda, cache)
  let ctx = buildViralContext(rows, parent, parentModel, cache)
  var history = open(cfg.outDir / "viral_trials.dat", fmAppend)
  defer: history.close()
  var attempts = newSeq[bool](viruses.len)
  for trial in 0..<cfg.virusTrials:
    var chosen = -1
    var top = -Inf
    # Explore untried viruses preferentially, then use UCB-like merit.
    for i in 0..<viruses.len:
      if attempts[i] and trial < viruses.len: continue
      let merit = virusMerit(viruses[i]) + 0.005*r.uniform()
      if merit > top:
        top = merit
        chosen = i
    if chosen < 0: chosen = virusTournament(viruses,r)
    attempts[chosen] = true
    let k = mutationCount(r, cfg.rules)
    var infected = infect(viruses[chosen].virus, parent, ctx, source,
                          cfg.virusPartnerSamples, k, r)
    var control = copyGenome(parent)
    mutateExactly(control, source, r, k)
    let virusFit = fit(rows, infected, cfg.ridgeLambda, cache, false).looPearson
    let randomFit = fit(rows, control, cfg.ridgeLambda, cache, false).looPearson
    let advantage = virusFit - randomFit
    let improvement = virusFit - parent.score
    let n = viruses[chosen].trials
    viruses[chosen].advantage += (advantage-viruses[chosen].advantage)/float64(n+1)
    viruses[chosen].improvement += (improvement-viruses[chosen].improvement)/float64(n+1)
    inc viruses[chosen].trials
    history.writeLine($gen & " " & $trial & " " & $chosen & " " & $k &
      " " & $parent.score & " " & $virusFit & " " & $randomFit &
      " " & $advantage & " " & $improvement)
    infected.score = virusFit
    # A child competes for injection on CURRENT batch, never on held-out.
    result.add(VirusTrial(genome: infected, score: virusFit, advantage: advantage,
                          parentId: gen mod min(3, population.len)))
  result.sort(proc(a,b: VirusTrial): int = cmp(b.score,a.score))

proc nextGeneration(population: seq[Genome], archive: seq[ArchiveEntry],
                    source: seq[seq[int]], r: var Rng,
                    cfg: Config): seq[Genome] =
  result = @[]
  # True generational elitism plus cumulative Hall-of-Fame injection.
  for i in 0..<min(cfg.elites, population.len):
    result.add(copyGenome(population[i]))
  for item in archive:
    if result.len >= cfg.elites + 1: break
    var found = false
    for current in result:
      if genomeEqual(current, item.genome): found = true
    if not found: result.add(copyGenome(item.genome))
  while result.len < cfg.population:
    let ia = selectParent(population, r)
    let ib = selectParent(population, r)
    var child: Genome
    if r.probability(0.30):
      let ic = selectParent(population, r)
      child = diff3Crossover(population[ia], population[ib], population[ic])
    else:
      child = twoParentCrossover(population[ia], population[ib], r)
    discard simultaneousMutation(child, source, r)
    result.add(child)


# ---- Structured recombination, residual-guided search and racing ----------
proc hasProducer(g: Gene, producer: int): bool {.inline.} =
  for token in g.pat:
    if token == 256+producer: return true
  false

proc dependencyCrossover(mother, father: Genome, ctx: ViralContext,
                         maxSize: int, r: var Rng): Genome =
  ## Grow a linkage group from actual producer->consumer references and,
  ## secondarily, strong co-firing correlations.  Do not reorder loci:
  ## intermediate marker IDs have positional semantics.
  result = copyGenome(mother)
  let n = mother.genes.len
  if n == 0: return
  let seed = r.randInt(n)
  var chosen = newSeq[bool](n)
  var group = @[seed]
  chosen[seed] = true
  for roundIndex in 1..<min(n,maxSize):
    var winner = -1
    var winningWeight = 0.6
    # Strong directed reference links are always considered, even if far away.
    for candidate in 0..<n:
      if chosen[candidate]: continue
      var dependency = false
      for previous in group:
        if (candidate > previous and hasProducer(mother.genes[candidate],previous)) or
           (candidate < previous and hasProducer(mother.genes[previous],candidate)) or
           (candidate > previous and hasProducer(father.genes[candidate],previous)) or
           (candidate < previous and hasProducer(father.genes[previous],candidate)):
          dependency = true
          break
      if dependency:
        winner = candidate
        winningWeight = 2.0
        break
      # Correlation is a heuristic, not proof of causal dependency.
      var weight = 0.0
      for previous in group:
        weight = max(weight,abs(pairCorrelation(ctx,previous,candidate)))
      if weight >= 0.86 and weight > winningWeight:
        winningWeight = weight
        winner = candidate
    if winner < 0: break
    group.add(winner)
    chosen[winner] = true
  for locus in group:
    result.genes[locus] = copyGene(father.genes[locus])
  result.score = -Inf

proc statesBefore(rows: seq[Row], genome: Genome, at: int,
                  cache: var FeatureCache): seq[seq[int]] =
  ## Reuse a trace checkpoint where possible. Prefix reconstruction uses
  ## immutable row snapshots and never contaminates the cache.
  result = newSeq[seq[int]](rows.len)
  var startAt = 0
  for i,row in rows: result[i] = row.data
  if cache.maxEntries > 0:
    for entry in cache.entries:
      if commonPrefix(entry.genes,genome.genes) != genome.genes.len: continue
      for cp in entry.checkpoints:
        if cp.ruleEnd <= at and cp.ruleEnd > startAt and
            cp.states.len == rows.len:
          startAt = cp.ruleEnd
          for i in 0..<rows.len: result[i] = cp.states[i]
  for locus in startAt..<at:
    let gene = genome.genes[locus]
    var program = compileSeqReplaceProgram(gene.pat,canonicalRhs(gene,locus),1)
    for i in 0..<rows.len:
      discard applyAndCount(result[i],program)

proc motifCount(data, motif: seq[int]): int =
  if motif.len == 0 or motif.len > data.len: return 0
  var pos = 0
  while pos + motif.len <= data.len:
    var identical = true
    for j in 0..<motif.len:
      if data[pos+j] != motif[j]:
        identical = false
        break
    if identical:
      inc result
      pos += motif.len
    else:
      inc pos

proc residualGuidedChild(parent: Genome, rows: seq[Row],
                         source: seq[seq[int]], candidates: int,
                         ridgeLambda: float64, r: var Rng,
                         cache: var FeatureCache): Genome =
  ## Residual-directed *proposal*, not a claim of exact global improvement:
  ## changing a rule can change every downstream activation. Candidate
  ## motifs are checked with a full, paired fitness evaluation afterwards.
  result = copyGenome(parent)
  let model = fit(rows,parent,ridgeLambda,cache)
  let x = getFeatures(rows,parent,cache)
  let n = rows.len
  if n < 3 or model.beta.len != parent.genes.len: return
  var residual = newSeq[float64](n)
  for i,row in rows:
    var estimate = model.intercept
    for j,col in x: estimate += model.beta[j]*col[i]
    residual[i] = row.target-estimate
  # Prefer genes whose current contribution is relatively small.
  var at = r.randInt(parent.genes.len)
  var bestContribution = Inf
  for repeatIndex in 0..<min(48,parent.genes.len):
    let k = r.randInt(parent.genes.len)
    var mean = 0.0
    var sq = 0.0
    for z in x[k]: mean += z
    mean /= float64(n)
    for z in x[k]: sq += (z-mean)*(z-mean)
    let contribution = abs(model.beta[k])*sqrt(sq/float64(n))
    if contribution < bestContribution:
      bestContribution = contribution
      at = k
  let states = statesBefore(rows,parent,at,cache)
  var ranks = newSeq[int](n)
  for i in 0..<n: ranks[i] = i
  ranks.sort(proc(a,b: int): int = cmp(abs(residual[b]),abs(residual[a])))
  var motifs: seq[seq[int]] = @[]
  for i in 0..<min(n,8):
    let state = states[ranks[i]]
    if state.len < 2: continue
    for tries in 0..<max(1,candidates div min(n,8)):
      let size = min(state.len,2+r.randInt(4))
      let first = r.randInt(state.len-size+1)
      let motif = shallowSegment(state,first,size)
      var valid = true
      for token in motif:
        if token < 0 or token >= 256+at:
          valid = false
          break
      if valid and motif.len > 0 and motif notin motifs: motifs.add(motif)
      if motifs.len >= candidates: break
    if motifs.len >= candidates: break
  var best = -Inf
  var bestMotif: seq[int] = @[]
  for motif in motifs:
    var fire = newSeq[float64](n)
    var mean = 0.0
    for i, state in states:
      fire[i] = float64(motifCount(state,motif))/float64(rows[i].sourceLen)
      mean += fire[i]
    mean /= float64(n)
    var variance = 0.0
    var dot = 0.0
    var oldDot = 0.0
    var oldSq = 0.0
    var oldMean = 0.0
    for z in x[at]: oldMean += z
    oldMean /= float64(n)
    for i in 0..<n:
      let centered = fire[i]-mean
      let previous = x[at][i]-oldMean
      variance += centered*centered
      dot += centered*residual[i]
      oldDot += centered*previous
      oldSq += previous*previous
    if variance < 1e-18: continue
    # Squared residual projection, with a penalty for an already redundant
    # rule. It is a cheap screening surrogate for regularized squared gain.
    let redundant = abs(oldDot)/sqrt(max(1e-18,variance*oldSq))
    let score = dot*dot/(ridgeLambda+variance) * (1.0-0.5*redundant)
    if score > best:
      best = score
      bestMotif = motif
  if bestMotif.len > 0:
    result.genes[at] = Gene(pat:bestMotif,mode:0)
    if geneEqual(result.genes[at],parent.genes[at]):
      result.genes[at] = mutateGene(parent.genes[at],source,r,at)
  else:
    result.genes[at] = mutateGene(parent.genes[at],source,r,at)
  result.score = -Inf

proc localizedChild(parent: Genome, ctx: ViralContext,
                    source: seq[seq[int]], r: var Rng): Genome =
  result = copyGenome(parent)
  let count = 1+r.randInt(4)
  var chosen = newSeq[bool](parent.genes.len)
  for step in 0..<min(count,parent.genes.len):
    var at = -1
    var best = Inf
    for sample in 0..<min(20,parent.genes.len):
      let locus = r.randInt(parent.genes.len)
      if chosen[locus]: continue
      let contribution = abs(ctx.contribution[locus]) + 0.03*r.uniform()
      if contribution < best:
        best = contribution
        at = locus
    if at < 0:
      for locus in 0..<parent.genes.len:
        if not chosen[locus]:
          at = locus
          break
    if at < 0: break
    chosen[at] = true
    result.genes[at] = mutateGene(parent.genes[at],source,r,at)
  result.score = -Inf

proc refineNextGeneration(population: seq[Genome], archive: seq[ArchiveEntry],
                          trainingSource, allSource: seq[seq[int]],
                          order: seq[int], gen: int, alphabet: seq[int],
                          currentRows: seq[Row], viralChildren: seq[VirusTrial],
                          cfg: Config, r: var Rng, trainCache: var FeatureCache,
                          stats: var SearchStats): seq[Genome] =
  ## Preserve true elitism and historical global best in nextGeneration.
  ## The new optimizer proposes additional children, screens cheaply,
  ## then promotes only independently checked improvements.
  let started = epochTime()
  result = nextGeneration(population,archive,trainingSource,r,cfg)
  if not cfg.multifidelity: return
  let baseModel = fit(currentRows,population[0],cfg.ridgeLambda,trainCache)
  let context = buildViralContext(currentRows,population[0],baseModel,trainCache)
  let proposalCount = max(cfg.finalists*2,
    (cfg.population-cfg.elites)*cfg.candidateFactor)
  var proposals: seq[Proposal] = @[]
  for candidateIndex in 0..<proposalCount:
    let ia = selectParent(population,r)
    let ib = selectParent(population,r)
    var child: Genome
    var methodName = "ordinary"
    case candidateIndex mod 5
    of 0:
      child = twoParentCrossover(population[ia],population[ib],r)
      discard simultaneousMutation(child,trainingSource,r)
    of 1:
      child = diff3Crossover(population[ia],population[ib],
                             population[selectParent(population,r)])
      discard simultaneousMutation(child,trainingSource,r)
    of 2:
      if cfg.localSearch:
        child = localizedChild(population[ia],context,trainingSource,r)
        methodName = "local"
      else:
        child = copyGenome(population[ia])
        discard simultaneousMutation(child,trainingSource,r)
    of 3:
      if cfg.linkageCross:
        child = dependencyCrossover(population[ia],population[ib],context,
                                    cfg.linkageSize,r)
        methodName = "linkage"
      else:
        child = twoParentCrossover(population[ia],population[ib],r)
    else:
      if cfg.residualSearch and candidateIndex < 5:
        child = residualGuidedChild(population[ia],currentRows,trainingSource,
                 cfg.residualCandidates,cfg.ridgeLambda,r,trainCache)
        methodName = "residual"
      else:
        child = copyGenome(population[ia])
        discard simultaneousMutation(child,trainingSource,r)
    if genomeEqual(child,population[ia]):
      discard simultaneousMutation(child,trainingSource,r)
    proposals.add(Proposal(genome:child,parent:population[ia],
                  parentId:ia,originKind:methodName))
  if cfg.localSearch:
    for localIndex in 0..<cfg.localTrials:
      let parentId = selectParent(population,r)
      proposals.add(Proposal(genome:localizedChild(population[parentId],context,
          trainingSource,r),parent:population[parentId],
          parentId:parentId,originKind:"local"))
  if cfg.virusInject > 0:
    for trial in viralChildren:
      let id = trial.parentId
      if id < 0 or id >= population.len: continue
      proposals.add(Proposal(genome:trial.genome,parent:population[id],
                     parentId:id,originKind:"viral"))
  stats.proposals = proposals.len
  let firstRows = samplingRows(allSource,order,gen,cfg,alphabet,
                               cfg.trainChunks,min(cfg.noiseSteps,cfg.stage1Noise))
  let middleRows = samplingRows(allSource,order,gen,cfg,alphabet,
                              cfg.stage2Chunks,min(cfg.noiseSteps,cfg.stage2Noise))
  let finalRows = samplingRows(allSource,order,gen,cfg,alphabet,
                              cfg.stage3Chunks,min(cfg.noiseSteps,cfg.stage3Noise))
  var stage1Cache = newFeatureCache(4,cfg.cacheStride,8,8,1024)
  var stage2Cache = newFeatureCache(4,cfg.cacheStride,16,16,2048)
  var stage3Cache = newFeatureCache(4,cfg.cacheStride,32,32,4096)
  var baseline1 = newSeq[float64](population.len)
  for i in 0..<population.len: baseline1[i] = NaN
  for k in 0..<proposals.len:
    let parentId = proposals[k].parentId
    if classify(baseline1[parentId]) == fcNan:
      baseline1[parentId] = fit(firstRows,population[parentId],
                       cfg.ridgeLambda,stage1Cache,false).looPearson
      inc stats.coarseEvaluations
    let measured = fit(firstRows,proposals[k].genome,
                        cfg.ridgeLambda,stage1Cache,false).looPearson
    inc stats.coarseEvaluations
    proposals[k].coarseDelta = measured-baseline1[parentId]
  proposals.sort(proc(a,b: Proposal): int = cmp(b.coarseDelta,a.coarseDelta))
  proposals.setLen(min(proposals.len,max(cfg.finalists*2,(proposals.len+1) div 2)))
  var baseline2 = newSeq[float64](population.len)
  for i in 0..<population.len: baseline2[i] = NaN
  for k in 0..<proposals.len:
    let parentId = proposals[k].parentId
    if classify(baseline2[parentId]) == fcNan:
      baseline2[parentId] = fit(middleRows,population[parentId],
                       cfg.ridgeLambda,stage2Cache,false).looPearson
      inc stats.mediumEvaluations
    let measured = fit(middleRows,proposals[k].genome,
                       cfg.ridgeLambda,stage2Cache,false).looPearson
    inc stats.mediumEvaluations
    proposals[k].mediumDelta = measured-baseline2[parentId]
  proposals.sort(proc(a,b: Proposal): int = cmp(b.mediumDelta,a.mediumDelta))
  proposals.setLen(min(proposals.len,cfg.finalists))
  var baseline3 = newSeq[seq[float64]](population.len)
  for k in 0..<proposals.len:
    let parentId = proposals[k].parentId
    if baseline3[parentId].len == 0:
      baseline3[parentId] = groupCvScores(finalRows,population[parentId],
                                              cfg.ridgeLambda,stage3Cache)
      inc stats.precisionEvaluations
    let childScores = groupCvScores(finalRows,proposals[k].genome,
                                    cfg.ridgeLambda,stage3Cache)
    inc stats.precisionEvaluations
    let parentScores = baseline3[parentId]
    proposals[k].refinedDelta = (if cfg.pairedSelection:
       pairedImprovement(childScores,parentScores)
       else: average(childScores)-average(parentScores))
  proposals.sort(proc(a,b: Proposal): int = cmp(b.refinedDelta,a.refinedDelta))
  var insertion = 0
  var viralAdded = 0
  for proposal in proposals:
    # Do not overwrite elites/archived genome; these offspring will be
    # re-evaluated on the next generation's rotating cohort as usual.
    if proposal.refinedDelta <= 0.0: continue
    if proposal.originKind == "viral" and viralAdded >= cfg.virusInject: continue
    if insertion >= min(cfg.finalists,cfg.population-cfg.elites-1): break
    var duplicate = false
    for existing in result:
      if genomeEqual(existing,proposal.genome): duplicate = true
    if duplicate: continue
    let slot = result.len-1-insertion
    result[slot] = copyGenome(proposal.genome)
    result[slot].score = -Inf
    inc insertion
    inc stats.accepted
    case proposal.originKind
    of "local": inc stats.localWins
    of "residual": inc stats.residualWins
    of "linkage": inc stats.linkageWins
    of "viral":
      inc stats.viralWins
      inc viralAdded
    else: discard
  stats.timeSeconds = epochTime()-started

proc selfTest() =
  var r = Rng(state: 42)
  let corpus = @[@[65,66,67,65,66,67,68], @[65,67,65,67,68], @[66,67,68,67,68], @[68,65,66,68]]
  let rows = makeRows(corpus, @[0, 1], 4, 123'u64, observedAlphabet(corpus))
  var shuffleRng = Rng(state: 713'u64)
  let order = shuffledTrainingIndices(17, shuffleRng)
  var visited = newSeq[bool](17)
  for gen in 0..<17:
    let ix = trainingIndices(order, gen, 2, 1, 1)
    for x in ix: visited[x] = true
    doAssert ix.len == 2 and ix[0] != ix[1]
  for seen in visited: doAssert seen
  doAssert trainingIndices(order, 0, 3, 1, 1) ==
           trainingIndices(order, 0, 3, 2, 1)
  doAssert trainingIndices(order, 1, 3, 2, 1) ==
           trainingIndices(order, 0, 3, 1, 1)
  doAssert trainingIndices(order, 17, 2, 1, 1) ==
           trainingIndices(order, 0, 2, 1, 1)
  var levelsRng = Rng(state: 119'u64)
  let randomLevels = diffusionLevels(101, 11, levelsRng, "random")
  doAssert randomLevels.len == 12
  doAssert randomLevels[0] == 0 and randomLevels[^1] == 101
  for k in 1..<randomLevels.len:
    doAssert randomLevels[k] > randomLevels[k-1]
  var repeatLevelsRng = Rng(state: 119'u64)
  doAssert randomLevels == diffusionLevels(101, 11, repeatLevelsRng, "random")
  var uniformLevelsRng = Rng(state: 119'u64)
  let uniformLevels = diffusionLevels(101, 11, uniformLevelsRng, "uniform")
  doAssert uniformLevels[0] == 0 and uniformLevels[^1] == 101
  doAssert uniformLevels != randomLevels
  # Global source id, not per-batch offset, determines the diffusion trajectory.
  let obs = observedAlphabet(corpus)
  # Changing sample count never changes the full corruption path.
  let fewer = makeRows(corpus,@[2],3,123'u64,obs)
  let more = makeRows(corpus,@[2],8,123'u64,obs)
  doAssert fewer[0].data == more[0].data
  doAssert fewer[^1].data == more[^1].data
  # Nested random stage selection: every low-fidelity sample is reproduced
  # bit-for-bit in the higher-fidelity batch with the same corpus chunk ID.
  for smallRow in fewer:
    var found = false
    for largeRow in more:
      if abs(largeRow.target-smallRow.target) < 1e-12:
        doAssert smallRow.data == largeRow.data
        found = true
    doAssert found
  let singleChunkRows = makeRows(corpus, @[2], 4, 123'u64, obs)
  let reorderedRows = makeRows(corpus, @[2, 0], 4, 123'u64, obs)
  for i in 0..<singleChunkRows.len:
    doAssert reorderedRows[i].data == singleChunkRows[i].data
    doAssert reorderedRows[i].target == singleChunkRows[i].target
  doAssert rows.len >= 8
  var a, b, c: Genome
  for i in 0..<16:
    a.genes.add(Gene(pat: @[65], mode: 0))
    b.genes.add(Gene(pat: @[65], mode: 0))
    c.genes.add(Gene(pat: @[65], mode: 0))
  b.genes[3] = Gene(pat: @[66], mode: 0)
  c.genes[7] = Gene(pat: @[67], mode: 0)
  let merged = diff3Crossover(a, b, c)
  doAssert merged.genes[3].pat == @[66] and merged.genes[7].pat == @[67]
  c.genes[3] = Gene(pat: @[68], mode: 0)
  doAssert diff3Crossover(a, b, c).genes[3].pat == @[65]
  for rep in 0..<50:
    let x = mutationCount(r, 1500)
    doAssert x >= 1 and x <= 1500
  var child = copyGenome(a)
  let mutationTotal = simultaneousMutation(child, corpus, r)
  doAssert mutationTotal in 1..16
  for i, g in child.genes: doAssert validGene(g, i)
  # Cached and uncached simulations must be bit-identical, including
  # intermediate-marker reuse and edits early/late in the 1500-rule array.
  var testCache = newFeatureCache(4, 3, 8)
  let baselineFeatures = getFeatures(rows, a)
  doAssert baselineFeatures == getFeaturesReference(rows, a)
  doAssert baselineFeatures == getFeatures(rows, a, testCache)
  doAssert baselineFeatures == getFeatures(rows, a, testCache)
  doAssert testCache.exactHits > 0
  var lateEdit = copyGenome(a)
  lateEdit.genes[14] = Gene(pat: @[68, 65], mode: 0)
  doAssert getFeatures(rows, lateEdit) == getFeatures(rows, lateEdit, testCache)
  doAssert testCache.savedRuleRows >= int64(12 * rows.len)
  var earlyEdit = copyGenome(a)
  earlyEdit.genes[0] = Gene(pat: @[67], mode: 0)
  doAssert getFeatures(rows, earlyEdit) == getFeatures(rows, earlyEdit, testCache)
  doAssert getFeaturesReference(rows, earlyEdit) == getFeatures(rows, earlyEdit, testCache)
  # Interval cache works even when the common PREFIX is zero. Compare against
  # the independent serial interpreter for both matching and diverging states.
  var intervalCache = newFeatureCache(0, 3, 0, 8, 300)
  doAssert getFeatures(rows, a, intervalCache) == baselineFeatures
  var invisibleEdit = copyGenome(a)
  invisibleEdit.genes[0] = Gene(pat: @[250,251], mode: 0)
  doAssert getFeatures(rows, invisibleEdit, intervalCache) ==
           getFeaturesReference(rows, invisibleEdit)
  # A second evaluation is guaranteed to share every interval, regardless
  # of whether the different first rule changed a particular test row.
  doAssert getFeatures(rows, invisibleEdit, intervalCache) ==
           getFeaturesReference(rows, invisibleEdit)
  doAssert intervalCache.blockHits > 0
  doAssert intervalCache.blockSavedRuleRows > 0
  # Exact verification must reject a deliberately wrong input even when
  # the caller supplies an existing cache hash (collision simulation).
  let collisionKey = blockRowHash(blockGeneHash(a.genes, 0, 3), rows[0].data)
  var rejected: BlockMemo
  doAssert not lookupBlock(intervalCache, collisionKey, a.genes, 0, 3,
                            @[999, 999], rejected)
  doAssert getFeatures(rows, earlyEdit, intervalCache) ==
           getFeaturesReference(rows, earlyEdit)
  doAssert getFeatures(rows, lateEdit, intervalCache) ==
           getFeaturesReference(rows, lateEdit)
  let hitsBefore = intervalCache.blockHits
  discard getFeatures(rows, earlyEdit, intervalCache)
  doAssert intervalCache.blockHits > hitsBefore
  var tinyBlocks = newFeatureCache(0, 3, 0, 1, 1)
  doAssert getFeatures(rows, a, tinyBlocks) == baselineFeatures
  doAssert tinyBlocks.blockEntries.len <= 1
  doAssert getFeatures(rows, earlyEdit, tinyBlocks) ==
           getFeaturesReference(rows, earlyEdit)
  doAssert tinyBlocks.blockEntries.len <= 1
  clearFeatureCache(intervalCache)
  doAssert intervalCache.blockEntries.len == 0
  doAssert intervalCache.blockUsedBytes == 0
  # Force a dependency on an intermediate marker whose producer never fired.
  var deadMarker = copyGenome(a)
  deadMarker.genes[0] = Gene(pat: @[255, 255], mode: 0)
  deadMarker.genes[1] = Gene(pat: @[256], mode: 0)
  doAssert getFeaturesReference(rows, deadMarker) ==
           getFeatures(rows, deadMarker, testCache)
  let baselineFit = fit(rows, lateEdit, 0.1)
  let cachedFit = fit(rows, lateEdit, 0.1, testCache)
  doAssert abs(baselineFit.looPearson - cachedFit.looPearson) < 1e-10
  let fastFit = fit(rows, lateEdit, 0.1, testCache, false)
  doAssert abs(baselineFit.looPearson - fastFit.looPearson) < 1e-10
  clearFeatureCache(testCache)
  doAssert testCache.entries.len == 0
  doAssert getFeatures(rows, lateEdit) == getFeatures(rows, lateEdit, testCache)
  let trained = fit(rows, a, 0.1)
  # Chunk-level CV must hold out all diffusion states from one text.
  var allRows = makeRows(corpus,@[0,1,2,3],4,123'u64,obs)
  var groupCache = newFeatureCache(2,4,8,8,100)
  let chunkScores = groupCvScores(allRows,a,0.1,groupCache)
  doAssert chunkScores.len == corpus.len
  for z in chunkScores: doAssert classify(z) notin {fcNan,fcInf,fcNegInf}
  doAssert abs(pairedImprovement(chunkScores,chunkScores)) < 1e-12
  let trainedContext = buildViralContext(rows,a,trained)
  let linked = dependencyCrossover(a,b,trainedContext,4,r)
  for locus,gene in linked.genes: doAssert validGene(gene,locus)
  let guided = residualGuidedChild(a,rows,corpus,8,0.1,r,testCache)
  for locus,gene in guided.genes: doAssert validGene(gene,locus)
  let local = localizedChild(a,trainedContext,corpus,r)
  var localChanges = 0
  for locus,gene in local.genes:
    doAssert validGene(gene,locus)
    if not geneEqual(gene,a.genes[locus]): inc localChanges
  doAssert localChanges in 1..4
  doAssert trained.beta.len == a.genes.len
  doAssert classify(trained.trainPearson) notin {fcNan, fcInf, fcNegInf}
  doAssert classify(trained.looPearson) notin {fcNan, fcInf, fcNegInf}
  # Save/load round-trip with a wildcard and inverse-sort capture.
  var rw = copyGenome(a)
  rw.genes[1] = Gene(pat: @[65,-1,67], mode: 3)
  let fn = getTempDir() / "lens_ga_selftest_rules.tsv"
  saveGenome(fn, rw)
  doAssert genomeEqual(rw, loadGenome(fn, 16))
  doAssert loadGenome(fn, 16).genes[1].mode == 3
  removeFile(fn)
  # Exhaustive coverage of every classical local edit operator, including
  # crossover/motif donors containing captures and earlier-rule markers.
  let wideParent = Gene(pat: @[65,66,-1,67,68,69], mode: 1)
  let wideDonor = Gene(pat: @[67,68,-1,69,70], mode: 2)
  for op in 0..<MutationOps:
    for rep in 0..<12:
      let changed = editGene(wideParent,wideDonor,corpus,r,9,op)
      doAssert validGene(changed,9)
      doAssert not geneEqual(changed,wideParent)
  for rep in 0..<30:
    var t1,t2: Genome
    for i in 0..<16:
      t1.genes.add(newGene(corpus,r,i))
      t2.genes.add(newGene(corpus,r,i))
    let rec = twoParentCrossover(t1,t2,r)
    doAssert rec.genes.len == 16
    for i,g in rec.genes: doAssert validGene(g,i)
  # PVE graph validity, exact K non-repeating infection, stages and operators.
  let v = randomVirus(r)
  let vf = getTempDir() / "lens_virus_selftest.vgen"
  writeFile(vf, serializeVirus(v))
  let loadedVirus = loadVirus(vf)
  doAssert serializeVirus(loadedVirus) == serializeVirus(v)
  removeFile(vf)
  let ctxt = buildViralContext(rows, a, trained)
  for rep in 0..<10:
    var u = copyVirus(v)
    mutateVirus(u, r)
    let crossed = crossoverVirus(v, u, r)
    let infected = infect(crossed, a, ctxt, corpus, 8, 1+rep, r)
    var edited = 0
    for i in 0..<a.genes.len:
      doAssert validGene(infected.genes[i], i)
      if not geneEqual(a.genes[i], infected.genes[i]): inc edited
    doAssert edited == 1+rep
  # Simple moving-average plot data: preserve generations, use trailing points.
  let maDir = getTempDir() / "lens_pve_ma_selftest"
  createDir(maDir)
  writeFile(maDir / "training.dat", "# training header\n" &
    "0 1 2 0 0 1 0 10 0 100 0\n" &
    "3 3 4 0 0 1 0 20 0 200 0\n" &
    "6 5 6 0 0 1 0 30 0 300 0\n" &
    "9 7 8 0 0 1 0 40 0 400 0\n")
  writeMovingAverages(maDir, 2)
  let maLines = readFile(maDir / "training_ma.dat").splitLines()
  let firstMa = maLines[1].splitWhitespace()
  let secondMa = maLines[2].splitWhitespace()
  let lastMa = maLines[4].splitWhitespace()
  doAssert firstMa[0] == "0" and abs(parseFloat(firstMa[1]) - 1.0) < 1e-10
  doAssert secondMa[0] == "3" and abs(parseFloat(secondMa[1]) - 2.0) < 1e-10
  doAssert abs(parseFloat(secondMa[2]) - 3.0) < 1e-10
  doAssert abs(parseFloat(secondMa[3]) - 15.0) < 1e-10
  doAssert abs(parseFloat(secondMa[4]) - 150.0) < 1e-10
  doAssert lastMa[0] == "9" and abs(parseFloat(lastMa[1]) - 6.0) < 1e-10
  writeMovingAverages(maDir, 1)
  let rawMa = readFile(maDir / "training_ma.dat").splitLines()[4].splitWhitespace()
  doAssert abs(parseFloat(rawMa[1]) - 7.0) < 1e-10
  removeFile(maDir / "training.dat")
  removeFile(maDir / "training_ma.dat")
  removeDir(maDir)
  echo "[self-test] paired CRN, chunk-heldout CV, local, linkage, residual, " &
       "rotation, interval cache, edits, PVE, Ridge, moving-average passed"

proc benchCache() =
  ## Compare identical work with and without reuse. This synthetic case
  ## deliberately changes only the tail of each genome, so it measures the
  ## maximum expected benefit from shared-prefix caching rather than a
  ## forecast for arbitrary multi-point crossover workloads.
  var rng = Rng(state: 931'u64)
  var source: seq[seq[int]] = @[]
  for i in 0..<5:
    var chunk = newSeq[int](250)
    for j in 0..<chunk.len: chunk[j] = 65 + rng.randInt(7)
    source.add(chunk)
  let rows = makeRows(source, @[0, 1, 2, 3], 5, 59'u64,
                      observedAlphabet(source))
  var base: Genome
  base.genes = newSeq[Gene](1500)
  for i in 0..<1500:
    base.genes[i] = newGene(source, rng, i)
  var variations: seq[Genome] = @[]
  variations.add(base)
  for i in 0..<12:
    var variant = copyGenome(base)
    let at = 1400 + i*7
    variant.genes[at] = Gene(pat: @[65 + (i mod 7), 66], mode: 0)
    variations.add(variant)
  var without = newFeatureCache(0, 64, 0)
  var withCache = newFeatureCache(16, 64, 128)
  var before = epochTime()
  var reference = newSeq[seq[seq[float64]]](variations.len)
  for i, variant in variations:
    reference[i] = getFeatures(rows, variant, without)
  let coldSeconds = epochTime() - before
  before = epochTime()
  for i, variant in variations:
    let x = getFeatures(rows, variant, withCache)
    doAssert x == reference[i]
  let warmSeconds = epochTime() - before
  echo "[bench-cache] variants=", variations.len,
    " rows=", rows.len, " rules=1500",
    " uncached_seconds=", coldSeconds,
    " cached_seconds=", warmSeconds,
    " speedup=", coldSeconds / max(1e-9, warmSeconds),
    " hits=", withCache.hits,
    " exact=", withCache.exactHits,
    " saved_rule_rows=", withCache.savedRuleRows

proc benchBlockCache() =
  # Divergence at rule zero prevents prefix reuse, while identical input
  # rows after no-op edits permit exact downstream interval reuse.
  var rng = Rng(state: 993'u64)
  var source: seq[seq[int]] = @[]
  for i in 0..<4:
    var chunk = newSeq[int](350)
    for j in 0..<chunk.len: chunk[j] = 65 + rng.randInt(7)
    source.add(chunk)
  let rows = makeRows(source, @[0, 1, 2, 3], 8, 171'u64,
                      observedAlphabet(source))
  var base: Genome
  base.genes = newSeq[Gene](300)
  for i in 0..<base.genes.len: base.genes[i] = newGene(source, rng, i)
  var variants: seq[Genome] = @[base]
  for i in 0..<10:
    var variant = copyGenome(base)
    variant.genes[0] = Gene(pat: @[200+i, 201+i], mode: 0)
    variants.add(variant)
  var disabled = newFeatureCache(0, 32, 0)
  var enabled = newFeatureCache(0, 32, 0, 128, 20000)
  var baseline: seq[seq[seq[float64]]] = @[]
  var before = epochTime()
  for genome in variants: baseline.add(getFeatures(rows, genome, disabled))
  let coldSeconds = epochTime() - before
  before = epochTime()
  for i, genome in variants:
    doAssert getFeatures(rows, genome, enabled) == baseline[i]
  let warmSeconds = epochTime() - before
  doAssert enabled.blockHits > 0
  echo "[bench-block-cache] variants=", variants.len,
       " rows=", rows.len, " rules=", base.genes.len,
       " uncached_seconds=", coldSeconds,
       " block_cached_seconds=", warmSeconds,
       " speedup=", coldSeconds / max(1e-9, warmSeconds),
       " block_hits=", enabled.blockHits,
       " avoided_rule_rows=", enabled.blockSavedRuleRows

proc main() =
  if "--self-test" in commandLineParams():
    selfTest()
    return
  if "--bench-cache" in commandLineParams():
    benchCache()
    return
  if "--bench-block-cache" in commandLineParams():
    benchBlockCache()
    return
  let cfg = config()
  createDir(cfg.outDir)
  let source = loadCorpus(cfg)
  let poolSize = source.len - cfg.validChunks
  if poolSize < 1: quit("No training chunks after excluding held-out validation", 1)
  if cfg.trainChunks > poolSize:
    quit("--train must not exceed the non-validation training pool", 1)
  # Build diffusion alphabet from TRAINING ONLY; no validation byte leakage.
  let alphabet = observedAlphabet(source[0..<poolSize])
  var r = Rng(state: cfg.seed)
  if r.state == 0: r.state = 1729'u64
  # Validation is ALWAYS the final --valid chunks. No selection / archive scores
  # use these chunks. Random rotation affects ONLY source indices 0..<poolSize.
  var validationIds: seq[int] = @[]
  for i in poolSize..<source.len: validationIds.add(i)
  let validRows = makeRows(source, validationIds, cfg.noiseSteps, cfg.seed, alphabet,
                           cfg.diffusionSampling)
  if validRows.len < 3: quit("Insufficient held-out diffusion rows", 1)

  let order = shuffledTrainingIndices(poolSize, r)
  let fixedIds = referenceIndices(order, cfg.anchorChunks)
  let fixedRows = makeRows(source, fixedIds, cfg.noiseSteps, cfg.seed, alphabet,
                           cfg.diffusionSampling)
  if fixedRows.len < 3: quit("Insufficient fixed-reference diffusion rows", 1)
  let trainSource = source[0..<poolSize]
  # RAM caps split across strictly isolated cohorts; no validation leakage.
  var trainCache = newFeatureCache(cfg.cacheEntries, cfg.cacheStride,
                                    cfg.cacheMb * 7 div 10,
                                    cfg.blockCacheMb * 7 div 10, cfg.blockCacheEntries)
  var fixedCache = newFeatureCache(cfg.cacheEntries, cfg.cacheStride,
                                    cfg.cacheMb * 2 div 10,
                                    cfg.blockCacheMb * 2 div 10, cfg.blockCacheEntries)
  var validCache = newFeatureCache(cfg.cacheEntries, cfg.cacheStride,
                                    cfg.cacheMb - cfg.cacheMb * 7 div 10 -
                                    cfg.cacheMb * 2 div 10,
                                    cfg.blockCacheMb - cfg.blockCacheMb * 7 div 10 -
                                    cfg.blockCacheMb * 2 div 10, cfg.blockCacheEntries)
  var ancestor: Genome
  if cfg.resume.len > 0:
    ancestor = loadGenome(cfg.resume, cfg.rules)
  else:
    ancestor.genes = newSeq[Gene](cfg.rules)
    for j in 0..<cfg.rules:
      ancestor.genes[j] = newGene(trainSource, r, j)
  var population: seq[Genome] = @[]
  for i in 0..<cfg.population:
    var genome = copyGenome(ancestor)
    if i > 0: discard simultaneousMutation(genome, trainSource, r)
    genome.score = -Inf
    population.add(genome)

  var virusRng = Rng(state: cfg.seed xor 0xD1B54A32D192ED03'u64)
  if virusRng.state == 0: virusRng.state = 17'u64
  var viruses: seq[VirusRecord] = @[]
  if cfg.virusEnabled:
    for i in 0..<cfg.virusPopulation:
      viruses.add(VirusRecord(virus: randomVirus(virusRng)))
    if cfg.virusResume.len > 0:
      viruses[0].virus = loadVirus(cfg.virusResume)
    writeFile(cfg.outDir / "viral_trials.dat",
      "# generation trial virus_index K parent_LOO infected_LOO random_LOO advantage improvement\n")
    writeFile(cfg.outDir / "virus_history.dat",
      "# generation best_advantage best_improvement total_trials active_viruses\n")
  var bestViralAdvantage = -Inf
  var archive: seq[ArchiveEntry] = @[]
  var highestFixed = -Inf
  writeFile(cfg.outDir / "training.dat",
    "# gen train_metric valid_metric train_pearson valid_pearson seconds loo_pearson loo_metric fixed_loo fixed_metric rotation_start\n")
  writeFile(cfg.outDir / "rotation.dat", "# generation rotation_start source_chunk_ids (0-based, never held-out)\n")
  writeFile(cfg.outDir / "multifidelity_stats.dat",
    "# generation proposals coarse_fits medium_fits group_cv_fits accepted " &
    "local_wins residual_wins linkage_wins viral_wins seconds\n")
  writeFile(cfg.outDir / "cache_stats.dat",
    "# generation train_hits train_exact train_misses train_saved_rule_rows " &
    "fixed_hits fixed_exact fixed_saved_rule_rows valid_hits valid_exact " &
    "valid_saved_rule_rows bytes_cached_total train_dead_marker_rule_rows " &
    "fixed_dead_marker_rule_rows valid_dead_marker_rule_rows " &
    "train_block_hits train_block_misses train_block_saved_rule_rows " &
    "fixed_block_hits fixed_block_saved_rule_rows valid_block_hits " &
    "valid_block_saved_rule_rows block_bytes_cached\n")
  var orderLog = "# shuffled_position source_chunk_id (0-based); validation chunk ids excluded\n"
  for i, id in order: orderLog.add($i & " " & $id & "\n")
  writeFile(cfg.outDir / "rotation_order.dat", orderLog)
  echo "[start] rules=", cfg.rules, " population=", cfg.population,
       " trainPool=", poolSize, " trainChunks=", cfg.trainChunks,
       " validChunks=", cfg.validChunks, " fixedReference=", indexString(fixedIds),
       " ridgeLambda=", cfg.ridgeLambda,
       " fitness=rotating-LOO archive=fixed-reference-LOO",
       " rotateEvery=", cfg.rotateEvery, " rotateStride=", cfg.rotateStride,
       " tokenMap=none coarseFine=none mutation=floor(exp(U(0,log(LEN))))",
       " virus=", cfg.virusEnabled, " virusTrials=", cfg.virusTrials,
       " cacheEntries=", cfg.cacheEntries, " cacheStride=", cfg.cacheStride,
       " cacheMiB=", cfg.cacheMb, " blockCacheMiB=", cfg.blockCacheMb,
       " blockCacheEntries=", cfg.blockCacheEntries,
       " diffusionSampling=", cfg.diffusionSampling,
       " multifidelity=", cfg.multifidelity, " paired=", cfg.pairedSelection,
       " local=", cfg.localSearch, " residual=", cfg.residualSearch,
       " linkage=", cfg.linkageCross
  if poolSize <= cfg.trainChunks:
    echo "[warning] Entire train pool is used each generation. Increase --chunks or decrease --train for actual rotation."
  for gen in 0..<cfg.generations:
    let started = epochTime()
    let batchIds = trainingIndices(order, gen, cfg.trainChunks,
                                   cfg.rotateEvery, cfg.rotateStride)
    let rotationStep = (gen div cfg.rotateEvery) mod order.len
    let rotationStart = (rotationStep * (cfg.rotateStride mod order.len)) mod order.len
    let batchChanged = gen == 0 or (gen mod cfg.rotateEvery == 0)
    if batchChanged:
      # Rotation changes ROW CONTENT: old intermediate states are invalid.
      clearFeatureCache(trainCache)
      # Even preserved elites and injected Hall-of-Fame candidates must be
      # re-scored on the NEW batch before tournaments and elitist selection.
      for i in 0..<population.len: population[i].score = -Inf
    let trainRows = makeRows(source, batchIds, cfg.noiseSteps, cfg.seed, alphabet,
                             cfg.diffusionSampling)
    if trainRows.len < 3: quit("Insufficient rotating training diffusion rows", 1)
    for i in 0..<population.len:
      if population[i].score == -Inf:
        let m = fit(trainRows, population[i], cfg.ridgeLambda, trainCache, false)
        population[i].score = m.looPearson
        if classify(population[i].score) in {fcNan, fcInf, fcNegInf}:
          population[i].score = -1.0
    population.sort(proc(a, b: Genome): int = cmp(b.score, a.score))

    # Only generation leaders are promoted into Hall-of-Fame, and ONLY after
    # evaluating them on the same fixed reference rows as all previous leaders.
    for i in 0..<min(cfg.elites + 1, population.len):
      var known = false
      for entry in archive:
        if genomeEqual(entry.genome, population[i]): known = true
      if not known:
        let score = fit(fixedRows, population[i], cfg.ridgeLambda, fixedCache, false).looPearson
        insertArchive(archive, population[i], score, cfg.archiveSize)
    let champion = archive[0].genome
    let championFixed = archive[0].referenceScore
    if championFixed > highestFixed:
      highestFixed = championFixed
      saveGenome(cfg.outDir / "best_rules.tsv", champion)
      writeFile(cfg.outDir / "best_score.txt",
        "fixed_reference_LOO=" & $highestFixed & "\n" &
        "generation=" & $gen & "\n" &
        "reference_chunk_ids=" & indexString(fixedIds) & "\n")
    var rotLog = open(cfg.outDir / "rotation.dat", fmAppend)
    rotLog.writeLine($gen & " " & $rotationStart & " " & indexString(batchIds))
    rotLog.close()

    if gen == 0 or gen == cfg.generations - 1 or (gen+1) mod cfg.evalEvery == 0:
      let m = fit(trainRows, champion, cfg.ridgeLambda, trainCache)
      let validScore = validate(validRows, champion, m, validCache)
      let duration = epochTime() - started
      let row = $gen & " " & $metric(m.trainPearson) & " " & $metric(validScore) &
                " " & $m.trainPearson & " " & $validScore & " " & $duration &
                " " & $m.looPearson & " " & $metric(m.looPearson) &
                " " & $championFixed & " " & $metric(championFixed) &
                " " & $rotationStart & "\n"
      var fp = open(cfg.outDir / "training.dat", fmAppend)
      fp.write(row)
      fp.close()
      echo "[generation] ", gen, " trainIds=", indexString(batchIds),
           " trainPearson=", m.trainPearson, " heldoutPearson=", validScore,
           " rotatingLOO=", m.looPearson, " generationBest=", population[0].score,
           " globalBestFixedLOO=", championFixed, " time=", duration, "s"
      if cfg.plotEvery > 0 and (gen == 0 or gen == cfg.generations-1 or
                                (gen+1) mod cfg.plotEvery == 0):
        plot(cfg.outDir, cfg.maWindow)
    else:
      echo "[generation] ", gen, " trainIds=", indexString(batchIds),
           " generationBest=", population[0].score,
           " globalBestFixedLOO=", championFixed,
           " heldout=skipped time=", epochTime() - started, "s"
    var viralChildren: seq[VirusTrial] = @[]
    if cfg.virusEnabled:
      viralChildren = runViralTrials(viruses, population, trainSource,
                                     trainRows, cfg, virusRng, gen, trainCache)
      var bestVirusId = 0
      for i in 1..<viruses.len:
        if viruses[i].trials > 0 and
            (viruses[bestVirusId].trials == 0 or
             viruses[i].advantage > viruses[bestVirusId].advantage):
          bestVirusId = i
      let topVirus = viruses[bestVirusId]
      var f = open(cfg.outDir / "virus_history.dat", fmAppend)
      f.writeLine($gen & " " & $topVirus.advantage & " " &
        $topVirus.improvement & " " & $topVirus.trials & " " & $viruses.len)
      f.close()
      if topVirus.trials > 0 and topVirus.advantage > bestViralAdvantage:
        bestViralAdvantage = topVirus.advantage
        writeFile(cfg.outDir / "best_virus.txt", viralDescription(topVirus.virus) &
          "# mean_paired_advantage=" & $topVirus.advantage &
          " n=" & $topVirus.trials & " gen=" & $gen & "\n")
        writeFile(cfg.outDir / "best_virus.vgen", serializeVirus(topVirus.virus))
      echo "[virus] gen=", gen, " pairedAdvantage=", topVirus.advantage,
           " trials=", topVirus.trials, " highestInfected=", viralChildren[0].score
      viruses = evolveViruses(viruses, virusRng)
    var cacheLog = open(cfg.outDir / "cache_stats.dat", fmAppend)
    let cachedBytes = trainCache.usedBytes + fixedCache.usedBytes + validCache.usedBytes
    cacheLog.writeLine($gen & " " & $trainCache.hits & " " & $trainCache.exactHits &
      " " & $trainCache.misses & " " & $trainCache.savedRuleRows &
      " " & $fixedCache.hits & " " & $fixedCache.exactHits &
      " " & $fixedCache.savedRuleRows & " " & $validCache.hits &
      " " & $validCache.exactHits & " " & $validCache.savedRuleRows &
      " " & $cachedBytes & " " & $trainCache.deadMarkerRuleRows &
      " " & $fixedCache.deadMarkerRuleRows &
      " " & $validCache.deadMarkerRuleRows &
      " " & $trainCache.blockHits & " " & $trainCache.blockMisses &
      " " & $trainCache.blockSavedRuleRows &
      " " & $fixedCache.blockHits & " " & $fixedCache.blockSavedRuleRows &
      " " & $validCache.blockHits & " " & $validCache.blockSavedRuleRows &
      " " & $(trainCache.blockUsedBytes + fixedCache.blockUsedBytes +
               validCache.blockUsedBytes))
    cacheLog.close()
    if gen+1 < cfg.generations:
      var searchStats: SearchStats
      population = refineNextGeneration(population,archive,trainSource,source,
                     order,gen,alphabet,trainRows,viralChildren,cfg,r,
                     trainCache,searchStats)
      var searchLog = open(cfg.outDir / "multifidelity_stats.dat",fmAppend)
      searchLog.writeLine($gen & " " & $searchStats.proposals & " " &
        $searchStats.coarseEvaluations & " " & $searchStats.mediumEvaluations &
        " " & $searchStats.precisionEvaluations & " " & $searchStats.accepted &
        " " & $searchStats.localWins & " " & $searchStats.residualWins &
        " " & $searchStats.linkageWins & " " & $searchStats.viralWins &
        " " & $searchStats.timeSeconds)
      searchLog.close()
      echo "[multifidelity] proposals=", searchStats.proposals,
           " stage1=", searchStats.coarseEvaluations,
           " stage2=", searchStats.mediumEvaluations,
           " groupCV=", searchStats.precisionEvaluations,
           " accepted=", searchStats.accepted,
           " time=", searchStats.timeSeconds, "s"
      # Inject at most virusInject children if they improved upon their matched
      # parent; no stale score is ever carried into the rotated generation.
      if not cfg.multifidelity:
        var added = 0
        for child in viralChildren:
          if added >= cfg.virusInject: break
          if population.len-1-added <= cfg.elites: break
          if child.advantage > 0.0:
            let slot = population.len-1-added
            var injected = copyGenome(child.genome)
            injected.score = -Inf
            population[slot] = injected
            inc added
  echo "[saved] ", cfg.outDir / "best_rules.tsv"

when isMainModule:
  main()
