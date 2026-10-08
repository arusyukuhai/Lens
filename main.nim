## Lens / Replacer -- incremental single-path greedy diff3 differential evolution.
## Depends only on replace.nim, ridge.nim, and the Nim standard library.
## Run: nim c -d:release main.nim && ./main --corpus github-code.txt

import std/[algorithm, json, math, os, osproc, random, sequtils,
            strformat, strutils, times, tables]
import replace, ridge

const
  SearchWidth = 1
  ReplacePasses = 1
  MaxEvaluationState = 8192  # Avoid exponential output-growth evaluation hangs.
  DefaultCorpus = "github-code.txt"
  HistoryHeader = "step,rule_count,rule_index,chunk_index,seconds,rho_current,rho_ma5,rho_best_ma5,infer_current,infer_ma5,infer_best_ma5,combined_current\n"

type
  Rule = object
    a, b: seq[int]
  Model = object
    rules: seq[Rule]
  Evaluation = object
    rho, inference, combined: float64
    readout: seq[float64]
  Candidate = object
    model: Model
    score: Evaluation
    # Compiled immutable patterns are carried across DE offspring; only the
    # single modified index is recompiled.
    patterns: seq[SeqPattern]
    literalTriggers: seq[int]
  FeaturePrefix = object
    state: seq[int]
    counts: seq[float64]
    active: int
    aborted: bool
  InferenceCase = object
    noisy: seq[int]
    clean: seq[int]               # Used ONLY by the final accuracy measurement.
    loci: seq[int]
  Config = object
    corpus: string
    outdir: string
    ruleCount: int
    population: int
    iterations, stagePatience: int
    minChunk, maxChunk, maxChunks: int
    diffusionSteps: int
    ridgeRows, maxRidgeFeatures: int
    ridgeLambda: float64
    inferCases, inferSpan, inferPopulation, inferGenerations, inferMaxLoci: int
    inferNoise, inferWeight: float64
    mutationRate: float64  # Extra mutation probability per DE trial (not per token).
    maxRuleLen: int
    seed, checkpointEvery, printEvery, plotEvery: int
    resume, selfTest: bool
  Context = object
    trajectory: seq[seq[int]]
    prefixes: seq[FeaturePrefix]   # Stage-local invariant prefix cache (single scan).
    cleanliness: seq[float64]
    inferenceCases: seq[InferenceCase]
    vocabulary: seq[int]
    chunkIndex, inferenceSeed: int
  HistoryPoint = object
    step, ruleCount, ruleIndex, chunkIndex: int
    seconds, rho, rhoMA, rhoBestMA, inference, inferMA, inferBestMA, combined: float64
  Trainer = object
    current: Candidate
    history: seq[HistoryPoint]
    completed: int
    baseSeed: int

proc defaults(): Config =
  result = Config(
    corpus: DefaultCorpus, outdir: ".", ruleCount: 1500,
    population: 10, iterations: 20, stagePatience: 20,
    minChunk: 16, maxChunk: 2000, maxChunks: 16384,
    diffusionSteps: 150, ridgeRows: 256, maxRidgeFeatures: 1024,
    ridgeLambda: 4.0, inferCases: 2, inferSpan: 2000,
    inferPopulation: 10, inferGenerations: 20, inferMaxLoci: 24,
    inferNoise: 0.10, inferWeight: 1.0, mutationRate: 0.25, maxRuleLen: 64,
    seed: 0, checkpointEvery: 1, printEvery: 1, plotEvery: 1,
    resume: false, selfTest: false
  )

proc usage() =
  echo """Incremental Lens trainer -- single-path greedy differential evolution.
  nim c -d:release main.nim
  ./main --corpus github-code.txt --rules 1500 --population 40 --iterations 40

Options (both --key=VALUE and --key VALUE):
  --corpus PATH            Corpus: ===SPLIT=== separated snippets (default github-code.txt)
  --outdir DIR             Graphs, CSV, greedy checkpoint (default .)
  --rules N               Total rules (default 1500; number of stages N*(N+1)/2)
  --population N          Single DE population size (default 40, minimum 4)
  --iterations N          DE iterations per stage (default 40)
  --stage-patience N      Stop stage after N iterations without a new max rho (default 5; 0 disables)
  --max-chunk N           Ignore snippets longer than N bytes (default 256)
  --min-chunk N           Skip shorter snippets (default 16)
  --max-chunks N          Maximum loaded chunks (default 8192)
  --diffusion-steps N     Fixed replacement steps per chunk (default 100)
  --ridge-rows N          Trajectories used to FIT rank ridge (default 48)
  --ridge-features N      Max rank-ridge features (default 128)
  --ridge-lambda X        Ridge regularization (default 4)
  --infer-cases N         Reconstruction cases per candidate (default 2)
  --infer-span N          Maximum context window for reconstruction (default 96)
  --infer-population N    Inner string-side GA population (default 10)
  --infer-generations N   Inner string-side GA generations (default 5)
  --infer-max-loci N      Max mutated positions per inference case (default 16)
  --infer-noise X         Corrupted positions fraction (default .10)
  --infer-weight X        Weight on accuracy for reporting joint only (default .5)
  --max-rule-len N        Max length of each pattern/replacement (default 16)
  --mutation-rate X       Additional mutation chance per DE trial (default .05; 0..1)
  --seed N                RNG seed; 0 uses time
  --checkpoint-every N    Save after N stage updates (default 1)
  --plot-every N          Redraw PNG every N stage updates (default 1)
  --print-every N         Progress every N DE iterations (default 5)
  --resume                Resume greedy_checkpoint.json (can migrate v2 beam checkpoint)
  --self-test             Small built-in corpus and short training smoke run
  --help                  Show this message
"""

proc parseConfig(): Config =
  result = defaults()
  let params = commandLineParams()
  var i = 0
  while i < params.len:
    let p = params[i]
    if p in ["-h", "--help"]:
      usage()
      quit(0)
    if p == "--resume":
      result.resume = true
      inc i
      continue
    if p == "--self-test":
      result.selfTest = true
      inc i
      continue
    if not p.startsWith("--"):
      quit("Unknown argument: " & p)
    let eq = p.find('=')
    var key, value: string
    if eq >= 0:
      key = p[2 ..< eq]
      value = p[eq+1 .. ^1]
    else:
      key = p[2 .. ^1]
      inc i
      if i >= params.len: quit("Missing value for --" & key)
      value = params[i]
    try:
      case key
      of "corpus": result.corpus = value
      of "outdir": result.outdir = value
      of "rules": result.ruleCount = parseInt(value)
      of "population": result.population = parseInt(value)
      of "iterations": result.iterations = parseInt(value)
      of "stage-patience": result.stagePatience = parseInt(value)
      of "min-chunk": result.minChunk = parseInt(value)
      of "max-chunk": result.maxChunk = parseInt(value)
      of "max-chunks": result.maxChunks = parseInt(value)
      of "diffusion-steps": result.diffusionSteps = parseInt(value)
      of "ridge-rows": result.ridgeRows = parseInt(value)
      of "ridge-features": result.maxRidgeFeatures = parseInt(value)
      of "ridge-lambda": result.ridgeLambda = parseFloat(value)
      of "infer-cases": result.inferCases = parseInt(value)
      of "infer-span": result.inferSpan = parseInt(value)
      of "infer-population": result.inferPopulation = parseInt(value)
      of "infer-generations": result.inferGenerations = parseInt(value)
      of "infer-max-loci": result.inferMaxLoci = parseInt(value)
      of "infer-noise": result.inferNoise = parseFloat(value)
      of "infer-weight": result.inferWeight = parseFloat(value)
      of "max-rule-len": result.maxRuleLen = parseInt(value)
      of "mutation-rate": result.mutationRate = parseFloat(value)
      of "seed": result.seed = parseInt(value)
      of "checkpoint-every": result.checkpointEvery = parseInt(value)
      of "print-every": result.printEvery = parseInt(value)
      of "plot-every": result.plotEvery = parseInt(value)
      else: quit("Unknown option: --" & key)
    except ValueError:
      quit("Bad option value: --" & key & "=" & value)
    inc i
  if result.ruleCount < 1 or result.population < 4 or result.iterations < 1 or
      result.stagePatience < 0 or
      result.minChunk < 2 or result.maxChunk < result.minChunk or
      result.maxChunks < 1 or result.diffusionSteps < 1 or
      result.ridgeRows < 2 or result.maxRidgeFeatures < 1 or
      result.ridgeLambda <= 0.0 or result.inferCases < 1 or
      result.inferSpan < 2 or result.inferPopulation < 3 or
      result.inferGenerations < 1 or result.inferMaxLoci < 1 or
      result.inferNoise <= 0.0 or result.inferNoise > 1.0 or
      result.inferWeight < 0 or result.inferWeight > 1 or
      result.maxRuleLen < 1 or result.maxRuleLen > 64 or
      classify(result.mutationRate) in {fcNan, fcInf, fcNegInf} or
      result.mutationRate < 0.0 or result.mutationRate > 1.0 or
      result.checkpointEvery < 1 or result.plotEvery < 1 or
      result.printEvery < 1:
    quit("Invalid configuration; use --help to see valid ranges")
  if result.selfTest:
    result.ruleCount = 2
    result.population = 4
    result.iterations = 2
    result.minChunk = 8
    result.maxChunk = 24
    result.ridgeRows = 8
    result.inferCases = 1
    result.inferSpan = 16
    result.inferPopulation = 4
    result.inferGenerations = 2
    result.inferMaxLoci = 3
    result.maxChunks = 2

proc byteSeq(s: string): seq[int] =
  result = newSeq[int](s.len)
  for i, ch in s:
    result[i] = ord(ch)

proc cloneInts(a: openArray[int]): seq[int] =
  result = newSeq[int](a.len)
  for i, x in a: result[i] = x

proc admitSnippet(buffer: string, cfg: Config, reservoirRng: var Rand,
                  seen: var int, chunks: var seq[seq[int]]) =
  # Admit the ORIGINAL snippet or discard it entirely. Never crop, and do not
  # let rejected snippets affect reservoir sampling probabilities.
  if buffer.len < cfg.minChunk or buffer.len > cfg.maxChunk: return
  inc seen
  if chunks.len < cfg.maxChunks:
    chunks.add(byteSeq(buffer))
  else:
    # Deterministic reservoir sampling avoids bias toward the start of a corpus.
    let slot = rand(reservoirRng, seen-1)
    if slot < cfg.maxChunks:
      chunks[slot] = byteSeq(buffer)

proc appendCorpusLine(buffer: var string, oversized: var bool,
                      line: string, maxChunk: int) =
  # Streaming guard: once a snippet exceeds the limit, stop buffering it
  # entirely until ===SPLIT===. The extra byte is the original newline.
  if oversized: return
  if line.len >= maxChunk - buffer.len:
    oversized = true
    buffer.setLen(0)
  else:
    buffer.add(line)
    buffer.add('\n')

proc loadChunks(cfg: Config): seq[seq[int]] =
  if cfg.selfTest:
    return @[byteSeq("alpha beta gamma delta"), byteSeq("int add(int a,int b)")]
  if not fileExists(cfg.corpus):
    quit("Corpus not found: " & cfg.corpus & " (supply --corpus PATH)")
  let f = open(cfg.corpus, fmRead)
  defer: f.close()
  var buffer = ""
  var oversized = false
  var seen = 0
  var reservoirRng = initRand(91823'i64) # Independent of stage RNG / resume.
  while not f.endOfFile:
    let line = f.readLine()
    if line == "===SPLIT===":
      if not oversized:
        admitSnippet(buffer, cfg, reservoirRng, seen, result)
      buffer.setLen(0)
      oversized = false
    else:
      appendCorpusLine(buffer, oversized, line, cfg.maxChunk)
  if not oversized:
    admitSnippet(buffer, cfg, reservoirRng, seen, result)
  if result.len == 0:
    quit("Corpus contains no chunks between " & $cfg.minChunk & " and " &
         $cfg.maxChunk & " bytes: " & cfg.corpus)

proc checkChunkAdmission() =
  # Equality is allowed: maxChunk is an inclusive upper bound.
  let cfg = Config(minChunk: 3, maxChunk: 8, maxChunks: 2)
  var rng = initRand(91823'i64)
  var seen = 0
  var chunks: seq[seq[int]] = @[]
  admitSnippet("12", cfg, rng, seen, chunks)
  admitSnippet("123456789", cfg, rng, seen, chunks)
  doAssert seen == 0 and chunks.len == 0,
           "short/oversized snippets must not enter the reservoir"
  admitSnippet("abc", cfg, rng, seen, chunks)
  admitSnippet("12345678", cfg, rng, seen, chunks)
  doAssert seen == 2 and chunks == @[byteSeq("abc"), byteSeq("12345678")],
           "valid snippets must remain whole, including maxChunk boundary"
  admitSnippet("ABCDEFGHI", cfg, rng, seen, chunks)
  doAssert seen == 2 and chunks.len == 2,
           "oversized snippets must not change reservoir probabilities"
  # Verify memory-bounded streaming: clear an oversized candidate, ignore its
  # remaining lines, and accept a fresh snippet after a split.
  var buffer = ""
  var oversized = false
  appendCorpusLine(buffer, oversized, "1234567", cfg.maxChunk)
  doAssert not oversized and buffer == "1234567\n"
  appendCorpusLine(buffer, oversized, "x", cfg.maxChunk)
  doAssert oversized and buffer.len == 0
  appendCorpusLine(buffer, oversized, "ignored", cfg.maxChunk)
  doAssert oversized and buffer.len == 0
  buffer.setLen(0)
  oversized = false
  appendCorpusLine(buffer, oversized, "ab", cfg.maxChunk)
  doAssert not oversized and buffer == "ab\n"
  echo "PASS: minChunk/maxChunk whole-snippet filtering and streaming skip"

proc vocabularyFrom(chunks: seq[seq[int]]): seq[int] =
  ## Weighted byte vocabulary, built before inference and independent of its clean target.
  ## Cap to avoid megabytes of repeated high-frequency bytes.
  for chunk in chunks:
    for v in chunk:
      if result.len < 32768: result.add(v)
  if result.len == 0:
    for k in 32..126: result.add(k)

proc logUniformLen(maxLen: int): int {.inline.} =
  if maxLen <= 1: return max(1, maxLen)
  clamp(int(round(exp(rand(1.0) * ln(float64(maxLen))))), 1, maxLen)

proc randomOther(orig: int, vocab: seq[int]): int =
  for attempts in 0..<12:
    let x = (if vocab.len > 0: vocab[rand(vocab.high)] else: rand(255))
    if x != orig: return x
  (orig + 1 + rand(254)) mod 256

proc randomOtherWithRng(orig: int, vocab: seq[int], rng: var Rand): int =
  for attempts in 0..<12:
    let x = (if vocab.len > 0: vocab[rand(rng, vocab.high)] else: rand(rng, 255))
    if x != orig: return x
  (orig + 1 + rand(rng, 254)) mod 256

proc logUniformWithRng(maxLen: int, rng: var Rand): int =
  if maxLen <= 1: return max(1, maxLen)
  clamp(int(round(exp(rand(rng, 1.0) * ln(float64(maxLen))))), 1, maxLen)

proc randomRule(source: seq[int], cfg: Config, vocab: seq[int]): Rule =
  let maxN = min(cfg.maxRuleLen, 10)
  let n = min(source.len, logUniformLen(max(1, maxN)))
  if source.len >= n and rand(99) < 85:
    let start = rand(source.len-n)
    result.a = cloneInts(source[start ..< start+n])
    result.b = cloneInts(result.a)
    let at = rand(result.b.high)
    result.b[at] = randomOther(result.b[at], vocab)
  else:
    let na = logUniformLen(maxN)
    let nb = logUniformLen(maxN)
    for j in 0..<na:
      if rand(99) < 12: result.a.add(-rand(1..3))
      else: result.a.add(vocab[rand(vocab.high)])
    for j in 0..<nb:
      if rand(99) < 15: result.b.add(-rand(1..3))
      else: result.b.add(vocab[rand(vocab.high)])

proc crossoverSegment(baseParent, donorParent: openArray[int], maxLength: int): seq[int] =
  if baseParent.len == 0: return cloneInts(donorParent)
  if donorParent.len == 0: return cloneInts(baseParent)
  let removeLen = logUniformLen(baseParent.len)
  let insertLen = logUniformLen(donorParent.len)
  let p = rand(baseParent.len-removeLen)
  let q = rand(donorParent.len-insertLen)
  result = newSeqOfCap[int](max(1, baseParent.len-removeLen+insertLen))
  for j in 0..<p: result.add(baseParent[j])
  for j in q..<q+insertLen: result.add(donorParent[j])
  for j in p+removeLen..<baseParent.len: result.add(baseParent[j])
  if result.len > maxLength: result.setLen(maxLength)

proc diff3Mix(target, donorBase, donorChanged: openArray[int], maxLength: int): seq[int] =
  ## True difference transfer: (donorBase -> donorChanged) applied to target.
  ## Empty diff3 result denotes a conflict in the provided replace.nim.
  result = diff3Apply(donorBase, donorChanged, target)
  if result.len == 0 or result.len > maxLength:
    case rand(3)
    of 0: result = crossoverSegment(target, donorChanged, maxLength)
    of 1: result = crossoverSegment(donorChanged, target, maxLength)
    of 2: result = cloneInts(target)
    else: result = cloneInts(donorChanged)
  if result.len == 0:
    result = cloneInts(target)

proc mutateSeq(seqIn: openArray[int], maxLength: int, vocab: seq[int], pattern: bool): seq[int] =
  result = cloneInts(seqIn)
  let count = logUniformLen(max(1, result.len))
  for mutation in 0..<count:
    let op = rand(4)
    if op == 0 and result.len < maxLength:
      let p = rand(result.len)
      let v = if pattern and rand(99) < 10: -rand(1..3)
              elif not pattern and rand(99) < 15: -rand(1..3)
              else: vocab[rand(vocab.high)]
      result.insert(v, p)
    elif op == 1 and result.len > 1:
      result.delete(rand(result.high))
    elif op == 2 and result.len > 1:
      let p = rand(result.high)
      let q = rand(result.high)
      swap(result[p], result[q])
    elif result.len > 0:
      let p = rand(result.high)
      result[p] = if pattern and rand(99) < 10: -rand(1..3)
                  elif not pattern and rand(99) < 15: -rand(1..3)
                  else: randomOther(result[p], vocab)
  if result.len == 0:
    result.add(vocab[rand(vocab.high)])
  if result.len > maxLength: result.setLen(maxLength)

proc mutateRuleOnce(r: Rule, cfg: Config, vocab: seq[int]): Rule =
  ## Modify exactly ONE of the rule's two sequences. An attempted mutation
  ## must actually change its content (a swap of equal tokens may be a no-op).
  result = r
  if rand(1) == 0:
    result.a = mutateSeq(r.a, cfg.maxRuleLen, vocab, true)
    if result.a == r.a:
      let pos = rand(result.a.high)
      result.a[pos] = randomOther(result.a[pos], vocab)
  else:
    result.b = mutateSeq(r.b, cfg.maxRuleLen, vocab, false)
    if result.b == r.b:
      let pos = rand(result.b.high)
      result.b[pos] = randomOther(result.b[pos], vocab)

proc trialRule(a, b, c: Rule, cfg: Config, vocab: seq[int]): Rule =
  ## DE core: transfer donor edit (b -> c) onto the target rule a.
  result.a = diff3Mix(a.a, b.a, c.a, cfg.maxRuleLen)
  result.b = diff3Mix(a.b, b.b, c.b, cfg.maxRuleLen)
  ## Rare, independent local mutation. The old 85% per-side mutations and
  ## 9% random restarts overwhelmed the differential-evolution operation.
  if cfg.mutationRate >= 1.0 or
      (cfg.mutationRate > 0.0 and rand(1.0) < cfg.mutationRate):
    result = mutateRuleOnce(result, cfg, vocab)

proc changeRule(m: Model, ruleIndex: int, replacement: Rule): Model =
  ## CHANGE only ruleIndex. In particular, rules AFTER ruleIndex remain
  ## present AND active during all fitness / inference evaluations. They are
  ## frozen, not masked, removed, skipped, or replaced by no-op rules.
  doAssert ruleIndex >= 0 and ruleIndex < m.rules.len
  result.rules = newSeq[Rule](m.rules.len)
  for i, rule in m.rules:
    result.rules[i] = (if i == ruleIndex: replacement else: rule)

  # Optional expensive invariant checks for regression / debugging builds.
  when defined(lensCheckFrozen):
    doAssert result.rules.len == m.rules.len
    for i in 0..<m.rules.len:
      if i != ruleIndex:
        doAssert result.rules[i].a == m.rules[i].a
        doAssert result.rules[i].b == m.rules[i].b

proc appendRule(m: Model, rule: Rule): Model =
  result.rules = newSeq[Rule](m.rules.len+1)
  for i, r in m.rules: result.rules[i] = r
  result.rules[m.rules.len] = rule

proc compileModel(m: Model): seq[SeqPattern] =
  result = newSeq[SeqPattern](m.rules.len)
  for i, rule in m.rules: result[i] = compileSeqPattern(rule.a)

proc firstLiteral(rule: Rule): int {.inline.} =
  ## Any real match MUST contain at least one occurrence of this literal.
  ## -1 means an all-wildcard pattern: never reject it by the byte filter.
  for v in rule.a:
    if v >= 0: return v
  -1

proc literalTriggers(m: Model): seq[int] =
  result = newSeq[int](m.rules.len)
  for i, r in m.rules: result[i] = firstLiteral(r)

proc tokenBits(state: openArray[int]): array[8, uint64] =
  ## Exact 512-token presence set, no false negatives for valid literal probes.
  ## Values outside 0..511 are handled by the fallback matcher.
  for v in state:
    if v >= 0 and v < 512:
      result[v shr 6] = result[v shr 6] or (1'u64 shl (v and 63))

proc mightMatch(bits: array[8, uint64], marker: int): bool {.inline.} =
  if marker < 0 or marker >= 512: return true
  (bits[marker shr 6] and (1'u64 shl (marker and 63))) != 0

proc replaceRuleCompatible(state: var seq[int], pat: SeqPattern,
                           replacement: openArray[int]): bool =
  ## Compatible with older replace.nim; the downloaded ZIP has the fast API.
  when compiles(replaceSeqCompiledInPlace(state, pat, replacement)):
    result = replaceSeqCompiledInPlace(state, pat, replacement)
  else:
    let transformed = replaceSeqCompiled(state, pat, replacement)
    if transformed.changed and transformed.data != state:
      state = transformed.data
      result = true

proc prefixFor(input: openArray[int], m: Model,
               patterns: seq[SeqPattern], markers: seq[int],
               active: int): FeaturePrefix =
  ## Compute the fixed single-scan prefix [0, active). Every offspring
  ## during a stage has precisely the same rules there.
  result.active = active
  result.state = cloneInts(input)
  result.counts = newSeq[float64](active)
  if active == 0: return
  var bits = tokenBits(result.state)
  for i in 0..<active:
    if not mightMatch(bits, markers[i]): continue
    if replaceRuleCompatible(result.state, patterns[i], m.rules[i].b):
      result.counts[i] += 1.0
      if result.state.len > MaxEvaluationState:
        result.aborted = true
        return
      bits = tokenBits(result.state)

proc predictFeatures(input: openArray[int], m: Model,
                     patterns: seq[SeqPattern],
                     markers: seq[int] = @[],
                     prefix: FeaturePrefix = FeaturePrefix(),
                     usePrefix = false): seq[float64] =
  ## EXACT full-model single-scan semantics. The invariant prefix is reused,
  ## then all remaining rules (including the edited rule and its suffix) run once.
  doAssert patterns.len == m.rules.len
  result = newSeq[float64](m.rules.len)
  var state: seq[int]
  if usePrefix:
    state = cloneInts(prefix.state)
    for i, count in prefix.counts: result[i] = count
    if prefix.aborted: return
  else:
    state = cloneInts(input)
  var bits = tokenBits(state)
  # Exactly ONE left-to-right traversal through the full rule list.
  let firstRule = if usePrefix: prefix.active else: 0
  for i in firstRule..<m.rules.len:
    let marker = if markers.len == m.rules.len: markers[i]
                 else: firstLiteral(m.rules[i])
    if not mightMatch(bits, marker): continue
    if replaceRuleCompatible(state, patterns[i], m.rules[i].b):
      result[i] += 1.0
      if state.len > MaxEvaluationState:
        return
      bits = tokenBits(state)

proc primeStagePrefixes(ctx: var Context, seed: Candidate,
                        active: int) =
  ## Stage-local memory is freed with ctx. Invalid across a change to frozen
  ## prefix rules, but safe during this one-rule-only DE stage.
  ctx.prefixes = newSeq[FeaturePrefix](ctx.trajectory.len)
  for i, input in ctx.trajectory:
    ctx.prefixes[i] = prefixFor(input, seed.model, seed.patterns,
                                seed.literalTriggers, active)

proc shuffleIndices(n: int): seq[int] =
  result = toSeq(0..<n)
  shuffle(result)

proc makeTrajectory(clean, vocab: seq[int], cfg: Config): tuple[x: seq[seq[int]], y: seq[float64]] =
  ## Fixed number of diffusion steps regardless of source length.
  ## Visit each byte once per shuffled sweep. For short chunks, reshuffle and
  ## revisit positions; hence the exact corruption fraction must be tracked,
  ## not inferred from k/length (a revisited position may become clean again).
  assert clean.len > 0
  var state = cloneInts(clean)
  var order = shuffleIndices(clean.len)
  var incorrect = 0
  result.x.add(cloneInts(state))
  result.y.add(1.0)
  for k in 0..<cfg.diffusionSteps:
    if k > 0 and k mod clean.len == 0:
      shuffle(order)
    let p = order[k mod clean.len]
    let before = state[p]
    state[p] = randomOther(before, vocab)
    if before == clean[p] and state[p] != clean[p]:
      inc incorrect
    elif before != clean[p] and state[p] == clean[p]:
      dec incorrect
    result.x.add(cloneInts(state))
    result.y.add(1.0 - float64(incorrect)/float64(clean.len))

proc makeInferenceCase(clean, vocab: seq[int], cfg: Config): InferenceCase =
  let span = min(clean.len, cfg.inferSpan)
  let start = if span < clean.len: rand(clean.len-span) else: 0
  result.clean = cloneInts(clean[start ..< start+span])
  result.noisy = cloneInts(result.clean)
  let changeN = clamp(int(round(float64(span)*cfg.inferNoise)), 1,
                       min(span, cfg.inferMaxLoci))
  let order = shuffleIndices(span)
  for i in 0..<changeN:
    let p = order[i]
    result.loci.add(p)
    result.noisy[p] = randomOther(result.noisy[p], vocab)

proc makeContext(chunk: seq[int], index: int, vocab: seq[int], cfg: Config): Context =
  let diffusion = makeTrajectory(chunk, vocab, cfg)
  result.chunkIndex = index
  result.inferenceSeed = rand(2_000_000_000)
  result.trajectory = diffusion.x
  result.cleanliness = diffusion.y
  result.vocabulary = vocab
  for i in 0..<cfg.inferCases:
    result.inferenceCases.add(makeInferenceCase(chunk, vocab, cfg))

proc selectRows(n, wanted: int): seq[int] =
  let sampleN = min(n, wanted)
  if sampleN <= 0: return
  if sampleN == 1: return @[0]
  for i in 0..<sampleN:
    result.add(i*(n-1) div (sampleN-1))

proc predictReadout(input: openArray[int], m: Model,
                    patterns: seq[SeqPattern], markers: seq[int],
                    weights: seq[float64]): float64 =
  let features = predictFeatures(input, m, patterns, markers)
  for i, feat in features:
    result += feat * weights[i]

proc mutateCandidate(input: openArray[int], loci, vocab: seq[int], rng: var Rand): seq[int] =
  result = cloneInts(input)
  if loci.len == 0: return
  let count = logUniformWithRng(loci.len, rng)
  for j in 0..<count:
    let p = loci[rand(rng, loci.high)]
    result[p] = randomOtherWithRng(result[p], vocab, rng)

proc recombineCandidate(a, b: openArray[int], loci: seq[int], rng: var Rand): seq[int] =
  result = cloneInts(a)
  for p in loci:
    if rand(rng, 1) == 1: result[p] = b[p]

proc evaluateInferenceOne(c: InferenceCase, m: Model,
                          patterns: seq[SeqPattern], markers: seq[int],
                          weights: seq[float64],
                          vocab: seq[int], cfg: Config, seed: int): float64 =
  var rng = initRand(int64(seed))
  ## The search below cannot inspect c.clean. It is used ONLY in the final loop.
  var pool = newSeq[seq[int]](cfg.inferPopulation)
  pool[0] = cloneInts(c.noisy)
  for i in 1..<pool.len:
    pool[i] = mutateCandidate(c.noisy, c.loci, vocab, rng)

  var best = cloneInts(c.noisy)
  var bestScore = -Inf
  # A score depends on this model AND fitted ridge weights; this cache
  # must be recreated for each outer evaluation, never shared across models.
  var inferenceMemo = initTable[string, float64]()
  for gen in 0..cfg.inferGenerations:
    var scores = newSeq[float64](pool.len)
    for j in 0..<pool.len:
      let key = $pool[j]  # exact integer-sequence serialization
      if inferenceMemo.hasKey(key):
        scores[j] = inferenceMemo[key]
      else:
        scores[j] = predictReadout(pool[j], m, patterns, markers, weights)
        if inferenceMemo.len < 2048: inferenceMemo[key] = scores[j]
      if scores[j] > bestScore:
        bestScore = scores[j]
        best = cloneInts(pool[j])
    if gen == cfg.inferGenerations: break
    var order = toSeq(0..<pool.len)
    shuffle(rng, order)
    order.sort(proc(a, b: int): int =
      if scores[a] > scores[b]: -1
      elif scores[a] < scores[b]: 1
      else: cmp(a, b)
    )
    var nextPool: seq[seq[int]] = @[cloneInts(pool[order[0]]), cloneInts(pool[order[1]])]
    let parentRange = max(2, pool.len div 2)
    while nextPool.len < cfg.inferPopulation:
      let p = order[rand(rng, parentRange-1)]
      let q = order[rand(rng, parentRange-1)]
      let base = recombineCandidate(pool[p], pool[q], c.loci, rng)
      nextPool.add(mutateCandidate(base, c.loci, vocab, rng))
    pool = move(nextPool)

  if c.loci.len == 0: return 0.0
  for p in c.loci:
    if best[p] == c.clean[p]: result += 1.0
  result /= float64(c.loci.len)

proc objective(rho, infAcc: float64, cfg: Config): float64 =
  ## Reporting ONLY. Neither DE acceptance nor greedy selection uses this score.
  (1.0-cfg.inferWeight) * clamp((rho+1.0)*0.5, 0.0, 1.0) +
      cfg.inferWeight * clamp(infAcc, 0.0, 1.0)

proc evaluate(m: Model, patterns: seq[SeqPattern], markers: seq[int],
              ctx: Context, cfg: Config): Evaluation =
  var fullX = newSeq[seq[float64]](ctx.trajectory.len)
  let prefixAvailable = ctx.prefixes.len == ctx.trajectory.len
  for i, txt in ctx.trajectory:
    if prefixAvailable:
      fullX[i] = predictFeatures(txt, m, patterns, markers,
                                 ctx.prefixes[i], true)
    else:
      fullX[i] = predictFeatures(txt, m, patterns, markers)
  let chosen = selectRows(fullX.len, cfg.ridgeRows)
  var fitX: seq[seq[float64]] = @[]
  var fitY: seq[float64] = @[]
  var caseIds: seq[int] = @[]
  for j in chosen:
    fitX.add(fullX[j])
    fitY.add(ctx.cleanliness[j])
    caseIds.add(0)
  let zeroBaseline = newSeq[float64](fitX.len)
  result.readout = fitPairwiseRankRidgeAllModel(
    fitX, zeroBaseline, fitY, caseIds,
    cfg.ridgeLambda, cfg.maxRidgeFeatures
  ).weights
  var predicted = newSeq[float64](fullX.len)
  for i in 0..<fullX.len:
    for k, count in fullX[i]:
      predicted[i] += count * result.readout[k]
  result.rho = spearman(predicted, ctx.cleanliness)
  if classify(result.rho) in {fcNan, fcInf, fcNegInf}: result.rho = -1.0
  for j, infCase in ctx.inferenceCases:
    result.inference += evaluateInferenceOne(
      infCase, m, patterns, markers, result.readout, ctx.vocabulary, cfg,
      ctx.inferenceSeed + j*7919)
  result.inference /= float64(ctx.inferenceCases.len)
  result.combined = objective(result.rho, result.inference, cfg)

proc dominates(a, b: Evaluation): bool =
  (a.rho >= b.rho and a.inference >= b.inference) and
    (a.rho > b.rho or a.inference > b.inference)

proc candidateFrom(m: Model, ctx: Context, cfg: Config): Candidate =
  result.model = m
  result.patterns = compileModel(m)
  result.literalTriggers = literalTriggers(m)
  result.score = evaluate(m, result.patterns, result.literalTriggers, ctx, cfg)

proc candidateWithEditedRule(parent: Candidate, ruleIndex: int,
                             replacement: Rule, ctx: Context,
                             cfg: Config): Candidate =
  ## Same rule -> same state and fitness in this fixed evaluation context.
  if replacement.a == parent.model.rules[ruleIndex].a and
      replacement.b == parent.model.rules[ruleIndex].b:
    return parent
  result.model = changeRule(parent.model, ruleIndex, replacement)
  # Do NOT recompile frozen rules. SeqPattern's internal sequences are
  # immutable; shallow assignment of each pattern is safe here.
  result.patterns = newSeq[SeqPattern](parent.patterns.len)
  result.literalTriggers = newSeq[int](parent.literalTriggers.len)
  for i in 0..<parent.patterns.len:
    if i == ruleIndex:
      result.patterns[i] = compileSeqPattern(replacement.a)
      result.literalTriggers[i] = firstLiteral(replacement)
    else:
      result.patterns[i] = parent.patterns[i]
      result.literalTriggers[i] = parent.literalTriggers[i]
  result.score = evaluate(result.model, result.patterns,
                          result.literalTriggers, ctx, cfg)

proc ruleKey(r: Rule): string =
  ## Unambiguous representation, including negative tokens and lengths.
  result = $r.a.len & ":"
  for x in r.a:
    result.add($x)
    result.add(',')
  result.add('|')
  result.add($r.b.len)
  result.add(':')
  for x in r.b:
    result.add($x)
    result.add(',')

proc greedyAccept(incumbent: var Candidate, proposal: Candidate): bool =
  ## Single incumbent: NEVER trade Spearman for inference, or vice versa.
  ## The objective is kept for logging only; it cannot override this gate.
  if dominates(proposal.score, incumbent.score):
    incumbent = proposal
    return true
  false

proc evolveGreedy(seed: Candidate, ruleIndex: int, ctx: Context,
                  cfg: Config, revisiting: bool): Candidate =
  ## One local differential-evolution population for exactly ONE rule.
  ## The model's other rules stay fixed AND active during full evaluation.
  ## The global incumbent follows a monotonic chain of Pareto improvements,
  ## independent of which population member discovers each improvement.
  var population = newSeq[Candidate](cfg.population)
  # Cache only inside this stage: the evaluated chunk, frozen rules and
  # inner-GA seed change between stages. A fixed cap limits memory usage.
  var stageMemo = initTable[string, Candidate]()
  var cacheHits = 0
  # Reserved elite: the original rule survives this entire stage unchanged.
  population[0] = seed
  stageMemo[ruleKey(seed.model.rules[ruleIndex])] = seed
  result = seed
  for j in 1..<population.len:
    var r: Rule
    if not revisiting and j mod 5 == 0:
      # Preserve broad exploration only when introducing a genuinely new rule.
      r = randomRule(ctx.trajectory[0], cfg, ctx.vocabulary)
    else:
      r = seed.model.rules[ruleIndex]
      r.a = mutateSeq(r.a, cfg.maxRuleLen, ctx.vocabulary, true)
      r.b = mutateSeq(r.b, cfg.maxRuleLen, ctx.vocabulary, false)
    let key = ruleKey(r)
    if stageMemo.hasKey(key):
      population[j] = stageMemo[key]
      inc cacheHits
    else:
      population[j] = candidateWithEditedRule(seed, ruleIndex, r, ctx, cfg)
      if stageMemo.len >= 128: stageMemo.clear()
      stageMemo[key] = population[j]
    discard greedyAccept(result, population[j])

  var highestRho = result.score.rho
  var staleIterations = 0
  for iter in 1..cfg.iterations:
    ## Asynchronous DE: accepted trials are immediately available as donors.
    var accepted = 0
    var greedyAdvances = 0
    for i in 1..<population.len:
      # Index 0 is the immutable source-rule elite, but remains a DE donor.
      var ids: seq[int] = @[]
      for j in 0..<population.len:
        if j != i: ids.add(j)
      shuffle(ids)
      let current = population[i]
      let r = trialRule(current.model.rules[ruleIndex],
        population[ids[0]].model.rules[ruleIndex],
        population[ids[1]].model.rules[ruleIndex], cfg, ctx.vocabulary)
      let key = ruleKey(r)
      var trial: Candidate
      if stageMemo.hasKey(key):
        trial = stageMemo[key]
        inc cacheHits
      else:
        trial = candidateWithEditedRule(current, ruleIndex, r, ctx, cfg)
        if stageMemo.len >= 128: stageMemo.clear()
        stageMemo[key] = trial
      # Each DE individual can improve ONLY by strict Pareto dominance.
      if dominates(trial.score, current.score):
        population[i] = trial
        inc accepted
        # Global incumbent is greedily updated on exactly the SAME criterion.
        if greedyAccept(result, trial):
          inc greedyAdvances
    # Count complete DE iterations, not individual trial evaluations.
    # The stage-local maximum includes the initial population, and is reset
    # only by a strictly higher rho (an inference-only improvement is not enough).
    if result.score.rho > highestRho:
      highestRho = result.score.rho
      staleIterations = 0
    else:
      inc staleIterations
    let earlyStop = cfg.stagePatience > 0 and staleIterations >= cfg.stagePatience and highestRho > 0
    if iter mod cfg.printEvery == 0 or iter == cfg.iterations or earlyStop:
      echo &"  greedy iter={iter}/{cfg.iterations} rho={result.score.rho:.6f} infer={result.score.inference:.4f} joint={result.score.combined:.6f} accepted={accepted} advances={greedyAdvances} cacheHits={cacheHits} staleRho={staleIterations}"
    if earlyStop:
      echo &"  stage early stop: max rho did not improve for {staleIterations} consecutive DE iterations (best={highestRho:.6f})"
      break

proc averageLast(history: seq[HistoryPoint], width: int, inference: bool): float64 =
  let fromIndex = max(0, history.len-width)
  let count = history.len-fromIndex
  if count == 0: return 0.0
  for i in fromIndex..<history.len:
    result += (if inference: history[i].inference else: history[i].rho)
  result /= float64(count)

proc addHistory(t: var Trainer, best: Candidate, ruleIndex, chunkIndex: int,
                elapsed: float64) =
  var p = HistoryPoint(step: t.completed, ruleCount: best.model.rules.len,
    ruleIndex: ruleIndex, chunkIndex: chunkIndex, seconds: elapsed,
    rho: best.score.rho, inference: best.score.inference,
    combined: best.score.combined)
  t.history.add(p)
  let last = t.history.high
  t.history[last].rhoMA = averageLast(t.history, 5, false)
  t.history[last].inferMA = averageLast(t.history, 5, true)
  # IMPORTANT: the running maximum is of RAW 5-point averages.
  # Log transform occurs ONLY when plotting the averaged rho.
  if last < 4:
    # A width-five best is undefined for the first four observations.
    t.history[last].rhoBestMA = -1.0
    t.history[last].inferBestMA = 0.0
  elif last == 4:
    t.history[last].rhoBestMA = t.history[last].rhoMA
    t.history[last].inferBestMA = t.history[last].inferMA
  else:
    t.history[last].rhoBestMA = max(t.history[last-1].rhoBestMA, t.history[last].rhoMA)
    t.history[last].inferBestMA = max(t.history[last-1].inferBestMA, t.history[last].inferMA)

proc saveHistory(t: Trainer, outdir: string) =
  let path = outdir / "training_history.csv"
  let f = open(path, fmWrite)
  defer: f.close()
  f.write(HistoryHeader)
  for p in t.history:
    f.write(&"{p.step},{p.ruleCount},{p.ruleIndex+1},{p.chunkIndex},{p.seconds:.6f},{p.rho:.10f},{p.rhoMA:.10f},{p.rhoBestMA:.10f},{p.inference:.10f},{p.inferMA:.10f},{p.inferBestMA:.10f},{p.combined:.10f}\n")

proc toJsonRule(rule: Rule): JsonNode =
  result = newJObject()
  result["a"] = newJArray()
  result["b"] = newJArray()
  for v in rule.a: result["a"].add(%v)
  for v in rule.b: result["b"].add(%v)

proc fromJsonRule(node: JsonNode): Rule =
  for x in node["a"]: result.a.add(x.getInt())
  for x in node["b"]: result.b.add(x.getInt())

proc toJsonModel(m: Model): JsonNode =
  result = newJArray()
  for r in m.rules: result.add(toJsonRule(r))

proc fromJsonModel(node: JsonNode): Model =
  for n in node: result.rules.add(fromJsonRule(n))

proc trainingConfig(cfg: Config): JsonNode =
  # Only values affecting the search distribution or scores. Other CLI options
  # (plot/print/checkpoint frequency) are safe to change on resume.
  %*{
    "corpus": cfg.corpus, "population": cfg.population,
    "iterations": cfg.iterations, "minChunk": cfg.minChunk,
    "maxChunk": cfg.maxChunk, "maxChunks": cfg.maxChunks,
    "chunkAdmission": "skip-oversize-v1",
    "diffusionSteps": cfg.diffusionSteps, "ridgeRows": cfg.ridgeRows,
    "maxRidgeFeatures": cfg.maxRidgeFeatures, "ridgeLambda": cfg.ridgeLambda,
    "inferCases": cfg.inferCases, "inferSpan": cfg.inferSpan,
    "inferPopulation": cfg.inferPopulation,
    "inferGenerations": cfg.inferGenerations,
    "inferMaxLoci": cfg.inferMaxLoci, "inferNoise": cfg.inferNoise,
    "inferWeight": cfg.inferWeight, "maxRuleLen": cfg.maxRuleLen,
    "mutationRate": cfg.mutationRate,
    "rules": cfg.ruleCount, "replacePasses": ReplacePasses,
    "selfTest": cfg.selfTest
  }

proc saveCheckpoint(t: Trainer, cfg: Config) =
  var root = newJObject()
  root["version"] = %3
  root["completed"] = %t.completed
  root["base_seed"] = %t.baseSeed
  root["config"] = trainingConfig(cfg)
  root["target_rules"] = %cfg.ruleCount
  root["model"] = toJsonModel(t.current.model)
  root["history"] = newJArray()
  for p in t.history:
    root["history"].add(%*{
      "step": p.step, "ruleCount": p.ruleCount, "ruleIndex": p.ruleIndex,
      "chunkIndex": p.chunkIndex, "seconds": p.seconds, "rho": p.rho,
      "rhoMA": p.rhoMA, "rhoBestMA": p.rhoBestMA, "inference": p.inference,
      "inferMA": p.inferMA, "inferBestMA": p.inferBestMA,
      "combined": p.combined
    })
  let file = cfg.outdir / "greedy_checkpoint.json"
  let tmp = file & ".tmp"
  writeFile(tmp, pretty(root))
  moveFile(tmp, file)

proc loadCheckpoint(cfg: Config): Trainer =
  let primary = cfg.outdir / "greedy_checkpoint.json"
  let legacy = cfg.outdir / "beam_checkpoint.json"
  let file = if fileExists(primary): primary else: legacy
  if not fileExists(file):
    quit("Checkpoint missing: " & primary & " (also checked " & legacy & ")")
  let root = parseJson(readFile(file))
  let version = root["version"].getInt()
  if version notin [2, 3]:
    quit("Unsupported checkpoint version: expected v2 beam or v3 greedy")
  result.completed = root["completed"].getInt()
  result.baseSeed = root["base_seed"].getInt()
  if root["target_rules"].getInt() != cfg.ruleCount:
    quit("Resume must use the original --rules value")
  # Old checkpoints contain fitness/history obtained from TWO passes. Their
  # numbers cannot be compared to the new one-pass objective. Refuse unsafe resume.
  if not root["config"].hasKey("replacePasses") or
      root["config"]["replacePasses"].getInt() != ReplacePasses:
    quit("Checkpoint Replace pass count differs (old checkpoints used 2). " &
         "Start a new --outdir for the 1-pass model; do not mix fitness histories.")
  if not root["config"].hasKey("chunkAdmission") or
      root["config"]["chunkAdmission"].getStr() != "skip-oversize-v1":
    quit("Checkpoint was created with a different chunk admission policy. " &
         "Start a new --outdir to avoid mixing training histories.")
  # Old one-pass checkpoints had hard-coded 85% mutation and 9% restarts.
  # Their FITNESS and HISTORY are comparable: only future proposals change.
  # Allow resuming them with the newly selected mutation rate.
  if not root["config"].hasKey("mutationRate"):
    root["config"]["mutationRate"] = %cfg.mutationRate
    echo "Migrating previous one-pass checkpoint to configurable mutation rate"
  if root["config"] != trainingConfig(cfg):
    quit("Training options differ from checkpoint; resume with original settings")
  if version == 3:
    result.current.model = fromJsonModel(root["model"])
  else:
    # Migrating the old 3-beam checkpoint: take the FIRST (selected) model.
    # Keep the old file untouched; the next save uses greedy_checkpoint.json.
    if root["beams"].len == 0: quit("Old beam checkpoint has no model")
    result.current.model = fromJsonModel(root["beams"][0])
    echo "Migrated v2 beam checkpoint: retained selected beam #1 as greedy incumbent"
  for n in root["history"]:
    result.history.add(HistoryPoint(
      step: n["step"].getInt(), ruleCount: n["ruleCount"].getInt(),
      ruleIndex: n["ruleIndex"].getInt(), chunkIndex: n["chunkIndex"].getInt(),
      seconds: n["seconds"].getFloat(), rho: n["rho"].getFloat(),
      rhoMA: n["rhoMA"].getFloat(), rhoBestMA: n["rhoBestMA"].getFloat(),
      inference: n["inference"].getFloat(), inferMA: n["inferMA"].getFloat(),
      inferBestMA: n["inferBestMA"].getFloat(),
      combined: n["combined"].getFloat()))

proc writePlot(cfg: Config) =
  ## File naming and columns remain easy to use with arbitrary gnuplot versions.
  let plotFile = cfg.outdir / "training_saturation.gp"
  let csvFile = cfg.outdir / "training_history.csv"
  let pngFile = cfg.outdir / "training_saturation.png"
  # Quote gnuplot single-quoted strings, not shell arguments.
  let csvName = csvFile.replace("'", "''")
  let pngName = pngFile.replace("'", "''")
  let script = """set terminal pngcairo size 1500,950 enhanced font 'Arial,11'
set output '""" & pngName & """'
set datafile separator comma
set grid xtics ytics y2tics
set key outside horizontal top center
set xlabel 'Completed optimization stages'
set ylabel '-log_2(1-Spearman)'
set y2label 'Reconstruction accuracy (corrupted positions only)'
set y2tics
set yrange [0:*]
set y2range [0:1]
set title 'Lens -- single-path greedy DE / 1-pass Replacer'
# log transforms AFTER raw-score 5-point averaging and running max.
trans(x) = -log( (1.0 - ((x >= 0.999999999999) ? 0.999999999999 : x)) )/log(2.0)
plot '""" & csvName & """' using 1:(trans(column(6))) with lines lw 1 lc rgb '#999999' title 'Spearman current', \
     '' using 1:(trans(column(7))) with lines lw 2 lc rgb '#157bb8' title 'Spearman MA(5)', \
     '' using 1:((column(1)>=5)?trans(column(8)):1/0) with lines lw 2 lc rgb '#145a32' title 'Best historical MA(5)', \
     '' using 1:9 axes x1y2 with lines lw 1 dt 2 lc rgb '#b0a9a0' title 'Inference current', \
     '' using 1:10 axes x1y2 with lines lw 2 lc rgb '#e67e22' title 'Inference MA(5)', \
     '' using 1:((column(1)>=5)?column(11):1/0) axes x1y2 with lines lw 2 lc rgb '#ba4a00' title 'Inference best MA(5)'
"""
  writeFile(plotFile, script)
  # Training must still proceed when gnuplot is not installed.
  if findExe("gnuplot").len > 0:
    let code = execCmd("gnuplot " & quoteShell(plotFile))
    if code != 0:
      stderr.writeLine("gnuplot exited with ", code, "; script: ", plotFile)

proc checkBackwardStageSemantics() =
  ## Regression test: editing an EARLIER rule must NOT disable later rules.
  ## A -> Z cannot trigger B -> C -> D. Change only rule 1: A -> B.
  ## Full evaluation must now fire ALL THREE rules in order.
  let original = Model(rules: @[
    Rule(a: @[ord('A')], b: @[ord('Z')]),
    Rule(a: @[ord('B')], b: @[ord('C')]),
    Rule(a: @[ord('C')], b: @[ord('D')])
  ])
  let before = predictFeatures(@[ord('A')], original, compileModel(original))
  doAssert before == @[1.0, 0.0, 0.0], "full-model baseline"

  let updated = changeRule(original, 0,
                           Rule(a: @[ord('A')], b: @[ord('B')]))
  doAssert updated.rules.len == 3, "rule count must not shrink"
  doAssert updated.rules[1].a == original.rules[1].a
  doAssert updated.rules[1].b == original.rules[1].b
  doAssert updated.rules[2].a == original.rules[2].a
  doAssert updated.rules[2].b == original.rules[2].b
  let after = predictFeatures(@[ord('A')], updated, compileModel(updated))
  doAssert after == @[1.0, 1.0, 1.0],
           "later frozen rules must still execute when an earlier rule changes"
  doAssert predictFeatures(@[ord('A')], original, compileModel(original)) == before,
           "optimizing a different model must not mutate the parent"

  # Re-optimizing the middle rule must also retain and execute its suffix.
  let middleInactive = changeRule(updated, 1,
                                  Rule(a: @[ord('B')], b: @[ord('Q')]))
  doAssert predictFeatures(@[ord('A')], middleInactive,
                           compileModel(middleInactive)) == @[1.0, 1.0, 0.0]
  let middleRestored = changeRule(middleInactive, 1, updated.rules[1])
  doAssert predictFeatures(@[ord('A')], middleRestored,
                           compileModel(middleRestored)) == @[1.0, 1.0, 1.0]
  echo "PASS: backwards optimization edits one rule while all later rules remain active"

proc slowReferenceFeatures(input: openArray[int], m: Model,
                           patterns: seq[SeqPattern]): seq[float64] =
  ## Intentionally has NO shortcut/cache; used only for self-test.
  result = newSeq[float64](m.rules.len)
  var state = cloneInts(input)
  for i in 0..<m.rules.len:
    if replaceRuleCompatible(state, patterns[i], m.rules[i].b):
      result[i] += 1.0
      if state.len > MaxEvaluationState: return

proc checkOptimizedFeatureSemantics() =
  ## Regression: the token-presence filter and cached invariant prefix must
  ## produce EXACTLY the uncached single-pass feature counts.
  let model = Model(rules: @[
    Rule(a: @[ord('A')], b: @[ord('B')]),
    Rule(a: @[ord('B')], b: @[ord('C')]),
    Rule(a: @[ord('C')], b: @[ord('D')]),
    Rule(a: @[ord('!')], b: @[ord('?')])
  ])
  let patterns = compileModel(model)
  let markers = literalTriggers(model)
  for input in [@[ord('A')], @[ord('A'), ord('A')],
                @[ord('Z')], @[ord('!'), ord('A')]]:
    let reference = slowReferenceFeatures(input, model, patterns)
    doAssert predictFeatures(input, model, patterns, markers) == reference,
             "literal-presence prefilter changed feature counts"
    for active in 0..<model.rules.len:
      let prefix = prefixFor(input, model, patterns, markers, active)
      doAssert predictFeatures(input, model, patterns,
                               markers, prefix, true) == reference,
               "cached prefix changed single-pass features"
  # Wildcard patterns must NEVER be rejected by a literal presence probe.
  let wildcardModel = Model(rules: @[
    Rule(a: @[-1], b: @[ord('X')]),
    Rule(a: @[ord('X')], b: @[ord('Y')])
  ])
  let wp = compileModel(wildcardModel)
  let wm = literalTriggers(wildcardModel)
  for input in [@[ord('A')], @[ord('B'), ord('C')]]:
    let reference = slowReferenceFeatures(input, wildcardModel, wp)
    doAssert predictFeatures(input, wildcardModel, wp, wm) == reference
    let prefix = prefixFor(input, wildcardModel, wp, wm, 1)
    doAssert predictFeatures(input, wildcardModel, wp,
                             wm, prefix, true) == reference
  # A <- B <- A cycle fires once per rule, NEVER a second traversal.
  let cyclic = Model(rules: @[
    Rule(a: @[ord('A')], b: @[ord('B')]),
    Rule(a: @[ord('B')], b: @[ord('A')])
  ])
  let cp = compileModel(cyclic)
  let cm = literalTriggers(cyclic)
  doAssert predictFeatures(@[ord('A')], cyclic, cp, cm) == @[1.0, 1.0],
           "cyclic rules must not trigger a second pass"
  let cached = prefixFor(@[ord('A')], cyclic, cp, cm, 1)
  doAssert predictFeatures(@[ord('A')], cyclic, cp, cm, cached, true) == @[1.0, 1.0],
           "prefix cache must not add an extra pass"
  echo "PASS: token filter and cached prefix preserve one-pass features"

proc checkDifferentialMutation() =
  ## DE without mutation must be identity for identical donors; with mutation
  ## forced, precisely one side changes and the original parents stay intact.
  let base = Rule(a: @[65, 66, 67], b: @[68, 69, 70])
  let vocab = @[65, 66, 67, 68, 69, 70, 71, 72]
  var cfg = Config(maxRuleLen: 8, mutationRate: 0.0)
  for i in 0..<30:
    let same = trialRule(base, base, base, cfg, vocab)
    doAssert same.a == base.a and same.b == base.b,
             "identical DE donors without mutation must not change a rule"
  cfg.mutationRate = 1.0
  for i in 0..<30:
    let changed = trialRule(base, base, base, cfg, vocab)
    let leftChanged = changed.a != base.a
    let rightChanged = changed.b != base.b
    doAssert leftChanged != rightChanged,
             "extra mutation should change exactly one rule side"
    doAssert changed.a.len >= 1 and changed.a.len <= cfg.maxRuleLen
    doAssert changed.b.len >= 1 and changed.b.len <= cfg.maxRuleLen
  doAssert base.a == @[65, 66, 67] and base.b == @[68, 69, 70],
           "trial mutation must not alter donor sequences"
  echo "PASS: no mutation at rate 0; one-side nontrivial mutation at rate 1"

proc checkGreedyAcceptance() =
  var chosen = Candidate(score: Evaluation(rho: 0.5, inference: 0.5))
  doAssert not greedyAccept(chosen, Candidate(score: Evaluation(rho: 0.9, inference: 0.49)))
  doAssert chosen.score.rho == 0.5 and chosen.score.inference == 0.5
  doAssert not greedyAccept(chosen, Candidate(score: Evaluation(rho: 0.5, inference: 0.5)))
  doAssert greedyAccept(chosen, Candidate(score: Evaluation(rho: 0.6, inference: 0.5)))
  doAssert chosen.score.rho == 0.6 and chosen.score.inference == 0.5
  doAssert not greedyAccept(chosen, Candidate(score: Evaluation(rho: 0.5, inference: 0.8)))
  doAssert greedyAccept(chosen, Candidate(score: Evaluation(rho: 0.6, inference: 0.8)))
  echo "PASS: single-incumbent greedy acceptance rejects tradeoffs and ties"

proc stageCoordinates(step: int): tuple[outer, active: int] =
  ## 0 -> (0,0); 1,2 -> (1,1),(1,0); 3,4,5 -> (2,2),(2,1),(2,0).
  var outer = max(0, int((sqrt(8.0*float64(step)+1.0)-1.0)*0.5))
  while outer*(outer+1) div 2 > step: dec outer
  while (outer+1)*(outer+2) div 2 <= step: inc outer
  let offset = step - outer*(outer+1) div 2
  (outer, outer-offset)

proc train(cfg: Config) =
  if cfg.selfTest:
    checkBackwardStageSemantics()
    checkGreedyAcceptance()
    checkDifferentialMutation()
    checkOptimizedFeatureSemantics()
    checkChunkAdmission()
  if not dirExists(cfg.outdir): createDir(cfg.outdir)
  let seed = if cfg.seed == 0: int(epochTime()) else: cfg.seed
  randomize(seed)
  echo &"seed={seed} passes={ReplacePasses} beam={SearchWidth} mode=greedy mutationRate={cfg.mutationRate:.3f}"
  let chunks = loadChunks(cfg)
  let vocabulary = vocabularyFrom(chunks)
  var trainer: Trainer
  if cfg.resume:
    trainer = loadCheckpoint(cfg)
    echo &"resumed: stages={trainer.completed} beam=1 seed={trainer.baseSeed}"
  else:
    trainer.baseSeed = seed
  let totalStages = cfg.ruleCount * (cfg.ruleCount+1) div 2
  if totalStages > 100_000:
    echo &"note: {totalStages} triangular stages; consider small --rules for initial testing."
  for step in trainer.completed..<totalStages:
    # Reseed each stage deterministically; checkpoint/resume reproduces future stages.
    randomize((trainer.baseSeed xor ((step+1) * 1_000_003)) and 0x7fffffff)
    let start = epochTime()
    let (outer, active) = stageCoordinates(step)
    let chunkIndex = rand(chunks.high)
    var ctx = makeContext(chunks[chunkIndex], chunkIndex, vocabulary, cfg)
    if trainer.current.model.rules.len == 0:
      let m = Model(rules: @[randomRule(chunks[chunkIndex], cfg, vocabulary)])
      trainer.current = candidateFrom(m, ctx, cfg)
    elif active == outer and trainer.current.model.rules.len == outer:
      ## A new stage appends exactly one rule to the ONE incumbent model.
      let m = appendRule(trainer.current.model,
                         randomRule(chunks[chunkIndex], cfg, vocabulary))
      trainer.current = candidateFrom(m, ctx, cfg)
    else:
      ## Re-evaluate on this stage's fixed trajectory. Different chunks
      ## are NOT directly comparable; monotonicity is within a stage.
      trainer.current = candidateFrom(trainer.current.model, ctx, cfg)
    doAssert trainer.current.model.rules.len == outer+1
    # The single-scan prefix before `active` is frozen during this stage.
    # Prime it once; no stale prefix is carried to another stage.
    primeStagePrefixes(ctx, trainer.current, active)
    # Only `active` is editable. Every installed rule is evaluated twice.
    echo &"stage={step+1}/{totalStages} rules={outer+1} active={outer+1} optimize_only={active+1} full_eval=1..{outer+1} passes={ReplacePasses} chunk={chunkIndex} levels={ctx.trajectory.len} beam=1"
    trainer.current = evolveGreedy(trainer.current, active, ctx, cfg, active < outer)
    trainer.completed = step+1
    addHistory(trainer, trainer.current, active, chunkIndex, epochTime()-start)
    let best = trainer.current.score
    echo &"completed={trainer.completed} rho={best.rho:+.6f} infer={best.inference:.4f} joint={best.combined:.6f} seconds={epochTime()-start:.2f}"
    saveHistory(trainer, cfg.outdir)
    if trainer.completed mod cfg.plotEvery == 0 or trainer.completed == totalStages:
      writePlot(cfg)
    if trainer.completed mod cfg.checkpointEvery == 0 or trainer.completed == totalStages:
      saveCheckpoint(trainer, cfg)
  echo "Training complete: ", cfg.outdir / "greedy_checkpoint.json"

when isMainModule:
  let cfg = parseConfig()
  train(cfg)
