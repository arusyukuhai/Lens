## Lens: serial wildcard replacements with an elitist, rotating-chunk genetic algorithm.
## Every node emits a NON-BINARY count of successful non-overlapping rewrites.
## diff3 is used ONLY during genome recombination; never during model inference.
## Nim 2.x; no external Nim packages.
import std/[os, strutils, random, math, json, algorithm, sequtils, sets, times]
import replace_core

const
  NoiseToken = 256
  DefaultNodes = 1296
  SnapshotEvery = 64
  MaxPattern = 18
  MaxReplacement = 24

type
  Node = object
    pattern, replacement: seq[int]
    compiled: CountRule
  Model = object
    nodes: seq[Node]
    maxOutput: int
  Example = object
    tokens: seq[int]
    target: float64
  Regression = object
    intercept: float64
    columns: seq[int]
    weights: seq[float64]
  RowCache = object
    values: seq[float64]
    checkpoints: seq[seq[int]] # states after 0,64,128,... rewrites
  EvalCache = object
    train, anchor: seq[RowCache]
  Individual = object
    model: Model
    reg: Regression
    trainScore, anchorScore, fitness: float64
    cache: EvalCache
    origin: int
  Operation = enum opPoint, opSpan, opGeneSplice, opBlock,
                   opDiff3, opImmigrant
  OpStat = object
    attempts, improvements: int
  Config = object
    corpus, savePath, loadPath, outDir: string
    nodes, maxOutput, selectedFeatures, seed: int
    trainChunks, anchorChunks, validChunks, testChunks: int
    rotateChunks, rotateStride, chunkLen, noiseLevels: int
    population, elites, offspring, immigrants, generations: int
    plotEvery, reportEvery: int
    selfTest, evaluateOnly: bool

proc defaults(): Config =
  Config(corpus: "github-code.txt", savePath: "lens_serial_ga.json",
    outDir: "lens_training", nodes: DefaultNodes, maxOutput: 2048,
    selectedFeatures: 16, seed: 42,
    trainChunks: 12, anchorChunks: 2, validChunks: 2, testChunks: 2,
    rotateChunks: 4, rotateStride: 1, chunkLen: 128, noiseLevels: 12,
    population: 16, elites: 3, offspring: 24, immigrants: 2,
    generations: 100, plotEvery: 5, reportEvery: 1)

proc usage() =
  echo "Lens serial GA: count-based wildcard rules / rotating chunks"
  echo "  nim c -d:release --opt:speed lens_ga.nim"
  echo "  ./lens_ga --corpus=github-code.txt --generations=100"
  echo "  ./lens_ga --load=lens_serial_ga.json --evaluate"
  echo "  ./lens_ga --self-test"
  echo "Options: --nodes=1296 --train-chunks=12 --anchor-chunks=2"
  echo "  --rotate-chunks=4 --rotate-stride=1 --valid-chunks=2 --test-chunks=2"
  echo "  --population=16 --elites=3 --offspring=24 --immigrants=2"
  echo "  --generations=100 --selected-features=16 --noise-levels=12"
  echo "  --chunk-len=128 --max-output=2048 --seed=42"
  echo "  --plot-every=5 --report-every=1 --out=lens_training"
  echo "  --save=lens_serial_ga.json --load=FILE --evaluate"

proc parseArgs(): Config =
  result = defaults()
  for arg in commandLineParams():
    if arg == "--help" or arg == "-h": usage(); quit(0)
    if arg == "--self-test": result.selfTest = true; continue
    if arg == "--evaluate": result.evaluateOnly = true; continue
    if not arg.startsWith("--") or not arg.contains('='):
      quit("unknown option: " & arg, 2)
    let kv = arg[2 .. ^1].split('=', 1)
    let key = kv[0]
    let v = kv[1]
    case key
    of "arch":
      if v != "chain": quit("only --arch=chain is supported", 2)
    of "corpus": result.corpus = v
    of "save": result.savePath = v
    of "load": result.loadPath = v
    of "out": result.outDir = v
    of "nodes": result.nodes = parseInt(v)
    of "max-output": result.maxOutput = parseInt(v)
    of "selected-features": result.selectedFeatures = parseInt(v)
    of "seed": result.seed = parseInt(v)
    of "train-chunks": result.trainChunks = parseInt(v)
    of "anchor-chunks": result.anchorChunks = parseInt(v)
    of "valid-chunks": result.validChunks = parseInt(v)
    of "test-chunks": result.testChunks = parseInt(v)
    of "rotate-chunks": result.rotateChunks = parseInt(v)
    of "rotate-stride": result.rotateStride = parseInt(v)
    of "chunk-len": result.chunkLen = parseInt(v)
    of "noise-levels": result.noiseLevels = parseInt(v)
    of "population": result.population = parseInt(v)
    of "elites": result.elites = parseInt(v)
    of "offspring": result.offspring = parseInt(v)
    of "immigrants": result.immigrants = parseInt(v)
    of "generations": result.generations = parseInt(v)
    of "plot-every": result.plotEvery = parseInt(v)
    of "report-every": result.reportEvery = parseInt(v)
    else: quit("unknown option: " & arg, 2)
  if result.nodes < 1 or result.nodes > 20000 or
      result.maxOutput < 8 or result.chunkLen < 8 or
      result.chunkLen > result.maxOutput or result.selectedFeatures < 1 or
      result.trainChunks < 2 or result.anchorChunks < 1 or
      result.anchorChunks >= result.trainChunks or result.validChunks < 1 or
      result.testChunks < 1 or result.rotateChunks < 1 or
      result.rotateStride < 1 or result.noiseLevels < 0 or
      result.population < 4 or result.elites < 1 or
      result.elites >= result.population or result.offspring < 1 or
      result.immigrants < 0 or result.elites + result.immigrants >= result.population or
      result.generations < 0 or result.plotEvery < 1 or result.reportEvery < 1:
    quit("invalid configuration: see --help", 2)
  if result.rotateChunks > result.trainChunks - result.anchorChunks:
    quit("rotate-chunks exceeds non-anchor training chunks", 2)

proc readCorpus(path: string, want, chunkLen: int,
                rng: var Rand): seq[seq[int]] =
  var f: File
  if not open(f, path, fmRead):
    quit("cannot open corpus: " & path, 2)
  defer: close(f)
  var chunk = ""
  proc appendChunk(outp: var seq[seq[int]], s: string,
                   lim: int, rng: var Rand) =
    if s.len < 8: return
    let n = min(s.len, lim)
    let start = (if s.len > n: rand(rng, s.len - n) else: 0)
    var row = newSeq[int](n)
    for i in 0 ..< n: row[i] = ord(s[start+i])
    outp.add(move(row))
  while not f.endOfFile and result.len < want:
    let line = f.readLine()
    if line == "===SPLIT===":
      appendChunk(result, chunk, chunkLen, rng)
      chunk.setLen(0)
    elif chunk.len < 2*chunkLen + 4096:
      chunk.add(line)
      chunk.add('\n')
  if result.len < want:
    appendChunk(result, chunk, chunkLen, rng)
  if result.len < want:
    quit("corpus has only " & $result.len & " chunks; need " & $want, 2)

proc makeExamples(original: seq[int], levels: int,
                  rng: var Rand): seq[Example] =
  let count = (if levels == 0: original.len+1 else: max(2, levels))
  var positions = newSeq[int](original.len)
  for i in 0 ..< original.len: positions[i] = i
  shuffle(rng, positions)
  var corrupted = newSeq[int](original.len)
  for i, x in original: corrupted[i] = x
  var masked = 0
  var prev = -1
  for k in 0 ..< count:
    let wanted = (if count <= 1: 0 else: k*original.len div (count-1))
    while masked < wanted:
      corrupted[positions[masked]] = NoiseToken
      inc masked
    if masked == prev: continue
    prev = masked
    var row = Example(tokens: newSeq[int](corrupted.len),
      target: float64(masked)/float64(original.len))
    for i in 0 ..< corrupted.len: row.tokens[i] = corrupted[i]
    result.add(move(row))

proc makeBank(chunks: seq[seq[int]], levels: int,
              rng: var Rand): seq[seq[Example]] =
  for chunk in chunks: result.add(makeExamples(chunk, levels, rng))

proc concatBank(bank: seq[seq[Example]], start, count: int): seq[Example] =
  for j in start ..< start+count:
    for row in bank[j]: result.add(row)

proc rotatingRows(bank: seq[seq[Example]], generation, stride,
                  window: int): tuple[rows: seq[Example], start: int] =
  let start = ((generation-1)*stride) mod bank.len
  result.start = start
  for step in 0 ..< window:
    let j = (start+step) mod bank.len
    for row in bank[j]: result.rows.add(row)

proc cloneInts(a: seq[int]): seq[int] =
  result = newSeq[int](a.len)
  for i in 0 ..< a.len: result[i] = a[i]

proc nodeOf(p, r: seq[int]): Node =
  result.pattern = cloneInts(p)
  result.replacement = cloneInts(r)
  result.compiled = compileCountRule(result.pattern, result.replacement)

proc initialNode(chunks: seq[seq[int]], rng: var Rand,
                 id: int): Node =
  let original = chunks[rand(rng, chunks.len-1)]
  let width = min(original.len, rand(rng, 1..5))
  let offset = rand(rng, original.len-width)
  var p: seq[int] = @[]
  for k in 0 ..< width: p.add(original[offset+k])
  if p.len >= 3 and rand(rng, 99) < 25:
    p[1] = -1
  var r = cloneInts(p)
  if rand(rng, 99) < 25:
    let k = rand(rng, r.len-1)
    r[k] = 257 + (id mod 197) # an internal token, never a reserved opcode
  nodeOf(p, r)

proc makeModel(cfg: Config, chunks: seq[seq[int]],
               rng: var Rand): Model =
  result.maxOutput = cfg.maxOutput
  for i in 0 ..< cfg.nodes: result.nodes.add(initialNode(chunks, rng, i))

proc cloneModel(m: Model): Model =
  ## Immutable-node sharing: all changes replace a complete Node with fresh seqs.
  result.maxOutput = m.maxOutput
  result.nodes = newSeq[Node](m.nodes.len)
  for i in 0 ..< m.nodes.len: result.nodes[i] = m.nodes[i]

proc recordRow(model: Model, row: Example,
               parent: ptr RowCache = nil, firstChanged: int = 0): RowCache =
  let total = model.nodes.len
  result.values = newSeq[float64](total)
  let checkpointCount = total div SnapshotEvery + 1
  result.checkpoints = newSeq[seq[int]](checkpointCount)
  var start = 0
  var current: seq[int]
  if parent != nil and firstChanged > 0:
    start = (firstChanged div SnapshotEvery)*SnapshotEvery
    if start > 0:
      current = parent[].checkpoints[start div SnapshotEvery]
      for i in 0 ..< start: result.values[i] = parent[].values[i]
      for i in 0 .. start div SnapshotEvery:
        result.checkpoints[i] = parent[].checkpoints[i]
  if start == 0:
    current = row.tokens
    result.checkpoints[0] = current
  var scratch: seq[int] = @[]
  for i in start ..< total:
    let r = applyCountRule(current, model.nodes[i].compiled,
                           scratch, model.maxOutput)
    result.values[i] = float64(r.count)
    if r.count > 0: current = move(scratch)
    if (i+1) mod SnapshotEvery == 0:
      result.checkpoints[(i+1) div SnapshotEvery] = current

proc evalRows(model: Model, rows: seq[Example],
              parent: ptr seq[RowCache] = nil,
              firstChanged: int = 0): seq[RowCache] =
  result = newSeq[RowCache](rows.len)
  for i in 0 ..< rows.len:
    if parent != nil:
      result[i] = recordRow(model, rows[i], addr parent[][i], firstChanged)
    else:
      result[i] = recordRow(model, rows[i])

proc dot(a, b: seq[float64]): float64 =
  for i in 0 ..< a.len: result += a[i]*b[i]

proc avg(values: seq[float64]): float64 =
  if values.len == 0: return 0
  for x in values: result += x
  result /= float64(values.len)

proc fitOls(x: seq[seq[float64]], rows: seq[Example],
            maxCols: int): Regression =
  ## Orthogonal Matching Pursuit (column selection) followed by EXACT OLS
  ## over the selected columns using modified Gram-Schmidt QR.
  ## No ridge penalty, no iterative weight GA. With 1296 columns and only
  ## dozens of rows, full unconstrained OLS is rank-deficient.
  let n = rows.len
  if n == 0: return
  let p = x[0].len
  var y = newSeq[float64](n)
  var means = newSeq[float64](p)
  for i in 0 ..< n:
    y[i] = rows[i].target
    for j in 0 ..< p: means[j] += x[i][j]
  for j in 0 ..< p: means[j] /= float64(n)
  let my = avg(y)
  for i in 0 ..< n: y[i] -= my
  var residual = y
  var qs: seq[seq[float64]] = @[]
  var rtri: seq[seq[float64]] = @[]
  var qy: seq[float64] = @[]
  var banned = newSeq[bool](p)
  let budget = min(maxCols, min(n - 1, p))
  for step in 0 ..< budget:
    var best = -1
    var bestScore = 1e-10
    for j in 0 ..< p:
      if banned[j]: continue
      var numer = 0.0
      var denom = 0.0
      for i in 0 ..< n:
        let v = x[i][j] - means[j]
        numer += v * residual[i]
        denom += v * v
      if denom > 1e-18:
        let score = abs(numer) / sqrt(denom)
        if score > bestScore:
          bestScore = score
          best = j
    if best < 0: break
    banned[best] = true
    var col = newSeq[float64](n)
    for i in 0 ..< n: col[i] = x[i][best] - means[best]
    var projection = newSeq[float64](qs.len)
    for k in 0 ..< qs.len:
      projection[k] = dot(col, qs[k])
      for i in 0 ..< n: col[i] -= projection[k] * qs[k][i]
    # Second orthogonalization pass reduces cancellation in nearly
    # linearly dependent count features.
    for k in 0 ..< qs.len:
      let extra = dot(col, qs[k])
      projection[k] += extra
      for i in 0 ..< n: col[i] -= extra * qs[k][i]
    let length = sqrt(dot(col, col))
    if length < 1e-8: continue
    for k in 0 ..< rtri.len: rtri[k].add(projection[k])
    var triangularRow = newSeq[float64](qs.len + 1)
    triangularRow[qs.len] = length
    rtri.add(move(triangularRow))
    for i in 0 ..< n: col[i] /= length
    qs.add(move(col))
    result.columns.add(best)
    let projY = dot(qs[^1], y)
    qy.add(projY)
    for i in 0 ..< n: residual[i] -= projY * qs[^1][i]
    if dot(residual, residual) < 1e-15: break
  result.weights = newSeq[float64](result.columns.len)
  for k in countdown(result.columns.len - 1, 0):
    var total = qy[k]
    for j in k + 1 ..< result.columns.len:
      total -= rtri[k][j] * result.weights[j]
    result.weights[k] = total / rtri[k][k]
  result.intercept = my
  for k in 0 ..< result.columns.len:
    result.intercept -= result.weights[k] * means[result.columns[k]]

proc predict(reg: Regression, x: seq[float64]): float64 =
  result = reg.intercept
  for k in 0 ..< reg.columns.len:
    result += reg.weights[k] * x[reg.columns[k]]

proc pearson(x, y: seq[float64]): float64 =
  if x.len < 2: return 0
  let mx = avg(x)
  let my = avg(y)
  var sx, sy, sxy: float64
  for i in 0 ..< x.len:
    let dx = x[i] - mx
    let dy = y[i] - my
    sx += dx * dx
    sy += dy * dy
    sxy += dx * dy
  if sx <= 1e-20 or sy <= 1e-20: return 0
  result = sxy / sqrt(sx * sy)
  result = max(-1.0, min(1.0, result))

proc pearsonScore(reg: Regression, x: seq[seq[float64]],
                  rows: seq[Example]): float64 =
  var actual = newSeq[float64](rows.len)
  var preds = newSeq[float64](rows.len)
  for i in 0 ..< rows.len:
    actual[i] = rows[i].target
    preds[i] = predict(reg, x[i])
  pearson(preds, actual)

proc matrixOf(caches: seq[RowCache]): seq[seq[float64]] =
  for rec in caches: result.add(rec.values)

proc scoreMatrix(reg: Regression, cache: seq[RowCache],
                 rows: seq[Example]): float64 =
  if cache.len != rows.len: raise newException(ValueError, "cache mismatch")
  var a = newSeq[float64](rows.len)
  var p = newSeq[float64](rows.len)
  for i in 0 ..< rows.len:
    a[i] = rows[i].target
    p[i] = predict(reg, cache[i].values)
  pearson(p, a)

proc evaluate(model: Model, trainRows, anchorRows: seq[Example],
              featureBudget: int, parent: ptr Individual = nil,
              firstChanged: int = 0): Individual =
  result.model = model
  if parent == nil:
    result.cache.train = evalRows(model, trainRows)
    result.cache.anchor = evalRows(model, anchorRows)
  else:
    result.cache.train = evalRows(model, trainRows,
                               addr parent[].cache.train, firstChanged)
    result.cache.anchor = evalRows(model, anchorRows,
                               addr parent[].cache.anchor, firstChanged)
  result.reg = fitOls(matrixOf(result.cache.train), trainRows, featureBudget)
  result.trainScore = scoreMatrix(result.reg, result.cache.train, trainRows)
  result.anchorScore = scoreMatrix(result.reg, result.cache.anchor, anchorRows)
  ## Anchor is a separate set of TRAINING chunks, never the heldout or validation.
  result.fitness = 0.35*result.trainScore + 0.65*result.anchorScore

proc standalonePearson(model: Model, reg: Regression,
                       rows: seq[Example]): float64 =
  let cc = evalRows(model, rows)
  scoreMatrix(reg, cc, rows)

proc ruleChanged(a, b: Node): bool =
  a.pattern != b.pattern or a.replacement != b.replacement

proc literal(rng: var Rand, samples: seq[seq[int]]): int =
  if rand(rng, 99) < 74:
    let s = samples[rand(rng, samples.len-1)]
    s[rand(rng, s.len-1)]
  elif rand(rng, 99) < 85:
    NoiseToken
  else:
    257 + rand(rng, 255)

proc normalizeRule(p, r: var seq[int], samples: seq[seq[int]],
                   rng: var Rand) =
  ## Captures are positional; rewrite backrefs to the available range.
  if p.len > MaxPattern: p.setLen(MaxPattern)
  if r.len > MaxReplacement: r.setLen(MaxReplacement)
  if p.len == 0: p.add(literal(rng, samples))
  var captures = 0
  var hasLiteral = false
  for i in 0 ..< p.len:
    if p[i] < 0:
      if captures < 15:
        inc captures
        p[i] = -captures
      else:
        p[i] = literal(rng, samples)
        hasLiteral = true
    else: hasLiteral = true
  if not hasLiteral:
    p[rand(rng, p.len-1)] = literal(rng, samples)
    # Re-index captures once a wildcard becomes a literal.
    captures = 0
    for i in 0 ..< p.len:
      if p[i] < 0:
        inc captures
        p[i] = -captures
  for i in 0 ..< r.len:
    if r[i] < 0:
      if captures == 0:
        r[i] = literal(rng, samples)
      else:
        r[i] = -(1 + ((-r[i]-1) mod captures))
  # The replacement is allowed to be empty (deletion).

proc modifiedNode(p0, r0: seq[int], samples: seq[seq[int]],
                  rng: var Rand): Node =
  var p = cloneInts(p0)
  var r = cloneInts(r0)
  normalizeRule(p, r, samples, rng)
  nodeOf(p, r)

proc mutateString(s: var seq[int], pattern: bool,
                  samples: seq[seq[int]], rng: var Rand) =
  ## Intra-string operators: replace, insert, delete, swap, invert,
  ## duplicate, copy a corpus literal, and mutate wildcard/backreferences.
  let action = rand(rng, 8)
  case action
  of 0: # substitution
    if s.len > 0:
      let k = rand(rng, s.len-1)
      s[k] = literal(rng, samples)
    else: s.add(literal(rng, samples))
  of 1: # insert
    let k = rand(rng, s.len)
    s.insert(literal(rng, samples), k)
  of 2: # delete
    if s.len > 0: s.delete(rand(rng, s.len-1))
  of 3: # adjacent swap
    if s.len > 1:
      let k = rand(rng, s.len-2)
      swap(s[k], s[k+1])
  of 4: # duplicate a short internal span
    if s.len > 0:
      let l = rand(rng, s.len-1)
      let n = rand(rng, 1 .. min(4, s.len-l))
      let at = rand(rng, s.len)
      var fragment: seq[int] = @[]
      for k in l ..< l+n: fragment.add(s[k])
      for i, x in fragment: s.insert(x, at+i)
  of 5: # invert a short internal span
    if s.len > 1:
      let l = rand(rng, s.len-2)
      let r = rand(rng, l+1 .. min(s.len-1, l+5))
      var i = l
      var j = r
      while i < j:
        swap(s[i], s[j]); inc i; dec j
  of 6: # wildcard / capture introduction or removal
    if s.len > 0:
      let k = rand(rng, s.len-1)
      if pattern:
        s[k] = (if s[k] >= 0: -1 else: literal(rng, samples))
      else:
        s[k] = (if s[k] >= 0: -rand(rng, 1..3) else: literal(rng, samples))
  of 7: # replace with short corpus phrase
    let src = samples[rand(rng, samples.len-1)]
    let n = rand(rng, 1 .. min(4, src.len))
    let fromPos = rand(rng, src.len-n)
    let at = rand(rng, s.len)
    for i in 0 ..< n: s.insert(src[fromPos+i], at+i)
  else: # small numeric perturbation, preserves literal domain
    if s.len > 0:
      let k = rand(rng, s.len-1)
      if s[k] >= 0: s[k] = max(0, min(512, s[k] + rand(rng, -3..3)))
  if s.len > (if pattern: MaxPattern else: MaxReplacement):
    s.setLen(if pattern: MaxPattern else: MaxReplacement)

proc mutateRule(n: Node, samples: seq[seq[int]],
                rng: var Rand, span: bool = false): Node =
  var p = cloneInts(n.pattern)
  var r = cloneInts(n.replacement)
  let nMut = (if span: rand(rng, 2..5) else: 1)
  for k in 0 ..< nMut:
    if rand(rng, 99) < 48:
      mutateString(p, true, samples, rng)
    else:
      mutateString(r, false, samples, rng)
  normalizeRule(p, r, samples, rng)
  result = nodeOf(p, r)

proc spliceString(x, y: seq[int], rng: var Rand): seq[int] =
  let cutX = rand(rng, x.len)
  let cutY = rand(rng, y.len)
  for i in 0 ..< cutX: result.add(x[i])
  for i in cutY ..< y.len: result.add(y[i])

proc pickLocus(parent: Individual, rng: var Rand): int =
  ## Feature-focused search: mutate selected features or a preceding rule
  ## with some probability. Global random moves prevent complete tunnel vision.
  let n = parent.model.nodes.len
  if parent.reg.columns.len > 0 and rand(rng, 99) < 57:
    let selected = parent.reg.columns[rand(rng, parent.reg.columns.len-1)]
    result = max(0, min(n-1, selected-rand(rng, 0..min(80, selected))))
  else: result = rand(rng, n-1)

proc crossLocus(a, b, c: Individual, rng: var Rand): int =
  ## Sample a locus at which the parents actually disagree. Uniformly
  ## scanning the genome is cheap relative to 1296 sequential rewrites and
  ## prevents silent/no-op crossovers in a sparse, mostly-shared population.
  let n = a.model.nodes.len
  var count = 0
  result = -1
  for i in 0 ..< n:
    if ruleChanged(a.model.nodes[i], b.model.nodes[i]) or
       ruleChanged(a.model.nodes[i], c.model.nodes[i]):
      inc count
      if rand(rng, count-1) == 0: result = i
  if result < 0: result = pickLocus(a, rng)

proc diff3Gene(a, b, c: Node, samples: seq[seq[int]],
               rng: var Rand): Node =
  ## Literal 3-way sequence merge (base=a, change=b, target=c).
  ## Conflicts fall back to a, exactly as requested. This is a symbolic
  ## DE-LIKE edit transfer; it is not numeric DE/rand/1/bin.
  var p, r: seq[int] = @[]
  discard diff3ApplyInto(a.pattern, b.pattern, c.pattern, p,
                          d3KeepBase, maxMyersD = 32)
  discard diff3ApplyInto(a.replacement, b.replacement, c.replacement, r,
                          d3KeepBase, maxMyersD = 32)
  result = modifiedNode(p, r, samples, rng)

proc deriveChild(a, b, c: Individual, op: Operation,
                 samples: seq[seq[int]], rng: var Rand
                ): tuple[model: Model, first: int] =
  result.model = cloneModel(a.model)
  let n = a.model.nodes.len
  var first = n
  case op
  of opPoint, opSpan:
    let count = (if op == opSpan: rand(rng, 2..8) else: 1)
    for k in 0 ..< count:
      let ix = pickLocus(a, rng)
      let v = mutateRule(result.model.nodes[ix], samples, rng, op==opSpan)
      if ruleChanged(v, result.model.nodes[ix]):
        result.model.nodes[ix] = v
        first = min(first, ix)
  of opGeneSplice:
    for t in 0 ..< rand(rng, 1..6):
      let ix = crossLocus(a, b, c, rng)
      let p = spliceString(a.model.nodes[ix].pattern,
                           b.model.nodes[ix].pattern, rng)
      let r = spliceString(a.model.nodes[ix].replacement,
                           b.model.nodes[ix].replacement, rng)
      let v = modifiedNode(p, r, samples, rng)
      if ruleChanged(v, result.model.nodes[ix]):
        result.model.nodes[ix] = v
        first = min(first, ix)
  of opBlock:
    let beginAt = crossLocus(a, b, c, rng)
    let size = min(rand(rng, 1..64), n-beginAt)
    for ix in beginAt ..< beginAt+size:
      if ruleChanged(result.model.nodes[ix], b.model.nodes[ix]):
        result.model.nodes[ix] = b.model.nodes[ix]
        first = min(first, ix)
    if rand(rng, 99) < 40:
      let ix = beginAt + rand(rng, size-1)
      let v = mutateRule(result.model.nodes[ix], samples, rng)
      if ruleChanged(v, result.model.nodes[ix]):
        result.model.nodes[ix] = v
        first = min(first, ix)
  of opDiff3:
    let count = rand(rng, 1..7)
    for t in 0 ..< count:
      let ix = crossLocus(a, b, c, rng)
      let v = diff3Gene(a.model.nodes[ix], b.model.nodes[ix],
                        c.model.nodes[ix], samples, rng)
      if ruleChanged(v, result.model.nodes[ix]):
        result.model.nodes[ix] = v
        first = min(first, ix)
  of opImmigrant:
    result.model = makeModel(Config(nodes: n, maxOutput: a.model.maxOutput),
                             samples, rng)
    first = 0
  if first == n:
    # Guarantee a genuine modification even if diff3 collapses to base.
    let ix = pickLocus(a, rng)
    var v = mutateRule(result.model.nodes[ix], samples, rng, true)
    if not ruleChanged(v, result.model.nodes[ix]):
      var p = cloneInts(v.pattern)
      var r = cloneInts(v.replacement)
      # Deterministic change, including when the replacement is at its cap.
      p[0] = (if p[0] == 0: 1 else: 0)
      v = modifiedNode(p, r, samples, rng)
    result.model.nodes[ix] = v
    first = ix
  result.first = first

proc fingerprint(m: Model): uint64 =
  ## Simple FNV-style stable genotype digest, only used to discourage
  ## duplicates. Independent of Nim's randomized runtime hash seed.
  result = 14695981039346656037'u64
  for nd in m.nodes:
    result = (result xor (uint64(nd.pattern.len)+1'u64)) * 1099511628211'u64
    for x in nd.pattern:
      result = (result xor uint64(x+64)) * 1099511628211'u64
    result = (result xor (uint64(nd.replacement.len)+3'u64)) * 1099511628211'u64
    for x in nd.replacement:
      result = (result xor uint64(x+64)) * 1099511628211'u64

proc better(a, b: Individual): bool =
  if abs(a.fitness - b.fitness) > 1e-12:
    a.fitness > b.fitness
  else:
    a.anchorScore > b.anchorScore

proc compareIndividuals(a, b: Individual): int =
  if better(a, b): -1
  elif better(b, a): 1
  else: 0

proc tournament(pop: seq[Individual], rng: var Rand): int =
  result = rand(rng, pop.len-1)
  for j in 1 ..< 3:
    let other = rand(rng, pop.len-1)
    if better(pop[other], pop[result]): result = other

proc chooseOperation(stats: array[Operation, OpStat], rng: var Rand): Operation =
  let base = [4.0, 2.5, 2.5, 2.0, 2.0, 0.0]
  var weights: array[Operation, float64]
  var total = 0.0
  for op in Operation:
    if op == opImmigrant: continue
    let ix = ord(op)
    let success = float64(stats[op].improvements+1) /
                  float64(stats[op].attempts+2)
    weights[op] = base[ix] * (0.6 + 1.7*success)
    total += weights[op]
  var x = rand(rng, 0.0 .. total)
  for op in Operation:
    if op == opImmigrant: continue
    x -= weights[op]
    if x <= 0: return op
  opDiff3

proc intArray(v: JsonNode): seq[int] =
  for x in v.items: result.add(x.getInt())

proc saveModel(m: Model, reg: Regression, path: string) =
  var root = newJObject()
  root["format"] = %"LensSerialGA-v1"
  root["arch"] = %"chain"
  root["maxOutput"] = %m.maxOutput
  root["intercept"] = %reg.intercept
  root["columns"] = %reg.columns
  root["weights"] = %reg.weights
  var nodes = newJArray()
  for nd in m.nodes:
    var obj = newJObject()
    obj["pattern"] = %nd.pattern
    obj["replacement"] = %nd.replacement
    nodes.add(obj)
  root["nodes"] = nodes
  writeFile(path, pretty(root))

proc loadModel(path: string): tuple[model: Model, reg: Regression] =
  let doc = parseFile(path)
  let fmt = doc["format"].getStr()
  if fmt != "LensSerialGA-v1" and fmt != "LensTripleGate-v1":
    quit("unsupported model format: " & fmt, 2)
  if fmt == "LensTripleGate-v1" and doc["arch"].getStr() != "chain":
    quit("grid checkpoints cannot be read as serial genomes", 2)
  result.model.maxOutput = doc["maxOutput"].getInt()
  for val in doc["nodes"].items:
    result.model.nodes.add(nodeOf(intArray(val["pattern"]),
                                  intArray(val["replacement"])))
  result.reg.intercept = doc["intercept"].getFloat()
  result.reg.columns = intArray(doc["columns"])
  for w in doc["weights"].items: result.reg.weights.add(w.getFloat())
  if result.reg.columns.len != result.reg.weights.len:
    quit("invalid regression", 2)
  for ix in result.reg.columns:
    if ix < 0 or ix >= result.model.nodes.len:
      quit("invalid regression feature", 2)

proc quality(r: float64): float64 =
  ## -log2(1-Pearson), clamped solely to avoid +/- Inf in output.
  let clipped = min(1.0 - 1e-12, max(-1.0, r))
  -ln(1.0-clipped)/ln(2.0)

proc fmt(r: float64): string = formatFloat(r, ffDecimal, 8)

proc writeGnuplot(outDir: string) =
  let gp = """
set terminal pngcairo size 1450,850 enhanced font ',13'
set output 'training.png'
set title 'Lens serial GA: -log2(1-Pearson)'
set xlabel 'Generation'
set ylabel '-log2(1-Pearson)'
set key outside right top
set grid
set datafile separator '\t'
set style line 1 lw 2 lc rgb '#d08040'
set style line 2 lw 2 lc rgb '#46a876'
set style line 3 lw 2 lc rgb '#488dcc'
set style line 4 lw 1 dt 2 lc rgb '#666666'
plot 'training.tsv' using 1:3 with lines ls 1 title 'Rotating train', \
     'training.tsv' using 1:4 with lines ls 2 title 'Fixed anchor (train pool)', \
     'training.tsv' using 1:5 with lines ls 3 title 'Fixed validation (monitor only)', \
     'training.tsv' using 1:6 with lines ls 4 title 'Best recorded anchor'
set output 'training_detail.png'
set title 'Lens serial GA: fixed cohorts (separate y axes)'
set y2tics
set ytics nomirror
set ylabel 'Anchor: -log2(1-Pearson)'
set y2label 'Validation: -log2(1-Pearson)'
plot 'training.tsv' using 1:4 with lines ls 2 title 'Anchor', \
     'training.tsv' using 1:5 axes x1y2 with lines ls 3 title 'Validation'
"""
  writeFile(outDir / "training.gp", gp)

proc plot(outDir: string) =
  if findExe("gnuplot").len == 0: return
  let command = "cd " & quoteShell(outDir) & " && gnuplot training.gp"
  let code = execShellCmd(command)
  if code != 0: echo "[warning] gnuplot returned ", code

proc appendMetrics(outDir: string, gen, start: int,
                   best: Individual, validScore, bestAnchor: float64) =
  var f: File
  if open(f, outDir / "training.tsv", fmAppend):
    f.writeLine($gen & "\t" & $start & "\t" &
      fmt(quality(best.trainScore)) & "\t" &
      fmt(quality(best.anchorScore)) & "\t" &
      fmt(quality(validScore)) & "\t" &
      fmt(quality(bestAnchor)))
    close(f)

proc selfTest() =
  var scratch: seq[int] = @[]
  let rule = compileCountRule(@[1, 2], @[9])
  let r = applyCountRule(@[1, 2, 1, 2, 3], rule, scratch)
  doAssert r.count == 2 and scratch == @[9,9,3]
  var rng = initRand(1234)
  let samples = @[@[1,2,3,4,5,6,7,8,9,10]]
  let cfg = Config(nodes: 10, maxOutput: 50)
  let model = makeModel(cfg, samples, rng)
  doAssert model.nodes.len == 10
  let rows = makeExamples(samples[0], 6, rng)
  let cache = evalRows(model, rows)
  doAssert cache.len == rows.len
  let reg = fitOls(matrixOf(cache), rows, 3)
  doAssert reg.columns.len <= 3
  var parent = evaluate(model, rows, rows, 3)
  let b = evaluate(makeModel(cfg, samples, rng), rows, rows, 3)
  let c = evaluate(makeModel(cfg, samples, rng), rows, rows, 3)
  for op in Operation:
    let child = deriveChild(parent, b, c, op, samples, rng)
    doAssert child.first >= 0 and child.first < 10
    let fast = evaluate(child.model, rows, rows, 3, addr parent, child.first)
    let slow = evaluate(child.model, rows, rows, 3)
    for i in 0 ..< rows.len:
      doAssert fast.cache.train[i].values == slow.cache.train[i].values
  let d1 = nodeOf(@[1,2,3], @[7,8])
  let d2 = nodeOf(@[1,2,4], @[7,8])
  let d3 = nodeOf(@[1,2,3], @[7,9])
  discard diff3Gene(d1, d2, d3, samples, rng)
  let path = getTempDir() / "lens_serial_ga_test.json"
  saveModel(model, reg, path)
  let loaded = loadModel(path)
  removeFile(path)
  doAssert loaded.model.nodes.len == model.nodes.len
  doAssert abs(quality(0.5)-1.0) < 1e-10
  echo "[self-test] mutation/crossover/diff3/OLS/cache/count/save/score PASS"

when isMainModule:
  let cfg = parseArgs()
  if cfg.selfTest: selfTest(); quit(0)
  var rng = initRand(int64(cfg.seed))
  let requested = cfg.trainChunks + cfg.validChunks + cfg.testChunks
  let chunks = readCorpus(cfg.corpus, requested, cfg.chunkLen, rng)
  let bank = makeBank(chunks, cfg.noiseLevels, rng)
  let rotatingCount = cfg.trainChunks - cfg.anchorChunks
  let rotateBank = bank[0 ..< rotatingCount]
  let anchorRows = concatBank(bank, rotatingCount, cfg.anchorChunks)
  let validRows = concatBank(bank, cfg.trainChunks, cfg.validChunks)
  let testRows = concatBank(bank, cfg.trainChunks+cfg.validChunks,
                            cfg.testChunks)
  let allTrainRows = concatBank(bank, 0, cfg.trainChunks)
  echo "[data] rotatingChunks=", rotatingCount, " anchorChunks=",
    cfg.anchorChunks, " validRows=", validRows.len,
    " heldoutRows=", testRows.len
  if cfg.evaluateOnly:
    if cfg.loadPath.len == 0: quit("--evaluate requires --load", 2)
    let loaded = loadModel(cfg.loadPath)
    echo "[load] nodes=", loaded.model.nodes.len
    echo "[heldout] Pearson=", fmt(standalonePearson(loaded.model,
      loaded.reg, testRows)), " Q=", fmt(quality(standalonePearson(
      loaded.model, loaded.reg, testRows)))
    quit(0)
  if not dirExists(cfg.outDir): createDir(cfg.outDir)
  writeFile(cfg.outDir / "training.tsv",
    "generation\trotate_start\ttrain_q\tanchor_q\tvalid_q\tbest_anchor_q\n")
  writeGnuplot(cfg.outDir)
  var population: seq[Individual] = @[]
  let boot = rotatingRows(rotateBank, 1, cfg.rotateStride, cfg.rotateChunks)
  if cfg.loadPath.len > 0:
    let loaded = loadModel(cfg.loadPath)
    if loaded.model.nodes.len != cfg.nodes:
      quit("loaded model node count differs from --nodes", 2)
    population.add(evaluate(loaded.model, boot.rows, anchorRows,
                            cfg.selectedFeatures))
  while population.len < cfg.population:
    var model: Model
    if population.len > 0 and rand(rng, 99) < 75:
      let base = population[0]
      let ix = rand(rng, base.model.nodes.len-1)
      model = cloneModel(base.model)
      model.nodes[ix] = mutateRule(model.nodes[ix], chunks[0 ..< rotatingCount], rng)
    else:
      model = makeModel(cfg, chunks[0 ..< rotatingCount], rng)
    population.add(evaluate(model, boot.rows, anchorRows,
                             cfg.selectedFeatures))
  var globalBest: Model
  var globalAnchor = -Inf
  var stats: array[Operation, OpStat]
  let started = epochTime()
  for gen in 1 .. cfg.generations:
    let rotation = rotatingRows(rotateBank, gen, cfg.rotateStride,
                                cfg.rotateChunks)
    if gen != 1:
      for i in 0 ..< population.len:
        population[i] = evaluate(population[i].model, rotation.rows,
                                  anchorRows, cfg.selectedFeatures)
      # Inject the hall-of-fame champion under the SAME rotating window,
      # so its global genotype is not permanently lost when rotations change.
      if globalBest.nodes.len > 0:
        population.add(evaluate(globalBest, rotation.rows,
                                 anchorRows, cfg.selectedFeatures))
    population.sort(compareIndividuals)
    if population.len > cfg.population: population.setLen(cfg.population)
    var candidates: seq[Individual] = @[]
    var next: seq[Individual] = @[]
    var seen = initHashSet[uint64]()
    for i in 0 ..< cfg.elites:
      let sig = fingerprint(population[i].model)
      if sig notin seen:
        seen.incl(sig)
        next.add(population[i])
    for i in 0 ..< cfg.offspring:
      let aa = tournament(population, rng)
      var bb = tournament(population, rng)
      var cc = tournament(population, rng)
      if population.len > 2:
        while bb == aa: bb = tournament(population, rng)
        while cc == aa or cc == bb: cc = tournament(population, rng)
      let op = chooseOperation(stats, rng)
      let child = deriveChild(population[aa], population[bb], population[cc],
        op, chunks[0 ..< rotatingCount], rng)
      let fit = evaluate(child.model, rotation.rows, anchorRows,
        cfg.selectedFeatures, addr population[aa], child.first)
      inc stats[op].attempts
      if fit.fitness > population[aa].fitness + 1e-8:
        inc stats[op].improvements
      candidates.add(fit)
    candidates.sort(compareIndividuals)
    for j in 0 ..< candidates.len:
      if next.len >= cfg.population-cfg.immigrants: break
      let sig = fingerprint(candidates[j].model)
      if sig notin seen:
        seen.incl(sig)
        next.add(candidates[j])
    # Preserve other distinct parents before falling back to fresh immigrants.
    for j in 0 ..< population.len:
      if next.len >= cfg.population-cfg.immigrants: break
      let sig = fingerprint(population[j].model)
      if sig notin seen:
        seen.incl(sig)
        next.add(population[j])
    while next.len < cfg.population-cfg.immigrants:
      let immigrant = makeModel(cfg, chunks[0 ..< rotatingCount], rng)
      next.add(evaluate(immigrant, rotation.rows, anchorRows,
                        cfg.selectedFeatures))
    for k in 0 ..< cfg.immigrants:
      let immigrant = makeModel(cfg, chunks[0 ..< rotatingCount], rng)
      next.add(evaluate(immigrant, rotation.rows, anchorRows,
                         cfg.selectedFeatures))
      inc stats[opImmigrant].attempts
    population = move(next)
    population.sort(compareIndividuals)
    let winner = population[0]
    if winner.anchorScore > globalAnchor + 1e-10:
      globalAnchor = winner.anchorScore
      globalBest = cloneModel(winner.model)
      # Save a globally comparable checkpoint after refitting on ALL train
      # chunks; selection nevertheless never consults independent valid/test.
      let full = evalRows(globalBest, allTrainRows)
      let fullReg = fitOls(matrixOf(full), allTrainRows, cfg.selectedFeatures)
      saveModel(globalBest, fullReg, cfg.savePath)
    let validScore = standalonePearson(winner.model, winner.reg, validRows)
    appendMetrics(cfg.outDir, gen, rotation.start, winner,
                  validScore, globalAnchor)
    if gen mod cfg.reportEvery == 0 or gen == 1 or gen == cfg.generations:
      echo "[gen] ", gen, " rotateStart=", rotation.start,
           " trainQ=", fmt(quality(winner.trainScore)),
           " anchorQ=", fmt(quality(winner.anchorScore)),
           " validQ=", fmt(quality(validScore)),
           " bestAnchorQ=", fmt(quality(globalAnchor)),
           " elapsed=", fmt(epochTime()-started), "s"
    if gen mod cfg.plotEvery == 0 or gen == cfg.generations:
      plot(cfg.outDir)
  if cfg.generations == 0:
    globalBest = population[0].model
  let fullRows = evalRows(globalBest, allTrainRows)
  let reg = fitOls(matrixOf(fullRows), allTrainRows, cfg.selectedFeatures)
  saveModel(globalBest, reg, cfg.savePath)
  let finalValid = standalonePearson(globalBest, reg, validRows)
  let testPearson = standalonePearson(globalBest, reg, testRows)
  echo "[final] validation Pearson=", fmt(finalValid),
       " Q=", fmt(quality(finalValid))
  echo "[heldout] Pearson=", fmt(testPearson),
       " Q=", fmt(quality(testPearson)), " (NEVER used for selection)"
  echo "[save] model=", cfg.savePath, " metrics=", cfg.outDir / "training.tsv"
  if findExe("gnuplot").len == 0:
    echo "[plot] gnuplot unavailable: run 'cd ", cfg.outDir,
         " && gnuplot training.gp' when installed"
  for op in Operation:
    echo "[operator] ", op, " tries=", stats[op].attempts,
         " improvement=", stats[op].improvements
