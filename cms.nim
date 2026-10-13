## Lens wildcard-presence rank-regression beam learner.
import std/[os, strutils, unicode, json, jsonutils, parseopt,
            math, times, sha1, osproc, random, sets, algorithm]
import lens_core, lens_search, cms_miner, spearman_rank_ridge, lens_validation, lens_rollback

type
  Options = object
    corpus, output, resume: string
    iterations, window, paths, noiseSteps, seed, holdout: int
    trajectoryLimitMiB, maxChunkTokens: int
    searchPool, acceptPool, searchChunks, acceptChunks, replayChunks, regrowEvery, regrowPatience: int
    acceptanceMargin, rulePenalty, anchorMargin, acceptFloor, minSourceWin: float64
    bytes, plot, shuffleCorpus, sharedNoiseAlphabet: bool
    anchorChunks, anchorPool, anchorRefresh, validationSteps, anchorSteps, monitorSteps: int
    fitReplayChunks, fitReplaySteps: int
    readoutRefreshEvery: int
    readoutBlend: float64
    search: SearchConfig

proc defaults(): Options =
  Options(corpus: "github-code.txt", output: "lens_run", iterations: 100,
    window: 3, paths: 2, noiseSteps: 128, seed: 20261013, holdout: 8,
    plot: true, shuffleCorpus: true, sharedNoiseAlphabet: true, trajectoryLimitMiB: 1024, maxChunkTokens: 4096,
    searchPool: 32, acceptPool: 64, searchChunks: 4, acceptChunks: 6,
    anchorChunks: 8, anchorPool: 32, anchorRefresh: 16,
    validationSteps: 64, anchorSteps: 48, monitorSteps: 96,
    fitReplayChunks: 8, fitReplaySteps: 32,
    readoutRefreshEvery: 4, readoutBlend: 0.5,
    anchorMargin: 0.002, acceptFloor: 0.0, minSourceWin: 0.5,
    acceptanceMargin: 0.002, rulePenalty: 0.001,
    replayChunks: 1, regrowEvery: 4, regrowPatience: 5,
    search: defaultSearchConfig())

const help = """
Build: nim c -d:release --opt:speed cms.nim
Run:   ./cms --corpus:github-code.txt --iterations:100
Corpus: ===SPLIT=== on a line separates chunks. Empty chunks are skipped.

--window:3             several clean chunks, rotate exactly one after each iteration
--iterations:100       number of additional iterations; 0 runs indefinitely
--beam:4               Maximum actual beam states (LHS + RHS swap variants combined)
--lookahead:2          new rules per beam lookahead (>=2 makes swaps observable)
--paths:2              cumulative corruption paths per chunk per train/selection set
--noise-steps:128      0 = record every one-character change; >0 explicit subsampling
--holdout:8            fixed final monitor; NEVER consulted by model selection
--search-pool:32       fixed, disjoint source-chunk reservoir for beam search
--accept-pool:64       disjoint source-chunk reservoir for independent acceptance
--replay-chunks:1      additional train-pool chunks for rehearsal (0 disables)
--search-chunks:4      rotate N source chunks from search pool each iteration
--accept-chunks:6      rotate N source chunks from acceptance pool each iteration
--anchor-pool:32       exclusive source pool (not in rotating acceptance)
--anchor-chunks:8      source chunks per anchor batch
--anchor-refresh:16    change anchor batch every 16 iterations
--fit-replay-chunks:8  diverse train-pool chunks ONLY for regression (not n-gram mining)
--fit-replay-steps:32  sampled stages for regression-only replay (0=all)
--balanced-rank-fit:true  each source path contributes equally to regression
--rank-scales:4       compare distances 1,2,4,8 within corruption trajectory (1=legacy)
--mine-start-stride:2  approximate n-gram start sampling (1=all starts)
--mine-gap-samples:6  stratified wildcard-gap proposal; 0=exhaustive mining
--screen-candidates:256  rank-residual supervised proposal shortlist; 0=all candidates
--screen-explore:48  reserve gradient-independent exploratory proposals
--readout-refresh-every:4  refit frozen readout on new train data if separately validated (0=off)
--readout-blend:0.5   interpolate frozen/updated readout to avoid abrupt shifts
--validation-steps:64  sampled noise stages for search/rotating acceptance (0=all)
--anchor-steps:48      sampled stages on held-fixed promotion anchors (0=all)
--monitor-steps:96     sampled stages on fixed protected heldout (0=all)
--anchor-margin:0.002  anchor merit improvement required to promote
--accept-floor:0.0     reject a candidate negatively ranked on either acceptance set
--min-source-win:0.5  require >=50% source chunks improve on acceptance and anchors (0=disabled)
--accept-margin:0.002  required acceptance merit improvement
--rule-penalty:0.001  acceptance penalty per additional rule
--max-chunk-tokens:4096  Skip entire chunks longer than 4096 tokens (0 disables filter)
--shuffle-corpus:true   Deterministic seeded shuffle BEFORE train/search/accept/holdout split; new runs only
--shared-noise-alphabet:true  Use same train-derived corruption symbol vocabulary across all data pools
--trajectory-limit-mib:1024  Fail fast if 2 noise-series minimum exceeds this MiB (0 = unlimited)
--min-ngram:1 --max-gram:4 --max-total-gram:6 --max-wildcards:4
--top-per-shape:0      0 = max(1, total transformed length div eligible Shape count)
--max-candidates:0     0 = score all heap candidates; positive = explicit frequency cap
--cms-width:262144 --cms-depth:7
--ridge:0.01 --repair-passes:1 --phi-threshold:0.5
--regrow-every:4      minimum iterations between attempted rollbacks (0 disables)
--regrow-patience:5   trigger after 5 consecutive iterations without robust promotion
--regrow-depth:2      regenerate this many downstream rules after rollback
--regrow-positions:0  rollback targets per iteration (0 disables; opt-in for new runs)
--regrow-ablate:4     candidate indices for full-replay ablation (small = fast)
--regrow-max-loss:0.04  maximum search-rho loss on root ablation (late fallback allowed)
--regrow-beam:2       hard cap on local regrow paths
--regrow-roots:4      root variants retained before their affected rules are repaired
--regrow-targets:2    highest-impact nonadjacent targets explored per wave
--regrow-samples:64   balanced train snapshots for dependency measurement
--regrow-mines:8     maximum unique mining prefixes per regrow transaction (0 = unlimited)
--rhs-probe:4         temporary RHS search branches kept until next lookahead
--stability-penalty:0.35  Penalize search-case Spearman disagreement, not acceptance rho
--repair-candidates:16  legacy local repair only (no longer called in main loop)
--repair-beam:2        Beam width inside repair only (separate from --beam)
--repair-full-sweep:false  Test every rule during first repair pass if true
--seed:20261013 --bytes:false --plot:true --verbose:false
--out:lens_run --resume:lens_run/checkpoint.json

Default tokens are Unicode code points; --bytes:true preserves the supplied byte mode.
Noise is drawn from observed source symbols, always different from the original.
Mean |phi| for EACH rule includes every other rule, before AND after it.
Rollback & Regrow replaces legacy repair in the main loop.
Model promotion requires higher out-of-source acceptance merit than the prior champion.
Old checkpoints (format=1/2) can resume (one-time champion refit); format=3 also preserves frozen readout weights.
New evaluation rows are logged in training_v3.tsv (no mixing with old TSV metrics).
Plot: gnuplot training.gnuplot (inside output directory)
Predict: ./cms --predict:lens_run/model.json --text:example.txt
"""

proc tokens(text: string, bytes: bool): seq[int] =
  if bytes:
    for c in text: result.add(ord(c))
  else:
    if validateUtf8(text) != -1: raise newException(ValueError, "Invalid UTF-8 (use --bytes:true for binary input)")
    for r in text.runes: result.add(int(r))

type ChunkSpan* = tuple[first, past: int64]

proc shuffledCorpusSpans*(spans: seq[ChunkSpan], seed: int): seq[ChunkSpan] =
  ## Defensive copy: shuffling the partition must not mutate the original list.
  result = newSeq[ChunkSpan](spans.len)
  for i, span in spans: result[i] = span
  var rng = initRand(seed)
  shuffle(rng, result)

proc indexCorpus*(path: string, bytes: bool, maxChunkTokens: int,
                  skipped: var int): seq[ChunkSpan] =
  ## Keep offsets for eligible chunks only. Count exactly the token units
  ## readChunk produces (including its synthesized newline per source line).
  ## For valid UTF-8, non-continuation bytes equal Unicode code point count;
  ## this avoids allocating seq[int] for chunks we will skip.
  var f = open(path, fmRead)
  defer: f.close()
  var first = 0'i64
  var nonempty = false
  var tokenCount = 0
  while not f.endOfFile:
    let lineStart = f.getFilePos()
    let line = f.readLine()
    if line == "===SPLIT===":
      if nonempty:
        if maxChunkTokens == 0 or tokenCount <= maxChunkTokens:
          result.add((first, lineStart))
        else:
          inc skipped
      first = f.getFilePos()
      nonempty = false
      tokenCount = 0
    else:
      nonempty = true
      if maxChunkTokens == 0 or tokenCount <= maxChunkTokens:
        inc tokenCount # readChunk always appends one newline
        if bytes:
          tokenCount += line.len
        else:
          for c in line:
            if (ord(c) and 0xC0) != 0x80:
              inc tokenCount
            if maxChunkTokens > 0 and tokenCount > maxChunkTokens:
              break # Once over the cap, ignore the rest of this chunk.
  if nonempty:
    if maxChunkTokens == 0 or tokenCount <= maxChunkTokens:
      result.add((first, f.getFilePos()))
    else:
      inc skipped

proc readChunk(path: string, span: ChunkSpan, bytes: bool): seq[int] =
  var f = open(path, fmRead)
  defer: f.close()
  f.setFilePos(span.first)
  var text = ""
  while f.getFilePos() < span.past:
    text.add(f.readLine() & "\n")
  tokens(text, bytes)

proc trajectoryBaseMiB(groups: seq[seq[seq[int]]], paths, steps: int): float64 =
  ## Lower-bound snapshot bytes across ALL current groups; no implicit sampling.
  for group in groups:
    for chunk in group:
      let n = chunk.len
      let stages = 1 + (if steps <= 0: n else: min(steps, n))
      result += float64(paths) * float64(stages) *
        float64(n) * float64(sizeof(int)) / (1024.0 * 1024.0)

proc validate(o: Options) =
  if o.window < 2 or o.paths < 1 or o.iterations < 0 or o.noiseSteps < 0 or
      o.holdout < 0 or o.trajectoryLimitMiB < 0 or
      o.maxChunkTokens < 0 or o.searchPool < 1 or o.acceptPool < 1 or
      o.searchChunks < 1 or o.acceptChunks < 1 or
      o.replayChunks < 0 or o.regrowEvery < 0 or o.regrowPatience < 1 or
      o.anchorChunks < 0 or o.anchorPool < 0 or o.anchorRefresh < 1 or
      o.fitReplayChunks < 0 or o.fitReplaySteps < 0 or
      o.readoutRefreshEvery < 0 or o.readoutBlend <= 0 or o.readoutBlend > 1 or
      o.search.rankScales < 1 or o.search.rankScales > 16 or
      o.search.mining.startStride < 1 or o.search.mining.gapSamples < 0 or
      o.search.screenCandidates < 0 or o.search.screenExplore < 0 or o.validationSteps < 0 or
      o.anchorSteps < 0 or o.monitorSteps < 0 or
      o.anchorMargin < 0 or o.acceptFloor < -1 or o.acceptFloor > 1 or
      o.minSourceWin < 0 or o.minSourceWin > 1 or
      o.acceptanceMargin < 0 or o.rulePenalty < 0 or
      o.search.regrowDepth < 0 or o.search.regrowPositions < 0 or
      o.search.regrowAblate < 0 or o.search.regrowBeam < 1 or
      o.search.regrowRoots < 1 or o.search.regrowTargets < 1 or
      o.search.regrowSamples < 1 or o.search.regrowMines < 0 or
      o.search.regrowMaxLoss < 0 or o.search.stabilityPenalty < 0 or
      o.search.rhsProbe < 1 or
      o.search.beam < 1 or o.search.lookahead < 1 or
      o.search.repairPasses < 0 or o.search.repairCandidates < 0 or
      o.search.repairBeam < 1 or o.search.ridge <= 0 or
      o.search.phiThreshold < 0 or o.search.phiThreshold > 1 or
      o.search.maxCandidates < 0 or o.search.mining.topPerShape < 0:
    raise newException(ValueError, "Invalid configuration; see --help")
  if classify(o.search.ridge) in {fcNan, fcInf, fcNegInf} or
      classify(o.search.phiThreshold) in {fcNan, fcInf, fcNegInf} or
      classify(o.acceptanceMargin) in {fcNan, fcInf, fcNegInf} or
      classify(o.rulePenalty) in {fcNan, fcInf, fcNegInf} or
      classify(o.anchorMargin) in {fcNan, fcInf, fcNegInf} or
      classify(o.acceptFloor) in {fcNan, fcInf, fcNegInf} or
      classify(o.minSourceWin) in {fcNan, fcInf, fcNegInf} or
      classify(o.search.regrowMaxLoss) in {fcNan, fcInf, fcNegInf} or
      classify(o.search.stabilityPenalty) in {fcNan, fcInf, fcNegInf} or
      classify(o.readoutBlend) in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "Nonfinite configuration")

proc saveAtomic(path, content: string) =
  writeFile(path & ".tmp", content)
  moveFile(path & ".tmp", path)

proc toRuleNode(rules: seq[Rule]): JsonNode =
  result = newJArray()
  for rule in rules: result.add(%*{"lhs": rule.lhs, "rhs": rule.rhs})

proc readRules(node: JsonNode): seq[Rule] =
  for n in node:
    let rule = Rule(lhs: n["lhs"].to(seq[int]), rhs: n["rhs"].to(seq[int]))
    if rule.lhs.len == 0 or rule.lhs[0] < 0 or rule.lhs[^1] < 0:
      raise newException(ValueError, "Checkpoint rule must have literal anchors")
    for i in 1..<rule.lhs.len:
      if rule.lhs[i] < 0 and rule.lhs[i-1] < 0:
        raise newException(ValueError, "Adjacent wildcards unsupported")
    let wc = wildcardCount(rule.lhs)
    if wc > 4 or rule.rhs.len > 64:
      raise newException(ValueError, "Invalid checkpoint rule size")
    for t in rule.rhs:
      if t < -wc: raise newException(ValueError, "Invalid capture reference")
    result.add(rule)

proc validateTokens(state: State, firstToken: int) =
  # Learned literals can only refer to markers introduced by earlier rules.
  var available = firstToken
  for rule in state.rules:
    for t in rule.lhs:
      if t >= available: raise newException(ValueError, "Forward/dangling learned token")
    for t in rule.rhs:
      if t >= available: available = t+1
  if state.nextToken < available:
    raise newException(ValueError, "Checkpoint nextToken overlaps learned tokens")

proc saveCheckpoint(o: Options, beam: seq[State], accepted: State,
                    iteration: int, fingerprint: string,
                    rollback: RollbackControl) =
  var states = newJArray()
  for s in beam:
    states.add(%*{"rules": toRuleNode(s.rules), "nextToken": s.nextToken})
  let acceptedNode = %*{"rules": toRuleNode(accepted.rules),
                         "nextToken": accepted.nextToken,
                         "weights": accepted.evaluation.weights}
  saveAtomic(o.output / "checkpoint.json", pretty(%*{
    "format": 3, "iteration": iteration, "corpusSha1": fingerprint,
    "options": toJson(o), "beam": states, "accepted": acceptedNode,
    "rollback": {"noPromotionStreak": rollback.noPromotionStreak,
                 "lastAttempt": rollback.lastAttempt}}))

proc saveModel(o: Options, state: State, iteration: int) =
  saveAtomic(o.output / "model.json", pretty(%*{
    "format": 1, "iteration": iteration, "bytes": o.bytes, "minWildcardLength": 1,
    "target": "noise_fraction", "rules": toRuleNode(state.rules),
    "weights": state.evaluation.weights, "trainSpearman": state.evaluation.train,
    "selectionSpearman": state.evaluation.score,
    "meanAbsPhiAllOtherRules": state.evaluation.means}))

proc emitPlot(o: Options) =
  let plotScript = """
# Lens plot styling, based on the classic clean scientific plot.
# ALL four series use the SAME plotted transform: -log2(1 - Spearman rho).
# The underlying TSV remains raw Spearman rho for scientific reproducibility.
set terminal pngcairo size 1600,870 enhanced font ',14'
set output 'training.png'
set datafile separator '\t'
set datafile missing 'nan'
set encoding utf8

# Matplotlib-like white canvas, delicate grid and restrained typography.
set border 3 back linewidth 1.2 linecolor rgb '#323232'
set tics nomirror out scale 0.65 textcolor rgb '#292929'
set grid xtics ytics back linewidth 0.7 linecolor rgb '#dddddd'
set lmargin 11
set rmargin 4
set tmargin 3
set bmargin 5
set title 'Lens / Training progress' font ',18' offset 0,0.4
set xlabel 'iteration' offset 0,0.7
set ylabel '-log2(1 - Spearman rho)' offset 1.0,0

# All y values have the same meaning: never mix raw rho with transformed rho.
# Keep a fixed x origin so even the first iteration has a nonzero x-range.
set xrange [0:*]
set autoscale y
set key left bottom opaque box linecolor rgb '#c4c4c4' linewidth 0.8 samplen 2.5 spacing 1.15

# The protected held-out curve is the primary generalization indicator.
set style line 1 lt 1 lw 2.9 pt 7  ps 0.65 lc rgb '#2b7bb9'
set style line 2 lt 1 lw 2.6 pt 5  ps 0.65 lc rgb '#008f80'
set style line 3 lt 1 lw 2.3 pt 9  ps 0.60 lc rgb '#e68018'
set style line 4 lt 1 lw 2.0 pt 13 ps 0.55 lc rgb '#8855a2'

# This is log BASE TWO (not the natural logarithm). For rho == 1,
# cap the visualization only, preventing Inf from blanking the plot.
# Negative Spearman rho is valid and maps to a negative plotted value.
f(r) = -log(1.0 - (r >= 1.0 ? 1.0 - 1.0e-12 : r)) / log(2.0)

# training_v3.tsv: 1 iteration, 3 search rho, 4 acceptance rho,
#                  5 anchor rho, 6 held-out rho.
plot 'training_v3.tsv' using 1:(f($6)) with linespoints ls 1 title 'Held-out (fixed)', \
     '' using 1:(f($5)) with linespoints ls 2 title 'Anchor (rotating)', \
     '' using 1:(f($4)) with linespoints ls 3 title 'Acceptance (rotating)', \
     '' using 1:(f($3)) with linespoints ls 4 title 'Search validation'
unset output
"""
  let scriptPath = o.output / "training.gnuplot"
  writeFile(scriptPath, plotScript)
  if not o.plot: return
  let logPath = o.output / "training_v3.tsv"
  if not fileExists(logPath): return
  var validRows = 0
  for line in lines(logPath):
    if line.len == 0 or line[0] == '#': continue
    let cols = line.split('\t')
    if cols.len >= 5:
      try:
        discard parseFloat(cols[3])
        inc validRows
      except ValueError: discard
  if validRows == 0: return
  let gp = findExe("gnuplot")
  if gp.len == 0:
    stderr.writeLine("[plot] gnuplot unavailable; script/TSV retained")
    return
  let stagingScript = o.output / "training.__render.gnuplot"
  let stagingPng = o.output / "training.__render.png"
  if fileExists(stagingPng): removeFile(stagingPng)
  writeFile(stagingScript,
    plotScript.replace("set output 'training.png'",
                       "set output 'training.__render.png'"))
  let process = startProcess(gp, workingDir = absolutePath(o.output),
                             args = @["training.__render.gnuplot"],
                             options = {poParentStreams})
  let code = waitForExit(process)
  close(process)
  if fileExists(stagingScript): removeFile(stagingScript)
  if code == 0 and fileExists(stagingPng) and getFileSize(stagingPng) > 3000:
    moveFile(stagingPng, o.output / "training.png")
  else:
    stderr.writeLine("[plot] rendering failed; kept previous training.png")
    if fileExists(stagingPng): removeFile(stagingPng)

proc canPromoteV8(candidate, champion: State,
                  rot, anchors: DataSet,
                  candidateRot, baselineRot, candidateAnchor, baselineAnchor: float64,
                  paths: int, o: Options): bool =
  if not canAcceptRobust(candidateRot,baselineRot,candidateAnchor,baselineAnchor,
        candidate.rules.len,champion.rules.len,
        o.rulePenalty,o.acceptanceMargin,o.anchorMargin,o.acceptFloor):
    return false
  if o.minSourceWin <= 0: return true
  let rotWins = sourceWinFraction(candidate,champion,rot,paths)
  let anchorWins = (if anchors.texts.len > 0:
                      sourceWinFraction(candidate,champion,anchors,paths)
                    else: rotWins)
  result = rotWins >= o.minSourceWin and anchorWins >= o.minSourceWin
  stderr.writeLine("[source consensus] rotating=",rotWins,
    " anchor=",anchorWins," required=",o.minSourceWin,
    " pass=",result)

proc prediction(modelPath, textPath: string) =
  let node = parseFile(modelPath)
  if node["format"].getInt != 1: raise newException(ValueError, "Unknown model format")
  let rules = readRules(node["rules"])
  let text = tokens(readFile(textPath), node["bytes"].getBool)
  let traced = traceRules(rules, DataSet(texts: @[text]))
  let weights = node["weights"].to(seq[float64])
  if weights.len != rules.len: raise newException(ValueError, "model weight mismatch")
  echo "rank_score=", predict(traced.columns, weights, 1)[0]
  echo "Higher score predicts more noise; not a calibrated noise probability."

proc main() =
  var o = defaults()
  var args: seq[tuple[key, value: string]]
  for kind, key, value in getopt():
    if kind in {cmdLongOption, cmdShortOption}: args.add((key, value))
    elif kind == cmdArgument: raise newException(ValueError, "Use --corpus:path")
  for arg in args:
    if arg.key in ["help", "h"]: echo help; return
    if arg.key == "resume":
      let checkpoint = parseFile(arg.value)
      # Old checkpoints do not contain the new repair flags. Fill missing
      # properties before deserialization to preserve resume compatibility.
      let savedOptions = checkpoint["options"]
      let defaultOptions = toJson(defaults())
      # Older checkpoints also lack the top-level diagnostic safety option.
      # Old checkpoints used file order. Reordering them during resume would
      # silently change every train/validation source and invalidate models.
      if not savedOptions.hasKey("shuffleCorpus"):
        savedOptions["shuffleCorpus"] = %false
      if not savedOptions.hasKey("sharedNoiseAlphabet"):
        savedOptions["sharedNoiseAlphabet"] = %false
      for key, value in defaultOptions:
        if not savedOptions.hasKey(key):
          savedOptions[key] = value
      for key, value in defaultOptions["search"]:
        if not savedOptions["search"].hasKey(key):
          savedOptions["search"][key] = value
      # Nested MiningConfig gained startStride in v7. Old format-3 files
      # require a deep merge here, not just SearchConfig field merging.
      for key, value in defaultOptions["search"]["mining"]:
        if not savedOptions["search"]["mining"].hasKey(key):
          savedOptions["search"]["mining"][key] = value
      o = savedOptions.to(Options)
      o.resume = arg.value
  var modelPath, textPath: string
  for (key, value) in args:
    case key
    of "corpus": o.corpus = value
    of "out": o.output = value
    of "resume": o.resume = value
    of "iterations": o.iterations = parseInt(value)
    of "window": o.window = parseInt(value)
    of "paths": o.paths = parseInt(value)
    of "noise-steps": o.noiseSteps = parseInt(value)
    of "trajectory-limit-mib": o.trajectoryLimitMiB = parseInt(value)
    of "max-chunk-tokens": o.maxChunkTokens = parseInt(value)
    of "seed": o.seed = parseInt(value)
    of "holdout": o.holdout = parseInt(value)
    of "search-pool": o.searchPool = parseInt(value)
    of "accept-pool": o.acceptPool = parseInt(value)
    of "search-chunks": o.searchChunks = parseInt(value)
    of "replay-chunks": o.replayChunks = parseInt(value)
    of "regrow-every": o.regrowEvery = parseInt(value)
    of "regrow-patience": o.regrowPatience = parseInt(value)
    of "regrow-max-loss": o.search.regrowMaxLoss = parseFloat(value)
    of "accept-chunks": o.acceptChunks = parseInt(value)
    of "accept-margin": o.acceptanceMargin = parseFloat(value)
    of "anchor-margin": o.anchorMargin = parseFloat(value)
    of "accept-floor": o.acceptFloor = parseFloat(value)
    of "min-source-win": o.minSourceWin = parseFloat(value)
    of "anchor-chunks": o.anchorChunks = parseInt(value)
    of "anchor-pool": o.anchorPool = parseInt(value)
    of "anchor-refresh": o.anchorRefresh = parseInt(value)
    of "fit-replay-chunks": o.fitReplayChunks = parseInt(value)
    of "fit-replay-steps": o.fitReplaySteps = parseInt(value)
    of "readout-refresh-every": o.readoutRefreshEvery = parseInt(value)
    of "readout-blend": o.readoutBlend = parseFloat(value)
    of "balanced-rank-fit": o.search.balanceCases = parseBool(value)
    of "rank-scales": o.search.rankScales = parseInt(value)
    of "mine-start-stride": o.search.mining.startStride = parseInt(value)
    of "mine-gap-samples": o.search.mining.gapSamples = parseInt(value)
    of "screen-candidates": o.search.screenCandidates = parseInt(value)
    of "screen-explore": o.search.screenExplore = parseInt(value)
    of "validation-steps": o.validationSteps = parseInt(value)
    of "anchor-steps": o.anchorSteps = parseInt(value)
    of "monitor-steps": o.monitorSteps = parseInt(value)
    of "rule-penalty": o.rulePenalty = parseFloat(value)
    of "bytes": o.bytes = parseBool(value)
    of "shuffle-corpus": o.shuffleCorpus = parseBool(value)
    of "shared-noise-alphabet": o.sharedNoiseAlphabet = parseBool(value)
    of "plot": o.plot = parseBool(value)
    of "beam": o.search.beam = parseInt(value)
    of "lookahead": o.search.lookahead = parseInt(value)
    of "ridge": o.search.ridge = parseFloat(value)
    of "stability-penalty": o.search.stabilityPenalty = parseFloat(value)
    of "repair-passes": o.search.repairPasses = parseInt(value)
    of "regrow-depth": o.search.regrowDepth = parseInt(value)
    of "regrow-positions": o.search.regrowPositions = parseInt(value)
    of "regrow-ablate": o.search.regrowAblate = parseInt(value)
    of "regrow-beam": o.search.regrowBeam = parseInt(value)
    of "regrow-roots": o.search.regrowRoots = parseInt(value)
    of "regrow-targets": o.search.regrowTargets = parseInt(value)
    of "regrow-samples": o.search.regrowSamples = parseInt(value)
    of "regrow-mines": o.search.regrowMines = parseInt(value)
    of "rhs-probe": o.search.rhsProbe = parseInt(value)
    of "repair-candidates": o.search.repairCandidates = parseInt(value)
    of "repair-beam": o.search.repairBeam = parseInt(value)
    of "repair-full-sweep": o.search.repairFullSweep = parseBool(value)
    of "phi-threshold": o.search.phiThreshold = parseFloat(value)
    of "max-candidates": o.search.maxCandidates = parseInt(value)
    of "top-per-shape": o.search.mining.topPerShape = parseInt(value)
    of "min-ngram": o.search.mining.minGram = parseInt(value)
    of "max-gram": o.search.mining.maxGram = parseInt(value)
    of "max-total-gram": o.search.mining.maxTotalGram = parseInt(value)
    of "max-wildcards": o.search.mining.maxWildcards = parseInt(value)
    of "cms-width": o.search.mining.width = parseInt(value)
    of "cms-depth": o.search.mining.depth = parseInt(value)
    of "verbose": o.search.mining.verbose = parseBool(value)
    of "predict": modelPath = value
    of "text": textPath = value
    else: raise newException(ValueError, "Unknown option: " & key)
  if modelPath.len > 0:
    if textPath.len == 0: raise newException(ValueError, "--text is required")
    prediction(modelPath, textPath)
    return
  validate(o)
  let fingerprint = $secureHashFile(o.corpus)
  var skippedChunks = 0
  var corpus = indexCorpus(o.corpus, o.bytes, o.maxChunkTokens, skippedChunks)
  # Split *after* shuffling. Input corpora may be sorted by repository,
  # language, or source date. A tail-only holdout can be systematically OOD.
  # The permutation depends solely on the saved seed and filtered chunk list.
  if o.resume.len > 0:
    let saved = parseFile(o.resume)["options"]
    let wasShuffled = (if saved.hasKey("shuffleCorpus"):
                         saved["shuffleCorpus"].getBool else: false)
    let usedSharedNoise = (if saved.hasKey("sharedNoiseAlphabet"):
                             saved["sharedNoiseAlphabet"].getBool else: false)
    if o.shuffleCorpus != wasShuffled or
       o.sharedNoiseAlphabet != usedSharedNoise:
      raise newException(ValueError,
        "Cannot change corpus partition or noise alphabet mode on resume; use new --out")
    if o.shuffleCorpus and (o.seed != saved["seed"].getInt or
        o.maxChunkTokens != saved["maxChunkTokens"].getInt or
        o.corpus != saved["corpus"].getStr):
      raise newException(ValueError,
        "Cannot change seed, corpus path or chunk limit for a shuffled checkpoint")
  if o.shuffleCorpus:
    corpus = shuffledCorpusSpans(corpus,o.seed)
  stderr.writeLine("[corpus] eligibleChunks=", corpus.len,
    " skippedOverLimit=", skippedChunks,
    " maxChunkTokens=", o.maxChunkTokens,
    " tokenMode=", (if o.bytes: "bytes" else: "unicode"),
    " shuffled=", o.shuffleCorpus,
    " sharedNoiseAlphabet=", o.sharedNoiseAlphabet)
  if corpus.len < 2:
    raise newException(ValueError,
      "Need at least two chunks within --max-chunk-tokens:" &
      $o.maxChunkTokens & " (found " & $corpus.len & ").")
  let window = min(o.window, corpus.len)
  let heldout = min(o.holdout, max(0, corpus.len-window-2))
  let plan = splitChunks(corpus.len, window, heldout, o.searchPool, o.acceptPool)
  stderr.writeLine("[partition] train=[0,",plan.trainEnd,") search=[",
    plan.trainEnd,",",plan.searchEnd,") accept=[",plan.searchEnd,",",
    plan.acceptEnd,") heldout=[",plan.acceptEnd,",",plan.total,")")
  # Select a shared noise alphabet only from the training pool, never from
  # validation or protected heldout texts. Split-specific noise vocabularies
  # can introduce avoidable distribution differences in evaluation.
  var globalNoiseAlphabet: seq[int]
  if o.sharedNoiseAlphabet:
    var seen = initHashSet[int]()
    for idx in 0..<min(plan.trainEnd, 256):
      for t in readChunk(o.corpus, corpus[idx], o.bytes): seen.incl(t)
    for t in seen: globalNoiseAlphabet.add(t)
    globalNoiseAlphabet.sort()
    if globalNoiseAlphabet.len < 2:
      raise newException(ValueError,"Need at least two distinct training symbols for shared noise alphabet")
    stderr.writeLine("[noise alphabet] sampledTrainChunks=",min(plan.trainEnd,256),
      " distinctSymbols=",globalNoiseAlphabet.len)
  var monitor: DataSet
  if heldout > 0:
    var holdoutTexts: seq[seq[int]]
    for i in plan.acceptEnd..<corpus.len:
      holdoutTexts.add(readChunk(o.corpus, corpus[i], o.bytes))
    let monitorMiB = trajectoryBaseMiB(@[holdoutTexts],o.paths,o.monitorSteps)
    if o.trajectoryLimitMiB > 0 and monitorMiB > float64(o.trajectoryLimitMiB):
      raise newException(ValueError,"Held-out snapshot lower bound exceeds memory cap")
    monitor = trajectories(holdoutTexts, o.seed+999983, o.paths, o.monitorSteps,
                           alphabetOverride=globalNoiseAlphabet)
  var beam = @[State(nextToken: (if o.bytes: 256 else: 0x110000))]
  var accepted = beam[0]
  var rollback: RollbackControl
  var iteration = 0
  var needsChampionRefit = false
  if o.resume.len > 0:
    let checkpoint = parseFile(o.resume)
    if checkpoint["format"].getInt notin [1,2,3] or
        checkpoint["corpusSha1"].getStr != fingerprint:
      raise newException(ValueError, "Checkpoint format or corpus fingerprint mismatch")
    if checkpoint["options"]["bytes"].getBool != o.bytes:
      raise newException(ValueError, "Cannot change tokenization on resume")
    iteration = checkpoint["iteration"].getInt
    beam = @[]
    for node in checkpoint["beam"]:
      let state = State(rules: readRules(node["rules"]), nextToken: node["nextToken"].getInt)
      validateTokens(state, if o.bytes: 256 else: 0x110000)
      beam.add(state)
    if beam.len == 0: raise newException(ValueError, "Empty checkpoint beam")
    if checkpoint["format"].getInt in [2,3]:
      let node = checkpoint["accepted"]
      accepted = State(rules: readRules(node["rules"]), nextToken: node["nextToken"].getInt)
      validateTokens(accepted, if o.bytes: 256 else: 0x110000)
      if checkpoint["format"].getInt == 3 and node.hasKey("weights"):
        accepted.evaluation.weights = node["weights"].to(seq[float64])
      if accepted.evaluation.weights.len != accepted.rules.len:
        needsChampionRefit = true
        stderr.writeLine("[resume] legacy champion lacks saved readout; one-time refit on first training window")
      # Format-2 checkpoints without controller metadata are supported.
      if checkpoint.hasKey("rollback"):
        let meta = checkpoint["rollback"]
        if meta.hasKey("noPromotionStreak"):
          rollback.noPromotionStreak = max(0, meta["noPromotionStreak"].getInt)
        if meta.hasKey("lastAttempt"):
          rollback.lastAttempt = max(0, min(iteration, meta["lastAttempt"].getInt))
    else:
      # Legacy V1 has no independent acceptance data: migrate its best beam
      # into the accepted slot; future promotions require distinct chunks.
      accepted = beam[0]
      needsChampionRefit = true
      stderr.writeLine("[resume] migrating V1 checkpoint, independent acceptance starts now")
  createDir(o.output)
  if o.resume.len == 0 and fileExists(o.output / "checkpoint.json"):
    raise newException(ValueError, "Existing run: use --resume or choose a new --out")
  saveAtomic(o.output / "config.json", pretty(toJson(o)))
  let logPath = o.output / "training_v3.tsv"
  if o.resume.len == 0 or not fileExists(logPath):
    writeFile(logPath, "#iteration\ttrain_rho\tsearch_rho\trot_accept_rho\tanchor_accept_rho\theldout_rho\tbaseline_rot_rho\tbaseline_anchor_rho\trules\taccepted\tseconds\tregrow\n")
  # Reserve an acceptance-only reservoir for periodically refreshed anchor
  # subsets. They NEVER appear in ordinary rotating acceptance in the same run.
  let fixedAnchorPool = anchorIndices(plan.searchEnd,plan.acceptEnd,o.anchorPool)
  var anchor: DataSet
  var currentAnchorBlock = -1
  stderr.writeLine("[config] trainPool=", plan.trainEnd, " window=", window,
    " searchChunks=", min(o.searchChunks,plan.searchEnd-plan.trainEnd),
    " acceptanceChunks=", min(o.acceptChunks,plan.acceptEnd-plan.searchEnd-fixedAnchorPool.len),
    " holdoutChunks=", heldout, " noiseSteps=", o.noiseSteps,
    " beamStates=", o.search.beam)
  let startIteration = iteration
  while o.iterations == 0 or iteration-startIteration < o.iterations:
    let started = epochTime()
    stderr.writeLine("[prepare] iteration=", iteration+1, " loading window=", window)
    let anchorBlock = iteration div o.anchorRefresh
    if anchorBlock != currentAnchorBlock:
      currentAnchorBlock = anchorBlock
      var anchorTexts: seq[seq[int]]
      var activeAnchorIds: seq[int]
      if fixedAnchorPool.len > 0:
        let take = min(o.anchorChunks,fixedAnchorPool.len)
        let offset = (anchorBlock*take) mod fixedAnchorPool.len
        for j in 0..<take:
          let idx = fixedAnchorPool[(offset+j) mod fixedAnchorPool.len]
          activeAnchorIds.add(idx)
          anchorTexts.add(readChunk(o.corpus,corpus[idx],o.bytes))
      if anchorTexts.len > 0 and o.trajectoryLimitMiB > 0 and
          trajectoryBaseMiB(@[anchorTexts],o.paths,o.anchorSteps) > float64(o.trajectoryLimitMiB):
        raise newException(ValueError,"Acceptance anchors exceed memory cap; lower --anchor-steps")
      anchor = trajectories(anchorTexts,o.seed+333177+anchorBlock*7919,
                             o.paths,o.anchorSteps,
                             alphabetOverride=globalNoiseAlphabet)
      stderr.writeLine("[anchors] epoch=",anchorBlock," ids=",activeAnchorIds,
        " rows=",anchor.texts.len, " frozenReadout=true")
    var clean, searchClean, acceptClean, replayFitClean: seq[seq[int]]
    var trainIds: seq[int]
    for j in 0..<window:
      let idx = (iteration+j) mod plan.trainEnd
      trainIds.add(idx)
      clean.add(readChunk(o.corpus, corpus[idx], o.bytes))
    # Rehearsal samples are always drawn from the TRAIN pool only.
    for j in 0..<min(o.replayChunks,plan.trainEnd-window):
      let idx = ((iteration + plan.trainEnd - window - 1 - j) mod plan.trainEnd)
      if idx notin trainIds:
        trainIds.add(idx)
        clean.add(readChunk(o.corpus, corpus[idx], o.bytes))
    let searchIds = rotatingIndices(plan.trainEnd, plan.searchEnd,
                                     o.searchChunks, iteration)
    let acceptIds = rotatingExcluding(plan.searchEnd, plan.acceptEnd,
                                      o.acceptChunks, iteration, fixedAnchorPool)
    # The extra replay sources inform the regression weights, but MUST NOT
    # enlarge the n-gram mining corpus. This costs only a small fixed number
    # of sampled trajectories regardless of source chunk length.
    let replayCount = min(o.fitReplayChunks,max(0,plan.trainEnd-trainIds.len))
    var replayFitIds: seq[int]
    if replayCount > 0:
      let stride = max(1,plan.trainEnd div replayCount)
      let offset = (iteration*7919) mod plan.trainEnd
      for j in 0..<replayCount:
        let idx = (offset + j*stride) mod plan.trainEnd
        if idx notin trainIds and idx notin replayFitIds:
          replayFitIds.add(idx)
          replayFitClean.add(readChunk(o.corpus,corpus[idx],o.bytes))
    for idx in searchIds: searchClean.add(readChunk(o.corpus, corpus[idx], o.bytes))
    for idx in acceptIds: acceptClean.add(readChunk(o.corpus, corpus[idx], o.bytes))
    let lowerMiB = trajectoryBaseMiB(@[clean],o.paths,o.noiseSteps) +
                   trajectoryBaseMiB(@[replayFitClean],o.paths,o.fitReplaySteps) +
                   trajectoryBaseMiB(@[searchClean,acceptClean],o.paths,o.validationSteps)
    stderr.writeLine("[trajectory budget] train+search+accept minimumMiB=",
      formatFloat(lowerMiB,ffDecimal,1), " capMiB=",o.trajectoryLimitMiB,
      " (heldout snapshots are additional)")
    if o.trajectoryLimitMiB > 0 and lowerMiB > float64(o.trajectoryLimitMiB):
      raise newException(ValueError, "Current train+search+accept noise snapshots exceed --trajectory-limit-mib. " &
        "Reduce --noise-steps or --max-chunk-tokens; checkpoint unchanged.")
    var train = trajectories(clean,o.seed+iteration*1009,o.paths,o.noiseSteps,true,"train",
                             alphabetOverride=globalNoiseAlphabet)
    if replayFitClean.len > 0:
      let fittedReplay = trajectories(replayFitClean,o.seed+iteration*1009+44801,
                                       o.paths,o.fitReplaySteps,
                                       alphabetOverride=globalNoiseAlphabet)
      var caseOffset = 0
      for cid in train.cases:
        if cid >= caseOffset: caseOffset = cid+1
      for i, row in fittedReplay.texts:
        train.texts.add(row)
        train.target.add(fittedReplay.target[i])
        train.cases.add(caseOffset + fittedReplay.cases[i])
    let selection = trajectories(searchClean,o.seed+iteration*1009+49979687,
                                 o.paths,o.validationSteps,true,"search",
                                 alphabetOverride=globalNoiseAlphabet)
    let acceptance = trajectories(acceptClean,o.seed+iteration*1009+99877771,
                                  o.paths,o.validationSteps,true,"accept",
                                  alphabetOverride=globalNoiseAlphabet)
    stderr.writeLine("[iteration] ",iteration+1," trainRows=",train.texts.len,
      " searchRows=",selection.texts.len, " acceptRows=",acceptance.texts.len,
      " trainSources=",trainIds," fitReplaySources=",replayFitIds,
      " searchSources=",searchIds,
      " acceptSources=",acceptIds)
    # Refit exploratory states on this iteration's train + regression-only
    # replay rows. No validation rows influence regression coefficients.
    # CRITICAL: the accepted champion's fitted weights must NEVER be refitted
    # merely because the training window rotated. V4 silently changed the
    # model even on rejected iterations. Legacy checkpoints require a single
    # explicit migration fit because they omitted champion weights.
    if needsChampionRefit:
      accepted.evaluate(train,selection,o.search.ridge,
                        o.search.stabilityPenalty,
                        o.search.balanceCases,o.search.rankScales)
      needsChampionRefit = false
    let baselineRho = outOfSampleRho(accepted,acceptance)
    let baselineAnchor = (if anchor.texts.len > 0:
                            outOfSampleRho(accepted,anchor) else: baselineRho)
    var parentChampion = accepted
    parentChampion.evaluate(train,selection,o.search.ridge,o.search.stabilityPenalty,
                            o.search.balanceCases,o.search.rankScales)
    for i in 0..<beam.len:
      beam[i].evaluate(train,selection,o.search.ridge,o.search.stabilityPenalty,
                       o.search.balanceCases,o.search.rankScales)
      beam[i].novelty = (if beam[i].evaluation.means.len > 0:
                           beam[i].evaluation.means[^1] else: 0.0)
    let parents = selectBeam(beam & @[parentChampion],o.search.beam,iteration)
    var expanded = addBeam(parents,clean,train,selection,o.search,iteration)
    expanded.add(parentChampion)
    # First evaluate ordinary beam candidates on disjoint acceptance chunks.
    # Rollback is NOT periodic: it is reserved for an acceptance plateau.
    var shortlist = selectBeam(expanded,o.search.beam,iteration)
    var nextAccepted = accepted
    var chosenRho = baselineRho
    var chosenAnchor = baselineAnchor
    for candidate in shortlist:
      if candidate.rules == accepted.rules: continue
      let rho = outOfSampleRho(candidate,acceptance)
      let anchorRho = (if anchor.texts.len > 0:
                         outOfSampleRho(candidate,anchor) else: rho)
      if canPromoteV8(candidate,nextAccepted,acceptance,anchor,
                      rho,chosenRho,anchorRho,chosenAnchor,o.paths,o):
        nextAccepted = candidate
        chosenRho = rho
        chosenAnchor = anchorRho
    var promoted = nextAccepted.rules != accepted.rules
    var readoutRefreshed = false
    # A frozen champion prevents unintended model drift, but also prevents
    # benign coefficient updates on new source chunks. Propose a conservative
    # weight-only interpolation; keep it ONLY if disjoint rotating acceptance
    # and independently sourced anchors both beat the exact incumbent.
    if not promoted and o.readoutRefreshEvery > 0 and
        (iteration+1) mod o.readoutRefreshEvery == 0 and
        accepted.rules.len > 0 and
        parentChampion.evaluation.weights.len == accepted.evaluation.weights.len:
      let probe = blendReadout(accepted,parentChampion,o.readoutBlend)
      let probeRho = outOfSampleRho(probe,acceptance)
      let probeAnchor = (if anchor.texts.len > 0:
                           outOfSampleRho(probe,anchor) else: probeRho)
      if canPromoteV8(probe,nextAccepted,acceptance,anchor,
                      probeRho,chosenRho,probeAnchor,chosenAnchor,o.paths,o):
        nextAccepted = probe
        chosenRho = probeRho
        chosenAnchor = probeAnchor
        readoutRefreshed = true
        promoted = true
        stderr.writeLine("[readout] validated coefficient refresh; blend=",
                         o.readoutBlend, " acceptance=",probeRho,
                         " anchor=",probeAnchor)
      else:
        stderr.writeLine("[readout] refresh rejected; frozen weights preserved")
    rollback.recordPromotion(promoted)
    let reason = (if o.search.regrowPositions == 0 or o.search.regrowDepth == 0 or
                        o.search.regrowAblate == 0:
                    "disabled"
                  else:
                    rollbackReason(rollback,iteration+1,accepted.rules.len,
                                   o.regrowEvery,o.regrowPatience))
    var attemptedRegrow = false
    var regrew = false
    if not promoted and reason == "plateau":
      attemptedRegrow = true
      rollback.recordAttempt(iteration+1)
      # Late, persistent plateaus permit rewriting strongly dependent roots.
      # Early attempts are restricted to removable/redundant roots only.
      let allowHardRoots = rollback.noPromotionStreak >= 2*o.regrowPatience
      stderr.writeLine("[rollback] triggered iteration=",iteration+1,
        " stagnation=",rollback.noPromotionStreak,
        " root=accepted-champion hardRoots=",allowHardRoots)
      let rebuilt = rollbackRegrow(parentChampion,clean,train,selection,
                                   o.search,iteration,allowHardRoots)
      regrew = rebuilt.rules != accepted.rules
      if regrew:
        let rho = outOfSampleRho(rebuilt,acceptance)
        let anchorRho = (if anchor.texts.len > 0:
                           outOfSampleRho(rebuilt,anchor) else: rho)
        if rebuilt.rules != accepted.rules and
            canPromoteV8(rebuilt,nextAccepted,acceptance,anchor,
                         rho,chosenRho,anchorRho,chosenAnchor,o.paths,o):
          nextAccepted = rebuilt
          chosenRho = rho
          chosenAnchor = anchorRho
          promoted = nextAccepted.rules != accepted.rules
          stderr.writeLine("[rollback accept] independent merit improvement")
        expanded.add(rebuilt)
    elif reason == "cooldown":
      stderr.writeLine("[rollback] deferred, cooldown; stagnation=",
        rollback.noPromotionStreak)
    if promoted:
      # Successful promotion after regrow resets the plateau counter too.
      rollback.noPromotionStreak = 0
      stderr.writeLine("[accept] promoted baselineRho=",baselineRho,
        " candidateRho=",chosenRho," anchorRho=",chosenAnchor,
        " baselineAnchor=",baselineAnchor," rules=",nextAccepted.rules.len)
    else:
      stderr.writeLine("[accept] rejected; baseline merit preserved",
        " stagnation=",rollback.noPromotionStreak,
        " rollbackAttempted=",attemptedRegrow)
    accepted = nextAccepted
    # The exploratory beam is refitted each iteration; the frozen champion
    # is only added to a *future refit*, never mutated in-place.
    beam = selectBeam(expanded,max(0,o.search.beam-1),iteration)
    beam.add(accepted)
    var auditScore = NaN
    if heldout > 0:
      auditScore = outOfSampleRho(accepted,monitor)
    inc iteration
    var maxPhi = 0.0
    for x in accepted.evaluation.means: maxPhi = max(maxPhi,x)
    var log = open(logPath,fmAppend)
    let frozenTrain = outOfSampleRho(accepted,train)
    let frozenSearch = outOfSampleRho(accepted,selection)
    # Diagnostics alone may rotate; no learned coefficient is changed.
    accepted.evaluation.train = frozenTrain
    accepted.evaluation.score = frozenSearch
    log.writeLine(@[$iteration,$frozenTrain,$frozenSearch,$chosenRho,$chosenAnchor,
      (if heldout > 0: $auditScore else: "nan"),$baselineRho,$baselineAnchor,
      $accepted.rules.len,(if promoted: "1" else: "0"),
      $(epochTime()-started),(if regrew: "1" else: "0"),
      (if readoutRefreshed: "1" else: "0")].join("\t"))
    log.close()
    saveModel(o,accepted,iteration)
    saveCheckpoint(o,beam,accepted,iteration,fingerprint,rollback)
    emitPlot(o)
    echo "iteration=",iteration," rules=",accepted.rules.len,
      " train=",frozenTrain," search=",frozenSearch,
      " acceptance=",chosenRho," anchor=",chosenAnchor,
      " heldout=",auditScore,
      " promoted=",promoted," readoutRefreshed=",readoutRefreshed,
      " regrew=",regrew,
      " stagnation=",rollback.noPromotionStreak,
      " rollbackAttempted=",attemptedRegrow,
      " heldoutLogScore=",(if heldout > 0: logScore(auditScore) else: NaN)

when isMainModule:
  try: main()
  except CatchableError as e:
    quit("ERROR: " & e.msg, 1)
