## Pareto beam addition and transactional repair of arbitrary rule positions.
import std/[algorithm, sequtils, tables, sets, times, math]
import lens_core, cms_miner, nsga2_pareto, batch_presence, spearman_rank_ridge, lens_dependency

type
  SearchConfig* = object
    mining*: MiningConfig
    balanceCases*: bool
    rankScales*, readoutRidgeProbes*: int
    beam*, lookahead*, maxCandidates*, screenCandidates*, screenExplore*, repairPasses*: int
    repairCandidates*, repairBeam*: int
    repairFullSweep*: bool
    regrowDepth*, regrowPositions*, regrowAblate*, regrowBeam*, rhsProbe*: int
    regrowRoots*, regrowTargets*, regrowSamples*, regrowMines*: int
    ridge*, phiThreshold*, epsilon*, regrowMaxLoss*, stabilityPenalty*: float64
  Candidate = object
    lhs: seq[int]
    sourceIndex: int
    ev: Evaluation
    novelty: float64

proc defaultSearchConfig*(): SearchConfig =
  SearchConfig(mining: defaultMiningConfig(), balanceCases: true,
    rankScales: 0, readoutRidgeProbes: 3, beam: 4, lookahead: 2,
    screenCandidates: 256, screenExplore: 48,
    maxCandidates: 0, repairPasses: 1, repairCandidates: 16,
    repairBeam: 2, repairFullSweep: false, ridge: 0.01,
    phiThreshold: 0.5, epsilon: 1e-10,
    regrowDepth: 2, regrowPositions: 0, regrowAblate: 4,
    regrowBeam: 2, rhsProbe: 4,
    regrowRoots: 4, regrowTargets: 2, regrowSamples: 64,
    regrowMines: 8, regrowMaxLoss: 0.04, stabilityPenalty: 0.35)

proc patterns(chunks: seq[seq[int]], cfg: SearchConfig): seq[PatternEntry] =
  var length = 0
  for chunk in chunks: length += chunk.len
  stderr.writeLine("[mine] chunks=", chunks.len, " tokens=", length)
  result = minePatterns(chunks, cfg.mining)
  if cfg.maxCandidates > 0 and result.len > cfg.maxCandidates:
    result.setLen(cfg.maxCandidates)

proc ruleKey(rules: seq[Rule], omitLastRhs = false): string =
  for i, rule in rules:
    result.add($rule.lhs & ":")
    if not omitLastRhs or i != rules.high: result.add($rule.rhs)
    result.add(";")

proc bestIndex*(states: seq[State]): int =
  for i in 1..<states.len:
    if states[i].evaluation.searchMerit > states[result].evaluation.searchMerit or
       (states[i].evaluation.searchMerit == states[result].evaluation.searchMerit and
        states[i].rules.len < states[result].rules.len): result = i

proc selectBeam*(states: seq[State], width: int,
                 variantOffset = 0): seq[State] =
  ## The width is a HARD CAP on actual states, not on LHS families.
  ## Keep a mix of Pareto-selected LHS families and RHS pair-swap branches.
  ## An offset cycles through the alternative RHS variants between iterations,
  ## without requiring every alternative to remain in the next frontier.
  if states.len == 0 or width <= 0: return
  var groups = initOrderedTable[string, seq[int]]()
  var seen = initHashSet[string]()
  for i, state in states:
    let full = ruleKey(state.rules)
    if full in seen: continue
    seen.incl(full)
    groups.mgetOrPut(ruleKey(state.rules, true), @[]).add(i)
  var families: seq[seq[int]]
  var objectives: seq[seq[float64]]
  var hasVariants = false
  for _, ids in groups:
    families.add(ids)
    if ids.len > 1: hasVariants = true
    let s = states[ids[0]]
    objectives.add(@[s.evaluation.searchMerit, s.novelty])
  let order = nsga2Sort(objectives, @[odMaximize, odMinimize]).order
  # The accuracy elite must survive, even when Pareto crowding discards it.
  var bestFamily = 0
  for i in 1..<families.len:
    if objectives[i][0] > objectives[bestFamily][0]: bestFamily = i

  # If swap variants are available, reserve roughly half the state slots for
  # them. Otherwise retain up to `width` distinct Pareto families.
  let familyBudget = min(families.len,
    (if hasVariants: max(1, (width+1) div 2) else: width))
  var chosen = @[bestFamily]
  for f in order:
    if chosen.len >= familyBudget: break
    if f notin chosen: chosen.add(f)
  for f in chosen:
    result.add(states[families[f][0]])

  # Spread remaining slots fairly across chosen families. Rotate *which*
  # alternate swap survives, rather than always selecting the first pair.
  let offset = max(0, variantOffset)
  var round = 0
  while result.len < width:
    var added = false
    for f in chosen:
      let ids = families[f]
      if round < ids.len-1:
        let variant = 1 + (offset + round) mod (ids.len-1)
        result.add(states[ids[variant]])
        added = true
        if result.len >= width: break
    if not added: break
    inc round

  # If no selected family has enough variants, use the remaining slots for
  # further Pareto LHS families instead of leaving the beam artificially small.
  if result.len < width:
    for f in order:
      if result.len >= width: break
      if f notin chosen:
        result.add(states[families[f][0]])


proc residualScreen*(candidateColumns: seq[seq[uint64]], rows: int,
                     parentPred: seq[float64], train: DataSet,
                     keep, explore: int): seq[int] =
  ## Candidate mining/screening is a proposal mechanism, NOT evaluation.
  ## Rank residual per independently-corrupted trajectory. Each trajectory
  ## contributes the same total mass, so long chunks do not dominate.
  ## The screen is based on *training* rows only, not search/accept/held-out.
  ## `keep=0` disables the screen and returns the full index set.
  if train.texts.len != rows or parentPred.len != rows or
      train.cases.len != rows or train.target.len != rows:
    raise newException(ValueError, "residualScreen: training row mismatch")
  let n = candidateColumns.len
  if keep <= 0 or keep + explore >= n:
    for i in 0..<n: result.add(i)
    return
  var residual = newSeq[float64](rows)
  var groups = initOrderedTable[int,seq[int]]()
  for id,cid in train.cases:
    groups.mgetOrPut(cid, @[]).add(id)
  let groupWeight = 1.0 / float64(max(1,groups.len))
  for _,ids in groups:
    let m = ids.len
    if m < 2: continue
    var values = newSeq[float64](m)
    var expected = newSeq[float64](m)
    for j,id in ids:
      values[j] = parentPred[id]
      expected[j] = train.target[id]
    let predictedRanks = averageRanks(values)
    let targetRanks = averageRanks(expected)
    let normalizer = groupWeight / float64(m*max(1,m-1))
    for j,id in ids:
      residual[id] = (targetRanks[j] - predictedRanks[j]) * normalizer
  type Entry = tuple[id: int, merit: float64]
  var ranked: seq[Entry]
  for ci,bits in candidateColumns:
    var count = 0
    var covariance = 0.0
    var coveredCases = initHashSet[int]()
    for i in 0..<rows:
      if (bits[i shr 6] and (1'u64 shl (i and 63))) != 0:
        inc count
        covariance += residual[i]
        coveredCases.incl(train.cases[i])
    # Favor residual reduction that appears in multiple noise paths, not
    # just a rare one-row accident. Exploration slots preserve rare probes.
    let variability = sqrt(float64(count) * float64(rows-count))
    let coverage = sqrt(float64(coveredCases.len) / float64(max(1,groups.len)))
    let merit = (if variability > 0:
                   abs(covariance)*float64(rows)/variability * coverage
                 else: 0.0)
    ranked.add((ci,merit))
  ranked.sort(proc(a,b: Entry): int =
    if a.merit > b.merit: -1
    elif a.merit < b.merit: 1
    else: cmp(a.id,b.id))
  let selected = min(keep,n)
  var included = newSeq[bool](n)
  for j in 0..<selected:
    let id = ranked[j].id
    result.add(id)
    included[id] = true
  # Deterministic exploration budget: do not make the learner incapable of
  # discovering a pattern whose immediate readout gradient is zero but which
  # may unlock a valuable downstream RHS rewrite.
  let extras = min(explore,n-selected)
  if extras > 0:
    var picked = 0
    for j in 0..<n:
      let id = (j*n div extras + (n div 7)) mod n
      if not included[id]:
        result.add(id)
        included[id] = true
        inc picked
        if picked >= extras: break
    if picked < extras:
      for id in 0..<n:
        if not included[id]:
          result.add(id)
          inc picked
          if picked >= extras: break

type
  MiningCacheEntry = object
    candidates: seq[PatternEntry]
    matcher: BatchMatcher

proc expandCached(parent: State, clean: seq[seq[int]], train, selection: DataSet,
                  cfg: SearchConfig, cache: var Table[string,MiningCacheEntry]): seq[State] =
  let cleanText = transform(parent.rules, clean)
  # Exact content key: two different rule chains can still yield the same
  # transformed mining corpus. No hash collisions or approximate reuse.
  let contentKey = $cleanText
  var entry: MiningCacheEntry
  if cache.hasKey(contentKey):
    entry = cache[contentKey]
    stderr.writeLine("[mine cache] hit candidates=",entry.candidates.len)
  else:
    entry.candidates = patterns(cleanText,cfg)
    var patternValues: seq[seq[int]]
    for candidate in entry.candidates: patternValues.add(candidate.value)
    entry.matcher = compileBatch(patternValues)
    # Bounded per-lookahead cache; no cross-iteration stale results.
    if cache.len < 8:
      cache[contentKey] = entry
  let candidates = entry.candidates
  let tr = traceRules(parent.rules, train)
  let va = traceRules(parent.rules, selection)
  let oldMeans = allRulePhi(tr.columns)
  let scoreStarted = epochTime()
  stderr.writeLine("[presence batch] candidates=", candidates.len, " rows=", train.texts.len+selection.texts.len)
  # The screening stage uses TRAIN ONLY. In the old implementation we
  # needlessly matched every mined pattern on SEARCH before this shortlist.
  let trainPresence = batchPresence(entry.matcher, tr.texts)
  var oldPacked: seq[seq[uint64]]
  var oldCounts: seq[int]
  for column in tr.columns:
    oldPacked.add(packColumn(column))
    oldCounts.add(bitCount(oldPacked[^1]))
  let rankCache = prepareRankAppend(tr.columns, train, cfg.ridge,
                                      cfg.balanceCases, cfg.rankScales)
  let trainRanks = prepareSpearman(train)
  let selectRanks = prepareSpearman(selection)
  var lastProgress = epochTime()
  var scored: seq[Candidate]
  var objectives: seq[seq[float64]]
  let preselected = residualScreen(trainPresence.columns, trainPresence.rows,
    predict(tr.columns, parent.evaluation.weights, train.texts.len),
    train, cfg.screenCandidates, cfg.screenExplore)
  if preselected.len == 0: return
  # Compile a *smaller*, exact matcher for selected proposals only.
  # Unlike supervised screening, this still computes the original wildcard
  # presence semantics, with no sampling on the SEARCH dataset.
  var shortlisted: seq[seq[int]]
  for ci in preselected: shortlisted.add(candidates[ci].value)
  let selectMatcher = (if preselected.len == candidates.len:
                         entry.matcher else: compileBatch(shortlisted))
  let selectPresence = batchPresence(selectMatcher, va.texts)
  let presenceSeconds = epochTime()-scoreStarted
  stderr.writeLine("[expand] rules=", parent.rules.len, " candidates=", candidates.len,
    " residualScreen=",preselected.len,
    " exactSearchMatches=",shortlisted.len)
  for localId,ci in preselected:
    let candidate = candidates[ci]
    let count = bitCount(trainPresence.columns[ci])
    if count == 0: continue
    let column = toColumn(trainPresence.columns[ci], trainPresence.rows)
    # Identical/constant feature columns may still yield useful future rewrites; keep them.
    let validation = toColumn(selectPresence.columns[localId], selectPresence.rows)
    var ev: Evaluation
    try:
      ev.weights = fitAppend(rankCache, column)
    except ValueError:
      # Rare ill-conditioned Schur complement: use the original full solve.
      ev.weights = fitPrimal(tr.columns & @[column], train, cfg.ridge,
                             cfg.balanceCases, cfg.rankScales)
    var clipped = false
    for weight in ev.weights:
      if abs(weight) >= WeightClip: clipped = true
    if clipped: ev.weights = fitPrimal(tr.columns & @[column], train, cfg.ridge,
                             cfg.balanceCases, cfg.rankScales)
    # Training rho does not rank candidates; compute it only for survivors.
    let stable = caseSpearmanStability(selectRanks,
      predictAppend(va.columns, validation, ev.weights), cfg.stabilityPenalty)
    ev.score = stable.mean
    ev.searchMerit = stable.merit
    let p = tr.columns.len
    ev.means = newSeq[float64](p+1)
    var phi = 0.0
    for j, old in tr.columns:
      let correlation = packedPhi(trainPresence.columns[ci], oldPacked[j], count, oldCounts[j], trainPresence.rows)
      phi += correlation
      ev.means[j] = (oldMeans[j]*float64(p-1) + correlation)/float64(p)
    if p > 0: phi /= float64(p)
    ev.means[p] = phi
    scored.add(Candidate(lhs: candidate.value, sourceIndex: ci, ev: ev, novelty: phi))
    objectives.add(@[ev.searchMerit, phi])
    if epochTime()-lastProgress >= 2.0:
      lastProgress = epochTime()
      stderr.writeLine("[score] ", ci+1, "/", candidates.len)
  stderr.writeLine("[score time] batchedPresence=", presenceSeconds,
    "s regressionAndRank=", epochTime()-scoreStarted-presenceSeconds, "s")
  if scored.len == 0: return
  var selected = nsga2Select(objectives, cfg.beam, @[odMaximize, odMinimize])
  var best = 0
  for i in 1..<scored.len:
    if scored[i].ev.searchMerit > scored[best].ev.searchMerit: best = i
  if best notin selected:
    if selected.len == cfg.beam: selected.setLen(selected.len-1)
    selected.add(best)
  for index in selected:
    var candidate = scored[index]
    let column = toColumn(trainPresence.columns[candidate.sourceIndex], trainPresence.rows)
    candidate.ev.train = scoreSpearman(trainRanks, predictAppend(tr.columns, column, candidate.ev.weights))
    for rhs in replacementVariants(candidate.lhs, parent.nextToken):
      var child = State(rules: copyRules(parent.rules),
        nextToken: parent.nextToken + wildcardCount(candidate.lhs) + 1,
        evaluation: candidate.ev, novelty: candidate.novelty)
      child.rules.add(Rule(lhs: candidate.lhs, rhs: rhs))
      result.add(child)

proc tuneReadout*(state: State, train, selection: DataSet,
                  cfg: SearchConfig): State =
  ## Only a handful of selected children reach this code. Try modest ridge
  ## strengths, score on SEARCH (never acceptance or heldout), and let the
  ## independent acceptance gate determine whether anything is promoted.
  result = state
  if cfg.readoutRidgeProbes <= 1 or state.rules.len == 0: return
  let tr = traceRules(state.rules,train)
  let va = traceRules(state.rules,selection)
  let validation = prepareSpearman(selection)
  let training = prepareSpearman(train)
  let scales = (if cfg.readoutRidgeProbes == 2: @[1.0,4.0]
                else: @[0.25,1.0,4.0])
  var bestMerit = -Inf
  for scale in scales:
    let weights = fitPrimal(tr.columns, train, cfg.ridge*scale,
                            cfg.balanceCases, cfg.rankScales)
    let searchPrediction = predict(va.columns, weights, selection.texts.len)
    let measure = caseSpearmanStability(validation,searchPrediction,cfg.stabilityPenalty)
    if measure.merit > bestMerit:
      bestMerit = measure.merit
      result.evaluation.weights = weights
      result.evaluation.score = measure.mean
      result.evaluation.searchMerit = measure.merit
      result.evaluation.train = scoreSpearman(training,
        predict(tr.columns,weights,train.texts.len))
  # Rule firing, phi correlations and novelty do not depend on readout weights.

proc expand*(parent: State, clean: seq[seq[int]], train, selection: DataSet,
             cfg: SearchConfig): seq[State] =
  var cache = initTable[string,MiningCacheEntry]()
  result = expandCached(parent,clean,train,selection,cfg,cache)

proc addBeam*(parents: seq[State], clean: seq[seq[int]], train, selection: DataSet,
              cfg: SearchConfig, iteration = 0): seq[State] =
  # Older checkpoints may contain much wider beams (previously `beam`
  # counted only LHS families). Clamp before ANY expensive expansion.
  var frontier = selectBeam(parents, cfg.beam, iteration)
  var archive = frontier
  var miningCache = initTable[string,MiningCacheEntry]()
  for depth in 0..<cfg.lookahead:
    var children: seq[State]
    for parent in frontier:
      children.add(expandCached(parent, clean, train, selection, cfg, miningCache))
    if children.len == 0: break
    # Hold provisional RHS variants until the *next* expansion evaluates
    # the representations they produce; prune to the real beam at final depth.
    let interimWidth = (if depth+1 < cfg.lookahead:
                          max(cfg.beam, cfg.rhsProbe) else: cfg.beam)
    frontier = selectBeam(children, interimWidth, iteration+depth)
    archive.add(frontier)
    stderr.writeLine("[beam] depth=", depth+1, " states=", frontier.len,
      " rho=", frontier[bestIndex(frontier)].evaluation.score,
      " searchMerit=", frontier[bestIndex(frontier)].evaluation.searchMerit)
  result = selectBeam(archive, cfg.beam, iteration+cfg.lookahead)

proc prefixRules(rules: seq[Rule], stop: int): seq[Rule] =
  for i in 0..<stop: result.add(rules[i])

proc compatible(lhs, rhs: seq[int]): bool =
  let k = wildcardCount(lhs)
  for t in rhs:
    if t < 0 and -t > k: return false
  true

type
  RepairContext = object
    ## The same rule prefix always produces the same CLEAN mining input.
    ## Cache only inside one repair transaction / one corpus window.
    miningCache: Table[string, seq[PatternEntry]]
    prefixCache: Table[string, tuple[trainTrace, selectionTrace: Trace]]
    miningHits, miningMisses, prefixHits, evaluations: int
  RegrowPath = object
    state: State
    visited: seq[int]
    repaired: int  # non-root rules actually rewritten along this path

proc repairCandidates*(mined: seq[PatternEntry], oldLhs, rhs: seq[int],
                       limit: int): seq[PatternEntry] =
  ## A deterministic shortlist: high-frequency candidates plus candidates
  ## spread across the remaining frequency ranks, retaining diversity.
  ## limit=0 reproduces exhaustive candidate selection.
  var eligible: seq[PatternEntry]
  for entry in mined:
    if entry.value != oldLhs and compatible(entry.value, rhs):
      eligible.add(entry)
  if limit <= 0 or eligible.len <= limit:
    return eligible
  let head = (limit + 1) div 2
  for i in 0..<head:
    result.add(eligible[i])
  let remaining = limit - head
  for i in 0..<remaining:
    let offset = ((2*i+1)*(eligible.len-head)) div (2*remaining)
    result.add(eligible[head+offset])

proc proposeRepair(parent: State, index: int, clean: seq[seq[int]],
                   train, selection: DataSet, cfg: SearchConfig,
                   ctx: var RepairContext): seq[State] =
  ## Mining results depend ONLY on earlier rules. No need to re-enumerate
  ## 10+ million occurrences for each parent with the same prefix.
  let prefix = prefixRules(parent.rules, index)
  let key = ruleKey(prefix)
  # Hard cap unique mining prefixes per transaction, while allowing cached
  # results to be reused freely. 0 explicitly disables the limit.
  if cfg.regrowMines > 0 and ctx.miningMisses >= cfg.regrowMines and
      not ctx.miningCache.hasKey(key):
    return
  var mined: seq[PatternEntry]
  if ctx.miningCache.hasKey(key):
    mined = ctx.miningCache[key]
    inc ctx.miningHits
  else:
    let stageText = transform(prefix, clean)
    mined = patterns(stageText, cfg)
    ctx.miningCache[key] = mined
    inc ctx.miningMisses
  let candidates = repairCandidates(mined, parent.rules[index].lhs,
                                    parent.rules[index].rhs, cfg.repairCandidates)
  if candidates.len == 0: return

  ## Prefix firing/replacement on the noisy train and selection rows is
  ## identical for every candidate here. Compute it just once.
  var prefixTrain, prefixSelect: Trace
  if ctx.prefixCache.hasKey(key):
    let cached = ctx.prefixCache[key]
    prefixTrain = cached.trainTrace
    prefixSelect = cached.selectionTrace
    inc ctx.prefixHits
  else:
    prefixTrain = traceRules(prefix, train)
    prefixSelect = traceRules(prefix, selection)
    ctx.prefixCache[key] = (trainTrace: prefixTrain, selectionTrace: prefixSelect)
  for candidate in candidates:
    var child = State(rules: copyRules(parent.rules), nextToken: parent.nextToken)
    child.rules[index].lhs = candidate.value
    let suffixRules = child.rules[index..<child.rules.len]
    # traceRules assigns row entries when replacing. Give it independent
    # row containers to protect the precomputed prefix traces.
    var trainRows = newSeq[seq[int]](prefixTrain.texts.len)
    var selectRows = newSeq[seq[int]](prefixSelect.texts.len)
    for i, row in prefixTrain.texts: trainRows[i] = row
    for i, row in prefixSelect.texts: selectRows[i] = row
    let suffixTrain = traceRules(suffixRules, DataSet(texts: trainRows))
    let suffixSelect = traceRules(suffixRules, DataSet(texts: selectRows))
    child.evaluation = scoreColumns(prefixTrain.columns & suffixTrain.columns,
                                    prefixSelect.columns & suffixSelect.columns,
                                    train, selection, cfg.ridge,
                                    stabilityPenalty = cfg.stabilityPenalty,
                                    balanceCases = cfg.balanceCases,
                                    rankScales = cfg.rankScales)
    child.novelty = child.evaluation.means[index]
    result.add(child)
    inc ctx.evaluations
  result = selectBeam(result, min(cfg.beam, cfg.repairBeam))

proc repair*(original: State, clean: seq[seq[int]], train, selection: DataSet,
             cfg: SearchConfig): State =
  result = original
  if original.rules.len < 2 or cfg.repairPasses == 0: return
  let started = epochTime()
  var ctx = RepairContext(
    miningCache: initTable[string, seq[PatternEntry]](),
    prefixCache: initTable[string, tuple[trainTrace, selectionTrace: Trace]]())
  var order = toSeq(0..<original.rules.len)
  order.sort(proc(a,b: int): int = cmp(original.evaluation.means[b], original.evaluation.means[a]))
  var seeds: seq[State]
  for index in order:
    if original.evaluation.means[index] > cfg.phiThreshold:
      stderr.writeLine("[repair] rule=", index, " meanAbsPhi=", original.evaluation.means[index])
      seeds.add(proposeRepair(original, index, clean, train, selection, cfg, ctx))
  if seeds.len == 0: return
  let width = min(cfg.beam, cfg.repairBeam)
  var frontier = selectBeam(seeds, width)
  if frontier.len == 0: return
  var bestTrial = frontier[bestIndex(frontier)]
  let baselineTrace = traceRules(original.rules, train)
  for pass in 0..<cfg.repairPasses:
    for index in 0..<original.rules.len:
      var trials = frontier
      for state in frontier:
        let current = traceRules(state.rules, train)
        let affected = current.columns[index] != baselineTrace.columns[index]
        # The original first pass unconditionally explored every position.
        # This is now optional: by default examine changed and highly
        # correlated positions only, without sacrificing final full replay.
        if affected or state.evaluation.means[index] > cfg.phiThreshold or
           (cfg.repairFullSweep and pass == 0):
          trials.add(proposeRepair(state, index, clean, train, selection, cfg, ctx))
      frontier = selectBeam(trials, width)
      let best = frontier[bestIndex(frontier)]
      if best.evaluation.score > bestTrial.evaluation.score: bestTrial = best
  if bestTrial.evaluation.score > original.evaluation.score + cfg.epsilon:
    stderr.writeLine("[repair commit] ", original.evaluation.score, " -> ", bestTrial.evaluation.score)
    result = bestTrial
  else:
    stderr.writeLine("[repair reject] unchanged score=", original.evaluation.score)
  stderr.writeLine("[repair stats] mined=", ctx.miningMisses, " cacheHits=", ctx.miningHits,
    " prefixTraceHits=", ctx.prefixHits, " fullEvaluations=", ctx.evaluations,
    " elapsed=", epochTime()-started, "s")


proc chooseRegrowFrontier(paths: seq[RegrowPath], width: int): seq[RegrowPath] =
  ## Don't kill every temporarily-worse intermediate repair. Keep one route
  ## that actually edited an affected rule until the next wave can assess it.
  if paths.len == 0 or width <= 0: return
  var sorted = paths
  sorted.sort(proc(a,b: RegrowPath): int =
    let c = cmp(b.state.evaluation.score,a.state.evaluation.score)
    if c != 0: c else: cmp(b.repaired,a.repaired))
  var seen = initHashSet[string]()
  for path in sorted:
    let k = ruleKey(path.state.rules) & "|" & $path.visited
    if k notin seen:
      seen.incl(k)
      result.add(path)
      break
  if result.len < width and result[0].repaired == 0:
    for path in sorted:
      if path.repaired == 0: continue
      let k = ruleKey(path.state.rules) & "|" & $path.visited
      if k notin seen:
        seen.incl(k)
        result.add(path)
        break
  for path in sorted:
    if result.len >= width: break
    let k = ruleKey(path.state.rules) & "|" & $path.visited
    if k notin seen:
      seen.incl(k)
      result.add(path)

proc rollbackRegrow*(original: State, clean: seq[seq[int]],
                     train, selection: DataSet, cfg: SearchConfig,
                     iteration = 0, allowHardRoots = false): State =
  ## Change a root and selectively revisit the empirically affected rules,
  ## including NONADJACENT targets. Multiple downstream targets are branched,
  ## and the beam retains a real repair path if greedy sorting would drop it.
  result = original
  if cfg.regrowPositions <= 0 or cfg.regrowDepth <= 0 or
     original.rules.len < 2: return
  let started = epochTime()
  let n = original.rules.len
  var order = toSeq(0..<n-1)
  order.sort(proc(a,b: int): int =
    let diff = cmp(original.evaluation.means[b], original.evaluation.means[a])
    if diff != 0: diff else: cmp(a,b))
  var targets: seq[int]
  let cap = min(n-1, cfg.regrowAblate)
  if cap <= 0: return
  for idx in order:
    if targets.len >= (cap+1) div 2: break
    targets.add(idx)
  for j in 0..<n-1:
    if targets.len >= cap: break
    let idx = (iteration + j*(n-1) div cap) mod (n-1)
    if idx notin targets: targets.add(idx)
  for idx in 0..<n-1:
    if targets.len >= cap: break
    if idx notin targets: targets.add(idx)

  # Ablation measures the marginal search score of removing a root, NOT the
  # intrinsic causal importance of that rule. All downstream rules rerun.
  type Ablation = tuple[index: int, score: float64]
  var trials: seq[Ablation]
  for idx in targets:
    var minus = State(rules: copyRules(original.rules),
                      nextToken: original.nextToken)
    minus.rules.delete(idx)
    minus.evaluate(train, selection, cfg.ridge, cfg.stabilityPenalty,
                   cfg.balanceCases, cfg.rankScales)
    trials.add((index: idx, score: minus.evaluation.score))
  trials.sort(proc(a,b: Ablation): int =
    let diff = cmp(b.score,a.score)
    if diff != 0: diff else: cmp(a.index,b.index))
  # Ablation loss is a counterfactual screen, not a causal proof.
  # Do NOT treat a large |phi| as sufficient reason to disturb an important root.
  # Persistent plateaus allow an explicit fallback to stronger roots.
  var eligible: seq[Ablation]
  for trial in trials:
    let loss = original.evaluation.score - trial.score
    if loss <= cfg.regrowMaxLoss or allowHardRoots:
      eligible.add(trial)
  if eligible.len == 0:
    stderr.writeLine("[regrow skip] all tested roots lost more than ",
      cfg.regrowMaxLoss," Spearman on ablation; wait for sustained plateau")
    return
  let positions = min(cfg.regrowPositions,eligible.len)
  var ctx = RepairContext(
    miningCache: initTable[string,seq[PatternEntry]](),
    prefixCache: initTable[string,tuple[trainTrace,selectionTrace: Trace]]())
  var impactCache = initTable[string,seq[RuleImpact]]()
  var impactHits = 0
  for j in 0..<positions:
    let idx = eligible[j].index
    stderr.writeLine("[regrow] root=",idx," ablatedScore=",eligible[j].score,
      " ablationLoss=",original.evaluation.score-eligible[j].score,
      " depth=",cfg.regrowDepth," rootBudget=",cfg.regrowRoots,
      " targetsPerWave=",cfg.regrowTargets)
    # Keep a broader root frontier until downstream interactions are known.
    # Otherwise a root with a temporary score loss would be eliminated before
    # any affected downstream rule has a chance to adapt.
    var rootCfg = cfg
    rootCfg.beam = max(cfg.beam, cfg.regrowRoots)
    rootCfg.repairBeam = max(cfg.repairBeam, cfg.regrowRoots)
    let rootVariants = proposeRepair(original,idx,clean,train,selection,
                                     rootCfg,ctx)
    var localBest = original
    for root in rootVariants:
      if root.rules[idx].lhs == original.rules[idx].lhs: continue
      var frontier = @[RegrowPath(state: root)]
      for wave in 0..<cfg.regrowDepth:
        var proposals: seq[RegrowPath]
        var foundAffected = false
        for path in frontier:
          # Memoize repeated empirical replays in this single transaction.
          let key = $idx & ":" & ruleKey(path.state.rules)
          var impacted: seq[RuleImpact]
          if impactCache.hasKey(key):
            impacted = impactCache[key]
            inc impactHits
          else:
            impacted = affectedRuleOrder(original.rules,path.state.rules,
                        train.texts,idx,cfg.regrowSamples,train.cases)
            impactCache[key] = impacted
          var inspected = 0
          for impact in impacted:
            if inspected >= cfg.regrowTargets: break
            if impact.index in path.visited: continue
            inc inspected
            foundAffected = true
            stderr.writeLine("[regrow affected] root=",idx," wave=",wave+1,
              " rule=",impact.index," drift=",impact.priority,
              " nonadjacent=",impact.index-idx>1)
            let touched = path.visited & @[impact.index]
            # Skipping a disturbed node is permitted; unlike the earlier
            # version, it does not consume ALL options for this wave.
            proposals.add(RegrowPath(state:path.state,visited:touched,
                                      repaired:path.repaired))
            for proposed in proposeRepair(path.state,impact.index,clean,
                                           train,selection,cfg,ctx):
              if proposed.rules[impact.index].lhs != path.state.rules[impact.index].lhs:
                proposals.add(RegrowPath(state:proposed,visited:touched,
                                          repaired:path.repaired+1))
          if inspected == 0: proposals.add(path)
        if not foundAffected or proposals.len == 0: break
        frontier = chooseRegrowFrontier(proposals,cfg.regrowBeam)
      for path in frontier:
        if path.state.evaluation.score > localBest.evaluation.score + cfg.epsilon:
          localBest = path.state
      if root.evaluation.score > localBest.evaluation.score + cfg.epsilon:
        localBest = root
    if localBest.evaluation.score > result.evaluation.score + cfg.epsilon:
      result = localBest
  stderr.writeLine("[regrow done] original=",original.evaluation.score,
    " candidate=",result.evaluation.score,
    " mining=",ctx.miningMisses," cacheHits=",ctx.miningHits,
    " fullEvaluations=",ctx.evaluations,
    " miningLimit=",cfg.regrowMines,
    " impactCacheHits=",impactHits,
    " seconds=",epochTime()-started)
