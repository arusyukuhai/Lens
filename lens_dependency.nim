## Dependency-sensitive regrowth priority measurement.
## Compare the same input rows under two FULL rule chains, preserving rule indices.
## Local edit signatures avoid treating every downstream inherited difference as
## a reason to rebuild that rule. Both firing differences and changed edits count.
import std/[algorithm]
import lens_core, replace

type
  RuleImpact* = object
    index*: int
    firingDrift*: float64
    editDrift*: float64
    priority*: float64
  EditSignature = object
    removed, inserted: int
    fingerprint: uint64

proc editSignature(before, after: seq[int]): EditSignature =
  ## Approximate local rewrite effect, ignoring absolute match offsets.
  ## Changed captures and rewrite lengths remain visible; pure upstream shifts
  ## are not misclassified as changes to this rule.
  var left = 0
  while left < min(before.len, after.len) and before[left] == after[left]:
    inc left
  var tail = 0
  while tail < min(before.len-left, after.len-left) and
      before[before.len-tail-1] == after[after.len-tail-1]:
    inc tail
  result.removed = before.len-left-tail
  result.inserted = after.len-left-tail
  var h = 1469598103934665603'u64
  for i in left..<after.len-tail:
    h = (h xor cast[uint64](after[i])) * 1099511628211'u64
  result.fingerprint = h

proc chooseImpactSamples*(rowCount, sampleLimit: int,
                          cases: seq[int] = @[]): seq[int] =
  ## Balance across independent corruption paths/chunks first, then across
  ## mutation stages WITHIN each path. Long chunks cannot monopolize samples.
  if rowCount <= 0 or sampleLimit <= 0: return
  let budget = min(rowCount, sampleLimit)
  type Span = tuple[first, past: int]
  var spans: seq[Span]
  if cases.len == rowCount:
    var first = 0
    for i in 1..rowCount:
      if i == rowCount or cases[i] != cases[i-1]:
        spans.add((first, i))
        first = i
  else:
    spans.add((0, rowCount))
  let groupCount = min(budget, spans.len)
  var chosen = newSeq[Span](groupCount)
  for g in 0..<groupCount:
    chosen[g] = spans[g * spans.len div groupCount]
  var quotas = newSeq[int](groupCount)
  var total = 0
  # Round robin up to each group capacity, reassigning unused slots to others.
  while total < budget:
    var grew = false
    for g in 0..<groupCount:
      if total >= budget: break
      if quotas[g] < chosen[g].past - chosen[g].first:
        inc quotas[g]
        inc total
        grew = true
    if not grew: break
  for g, span in chosen:
    for q in 0..<quotas[g]:
      let n = span.past-span.first
      # Midpoints of equally-sized stage intervals; no repeated indices.
      result.add(span.first + ((2*q+1)*n) div (2*quotas[g]))

proc affectedRuleOrder*(baseline, revised: seq[Rule],
                        rows: seq[seq[int]], changedIndex: int,
                        sampleLimit = 64,
                        cases: seq[int] = @[]): seq[RuleImpact] =
  ## Empirical dependency detector, not a formal causal proof.
  ## Measure the rule-local firing and rewrite difference on stratified rows.
  ## Rule arrays must be aligned and the edited root stays at changedIndex.
  if baseline.len != revised.len or changedIndex < 0 or
      changedIndex >= baseline.len:
    raise newException(ValueError, "affectedRuleOrder needs aligned rule chains")
  let samples = chooseImpactSamples(rows.len, sampleLimit, cases)
  if samples.len == 0: return
  let n = baseline.len
  if changedIndex+1 >= n: return
  var drift = newSeq[RuleImpact](n)
  for i in changedIndex+1..<n: drift[i].index = i
  let count = samples.len
  var baseRows = newSeq[seq[int]](count)
  var altRows = newSeq[seq[int]](count)
  for j, idx in samples:
    baseRows[j] = rows[idx]
    altRows[j] = rows[idx]
  for i in 0..<n:
    let basePattern = compileSeqPattern(baseline[i].lhs, 1)
    # Shared rule patterns only need one compilation.
    let equalPattern = baseline[i].lhs == revised[i].lhs
    let altPattern = (if equalPattern: basePattern else:
                        compileSeqPattern(revised[i].lhs, 1))
    for j in 0..<count:
      let b = baseRows[j]
      let a = altRows[j]
      let onB = containsSeqCompiled(b, basePattern)
      let onA = containsSeqCompiled(a, altPattern)
      if i > changedIndex and onB != onA:
        drift[i].firingDrift += 1.0
      if onB:
        let after = replaceSeqStrict(b, basePattern, baseline[i].rhs)
        if i > changedIndex and onA:
          let otherAfter = replaceSeqStrict(a, altPattern, revised[i].rhs)
          # Do not include absolute match position: an unrelated earlier
          # insertion can shift an unchanged downstream rule's match.
          if editSignature(b, after) != editSignature(a, otherAfter):
            drift[i].editDrift += 1.0
          altRows[j] = otherAfter
        baseRows[j] = after
      if onA and not (i > changedIndex and onB):
        altRows[j] = replaceSeqStrict(a, altPattern, revised[i].rhs)
  for i in changedIndex+1..<n:
    var impact = drift[i]
    impact.firingDrift /= float64(count)
    impact.editDrift /= float64(count)
    # Prefer altered firing, but preserve equal-presence capture changes.
    impact.priority = 2.0*impact.firingDrift + impact.editDrift
    if impact.priority > 0.0: result.add(impact)
  result.sort(proc(a,b: RuleImpact): int =
    let c = cmp(b.priority,a.priority)
    if c != 0: c else: cmp(a.index,b.index))
