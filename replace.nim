## Standalone extraction of the heavily optimized sequence replacement engine
## from at_jev(20261007-072614).nim.
##
## Pattern syntax:
##   any negative value in `pattern` is a wildcard (up to 16 wildcards).
##
## Replacement syntax:
##    >= 0       literal token
##    -1..-15    capture 0..14
##   -16..-31    sorted capture 0..15
##   -32..-47    reversed capture 0..15
##   -48..-63    capture +1 mod 512
##   -64..-79    capture -1 mod 512
##   -80..-95    capture *2 mod 512
##   -96..-111   capture floor-div 2
##
## The implementation preserves the original hot-path optimizations:
## - 1/2-token direct literal scan
## - short literal scan for <= 8 tokens
## - KMP for longer literals
## - lazy wildcard-delimiter search (no occurrence arrays)
## - fixed capture/cursor storage
## - compiled replacement plan
## - bulk copy via copyMem
## - reusable scratch buffers

import std/algorithm

type
  SeqReplaceError* = object of CatchableError

  LiteralPart = object
    data: seq[int]
    prefix: seq[int]

  SeqPattern* = object
    parts: seq[LiteralPart]
    wildcardCount*: int
    hasLeadingLiteral: bool
    allWildcard: bool
    emptyPattern: bool

  Capture = object
    start: int
    len: int

  LiteralCursor = object
    lastMatch: int
    nextSearch: int
    exhausted: bool

  ReplacementOp = object
    kind: uint8       # 0 literal, 1 capture, 2 sort, 3 reverse, 4..7 arithmetic
    arg: uint8        # capture index 0..15
    value: int        # literal token when kind == 0

  ReplacementPlan = object
    ops: array[64, ReplacementOp]
    len: uint8
    allLiteral: bool

  ReplaceScratch = object
    cursors: array[17, LiteralCursor]
    captures: array[16, Capture]
    sortBuf: seq[int]

const
  EMBEDDING_CODE_COUNT = 512
  MAX_REPLACE_OUTPUT* = 1_048_576

proc buildPrefix(p: openArray[int]): seq[int] =
  result = newSeq[int](p.len)
  var j = 0
  for i in 1 ..< p.len:
    while j > 0 and p[i] != p[j]:
      j = result[j - 1]
    if p[i] == p[j]:
      inc j
    result[i] = j

proc makeLiteral(data: openArray[int]): LiteralPart =
  result.data = @data
  result.prefix = buildPrefix(data)

proc compileSeqPattern*(pattern: openArray[int]): SeqPattern =
  ## Compile a sequence pattern. Every negative token is a wildcard.
  var current: seq[int] = @[]
  for x in pattern:
    if x <= -1:
      result.parts.add(makeLiteral(current))
      current.setLen(0)
      inc result.wildcardCount
    else:
      current.add(x)
  result.parts.add(makeLiteral(current))

  result.hasLeadingLiteral =
    result.parts.len > 0 and result.parts[0].data.len > 0

  result.allWildcard = result.wildcardCount > 0
  if result.allWildcard:
    for part in result.parts:
      if part.data.len != 0:
        result.allWildcard = false
        break

  result.emptyPattern =
    result.wildcardCount == 0 and
    result.parts.len == 1 and
    result.parts[0].data.len == 0

proc findLiteralFrom(
  text: openArray[int],
  p: LiteralPart,
  start: int
): int {.inline.} =
  let plen = p.data.len
  if plen == 0 or start < 0 or start >= text.len or
      plen > text.len - start:
    return -1

  if plen == 1:
    let first = p.data[0]
    let textPtr = cast[ptr UncheckedArray[int]](unsafeAddr text[0])
    var i = start
    while i < text.len:
      if textPtr[i] == first:
        return i
      inc i
    return -1

  if plen == 2:
    let first = p.data[0]
    let second = p.data[1]
    let textPtr = cast[ptr UncheckedArray[int]](unsafeAddr text[0])
    let last = text.len - 2
    var i = start
    while i <= last:
      if textPtr[i] == first and textPtr[i + 1] == second:
        return i
      inc i
    return -1

  if plen <= 8:
    let first = p.data[0]
    let last = text.len - plen
    var i = start
    while i <= last:
      if text[i] == first:
        var j = 1
        while j < plen and text[i + j] == p.data[j]:
          inc j
        if j == plen:
          return i
      inc i
    return -1

  var j = 0
  var i = start
  while i < text.len:
    while j > 0 and text[i] != p.data[j]:
      j = p.prefix[j - 1]
    if text[i] == p.data[j]:
      inc j
    if j == plen:
      return i - plen + 1
    inc i
  -1

proc nextOccurrenceLazy(
  text: openArray[int],
  p: LiteralPart,
  cursor: var LiteralCursor,
  pos: int
): int {.inline.} =
  if cursor.lastMatch >= pos:
    return cursor.lastMatch
  if cursor.exhausted:
    return -1

  var searchPos = cursor.nextSearch
  while true:
    let found = findLiteralFrom(text, p, searchPos)
    if found < 0:
      cursor.exhausted = true
      cursor.lastMatch = -1
      return -1

    cursor.lastMatch = found
    cursor.nextSearch = found + 1
    if found >= pos:
      return found
    searchPos = cursor.nextSearch

proc matchOccurrences(
  input: openArray[int],
  pattern: SeqPattern,
  cursors: var openArray[LiteralCursor],
  start: int,
  captures: var array[16, Capture]
): tuple[ok: bool, finish: int] {.inline.} =
  if pattern.wildcardCount > captures.len or pattern.parts.len == 0 or
      start < 0 or start >= input.len:
    return (false, start)

  for w in 0 ..< pattern.wildcardCount:
    captures[w] = Capture(start: start, len: 0)

  var pos = start
  let first = pattern.parts[0]
  if first.data.len > 0:
    let found = nextOccurrenceLazy(input, first, cursors[0], pos)
    if found != pos or first.data.len > input.len - pos:
      return (false, start)
    pos += first.data.len

  for w in 0 ..< pattern.wildcardCount:
    let literal = pattern.parts[w + 1]
    if literal.data.len == 0:
      if w == pattern.wildcardCount - 1:
        captures[w] = Capture(start: pos, len: input.len - pos)
        pos = input.len
      else:
        captures[w] = Capture(start: pos, len: 0)
    else:
      let found = nextOccurrenceLazy(input, literal, cursors[w + 1], pos)
      if found < pos or found > input.len or
          literal.data.len > input.len - found:
        return (false, start)
      captures[w] = Capture(start: pos, len: found - pos)
      pos = found + literal.data.len

  (true, pos)

proc isAllWildcard(pat: SeqPattern): bool {.inline.} = pat.allWildcard
proc isEmptyPattern(pat: SeqPattern): bool {.inline.} = pat.emptyPattern

proc appendIntsBulk(
  outp: var seq[int],
  input: openArray[int],
  start, len: int
) {.inline.} =
  if len <= 0:
    return
  if start < 0 or start > input.len or len > input.len - start:
    raise newException(SeqReplaceError,
      "bulk copy out of bounds: start=" & $start & " len=" & $len &
      " input.len=" & $input.len)
  let oldLen = outp.len
  outp.setLen(oldLen + len)
  copyMem(addr outp[oldLen], unsafeAddr input[start], len * sizeof(int))

proc appendIntValuesBulk(
  outp: var seq[int],
  values: openArray[int]
) {.inline.} =
  if values.len == 0:
    return
  let oldLen = outp.len
  outp.setLen(oldLen + values.len)
  copyMem(addr outp[oldLen], unsafeAddr values[0], values.len * sizeof(int))

proc compileReplacementPlan(
  b: openArray[int],
  plan: var ReplacementPlan
) {.inline.} =
  plan.len = uint8(min(64, b.len))
  plan.allLiteral = true

  for i in 0 ..< int(plan.len):
    let t = b[i]
    if t >= 0:
      plan.ops[i] = ReplacementOp(kind: 0'u8, arg: 0'u8, value: t)
    elif t >= -15:
      plan.allLiteral = false
      plan.ops[i] = ReplacementOp(kind: 1'u8, arg: uint8(-t - 1), value: 0)
    elif t >= -31:
      plan.allLiteral = false
      plan.ops[i] = ReplacementOp(kind: 2'u8, arg: uint8(-t - 16), value: 0)
    elif t >= -47:
      plan.allLiteral = false
      plan.ops[i] = ReplacementOp(kind: 3'u8, arg: uint8(-t - 32), value: 0)
    elif t >= -111:
      plan.allLiteral = false
      let offset = -t - 48
      plan.ops[i] = ReplacementOp(
        kind: uint8(4 + offset div 16),
        arg: uint8(offset mod 16),
        value: 0
      )
    else:
      raise newException(
        SeqReplaceError,
        "Invalid replacement opcode: " & $t
      )

proc appendReplacementPlan(
  outp: var seq[int],
  planPtr: ptr ReplacementPlan,
  input: openArray[int],
  captures: openArray[Capture],
  wildcardCount: int,
  scratch: var ReplaceScratch,
  overflow: var bool,
  maxOutput: int = MAX_REPLACE_OUTPUT
) {.inline, gcsafe.} =
  if planPtr[].allLiteral:
    let planLen = int(planPtr[].len)
    if outp.len + planLen > maxOutput:
      overflow = true
      return
    let oldLen = outp.len
    outp.setLen(oldLen + planLen)
    for i in 0 ..< planLen:
      outp[oldLen + i] = planPtr[].ops[i].value
    return

  let planLen = int(planPtr[].len)
  for oi in 0 ..< planLen:
    let op = planPtr[].ops[oi]
    case op.kind
    of 0'u8:
      if outp.len >= maxOutput:
        overflow = true
        return
      outp.add(op.value)

    of 1'u8:
      var n = int(op.arg)
      if wildcardCount == 0 or captures.len == 0:
        continue
      if n >= wildcardCount or n >= captures.len:
        n = 0
      let c = captures[n]
      if c.start < 0 or c.len < 0 or c.start > input.len or
          c.len > input.len - c.start:
        raise newException(SeqReplaceError,
          "invalid capture: start=" & $c.start & " len=" & $c.len &
          " input.len=" & $input.len & " op=" & $op.kind)
      if c.len > 0:
        if c.len > maxOutput - outp.len:
          overflow = true
          return
        appendIntsBulk(outp, input, c.start, c.len)

    of 2'u8:
      var n = int(op.arg)
      if wildcardCount == 0 or captures.len == 0:
        continue
      if n >= wildcardCount or n >= captures.len:
        n = 0
      let c = captures[n]
      if c.start < 0 or c.len < 0 or c.start > input.len or
          c.len > input.len - c.start:
        raise newException(SeqReplaceError,
          "invalid capture: start=" & $c.start & " len=" & $c.len &
          " input.len=" & $input.len & " op=" & $op.kind)
      if c.len > 0:
        if c.len > maxOutput - outp.len:
          overflow = true
          return
        scratch.sortBuf.setLen(c.len)
        copyMem(addr scratch.sortBuf[0], unsafeAddr input[c.start], c.len * sizeof(int))
        sort(scratch.sortBuf)
        appendIntValuesBulk(outp, scratch.sortBuf)

    of 3'u8:
      var n = int(op.arg)
      if wildcardCount == 0 or captures.len == 0:
        continue
      if n >= wildcardCount or n >= captures.len:
        n = 0
      let c = captures[n]
      if c.start < 0 or c.len < 0 or c.start > input.len or
          c.len > input.len - c.start:
        raise newException(SeqReplaceError,
          "invalid capture: start=" & $c.start & " len=" & $c.len &
          " input.len=" & $input.len & " op=" & $op.kind)
      if c.len > maxOutput - outp.len:
        overflow = true
        return
      var ri = c.start + c.len
      while ri > c.start:
        dec ri
        outp.add(input[ri])

    of 4'u8 .. 7'u8:
      var n = int(op.arg)
      if wildcardCount == 0 or captures.len == 0:
        continue
      if n >= wildcardCount or n >= captures.len:
        n = 0
      let c = captures[n]
      if c.start < 0 or c.len < 0 or c.start > input.len or
          c.len > input.len - c.start:
        raise newException(SeqReplaceError,
          "invalid capture: start=" & $c.start & " len=" & $c.len &
          " input.len=" & $input.len & " op=" & $op.kind)
      if c.len > maxOutput - outp.len:
        overflow = true
        return

      let dst = outp.len
      outp.setLen(dst + c.len)
      case op.kind
      of 4'u8: # +1 modulo 512
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] =
            (if v >= 0 and v < EMBEDDING_CODE_COUNT:
               (v + 1) mod EMBEDDING_CODE_COUNT
             else:
               v)

      of 5'u8: # -1 modulo 512
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] =
            (if v >= 0 and v < EMBEDDING_CODE_COUNT:
               (v + EMBEDDING_CODE_COUNT - 1) mod EMBEDDING_CODE_COUNT
             else:
               v)

      of 6'u8: # *2 modulo 512
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] =
            (if v >= 0 and v < EMBEDDING_CODE_COUNT:
               (v * 2) mod EMBEDDING_CODE_COUNT
             else:
               v)

      of 7'u8: # floor division; high OOV/out-of-domain tokens stay untouched
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          if v >= EMBEDDING_CODE_COUNT:
            outp[dst + j] = v
          else:
            let q = v div 2
            outp[dst + j] =
              (if v < 0 and v mod 2 != 0: q - 1 else: q)

      else:
        discard

    else:
      raise newException(SeqReplaceError, "Unknown replacement opcode")

proc replaceSeqCompiledInto(
  input: openArray[int],
  pat: SeqPattern,
  replacement: openArray[int],
  outp: var seq[int],
  scratch: var ReplaceScratch,
  replacementPlan: ptr ReplacementPlan,
  overflowed: ptr bool = nil,
  maxOutput: int = MAX_REPLACE_OUTPUT
): bool {.gcsafe.} =
  ## Core replacement routine. Returns whether a replacement actually occurred.
  if isAllWildcard(pat) or isEmptyPattern(pat) or pat.wildcardCount > 16:
    return false

  var overflow = false
  if overflowed != nil:
    overflowed[] = false

  if pat.wildcardCount == 0:
    let literal = pat.parts[0]
    let plen = literal.data.len
    if plen == 0 or input.len < plen:
      return false

    let firstMatch = findLiteralFrom(input, literal, 0)
    if firstMatch < 0:
      return false

    outp.setLen(0)
    var copyPos = 0
    var i = firstMatch

    while i >= 0:
      if copyPos < i:
        if outp.len + i - copyPos > maxOutput:
          if overflowed != nil:
            overflowed[] = true
          outp.setLen(0)
          return false
        appendIntsBulk(outp, input, copyPos, i - copyPos)

      if replacementPlan[].allLiteral:
        if outp.len + replacement.len > maxOutput:
          if overflowed != nil:
            overflowed[] = true
          outp.setLen(0)
          return false
        appendIntValuesBulk(outp, replacement)
      else:
        appendReplacementPlan(
          outp, replacementPlan, input,
          scratch.captures, pat.wildcardCount,
          scratch, overflow, maxOutput
        )
        if overflow:
          if overflowed != nil:
            overflowed[] = true
          outp.setLen(0)
          return false

      copyPos = i + plen
      i = findLiteralFrom(input, literal, copyPos)

    if copyPos < input.len:
      if outp.len + input.len - copyPos > maxOutput:
        if overflowed != nil:
          overflowed[] = true
        outp.setLen(0)
        return false
      appendIntsBulk(outp, input, copyPos, input.len - copyPos)

    return true

  let hasLeadingLiteral = pat.hasLeadingLiteral

  for i in 0 ..< pat.parts.len:
    scratch.cursors[i] = LiteralCursor(
      lastMatch: -1,
      nextSearch: 0,
      exhausted: false
    )

  var changed = false
  var pos = 0

  var cachedStart = -1
  var cachedFinish = -1
  var cachedOk = false
  var hasCached = false
  var outStarted = false

  while pos < input.len:
    var start = pos

    if hasLeadingLiteral:
      let first = nextOccurrenceLazy(input, pat.parts[0], scratch.cursors[0], pos)
      if first < 0:
        break
      start = first

    var m = (ok: false, finish: start)
    if hasCached and cachedStart == start:
      m = (ok: cachedOk, finish: cachedFinish)
    else:
      m = matchOccurrences(input, pat, scratch.cursors, start, scratch.captures)

    if not m.ok:
      if hasLeadingLiteral:
        let nextStart = nextOccurrenceLazy(
          input, pat.parts[0], scratch.cursors[0], start + 1
        )
        if nextStart < 0:
          break

        let m2 = matchOccurrences(
          input, pat, scratch.cursors, nextStart, scratch.captures
        )

        if not m2.ok:
          pos = nextStart
          cachedStart = nextStart
          cachedFinish = m2.finish
          cachedOk = m2.ok
          hasCached = true
          continue

        if not outStarted:
          outp.setLen(0)
          outStarted = true

        if pos < nextStart:
          if outp.len + nextStart - pos > maxOutput:
            if overflowed != nil:
              overflowed[] = true
            outp.setLen(0)
            return false
          appendIntsBulk(outp, input, pos, nextStart - pos)

        appendReplacementPlan(
          outp, replacementPlan, input,
          scratch.captures, pat.wildcardCount,
          scratch, overflow, maxOutput
        )
        if overflow:
          if overflowed != nil:
            overflowed[] = true
          outp.setLen(0)
          return false

        changed = true
        pos = m2.finish
        hasCached = false
        continue
      else:
        break

    if not outStarted:
      outp.setLen(0)
      outStarted = true

    if pos < start:
      if outp.len + start - pos > maxOutput:
        if overflowed != nil:
          overflowed[] = true
        outp.setLen(0)
        return false
      appendIntsBulk(outp, input, pos, start - pos)

    appendReplacementPlan(
      outp, replacementPlan, input,
      scratch.captures, pat.wildcardCount,
      scratch, overflow, maxOutput
    )
    if overflow:
      if overflowed != nil:
        overflowed[] = true
      outp.setLen(0)
      return false

    changed = true
    pos = m.finish
    hasCached = false

  if not changed:
    return false

  if pos < input.len:
    if outp.len + input.len - pos > maxOutput:
      if overflowed != nil:
        overflowed[] = true
      outp.setLen(0)
      return false
    appendIntsBulk(outp, input, pos, input.len - pos)

  true

proc replaceSeqCompiled*(
  input: openArray[int],
  pat: SeqPattern,
  replacement: openArray[int]
): tuple[changed: bool, data: seq[int]] {.gcsafe.} =
  var outp = newSeqOfCap[int](max(16, input.len))
  var scratch = ReplaceScratch(sortBuf: @[])
  var plan: ReplacementPlan
  compileReplacementPlan(replacement, plan)

  let changed = replaceSeqCompiledInto(
    input, pat, replacement, outp, scratch, addr plan
  )
  if not changed:
    # Public API semantics: no match means identity, not an empty sequence.
    # `outp` may contain a partial speculative prefix, so reset it first.
    outp.setLen(0)
    appendIntsBulk(outp, input, 0, input.len)
    return (false, outp)
  (true, outp)

proc replaceSeqCompiledInPlace*(
  input: var seq[int],
  pat: SeqPattern,
  replacement: openArray[int]
): bool =
  ## Allocation-sparing compiled-pattern variant for repeated training evaluation.
  ## A pattern miss leaves input intact without materializing an identity copy.
  ## The bool reports a REAL byte-sequence change, not merely a matched pattern.
  var output: seq[int] = @[]
  var scratch = ReplaceScratch(sortBuf: @[])
  var plan: ReplacementPlan
  compileReplacementPlan(replacement, plan)
  if replaceSeqCompiledInto(input, pat, replacement, output, scratch, addr plan):
    if output != input:
      input = move(output)
      return true
  false

proc replaceSeq*(
  input: openArray[int],
  pattern: openArray[int],
  replacement: openArray[int]
): seq[int] =
  ## Replace all matches. If nothing matches, return an unchanged copy of input.
  let pat = compileSeqPattern(pattern)
  let r = replaceSeqCompiled(input, pat, replacement)
  r.data

proc replaceSeqInPlace*(
  input: var seq[int],
  pattern: openArray[int],
  replacement: openArray[int]
) =
  ## In-place convenience wrapper. A non-match is a true no-op and therefore
  ## avoids allocating/copying the unchanged input.
  let pat = compileSeqPattern(pattern)
  var outp = newSeqOfCap[int](max(16, input.len))
  var scratch = ReplaceScratch(sortBuf: @[])
  var plan: ReplacementPlan
  compileReplacementPlan(replacement, plan)
  if replaceSeqCompiledInto(input, pat, replacement, outp, scratch, addr plan):
    input = move(outp)

# -----------------------------------------------------------------------------
# Efficient diff3-style patch transfer for integer/token sequences
# -----------------------------------------------------------------------------
#
# diff3Apply(base, changed, target) means:
#   1) compute the edits base -> changed
#   2) preserve independent edits already present in target
#   3) transplant the base -> changed edits onto target
#   4) detect genuinely overlapping edits as conflicts
#
# All public inputs are openArray[int], so slices do not need to be materialized
# before calling. The hot path only allocates the two compact hunk lists plus the
# output. Temporary segment buffers are used only for overlapping diff3 regions.
#
# The diff is exact Myers up to `maxMyersD`. If the edit distance exceeds that
# guard, the remaining changed middle is represented as one coarse replacement
# hunk. That still transforms base -> changed exactly; it only makes conflict
# granularity coarser while bounding worst-case trace memory. Set maxMyersD <= 0
# for an unbounded exact Myers trace.

type
  DiffHunk* = object
    ## Replace base[baseStart ..< baseStart+baseLen] with
    ## newer[newStart ..< newStart+newLen].
    baseStart*: int
    baseLen*: int
    newStart*: int
    newLen*: int

  DiffEditKind = enum
    dekInsert,
    dekDelete

  DiffEdit = object
    kind: DiffEditKind
    aPos: int
    bPos: int

  Diff3ConflictChoice* = enum
    ## Keep the target-side edit when both sides changed the same base region.
    d3KeepTarget,
    ## Prefer the transplanted base -> changed edit.
    d3KeepChanged,
    ## Revert the conflicting region to the common base.
    d3KeepBase

  Diff3Conflict* = object
    baseStart*: int
    baseLen*: int
    baseData*: seq[int]
    changedData*: seq[int]
    targetData*: seq[int]

  Diff3ApplyResult* = object
    data*: seq[int]
    conflicts*: seq[Diff3Conflict]

  Diff3ConflictResolver* = proc(
    basePart: openArray[int],
    changedPart: openArray[int],
    targetPart: openArray[int]
  ): Diff3ConflictChoice {.closure.}

const
  DEFAULT_MAX_MYERS_D* = 768

proc cloneInts(a: openArray[int]): seq[int] {.inline.} =
  result = newSeq[int](a.len)
  if a.len > 0:
    copyMem(addr result[0], unsafeAddr a[0], a.len * sizeof(int))

proc rangesEqual(
  a: openArray[int], aStart, aLen: int,
  b: openArray[int], bStart, bLen: int
): bool {.inline.} =
  if aLen != bLen:
    return false
  if aLen == 0:
    return true
  if aStart < 0 or bStart < 0 or
      aStart > a.len or bStart > b.len or
      aLen > a.len - aStart or bLen > b.len - bStart:
    return false
  for i in 0 ..< aLen:
    if a[aStart + i] != b[bStart + i]:
      return false
  true

proc seqEqual(a, b: openArray[int]): bool {.inline.} =
  rangesEqual(a, 0, a.len, b, 0, b.len)

proc coarseMiddleHunk(
  prefix, aLen, bLen: int
): seq[DiffHunk] {.inline.} =
  if aLen == 0 and bLen == 0:
    return @[]
  @[DiffHunk(
    baseStart: prefix,
    baseLen: aLen,
    newStart: prefix,
    newLen: bLen
  )]

proc diffHunks*(
  base: openArray[int],
  newer: openArray[int],
  maxMyersD: int = DEFAULT_MAX_MYERS_D
): seq[DiffHunk] =
  ## Myers shortest-edit diff collapsed directly into replacement hunks.
  ## No copy of either input openArray is made.

  var prefix = 0
  let commonMax = min(base.len, newer.len)
  while prefix < commonMax and base[prefix] == newer[prefix]:
    inc prefix

  var aEnd = base.len
  var bEnd = newer.len
  while aEnd > prefix and bEnd > prefix and
      base[aEnd - 1] == newer[bEnd - 1]:
    dec aEnd
    dec bEnd

  let n = aEnd - prefix
  let m = bEnd - prefix
  if n == 0 or m == 0:
    return coarseMiddleHunk(prefix, n, m)

  let maxEdit = n + m
  let limit =
    if maxMyersD <= 0: maxEdit
    else: min(maxEdit, maxMyersD)

  # V only needs diagonals [-limit .. +limit], even when n+m is huge.
  let offset = limit + 1
  var v = newSeq[int](2 * limit + 3)
  for i in 0 ..< v.len:
    v[i] = -1
  v[offset + 1] = 0

  # Compact trace: at depth d only d+1 parity-valid diagonals exist.
  # Total trace storage is O(D^2/2), bounded by maxMyersD by default.
  var trace: seq[seq[int]] = @[]
  var foundD = -1

  for d in 0 .. limit:
    var k = -d
    while k <= d:
      var x: int
      if k == -d or
          (k != d and v[offset + k - 1] < v[offset + k + 1]):
        x = v[offset + k + 1]
      else:
        x = v[offset + k - 1] + 1

      var y = x - k
      while x < n and y < m and
          base[prefix + x] == newer[prefix + y]:
        inc x
        inc y

      v[offset + k] = x
      if x >= n and y >= m:
        foundD = d
        break
      k += 2

    var row = newSeq[int](d + 1)
    k = -d
    var ri = 0
    while k <= d:
      row[ri] = v[offset + k]
      inc ri
      k += 2
    trace.add(row)

    if foundD >= 0:
      break

  if foundD < 0:
    # Guard hit: exact transformation, coarser conflict region.
    return coarseMiddleHunk(prefix, n, m)
  if foundD == 0:
    return @[]

  var steps: seq[DiffEdit] = @[]
  steps = newSeqOfCap[DiffEdit](foundD)
  var x = n
  var y = m

  for d in countdown(foundD, 1):
    let prevD = d - 1
    let row = trace[prevD]
    let k = x - y

    template prevAt(kk: int): int =
      (if kk < -prevD or kk > prevD or ((kk + prevD) and 1) != 0:
         int.low div 4
       else:
         row[(kk + prevD) div 2])

    var prevK: int
    if k == -d or
        (k != d and prevAt(k - 1) < prevAt(k + 1)):
      prevK = k + 1
    else:
      prevK = k - 1

    let prevX = prevAt(prevK)
    let prevY = prevX - prevK

    # Walk backwards over the equal "snake".
    while x > prevX and y > prevY:
      dec x
      dec y

    if x == prevX:
      steps.add(DiffEdit(kind: dekInsert, aPos: prevX, bPos: prevY))
    else:
      steps.add(DiffEdit(kind: dekDelete, aPos: prevX, bPos: prevY))

    x = prevX
    y = prevY

  reverse(steps)

  # Collapse adjacent insert/delete edits into base-coordinate hunks.
  var have = false
  var cur: DiffHunk
  var afterA = -1
  var afterB = -1

  for st in steps:
    if not have or st.aPos != afterA or st.bPos != afterB:
      if have:
        result.add(cur)
      cur = DiffHunk(
        baseStart: prefix + st.aPos,
        baseLen: 0,
        newStart: prefix + st.bPos,
        newLen: 0
      )
      have = true

    case st.kind
    of dekDelete:
      inc cur.baseLen
      afterA = st.aPos + 1
      afterB = st.bPos
    of dekInsert:
      inc cur.newLen
      afterA = st.aPos
      afterB = st.bPos + 1

  if have:
    result.add(cur)

proc hunkEnd(h: DiffHunk): int {.inline.} =
  h.baseStart + h.baseLen

proc hunksConflict(a, b: DiffHunk): bool {.inline.} =
  ## Boundary-only contact is composable and therefore not a conflict.
  ## Two insertions at exactly the same base position do conflict.
  let ae = hunkEnd(a)
  let be = hunkEnd(b)

  if a.baseLen == 0 and b.baseLen == 0:
    return a.baseStart == b.baseStart
  if a.baseLen == 0:
    return a.baseStart > b.baseStart and a.baseStart < be
  if b.baseLen == 0:
    return b.baseStart > a.baseStart and b.baseStart < ae
  a.baseStart < be and b.baseStart < ae

proc hunkIntersectsCluster(
  h: DiffHunk,
  clusterStart, clusterEnd: int
): bool {.inline.} =
  if clusterStart == clusterEnd:
    return h.baseLen == 0 and h.baseStart == clusterStart
  if h.baseLen == 0:
    return h.baseStart > clusterStart and h.baseStart < clusterEnd
  h.baseStart < clusterEnd and hunkEnd(h) > clusterStart

proc renderDiffRegion(
  base: openArray[int],
  side: openArray[int],
  hunks: openArray[DiffHunk],
  firstHunk, pastHunk: int,
  regionStart, regionEnd: int,
  outp: var seq[int]
) =
  outp.setLen(0)
  var pos = regionStart

  for i in firstHunk ..< pastHunk:
    let h = hunks[i]
    if h.baseStart < pos:
      raise newException(SeqReplaceError,
        "overlapping diff hunks at base position " & $h.baseStart)

    if pos < h.baseStart:
      appendIntsBulk(outp, base, pos, h.baseStart - pos)
    if h.newLen > 0:
      appendIntsBulk(outp, side, h.newStart, h.newLen)
    pos = hunkEnd(h)

  if pos < regionEnd:
    appendIntsBulk(outp, base, pos, regionEnd - pos)

proc appendConflictChoice(
  outp: var seq[int],
  choice: Diff3ConflictChoice,
  base: openArray[int],
  baseStart, baseEnd: int,
  changedPart, targetPart: openArray[int]
) {.inline.} =
  case choice
  of d3KeepTarget:
    appendIntsBulk(outp, targetPart, 0, targetPart.len)
  of d3KeepChanged:
    appendIntsBulk(outp, changedPart, 0, changedPart.len)
  of d3KeepBase:
    appendIntsBulk(outp, base, baseStart, baseEnd - baseStart)

proc diff3ApplyInto*(
  base: openArray[int],
  changed: openArray[int],
  target: openArray[int],
  outp: var seq[int],
  defaultChoice: Diff3ConflictChoice = d3KeepTarget,
  conflicts: ptr seq[Diff3Conflict] = nil,
  resolver: Diff3ConflictResolver = nil,
  maxMyersD: int = DEFAULT_MAX_MYERS_D
): bool =
  ## Apply the edit base -> changed to target, preserving independent target edits.
  ## Returns true iff at least one real diff3 conflict occurred.
  ##
  ## `outp` is reused: setLen(0) preserves its capacity.
  ## Pass conflicts=nil for the cheapest GA/hot-loop path.

  let changedHunks = diffHunks(base, changed, maxMyersD)
  let targetHunks = diffHunks(base, target, maxMyersD)

  outp.setLen(0)
  if conflicts != nil:
    conflicts[].setLen(0)

  var ci = 0
  var ti = 0
  var pos = 0
  var hadConflict = false
  var changedBuf: seq[int] = @[]
  var targetBuf: seq[int] = @[]
  var baseBuf: seq[int] = @[]

  while ci < changedHunks.len or ti < targetHunks.len:
    let haveC = ci < changedHunks.len
    let haveT = ti < targetHunks.len

    if haveC and haveT and hunksConflict(changedHunks[ci], targetHunks[ti]):
      var clusterStart = min(
        changedHunks[ci].baseStart,
        targetHunks[ti].baseStart
      )
      var clusterEnd = max(
        hunkEnd(changedHunks[ci]),
        hunkEnd(targetHunks[ti])
      )

      let cFirst = ci
      let tFirst = ti
      inc ci
      inc ti

      # Expand transitively across all edits that overlap the conflicting base
      # interior. Edits that merely touch a boundary remain independently
      # composable and are deliberately left out of the cluster.
      var expanded = true
      while expanded:
        expanded = false
        while ci < changedHunks.len and
            hunkIntersectsCluster(changedHunks[ci], clusterStart, clusterEnd):
          clusterEnd = max(clusterEnd, hunkEnd(changedHunks[ci]))
          inc ci
          expanded = true

        while ti < targetHunks.len and
            hunkIntersectsCluster(targetHunks[ti], clusterStart, clusterEnd):
          clusterEnd = max(clusterEnd, hunkEnd(targetHunks[ti]))
          inc ti
          expanded = true

      if pos < clusterStart:
        appendIntsBulk(outp, base, pos, clusterStart - pos)

      renderDiffRegion(
        base, changed, changedHunks,
        cFirst, ci, clusterStart, clusterEnd, changedBuf
      )
      renderDiffRegion(
        base, target, targetHunks,
        tFirst, ti, clusterStart, clusterEnd, targetBuf
      )

      if seqEqual(changedBuf, targetBuf):
        appendIntsBulk(outp, changedBuf, 0, changedBuf.len)
      else:
        hadConflict = true
        var choice = defaultChoice
        if resolver != nil:
          baseBuf.setLen(0)
          if clusterEnd > clusterStart:
            appendIntsBulk(
              baseBuf, base, clusterStart, clusterEnd - clusterStart
            )
          choice = resolver(baseBuf, changedBuf, targetBuf)

        appendConflictChoice(
          outp, choice, base, clusterStart, clusterEnd,
          changedBuf, targetBuf
        )

        if conflicts != nil:
          var c = Diff3Conflict(
            baseStart: clusterStart,
            baseLen: clusterEnd - clusterStart,
            baseData: @[],
            changedData: cloneInts(changedBuf),
            targetData: cloneInts(targetBuf)
          )
          if clusterEnd > clusterStart:
            c.baseData = newSeq[int](clusterEnd - clusterStart)
            copyMem(
              addr c.baseData[0],
              unsafeAddr base[clusterStart],
              c.baseData.len * sizeof(int)
            )
          conflicts[].add(c)

      pos = clusterEnd
      continue

    # No overlap: emit whichever edit occurs first in base coordinates.
    # If both start at the same point and only one is an insertion, emit the
    # insertion first so that a replacement/deletion beginning there can follow.
    var chooseChanged: bool
    if not haveT:
      chooseChanged = true
    elif not haveC:
      chooseChanged = false
    else:
      let ch = changedHunks[ci]
      let th = targetHunks[ti]
      if ch.baseStart < th.baseStart:
        chooseChanged = true
      elif th.baseStart < ch.baseStart:
        chooseChanged = false
      elif ch.baseLen == 0 and th.baseLen != 0:
        chooseChanged = true
      elif th.baseLen == 0 and ch.baseLen != 0:
        chooseChanged = false
      else:
        # Same-start non-insertion edits should already have conflicted.
        chooseChanged = true

    if chooseChanged:
      let h = changedHunks[ci]
      if pos < h.baseStart:
        appendIntsBulk(outp, base, pos, h.baseStart - pos)
      if h.newLen > 0:
        appendIntsBulk(outp, changed, h.newStart, h.newLen)
      pos = hunkEnd(h)
      inc ci
    else:
      let h = targetHunks[ti]
      if pos < h.baseStart:
        appendIntsBulk(outp, base, pos, h.baseStart - pos)
      if h.newLen > 0:
        appendIntsBulk(outp, target, h.newStart, h.newLen)
      pos = hunkEnd(h)
      inc ti

  if pos < base.len:
    appendIntsBulk(outp, base, pos, base.len - pos)

  hadConflict

proc diff3Apply*(
  base: openArray[int],
  changed: openArray[int],
  target: openArray[int],
  defaultChoice: Diff3ConflictChoice = d3KeepTarget,
  resolver: Diff3ConflictResolver = nil,
  maxMyersD: int = DEFAULT_MAX_MYERS_D
): seq[int] =
  ## Fast convenience API. Does not retain conflict payloads.
  result = newSeqOfCap[int](max(target.len, changed.len))
  discard diff3ApplyInto(
    base, changed, target, result,
    defaultChoice, nil, resolver, maxMyersD
  )

proc diff3ApplyDetailed*(
  base: openArray[int],
  changed: openArray[int],
  target: openArray[int],
  defaultChoice: Diff3ConflictChoice = d3KeepTarget,
  resolver: Diff3ConflictResolver = nil,
  maxMyersD: int = DEFAULT_MAX_MYERS_D
): Diff3ApplyResult =
  ## Diagnostic API retaining copies of conflicting regions.
  result.data = newSeqOfCap[int](max(target.len, changed.len))
  discard diff3ApplyInto(
    base, changed, target, result.data,
    defaultChoice, addr result.conflicts, resolver, maxMyersD
  )
