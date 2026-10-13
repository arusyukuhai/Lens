## NSGA-II fast non-dominated sorting + crowding distance.
## Supports any number of objectives (2, 3, 4, ...), including mixed directions.
## Nim 1.6+; no dependencies beyond the standard library.
##
## Complexity: O(N^2 * M + M * N log N) time,
##             O(N^2 + N * M) worst-case memory.
## N = number of individuals, M = number of objectives.

import std/[algorithm, math]

type
  ObjectiveSense* = enum
    odMinimize, odMaximize

  Nsga2Result* = object
    ## Rank 0 is the best (first non-dominated front).
    rank*: seq[int]
    ## Larger distance is better within a front. Boundary solutions get +Inf.
    crowding*: seq[float64]
    ## fronts[0] is the first Pareto front. IDs are original row indices.
    fronts*: seq[seq[int]]
    ## IDs ordered by (rank ASC, crowding DESC, original index ASC).
    order*: seq[int]

proc dominance(values: seq[float64], m, a, b: int): int {.inline.} =
  ## +1: a dominates b; -1: b dominates a; 0: incomparable/equal.
  ## Values are pre-normalized so that all objectives are minimized.
  let a0 = a * m
  let b0 = b * m
  var aBetter = false
  var bBetter = false
  for k in 0 ..< m:
    let x = values[a0 + k]
    let y = values[b0 + k]
    if x < y:
      aBetter = true
    elif x > y:
      bBetter = true
    if aBetter and bBetter:
      return 0
  if aBetter:
    return 1
  if bBetter:
    return -1
  return 0

proc assignCrowding(
    front: seq[int], values: seq[float64], m: int,
    distances: var seq[float64]) =
  ## NSGA-II normalized crowding distance, computed independently per front.
  let count = front.len
  if count == 0:
    return
  if count <= 2:
    for id in front:
      distances[id] = Inf
    return

  var permutation = newSeq[int](count)
  for objective in 0 ..< m:
    for i in 0 ..< count:
      permutation[i] = front[i]

    let axis = objective
    permutation.sort(proc(a, b: int): int =
      let va = values[a * m + axis]
      let vb = values[b * m + axis]
      if va < vb: return -1
      if va > vb: return 1
      # Deterministic ordering for tied objective values.
      if a < b: return -1
      if a > b: return 1
      return 0
    )

    let lo = values[permutation[0] * m + objective]
    let hi = values[permutation[count - 1] * m + objective]
    if hi == lo:
      # A constant objective supplies no diversity information.
      continue

    distances[permutation[0]] = Inf
    distances[permutation[count - 1]] = Inf

    # Avoid overflow in hi - lo for very large (but finite) scores.
    let scale = if hi - lo == Inf: 0.5 else: 1.0
    let span = hi * scale - lo * scale
    for i in 1 ..< count - 1:
      let id = permutation[i]
      if distances[id] != Inf:
        let before = values[permutation[i - 1] * m + objective] * scale
        let after = values[permutation[i + 1] * m + objective] * scale
        distances[id] += (after - before) / span

proc nsga2SortGeneral*(
    scores: openArray[seq[float64]],
    senses: openArray[ObjectiveSense]): Nsga2Result =
  ## scores[individual][objective].
  ## An empty `senses` means "minimize all"; otherwise provide one per axis.
  ## All rows must have the same nonzero length and all scores must be finite.
  let n = scores.len
  if n == 0:
    return
  let m = scores[0].len
  if m == 0:
    raise newException(ValueError, "at least one objective is required")
  if senses.len != 0 and senses.len != m:
    raise newException(ValueError, "senses must be empty or have one entry per objective")

  # Flatten for better locality in the O(N^2 * M) dominance pass.
  var values = newSeq[float64](n * m)
  for i in 0 ..< n:
    if scores[i].len != m:
      raise newException(ValueError, "all score rows must have equal length")
    for k in 0 ..< m:
      let v = scores[i][k]
      if v != v or abs(v) == Inf:
        raise newException(ValueError, "objective values must be finite (no NaN/Inf)")
      if senses.len != 0 and senses[k] == odMaximize:
        values[i * m + k] = -v
      else:
        values[i * m + k] = v

  result.rank = newSeq[int](n)
  result.crowding = newSeq[float64](n)

  # Deb's fast non-dominated sorting: compare each unordered pair once.
  var dominates = newSeq[seq[int]](n)
  var dominationCount = newSeq[int](n)
  for i in 0 ..< n:
    for j in i + 1 ..< n:
      case dominance(values, m, i, j)
      of 1:
        dominates[i].add(j)
        inc dominationCount[j]
      of -1:
        dominates[j].add(i)
        inc dominationCount[i]
      else:
        discard

  var current: seq[int] = @[]
  for i in 0 ..< n:
    if dominationCount[i] == 0:
      current.add(i)

  while current.len > 0:
    let rankNumber = result.fronts.len
    result.fronts.add(current)
    var nextFront: seq[int] = @[]
    for i in current:
      for j in dominates[i]:
        dec dominationCount[j]
        if dominationCount[j] == 0:
          result.rank[j] = rankNumber + 1
          nextFront.add(j)
    current = nextFront

  # Diversity + deterministic survivor order.
  result.order = newSeqOfCap[int](n)
  for front in result.fronts:
    assignCrowding(front, values, m, result.crowding)
    var ranked = newSeq[int](front.len)
    for i, id in front:
      ranked[i] = id
    let distances = result.crowding
    ranked.sort(proc(a, b: int): int =
      let da = distances[a]
      let db = distances[b]
      if da > db: return -1
      if da < db: return 1
      if a < b: return -1
      if a > b: return 1
      return 0
    )
    for id in ranked:
      result.order.add(id)

proc nsga2Sort*(scores: openArray[seq[float64]],
                senses: openArray[ObjectiveSense]): Nsga2Result =
  ## Two-objective Fenwick rank sweep: O(N log N) time and O(N) memory.
  ## The general >=3 objective implementation remains available unchanged.
  if scores.len == 0: return
  if scores[0].len != 2: return nsga2SortGeneral(scores, senses)
  if senses.len != 0 and senses.len != 2:
    raise newException(ValueError, "senses length mismatch")
  let n = scores.len
  var values = newSeq[float64](2*n)
  var ys = newSeq[float64](n)
  var order = newSeq[int](n)
  for i in 0..<n:
    if scores[i].len != 2: raise newException(ValueError, "objective shape mismatch")
    order[i] = i
    for k in 0..1:
      let v = scores[i][k]
      if v != v or abs(v) == Inf: raise newException(ValueError, "nonfinite objective")
      values[2*i+k] = if senses.len > 0 and senses[k] == odMaximize: -v else: v
    ys[i] = values[2*i+1]
  order.sort(proc(a,b: int): int =
    let c = cmp(values[2*a],values[2*b])
    if c != 0: return c
    let d = cmp(values[2*a+1],values[2*b+1])
    if d != 0: d else: cmp(a,b))
  ys.sort()
  var uniqueY: seq[float64]
  for y in ys:
    if uniqueY.len == 0 or uniqueY[^1] != y: uniqueY.add(y)
  var tree = newSeq[int](uniqueY.len+1) # rank+1; zero means no dominator
  result.rank = newSeq[int](n)
  result.crowding = newSeq[float64](n)
  var start = 0
  while start < n:
    var stop = start+1
    let id = order[start]
    while stop < n and values[2*order[stop]] == values[2*id] and
        values[2*order[stop]+1] == values[2*id+1]: inc stop
    let coordinate = lowerBound(uniqueY,values[2*id+1])+1
    var q = coordinate
    var rank = 0
    while q > 0:
      rank = max(rank,tree[q])
      q -= q and -q
    # Identical points do not dominate each other: update after the entire group.
    for at in start..<stop: result.rank[order[at]] = rank
    q = coordinate
    while q < tree.len:
      tree[q] = max(tree[q],rank+1)
      q += q and -q
    if result.fronts.len <= rank: result.fronts.setLen(rank+1)
    for at in start..<stop: result.fronts[rank].add(order[at])
    start = stop
  for front in result.fronts:
    assignCrowding(front,values,2,result.crowding)
    var ranked = front
    let distances = result.crowding
    ranked.sort(proc(a,b: int): int =
      let c = cmp(distances[b],distances[a])
      if c != 0: c else: cmp(a,b))
    result.order.add(ranked)

proc nsga2Sort*(scores: openArray[seq[float64]]): Nsga2Result =
  ## Convenience overload: every objective is minimized.
  let senses: seq[ObjectiveSense] = @[]
  return nsga2Sort(scores, senses)

proc nsga2Select*(
    scores: openArray[seq[float64]], survivorCount: int,
    senses: openArray[ObjectiveSense]): seq[int] =
  ## Return up to survivorCount original individual indices in NSGA-II order.
  if survivorCount < 0:
    raise newException(ValueError, "survivorCount must be >= 0")
  if survivorCount == 0:
    return @[]
  let sorted = nsga2Sort(scores, senses)
  let takeCount = min(survivorCount, sorted.order.len)
  result = newSeq[int](takeCount)
  for i in 0 ..< takeCount:
    result[i] = sorted.order[i]

proc nsga2Select*(
    scores: openArray[seq[float64]], survivorCount: int): seq[int] =
  ## Convenience overload: every objective is minimized.
  let senses: seq[ObjectiveSense] = @[]
  return nsga2Select(scores, survivorCount, senses)

when isMainModule:
  block:
    # 3-objective minimization; 0-3 are mutually non-dominating.
    let scores = @[
      @[1.0, 4.0, 4.0],
      @[4.0, 1.0, 4.0],
      @[4.0, 4.0, 1.0],
      @[2.0, 2.0, 2.0],
      @[5.0, 5.0, 5.0]
    ]
    let res = nsga2Sort(scores)
    doAssert res.rank == @[0, 0, 0, 0, 1]
    doAssert res.fronts.len == 2
    doAssert res.fronts[0].len == 4
    doAssert res.fronts[1] == @[4]
    doAssert res.order.len == scores.len
    doAssert res.order[^1] == 4
    doAssert nsga2Select(scores, 2).len == 2

  block:
    # Mixed objectives: minimize first, maximize second, minimize third.
    let scores = @[
      @[2.0, 10.0, 4.0],
      @[3.0, 9.0, 4.0],
      @[1.0, 10.0, 3.0]
    ]
    let res = nsga2Sort(scores, @[odMinimize, odMaximize, odMinimize])
    doAssert res.rank == @[1, 2, 0]
    doAssert res.order == @[2, 0, 1]

  block:
    # All equal: all rank 0, zero crowding, deterministic original order.
    let scores = @[@[1.0, 1.0, 1.0], @[1.0, 1.0, 1.0], @[1.0, 1.0, 1.0]]
    let res = nsga2Sort(scores)
    doAssert res.rank == @[0, 0, 0]
    doAssert res.crowding == @[0.0, 0.0, 0.0]
    doAssert res.order == @[0, 1, 2]

  echo "NSGA-II self-tests passed."
