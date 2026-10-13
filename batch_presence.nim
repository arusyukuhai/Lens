## Exact batched existence for anchored wildcard patterns with positive gaps.
## Aho-Corasick shares literal scans across every candidate, one text at a time.
import std/[tables, algorithm, deques, bitops, math]

type
  ACNode = object
    edges: Table[int,int]
    fail, output: int
    terminal: int
  PackedPresence* = object
    rows*: int
    columns*: seq[seq[uint64]]
  BatchMatcher* = object
    nodes: seq[ACNode]
    lengths: seq[int]
    patterns: seq[seq[int]]  # literal IDs in order
    byFirstLiteral: Table[int,seq[int]]

proc newNode(): ACNode =
  ACNode(edges: initTable[int,int](), output: -1, terminal: -1)

proc compileBatch*(patterns: seq[seq[int]]): BatchMatcher =
  result.byFirstLiteral = initTable[int,seq[int]]()
  result.nodes.add(newNode())
  var ids = initTable[seq[int],int]()
  for pattern in patterns:
    if pattern.len == 0 or pattern[0] < 0 or pattern[^1] < 0:
      raise newException(ValueError, "Batch matcher requires literal anchors")
    var literalIds: seq[int]
    var start = 0
    for finish in 0..pattern.len:
      if finish < pattern.len and pattern[finish] >= 0: continue
      if finish == start: raise newException(ValueError, "Adjacent wildcards unsupported")
      let literal = pattern[start..<finish]
      var id = ids.getOrDefault(literal, -1)
      if id < 0:
        id = result.lengths.len
        ids[literal] = id
        result.lengths.add(literal.len)
        var node = 0
        for token in literal:
          var next = result.nodes[node].edges.getOrDefault(token, -1)
          if next < 0:
            next = result.nodes.len
            result.nodes.add(newNode())
            result.nodes[node].edges[token] = next
          node = next
        result.nodes[node].terminal = id
      literalIds.add(id)
      start = finish+1
    result.byFirstLiteral.mgetOrPut(literalIds[0], @[]).add(result.patterns.len)
    result.patterns.add(literalIds)
  var queue = initDeque[int]()
  for _, child in result.nodes[0].edges: queue.addLast(child)
  while queue.len > 0:
    let node = queue.popFirst()
    for token, child in result.nodes[node].edges:
      var f = result.nodes[node].fail
      while f != 0 and not result.nodes[f].edges.hasKey(token):
        f = result.nodes[f].fail
      result.nodes[child].fail = result.nodes[f].edges.getOrDefault(token, 0)
      let link = result.nodes[child].fail
      result.nodes[child].output =
        if result.nodes[link].terminal >= 0: link else: result.nodes[link].output
      queue.addLast(child)

proc batchPresence*(matcher: BatchMatcher, texts: seq[seq[int]]): PackedPresence =
  result.rows = texts.len
  result.columns = newSeq[seq[uint64]](matcher.patterns.len)
  for col in result.columns.mitems: col = newSeq[uint64]((texts.len+63) div 64)
  for row, text in texts:
    # Reuse occurrence storage for this row only: O(L * maximum literal length).
    var occurrences = initTable[int,seq[int]]()
    var node = 0
    for pos, token in text:
      while node != 0 and not matcher.nodes[node].edges.hasKey(token):
        node = matcher.nodes[node].fail
      node = matcher.nodes[node].edges.getOrDefault(token, 0)
      var outNode = node
      while outNode >= 0:
        let id = matcher.nodes[outNode].terminal
        if id >= 0:
          occurrences.mgetOrPut(id, @[]).add(pos-matcher.lengths[id]+1)
        outNode = matcher.nodes[outNode].output
    # Any matching candidate must contain its first anchor. Iterate only
    # candidates indexed by actually observed first anchors in this row.
    # The following gap-aware checks are unchanged, so this is exact.
    for firstId, _ in occurrences:
      if not matcher.byFirstLiteral.hasKey(firstId): continue
      for c in matcher.byFirstLiteral[firstId]:
        let parts = matcher.patterns[c]
        var position = 0
        var foundAll = true
        for id in parts:
          if not occurrences.hasKey(id):
            foundAll = false; break
          let positions = occurrences[id]
          let index = lowerBound(positions, position)
          if index == positions.len:
            foundAll = false; break
          position = positions[index] + matcher.lengths[id] + 1
        if foundAll:
          result.columns[c][row shr 6] =
            result.columns[c][row shr 6] or (1'u64 shl (row and 63))

proc toColumn*(bits: seq[uint64], rows: int): seq[float64] =
  result = newSeq[float64](rows)
  for i in 0..<rows: result[i] = float64((bits[i shr 6] shr (i and 63)) and 1'u64)

proc packColumn*(column: seq[float64]): seq[uint64] =
  result = newSeq[uint64]((column.len+63) div 64)
  for i,v in column:
    if v != 0: result[i shr 6] = result[i shr 6] or (1'u64 shl (i and 63))

proc bitCount*(bits: seq[uint64]): int =
  for word in bits: result += countSetBits(word)

proc packedPhi*(a,b: seq[uint64], countA,countB,rows: int): float64 =
  if countA == 0 or countB == 0 or countA == rows or countB == rows: return 0.0
  var both = 0
  for i in 0..<a.len: both += countSetBits(a[i] and b[i])
  let sa = float64(countA)
  let sb = float64(countB)
  let n = float64(rows)
  min(1.0,abs((n*float64(both)-sa*sb)/sqrt(sa*(n-sa)*sb*(n-sb))))
