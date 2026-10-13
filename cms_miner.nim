## Wildcard n-gram miner extracted from the supplied cms(3).nim.
## Positive gaps and original product/root constraints; no enumeration cutoff.
import std/[tables, hashes, algorithm, math, sequtils, times]
import fast_count_min_sketch

type
  PatternEntry* = object
    value*: seq[int]
    rank*: uint32
  RankEntry = PatternEntry
  RankBucket = ref object
    ranks: seq[RankEntry]
    positions: Table[seq[int], int]
  MiningConfig* = object
    minGram*, maxGram*, maxTotalGram*, maxWildcards*: int
    topPerShape*, width*, depth*, startStride*, gapSamples*: int
    verbose*: bool

proc defaultMiningConfig*(): MiningConfig =
  MiningConfig(minGram: 1, maxGram: 4, maxTotalGram: 6,
    maxWildcards: 4, topPerShape: 0, width: 262144, depth: 7,
    startStride: 2, gapSamples: 6)

proc minePatterns*(chunks: seq[seq[int]], cfg: MiningConfig): seq[PatternEntry] =
  if cfg.minGram < 1 or cfg.maxGram < cfg.minGram or cfg.maxTotalGram < 1 or
      cfg.maxWildcards < 0 or cfg.maxWildcards > 4 or cfg.maxGram > 6 or
      cfg.startStride < 1 or cfg.gapSamples < 0:
    raise newException(ValueError, "Invalid mining configuration (maxGram <= 6, wildcards <= 4)")
  var buckets = initTable[uint64, RankBucket]()
  var cms = initCountMinSketch(width = cfg.width, depth = cfg.depth)
  let miningStartedAt = epochTime()
  var lastProgress = miningStartedAt
  var enumerated: int64 = 0
  var a: seq[int]
  var totalLength = 0
  for chunk in chunks: totalLength += chunk.len
  var maxChunkLength = 0
  for chunk in chunks: maxChunkLength = max(maxChunkLength, chunk.len)
  var capacity = cfg.topPerShape
  var shapes: seq[seq[int]] = @[]
  proc newRankBucket(): RankBucket =
    new(result)
    result.ranks = @[]
    result.positions = initTable[seq[int], int]()

  proc siftUpFast(
    bucket: RankBucket,
    start: int
  ) =
    var i = start
    let item = bucket.ranks[i]

    while i > 0:
      let p = (i - 1) shr 1
      if bucket.ranks[p].rank <= item.rank:
        break

      bucket.ranks[i] = bucket.ranks[p]
      bucket.positions[bucket.ranks[i].value] = i
      i = p

    bucket.ranks[i] = item
    bucket.positions[item.value] = i

  proc siftDownFast(
    bucket: RankBucket,
    start: int
  ) =
    var i = start
    let n = bucket.ranks.len
    let item = bucket.ranks[i]

    while true:
      let l = (i shl 1) + 1
      if l >= n:
        break

      let r = l + 1
      var child = l

      if r < n and
         bucket.ranks[r].rank < bucket.ranks[l].rank:
        child = r

      if item.rank <= bucket.ranks[child].rank:
        break

      bucket.ranks[i] = bucket.ranks[child]
      bucket.positions[bucket.ranks[i].value] = i
      i = child

    bucket.ranks[i] = item
    bucket.positions[item.value] = i

  # ランキングに保存するキーは再帰探索中の可変バッファと共有しない。
  proc snapshotPattern(value: seq[int]): seq[int] =
    result = newSeq[int](value.len)
    for i in 0..<value.len:
      result[i] = value[i]

  proc updateRank(
    bucket: RankBucket,
    value: seq[int],
    score: uint32,
    lim: int
  ) =
    if lim <= 0:
      return

    let n = bucket.ranks.len

    # 満杯で最下位以下なら即棄却。
    # 同点の既存候補も更新不要。
    if n >= lim and score <= bucket.ranks[0].rank:
      return

    let idx = bucket.positions.getOrDefault(value, -1)

    if idx >= 0:
      let old = bucket.ranks[idx].rank

      if score > old:
        bucket.ranks[idx].rank = score
        bucket.siftDownFast(idx)

      return

    if n < lim:
      bucket.ranks.add(
        RankEntry(value: snapshotPattern(value), rank: score)
      )
      bucket.siftUpFast(n)
      return

    bucket.positions.del(bucket.ranks[0].value)

    bucket.ranks[0] = RankEntry(
      value: snapshotPattern(value),
      rank: score
    )
    bucket.siftDownFast(0)

  # ============================================================
  # 2. Shape の識別
  #
  # 例:
  # [2, 2]    = AB_CD
  # [3, 1, 2] = ABC_D_EF
  #
  # 各長さは 1..6 なので 7 進符号化。
  # ============================================================

  proc shapeKey(lens: seq[int]): uint64 =
    var key = 0'u64
    for n in lens:
      key = key * uint64(cfg.maxGram + 1) + uint64(n)
    result = key

  proc shapeText(lens: seq[int]): string =
    result = ""
    for i, n in lens:
      if i > 0:
        result.add("_")
      result.add($n)

  # ============================================================
  # 3. wildcard 距離の制約
  #
  # k=2:
  #   ab <= N/ln(N)
  #   OR a,b <= sqrt(N)
  #
  # k=3:
  #   abc <= 2N/ln(N)^2
  #   OR a,b,c <= N^(1/3)
  #
  # 一般化:
  #   product <= (k-1)! N / ln(N)^(k-1)
  #   OR all gaps <= N^(1/k)
  # ============================================================

  proc factorial(n: int): float64 =
    result = 1.0
    for i in 2..n:
      result *= float64(i)

  proc productLimit(
    n: int,
    k: int
  ): float64 =
    if k <= 1:
      return Inf

    let nf = float64(max(n, 2))
    let logn = ln(nf)

    result =
      factorial(k - 1) * nf /
      pow(logn, float64(k - 1))

  proc rootLimit(
    n: int,
    k: int
  ): int =
    if k <= 0:
      return n

    let nf = float64(max(n, 1))
    result = int(
      floor(pow(nf, 1.0 / float64(k)))
    )

    # 浮動小数点誤差の補正
    while pow(float64(result + 1), float64(k)) <= nf:
      inc result

    while result > 0 and
          pow(float64(result), float64(k)) > nf:
      dec result

  # ============================================================
  # 4. 1 件の候補を Count-Min Sketch に追加
  # ============================================================

  proc recordPattern(
    bucket: RankBucket,
    pattern: seq[int],
    completedHash: uint64
  ) =
    # No repeated whole-pattern hashing or double CMS row accesses.
    # Every enumerated occurrence still increments the same counters in
    # precisely the same order as before.
    inc enumerated
    let score = cms.addHashedAndEstimate(completedHash)
    bucket.updateRank(pattern, score, capacity)

  # ============================================================
  # 5. wildcard の長さを再帰的に列挙
  #
  # part       : 次に追加する gram のインデックス
  # prevEnd    : 直前の gram の終端（exclusive）
  # product    : これまでの gap 長の積
  # allShort   : すべての gap が rootLimit 以下か
  #
  # 不成立が確定した枝は早期棄却。
  # ============================================================

  proc enumerateSuffix(
    bucket: RankBucket,
    lens: seq[int],
    offsets: seq[int],
    part: int,
    prevEnd: int,
    product: float64,
    allShort: bool,
    prodLim: float64,
    rootLim: int,
    prefixHash: uint64,
    pattern: var seq[int]
  ) =
    if part >= lens.len:
      bucket.recordPattern(pattern, finishArrayHash(prefixHash))
      return

    let k = lens.len - 1

    var remainingChars = 0
    for t in part..<lens.len:
      remainingChars += lens[t]

    let remainingGaps = lens.len - part
    let maxAllowedGap =
      a.len - prevEnd - remainingChars - (remainingGaps - 1)
    if maxAllowedGap < 1:
      return

    var maxGap = maxAllowedGap
    if k >= 2:
      var byProduct = 0
      if product <= prodLim:
        byProduct = int(min(float64(maxGap), floor(prodLim / product)))

      var byRoot = 0
      if allShort:
        byRoot = min(maxGap, rootLim)
      maxGap = min(maxGap, max(byProduct, byRoot))

    if maxGap < 1:
      return

    let gramLen = lens[part]
    let offset = offsets[part]
    pattern[offset - 1] = -1
    let markedHash = extendArrayHash(prefixHash, -1)

    # Approximate miner proposal: explore geometrically spaced gap lengths
    # rather than enumerating O(L^wildcards) occurrences. Full exhaustive
    # enumeration is still available with --mine-gap-samples:0.
    # This changes candidate *proposal* frequencies, never exact matching,
    # training targets, or the subsequent validation procedure.
    let samples = (if cfg.gapSamples == 0: maxGap
                   else: min(cfg.gapSamples, maxGap))
    var lastGap = 0
    for sample in 0..<samples:
      var gap = sample + 1
      if samples < maxGap:
        # Always include 1..3 when there is room, then stratify remaining
        # gaps logarithmically. The small deterministic jitter depends on
        # the source prefix: across starts we visit *different* long gaps
        # instead of permanently excluding e.g. gaps 4..40.
        if samples <= 4 and sample == samples-1 and samples > 1:
          gap = maxGap
        elif samples >= 5 and sample >= 3:
          let groups = samples-3
          let jitter = float64((prefixHash shr (sample mod 16)) and 1023'u64) / 1024.0
          let ratio = (float64(sample-3) + jitter) / float64(groups)
          gap = max(4,min(maxGap,int(round(exp(ln(4.0) +
              (ln(float64(maxGap))-ln(4.0))*ratio)))))
          gap = min(maxGap,max(lastGap+1,gap))
      if gap == lastGap: continue
      lastGap = gap
      let newProduct = product * float64(gap)
      let newAllShort = allShort and (gap <= rootLim)
      if k >= 2 and newProduct > prodLim and not newAllShort:
        continue

      let startPos = prevEnd + gap
      var nextHash = markedHash
      for t in 0..<gramLen:
        let value = a[startPos + t]
        pattern[offset + t] = value
        nextHash = extendArrayHash(nextHash, value)

      enumerateSuffix(
        bucket, lens, offsets, part + 1,
        startPos + gramLen, newProduct, newAllShort,
        prodLim, rootLim, nextHash, pattern
      )

  # ============================================================
  # 6. ある Shape 全体を探索
  # ============================================================

  proc processShape(lens: seq[int]) =
    if lens.len == 0:
      return

    let gramTotal = lens.foldl(a + b, 0)
    let k = lens.len - 1

    if gramTotal > cfg.maxTotalGram:
      return

    if k > cfg.maxWildcards:
      return

    # すべての gram のほかに
    # wildcard ごとに最低 1 文字飛ばす。
    if gramTotal + k > a.len:
      return

    let key = shapeKey(lens)

    if not buckets.hasKey(key):
      buckets[key] = newRankBucket()
    let bucket = buckets[key]

    let prodLim = productLimit(a.len, k)
    let rootLim = rootLimit(a.len, k)

    let firstLen = lens[0]
    let maxStart = a.len - gramTotal - k
    let startedAt = epochTime()
    let enumeratedBefore = enumerated
    if cfg.verbose: stderr.writeLine("[start] Shape ", shapeText(lens),
                     " (", maxStart + 1, " starting positions; no search cutoff)")

    # Fixed length/offsets; reuse one mutable pattern buffer for every start.
    # Only the rare top-ranked patterns are copied by updateRank.
    let finalLength = gramTotal + k
    var offsets = newSeq[int](lens.len)
    var curOffset = 0
    for i in 0..<lens.len:
      if i > 0: inc curOffset # wildcard immediately before this gram
      offsets[i] = curOffset
      curOffset += lens[i]
    var pattern = newSeq[int](finalLength)
    let hashSeed = cms.beginArrayHash(finalLength)

    # Approximate mining when stride > 1: sample one rotating residue per
    # shape, reducing enumeration without altering wildcard/gap constraints.
    # stride=1 restores complete, identical enumeration.
    let startOffset = (if cfg.startStride == 1: 0
                       else: min(maxStart, int(key mod uint64(cfg.startStride))))
    for start in countup(startOffset, maxStart, cfg.startStride):
      if (start and 63) == 0 and epochTime()-lastProgress >= 10.0:
        lastProgress = epochTime()
        stderr.writeLine("[mining] Shape ", shapeText(lens), " starts=", start, "/", maxStart+1)
      if (start and 1023) == 0 and start > 0:
        if cfg.verbose: stderr.writeLine("[progress] Shape ", shapeText(lens),
                         " starts=", start, "/", maxStart + 1,
                         " elapsed=", int(epochTime() - startedAt), "s")

      var prefixHash = hashSeed
      for t in 0..<firstLen:
        let v = a[start + t]
        pattern[t] = v
        prefixHash = extendArrayHash(prefixHash, v)

      if k == 0:
        bucket.recordPattern(pattern, finishArrayHash(prefixHash))
      else:
        enumerateSuffix(
          bucket, lens, offsets, 1, start + firstLen,
          1.0, true, prodLim, rootLim, prefixHash, pattern
        )

    if cfg.verbose: stderr.writeLine("[done] Shape ", shapeText(lens),
                     " elapsed=", int(epochTime() - startedAt), "s",
                     " occurrences=", enumerated - enumeratedBefore,
                     " ranked=", bucket.ranks.len)

  # ============================================================
  # 7. すべての Shape を生成
  #
  # 各 gram は cfg.minGram..cfg.maxGram 文字。
  # gram 長の合計は cfg.maxTotalGram 以下。
  # wildcard は最大 cfg.maxWildcards 個。
  # ============================================================

  proc generateShapes(
    lens: var seq[int],
    total: int
  ) =
    if lens.len > 0 and total + lens.len - 1 <= maxChunkLength:
      shapes.add(snapshotPattern(lens))

    if lens.len >= cfg.maxWildcards + 1:
      return

    for n in cfg.minGram..cfg.maxGram:
      if total + n > cfg.maxTotalGram:
        break

      lens.add(n)
      generateShapes(lens, total + n)
      lens.setLen(lens.len - 1)

  var shape: seq[int] = @[]
  generateShapes(shape, 0)
  if shapes.len == 0: return
  if capacity <= 0:
    capacity = max(1, totalLength div shapes.len)
  stderr.writeLine("[budget] len=", totalLength, " shapes=", shapes.len,
    " perShape=", capacity, " maxCandidates=", capacity * shapes.len,
    " gapSamples=", cfg.gapSamples)
  for chunk in chunks:
    a = chunk
    for lens in shapes:
      processShape(lens)
  var keys = toSeq(buckets.keys)
  keys.sort()
  for key in keys:
    for entry in buckets[key].ranks:
      result.add(entry)
  result.sort(proc(x, y: PatternEntry): int =
    let c = cmp(y.rank, x.rank)
    if c != 0: return c
    for i in 0..<min(x.value.len, y.value.len):
      let d = cmp(x.value[i], y.value[i])
      if d != 0: return d
    cmp(x.value.len, y.value.len))
  stderr.writeLine("[mining complete] occurrences=", enumerated,
    " ranked=", result.len, " elapsed=", epochTime()-miningStartedAt, "s")
