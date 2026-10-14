
import std/[tables, hashes, os, streams, algorithm, math, strutils, sequtils, times]
import fast_count_min_sketch

const
  MIN_NGRAM = 2  # 各 N-gram（各連続区間）の最小文字数
  MAX_GRAM = 4
  MAX_TOTAL_GRAM = 8
  MAX_WILDCARDS = 3


type
  RankEntry = object
    value: seq[int]
    rank: uint32

  RankBucket = ref object
    ranks: seq[RankEntry]
    positions: Table[seq[int], int]

  Shape = object
    lens: seq[int]
    key: uint64

var
  buckets = initTable[uint64, RankBucket]()
  sourceChunks: seq[seq[int]] = @[]

  cms = initCountMinSketch(
    width = 262144,
    depth = 7
  )

proc loadTrainingCorpus(n: int) =
  var f = open("github-code.txt", fmRead)
  defer: f.close()

  var j = 0
  while not f.endOfFile:
    var txt = ""
    while not f.endOfFile:
      let line = f.readLine()
      if line == "===SPLIT===":
        break
      txt.add(line)
      txt.add("\n")

    if txt.len == 0:
      continue

    var full = newSeq[int](txt.len)
    for idx, c in txt:
      full[idx] = ord(c)

    sourceChunks.add(full)
    inc j
    if j > n:
      break

  if sourceChunks.len == 0:
    quit("No source chunks found.")

loadTrainingCorpus(1)

let a = sourceChunks[0]
let LEN = a.len

# ============================================================
# 1. ランキング管理
# ============================================================

proc newRankBucket(): RankBucket =
  new(result)
  result.ranks = @[]
  result.positions = initTable[seq[int], int]()

proc siftUpFast(
  bucket: RankBucket,
  start: int
) {.inline.} =
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
) {.inline.} =
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
proc snapshotPattern(value: seq[int]): seq[int] {.inline.} =
  result = newSeq[int](value.len)
  for i in 0..<value.len:
    result[i] = value[i]

proc updateRank(
  bucket: RankBucket,
  value: seq[int],
  score: uint32,
  lim: int
) {.inline.} =
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
    key = key * 7'u64 + uint64(n)
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
  pattern: seq[int]
) {.inline.} =
  cms.add(pattern)
  let score = cms.estimate(pattern)
  bucket.updateRank(pattern, score, LEN)

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
  part: int,
  prevEnd: int,
  product: float64,
  allShort: bool,
  prodLim: float64,
  rootLim: int,
  pattern: var seq[int]
) =
  if part >= lens.len:
    bucket.recordPattern(pattern)
    return

  let k = lens.len - 1

  var remainingChars = 0
  for t in part..<lens.len:
    remainingChars += lens[t]

  # この後の gap は現在のものを含めて
  # lens.len - part 個必要。
  let remainingGaps = lens.len - part

  let maxAllowedGap =
    a.len - prevEnd -
    remainingChars -
    (remainingGaps - 1)

  if maxAllowedGap < 1:
    return

  var maxGap = maxAllowedGap

  # product と allShort の両条件が壊れる
  # 距離より先を探索しない。
  #
  # k=1 は条件を設けない。
  if k >= 2:
    var byProduct = 0
    if product <= prodLim:
      byProduct = int(min(
        float64(maxGap),
        floor(prodLim / product)
      ))

    var byRoot = 0
    if allShort:
      byRoot = min(maxGap, rootLim)

    maxGap = min(
      maxGap,
      max(byProduct, byRoot)
    )

  if maxGap < 1:
    return

  let oldLen = pattern.len
  let gramLen = lens[part]

  # 同じ深さでは同じ長さのパターンを使い回し、
  # 各 gap ごとの seq 拡張・縮小を避ける（探索対象は変えない）。
  pattern.setLen(oldLen + 1 + gramLen)
  pattern[oldLen] = -1

  for gap in 1..maxGap:
    let newProduct = product * float64(gap)
    let newAllShort = allShort and (gap <= rootLim)

    if k >= 2:
      if newProduct > prodLim and not newAllShort:
        continue

    let startPos = prevEnd + gap

    # wildcard token はループ前に設定済み。
    for t in 0..<gramLen:
      pattern[oldLen + 1 + t] = a[startPos + t]

    enumerateSuffix(
      bucket,
      lens,
      part + 1,
      startPos + gramLen,
      newProduct,
      newAllShort,
      prodLim,
      rootLim,
      pattern
    )

  pattern.setLen(oldLen)

# ============================================================
# 6. ある Shape 全体を探索
# ============================================================

proc processShape(lens: seq[int]) =
  if lens.len == 0:
    return

  let gramTotal = lens.foldl(a + b, 0)
  let k = lens.len - 1

  if gramTotal > MAX_TOTAL_GRAM:
    return

  if k > MAX_WILDCARDS:
    return

  # すべての gram のほかに
  # wildcard ごとに最低 1 文字飛ばす。
  if gramTotal + k > a.len:
    return

  let key = shapeKey(lens)

  var bucket = newRankBucket()
  buckets[key] = bucket

  let prodLim = productLimit(a.len, k)
  let rootLim = rootLimit(a.len, k)

  let firstLen = lens[0]
  let maxStart = a.len - gramTotal - k
  let startedAt = epochTime()
  stderr.writeLine("[start] Shape ", shapeText(lens),
                   " (", maxStart + 1, " starting positions; no search cutoff)")

  for start in 0..maxStart:
    # 進捗出力は外側ループのみ。列挙・集計結果には影響しない。
    if (start and 1023) == 0 and start > 0:
      stderr.writeLine("[progress] Shape ", shapeText(lens),
                       " starts=", start, "/", maxStart + 1,
                       " elapsed=", int(epochTime() - startedAt), "s")
    var pattern = newSeqOfCap[int](
      gramTotal + k
    )

    for t in 0..<firstLen:
      pattern.add(a[start + t])

    if k == 0:
      bucket.recordPattern(pattern)
    else:
      enumerateSuffix(
        bucket,
        lens,
        1,
        start + firstLen,
        1.0,
        true,
        prodLim,
        rootLim,
        pattern
      )

  stderr.writeLine("[done] Shape ", shapeText(lens),
                   " elapsed=", int(epochTime() - startedAt), "s",
                   " ranked=", bucket.ranks.len)

# ============================================================
# 7. すべての Shape を生成
#
# 各 gram は MIN_NGRAM..MAX_GRAM 文字。
# gram 長の合計は MAX_TOTAL_GRAM 以下。
# wildcard は最大 MAX_WILDCARDS 個。
# ============================================================

proc generateShapes(
  lens: var seq[int],
  total: int
) =
  if lens.len > 0:
    processShape(lens)

  if lens.len >= MAX_WILDCARDS + 1:
    return

  for n in MIN_NGRAM..MAX_GRAM:
    if total + n > MAX_TOTAL_GRAM:
      break

    lens.add(n)
    generateShapes(lens, total + n)
    lens.setLen(lens.len - 1)

# ============================================================
# 8. 全探索
# ============================================================

var shape: seq[int] = @[]
generateShapes(shape, 0)

# ============================================================
# 9. ランキング出力
# ============================================================

proc patternText(value: seq[int]): string =
  result = newStringOfCap(value.len)

  for x in value:
    if x == -1:
      result.add("_")
    elif x >= 0 and x <= 255:
      result.add(char(x))
    else:
      result.add("[" & $x & "]")

var keys = toSeq(buckets.keys)
keys.sort()

for key in keys:
  let bucket = buckets[key]

  bucket.ranks.sort(
    proc(x, y: RankEntry): int =
      cmp(y.rank, x.rank)
  )

  echo ""
  echo "=================================="
  echo "Shape key: ", key
  echo "Entries: ", bucket.ranks.len
  echo "=================================="

  for entry in bucket.ranks:
    echo entry.rank, " ", patternText(entry.value)
