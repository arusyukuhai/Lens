# Single-trajectory replacement GA (v16): one mutable text per evaluation.
# Remove evolved branch mask/coefficient, forked frontiers and mode mutations.
# Preserve ordered replacements, sqrt(input length) rounds, two-gram termination,
# exact feature/credit accumulation, length limits and candidate indexing.
# Checkpoint v3..v9/v13..v22 and energy models V1..V8 remain readable;
# legacy branching alleles are discarded on load. v17/V4 has no branch fields.
# A migrated genome is reevaluated under the new objective; old scores reset.
# v17: nine cases, sparse exact evaluation and log-uniform mutation counts.
# v20: one-token 256->256 evolvable byte permutation; legacy v18/v19
#       octal checkpoints are decoded; avoid threefold evaluation expansion.
# Strict matcher/copy bounds and capture bounds validation are retained.
# v21: capture +1/-1/*2 operate modulo 256 on encoded token IDs.
# v22: 256 raw bytes injected into a 1024-symbol internal space.
# v23: shrink that space to 512 symbols (256 mapped + 256 latent); arithmetic
#      closes modulo 512 and the sparse two-token candidate index is retained.
# v24-opt: dedupe pair buckets before lookup, reuse per-thread sample memo tables,
#      precompute candidate anchors, incremental diversity selection, no-op retry,
#      and move the expensive Jev/fusion cadence from 5 to 36 generations.
# v25-genetic-opt: make archive injection stagnation-adaptive instead of flooding
#      every generation; reduce compulsory diagnostic FULL probes; deduplicate
#      parent genotypes; adapt crossover/asexual/distant-mating pressure to
#      stagnation/diversity; and reject duplicate offspring before evaluation.
# v26-precision-stable: synchronize Jev/fusion to 36 generations; add a paired
#      three-regime mutation bandit (local/balanced/explore), conservative
#      cross-case parent scoring, stagnation-triggered credit refresh, and
#      directed hypermutation restarts so evaluation budget is spent on more
#      informative offspring without changing the FULL fitness objective.
# Build: nim c -d:release --threads:on at.nim
# Verify: ./at --regression-test && ./at --self-test

import sequtils
import std/random
import std/math
import std/times
import std/complex
import std/algorithm
import std/tables
import std/hashes
import std/os
import std/streams
import std/strutils
import progress
import std/threadpool
import std/cpuinfo
import std/sets
import std/json
import std/unicode as unicode

# Weight sign is a discrete semantic property: positive/negative rules have
# opposite effects.  Continuous/local optimizers are therefore not allowed to
# cross zero accidentally.  Sign changes, when desired, are handled by an
# explicit rare mutation below.
# Upper bound passed by all ordinary callers. The ACTUAL per-input number of
# sweeps is ceil(sqrt(initial input length)), computed once in scoreRaw.
# An explicit maxRounds=1 still runs exactly one sweep for comparison tests.
const SCORE_ROUNDS = high(int)
const WEIGHT_ABS_LIMIT = 32.0
const WEIGHT_SIGN_FLOOR = 1.0e-6

proc clampWeight(w: float): float {.inline.} =
  if w != w:
    return 0.0
  max(-WEIGHT_ABS_LIMIT, min(WEIGHT_ABS_LIMIT, w))

proc signPreservingWeight(oldWeight, proposedWeight: float): float {.inline.} =
  ## Keep a rule on the same side of zero during continuous tuning.
  ## If an additive proposal overshoots zero, stop just before zero instead of
  ## silently turning an excitatory rule into an inhibitory one (or vice versa).
  let p = clampWeight(proposedWeight)
  if oldWeight > 0.0:
    max(WEIGHT_SIGN_FLOOR, p)
  elif oldWeight < 0.0:
    min(-WEIGHT_SIGN_FLOOR, p)
  else:
    # Exact zero has no sign. Keep it disabled until an explicit structural/sign
    # mutation chooses a direction rather than letting numerical noise do it.
    0.0

proc oppositeNonZeroSign(a, b: float): bool {.inline.} =
  (a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0)

# ------------------------------------------------------------
# ★このファイルは元のGAスクリプトに「世代内の個体評価の並列化」を
# 追加したものです。コンパイル時に必ず --threads:on を付けてください。
#
#   nim c -d:release --threads:on ga_parallel.nim
#
# (--threads:on を付け忘れると spawn/FlowVar がコンパイルエラーになります)
#
# ★速度優先ビルド(動作確認が済んだら推奨):
#   -d:danger は -d:release よりさらに境界チェック等の実行時チェックを
#   外すため、添字アクセスが大量にあるこのコード(replaceSeqCompiledInto
#   や scanCandidateIds など)では体感できる差が出ます。
#   --mm:orc は既定の refc よりマルチスレッド下でのアロケータ性能が
#   良いことが多く、あわせて指定する価値があります。
#   ただし -d:danger は不正な添字アクセス等を検出しなくなるので、
#   まず -d:release のまま十分な世代を回して問題が出ないことを
#   確認してから切り替えてください。
#
#   nim c -d:release -d:danger --mm:orc --threads:on ga_parallel.nim
#
# ★v13で以下を追加:
#   5. Rule / Genome の型定義を前方へ移動し、getReplacementPlanPtr の
#      「宣言前の Genome 参照」によるコンパイルエラーを修正。
#   1. evaluateIndividual 内の作業列 aggggg に長さ上限を設け、
#      置換が膨張し続ける「暴走個体」を検出したら即座に打ち切る。
#      (gen=4以降でtqdmが0.00%のまま固まる主因と推測されるもの)
#   2. FlowVar の回収を jp=0 から順番にブロッキング待ちする方式から、
#      isReady によるポーリング方式に変更。これにより
#        - どの個体が終わらず残っているかを一定間隔でログできる
#        - 一部が重くても、終わった個体から進捗バーが進む
#   3. 毎世代、ファイルから読んだチャンク長 (txt.len) をログする。
#      チャンクサイズのばらつきが原因である可能性も切り分けられるように。
#
# ★3回目の修正(今回)で以下を追加:
#   4. evaluateIndividual / evaluateIndividualCases が呼ばれるたびに
#      newEvalScratch() で pairSeen bitsetを含む作業領域を
#      新規確保・ゼロ初期化していたのを、getEvalScratchによる
#      スレッドローカル再利用に変更。中身は元々 stamp ベースで
#      使い回せる設計だったため、スレッド単位で1回だけ確保すれば
#      個体をまたいでも安全に再利用できる。世代あたり数百回の
#      不要な大きなアロケーション/ゼロ埋めを削減する。
# ------------------------------------------------------------

type
  SeqReplaceError* = object of CatchableError

  LiteralPart = object
    data: seq[int]
    prefix: seq[int]

  SeqPattern* = object
    parts: seq[LiteralPart]
    wildcardCount*: int
    nonEmptyPartCount: int
    hasLeadingLiteral: bool
    allWildcard: bool
    emptyPattern: bool
    # Candidate-index anchor is derived once at compile time. Rebuilding an
    # index for a child must not rescan every literal part of every rule.
    anchorFirst: int
    anchorSecond: int
    anchorLen: int

  Capture = object
    start: int
    len: int

  # Rule / Genome は getReplacementPlanPtr より前に宣言する必要がある。
  # v12ではこの型定義がファイル後半に移動していたため、Nimの
  # 「宣言前の型名参照」により Genome undeclared が発生していた。
  Rule = object
    a: seq[int]
    b: seq[int]
    weight: float
    patternRevision: uint64
    replacementRevision: uint64
    # Only genome[0] owns the mapping; ordinary rules do not duplicate it.
    # 256 input bytes inject into 512 internal token IDs. The unused IDs are
    # genuine latent/work symbols available to the rewrite system.
    embedding: seq[int]

  Genome = seq[Rule]

# External data remains byte-oriented, but the rewrite machine operates in a
# larger discrete latent space. `embedding` is a 256-entry injection into the
# 512-symbol internal vocabulary.  Exactly half of the vocabulary is therefore
# initially free for latent/work symbols; 512 itself is the OOV sentinel.
const INPUT_BYTE_COUNT = 256
const EMBEDDING_ENTRY_COUNT = INPUT_BYTE_COUNT # dictionary entries: one per raw byte
const EMBEDDING_SIZE = 512                     # internal token vocabulary 0..511
const EMBEDDING_WIDTH = 1
const EMBEDDING_CODE_COUNT = EMBEDDING_SIZE
const EMBEDDING_OOV = EMBEDDING_CODE_COUNT     # sentinel 512
const INDEX_TOKEN_COUNT = EMBEDDING_CODE_COUNT + 1
# v22 used a 1024-symbol internal vocabulary. It remains loadable through an
# explicit lossy latent-space migration which preserves all 256 byte identities.
const V22_EMBEDDING_CODE_COUNT = 1024
const V22_EMBEDDING_OOV = V22_EMBEDDING_CODE_COUNT
const LEGACY_OCTAL3_CODE_COUNT = 512
const LEGACY_OCTAL4_CODE_COUNT = 4096

proc defaultEmbedding(): seq[int] =
  ## Backward-compatible aligned start: bytes initially occupy tokens 0..255;
  ## 256..511 begin as free latent coordinates.
  result = newSeq[int](EMBEDDING_ENTRY_COUNT)
  for i in 0 ..< EMBEDDING_ENTRY_COUNT: result[i] = i

proc cloneEmbedding(e: openArray[int]): seq[int] =
  result = newSeq[int](e.len)
  for i, v in e: result[i] = v

proc validEmbedding(e: openArray[int]): bool =
  ## Injection: exactly 256 distinct destinations chosen from 512 tokens.
  if e.len != EMBEDDING_ENTRY_COUNT: return false
  var seen: array[EMBEDDING_CODE_COUNT, bool]
  for code in e:
    if code < 0 or code >= EMBEDDING_CODE_COUNT or seen[code]: return false
    seen[code] = true
  true

proc validLegacyBytePermutation(e: openArray[int]): bool =
  ## v20/v21 stored a strict permutation of 0..255. Validate that old contract
  ## before widening it into the v22 injection space.
  if e.len != EMBEDDING_ENTRY_COUNT: return false
  var seen: array[INPUT_BYTE_COUNT, bool]
  for code in e:
    if code < 0 or code >= INPUT_BYTE_COUNT or seen[code]: return false
    seen[code] = true
  true

proc validV22Embedding(e: openArray[int]): bool =
  ## v22 stored a 256-entry injection into 0..1023.
  if e.len != EMBEDDING_ENTRY_COUNT: return false
  var seen: array[V22_EMBEDDING_CODE_COUNT, bool]
  for code in e:
    if code < 0 or code >= V22_EMBEDDING_CODE_COUNT or seen[code]: return false
    seen[code] = true
  true

proc migrateV22Embedding(oldMap: openArray[int]): tuple[mapping, translation: seq[int]] =
  ## Shrink 1024 -> 512 without ever aliasing a raw byte with another raw byte.
  ## 256 byte identities remain injective. The 768 historical latent symbols
  ## necessarily fold onto the 256 new latent coordinates; they are NEVER folded
  ## onto byte-owned coordinates, so the byte/latent distinction is preserved.
  if not validV22Embedding(oldMap):
    raise newException(IOError, "Invalid v22 1024-token embedding.")
  result.mapping = newSeq[int](EMBEDDING_ENTRY_COUNT)
  result.translation = newSeq[int](V22_EMBEDDING_CODE_COUNT + 1)
  result.translation.fill(-1)
  var occupied: array[EMBEDDING_CODE_COUNT, bool]
  for b in 0 ..< EMBEDDING_ENTRY_COUNT:
    var target = oldMap[b] mod EMBEDDING_CODE_COUNT
    if occupied[target]:
      let start = target
      while occupied[target]:
        target = (target + 1) mod EMBEDDING_CODE_COUNT
        if target == start:
          raise newException(IOError, "No free token while migrating v22 embedding.")
    result.mapping[b] = target
    occupied[target] = true
    result.translation[oldMap[b]] = target
  var latentTargets = newSeqOfCap[int](EMBEDDING_CODE_COUNT - EMBEDDING_ENTRY_COUNT)
  for token in 0 ..< EMBEDDING_CODE_COUNT:
    if not occupied[token]: latentTargets.add(token)
  if latentTargets.len == 0:
    raise newException(IOError, "No latent targets while migrating v22 embedding.")
  var latentOrdinal = 0
  for token in 0 ..< V22_EMBEDDING_CODE_COUNT:
    if result.translation[token] < 0:
      result.translation[token] = latentTargets[latentOrdinal mod latentTargets.len]
      inc latentOrdinal
  result.translation[V22_EMBEDDING_OOV] = EMBEDDING_OOV
  doAssert validEmbedding(result.mapping)

proc embeddingContains(e: openArray[int], token: int): bool {.inline.} =
  if token < 0 or token >= EMBEDDING_CODE_COUNT: return false
  for code in e:
    if code == token: return true
  false

proc sampleUnusedEmbeddingCode(e: openArray[int]): int =
  ## Half of the 512-space is free, so rejection normally succeeds quickly.
  ## A deterministic bounded fallback keeps this total even for malformed callers.
  if e.len != EMBEDDING_ENTRY_COUNT: return rand(EMBEDDING_CODE_COUNT - 1)
  for _ in 0 ..< 16:
    let candidate = rand(EMBEDDING_CODE_COUNT - 1)
    if not embeddingContains(e, candidate): return candidate
  let start = rand(EMBEDDING_CODE_COUNT - 1)
  for d in 0 ..< EMBEDDING_CODE_COUNT:
    let candidate = (start + d) mod EMBEDDING_CODE_COUNT
    if not embeddingContains(e, candidate): return candidate
  raise newException(ValueError, "embedding has no unused internal token")

proc embeddingTranslation(fromMap, toMap: openArray[int]): array[EMBEDDING_CODE_COUNT, int] =
  ## Extend the 256 byte correspondences to a full 512-token permutation.
  ## This makes dictionary mutation/crossover a coordinate change for BOTH
  ## mapped and latent symbols, avoiding accidental collisions with a latent
  ## token when a byte moves into a previously unused coordinate.
  doAssert validEmbedding(fromMap) and validEmbedding(toMap)
  result.fill(-1)
  var targetTaken: array[EMBEDDING_CODE_COUNT, bool]
  for b in 0 ..< EMBEDDING_ENTRY_COUNT:
    result[fromMap[b]] = toMap[b]
    targetTaken[toMap[b]] = true
  # Preserve latent coordinates that remain free in both systems.
  for token in 0 ..< EMBEDDING_CODE_COUNT:
    if result[token] < 0 and not targetTaken[token]:
      result[token] = token
      targetTaken[token] = true
  # Pair only the displaced latent coordinates with the remaining free targets.
  var freeTargets = newSeqOfCap[int](EMBEDDING_CODE_COUNT - EMBEDDING_ENTRY_COUNT)
  for token in 0 ..< EMBEDDING_CODE_COUNT:
    if not targetTaken[token]: freeTargets.add(token)
  var ti = 0
  for token in 0 ..< EMBEDDING_CODE_COUNT:
    if result[token] < 0:
      doAssert ti < freeTargets.len
      result[token] = freeTargets[ti]
      inc ti
  doAssert ti == freeTargets.len

proc encodeLiteralRuns(x: openArray[int], e: openArray[int], maxLen = 64): seq[int] =
  ## Byte-level seeding and conversion of pre-embedding checkpoints.
  result = newSeqOfCap[int](min(maxLen, x.len))
  for v in x:
    if result.len >= maxLen: break
    let mapped = (if v < 0: v
                  elif v < INPUT_BYTE_COUNT: e[v]
                  else: EMBEDDING_OOV)
    result.add(mapped)

proc embedBytesInto(input: openArray[int], e: openArray[int], dst: var seq[int]) {.inline.} =
  ## Never expand text. Sentinel 512 is distinct from all internal tokens.
  dst.setLen(input.len)
  for i, v in input:
    dst[i] = (if v >= 0 and v < INPUT_BYTE_COUNT: e[v] else: EMBEDDING_OOV)

proc recodeLiteralRunsWithTranslation(
  x: openArray[int],
  translation: array[EMBEDDING_CODE_COUNT, int]
): seq[int] {.inline.} =
  result = newSeq[int](x.len)
  for i, v in x:
    result[i] = (if v >= 0 and v < EMBEDDING_CODE_COUNT:
                   translation[v]
                 else: v)

proc recodeLiteralRuns(x: openArray[int], fromMap, toMap: openArray[int]): seq[int] =
  ## Re-express an entire rule in another genome's internal coordinate system.
  ## Negative opcodes and OOV/out-of-domain values are never touched.
  let translation = embeddingTranslation(fromMap, toMap)
  recodeLiteralRunsWithTranslation(x, translation)

proc migrateLegacyMap(oldMap: openArray[int], oldCodeCount: int): seq[int] =
  ## Rank the old distinct codewords to obtain the aligned 0..255 subset of
  ## the new byte-to-latent injection. Default old identity maps remain identity.
  if oldMap.len != EMBEDDING_ENTRY_COUNT or
      oldCodeCount notin [LEGACY_OCTAL3_CODE_COUNT, LEGACY_OCTAL4_CODE_COUNT]:
    raise newException(IOError, "Invalid legacy embedding length/radix")
  var owner = newSeq[int](oldCodeCount)
  for i in 0 ..< oldCodeCount: owner[i] = -1
  for byte, code in oldMap:
    if code < 0 or code >= oldCodeCount - 1 or owner[code] != -1:
      raise newException(IOError, "Invalid legacy embedding code or collision")
    owner[code] = byte
  result = newSeq[int](EMBEDDING_ENTRY_COUNT)
  var rank = 0
  for byte in owner:
    if byte >= 0:
      result[byte] = rank
      inc rank
  doAssert rank == EMBEDDING_ENTRY_COUNT and validEmbedding(result)

proc migrateLegacyEncodedLiterals(
    x: openArray[int], oldMap, newMap: openArray[int],
    width, codeCount: int, maxLen = 64): seq[int] =
  ## Decode complete 3-/4-octal byte codewords to ONE mapped byte.
  ## A negative opcode breaks alignment; malformed fragments are retained
  ## separately, so no wildcard/sort/reverse/arithmetic opcode is decoded.
  var owner = newSeq[int](codeCount)
  for i in 0 ..< codeCount: owner[i] = -1
  for byte, code in oldMap: owner[code] = byte
  result = newSeqOfCap[int](min(maxLen, x.len))
  var i = 0
  while i < x.len and result.len < maxLen:
    if x[i] < 0:
      result.add(x[i])
      inc i
      continue
    let start = i
    while i < x.len and x[i] >= 0: inc i
    var j = start
    while j < i and result.len < maxLen:
      if j + width <= i:
        var code = 0
        var validDigits = true
        for k in 0 ..< width:
          if x[j+k] > 7:
            validDigits = false
            break
          code = (code shl 3) or x[j+k]
        if validDigits and owner[code] >= 0:
          result.add(newMap[owner[code]])
          j += width
          continue
        if validDigits and code == codeCount - 1:
          result.add(EMBEDDING_OOV)
          j += width
          continue
      # Sub-byte fragment / arithmetic-produced value: preserve best-effort.
      result.add(x[j])
      inc j

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
  ## -1以下をワイルドカードとしてコンパイル。
  result.anchorFirst = -1
  result.anchorSecond = -1
  result.anchorLen = 0
  var current: seq[int] = @[]
  for x in pattern:
    if x <= -1:
      result.parts.add(makeLiteral(current))
      current.setLen(0)
      inc result.wildcardCount
    else:
      current.add(x)
  result.parts.add(makeLiteral(current))

  result.nonEmptyPartCount = 0
  for part in result.parts:
    if part.data.len > 0:
      inc result.nonEmptyPartCount
      if part.data.len > result.anchorLen:
        result.anchorLen = part.data.len
        result.anchorFirst = part.data[0]
        result.anchorSecond = if part.data.len >= 2: part.data[1] else: -1

  result.hasLeadingLiteral =
    result.parts.len > 0 and result.parts[0].data.len > 0
  result.allWildcard =
    result.wildcardCount > 0 and result.parts.allIt(it.data.len == 0)
  result.emptyPattern =
    result.wildcardCount == 0 and
    result.parts.len == 1 and
    result.parts[0].data.len == 0


# ------------------------------------------------------------
# パターンコンパイル結果のメモ化キャッシュ
#
# GAのエリート個体はクローンされるだけで .a 列の中身は前世代と
# 完全に同一になる。crossover の子個体もほとんどの行は親から
# そのままコピーされる(交差点の外側)。そのため、同じ .a (seq[int])
# に対して compileSeqPattern を毎世代呼び直すのは大きな無駄になる。
#
# ここでは内容(seq[int])をキーにしたグローバルキャッシュを持ち、
# 既に計算済みならそれを再利用する。停滞世代(stagnationが大きい)
# ほど個体群の多様性が下がりヒット率が上がる。
#
# ★注意: これはグローバルな Table なのでスレッドセーフではない。
# そのため、個体の並列評価(evaluateIndividual)側では絶対に
# 触らず、コンパイル段階(メインスレッド・逐次)でのみ使用する。
#
# 無制限に肥大化しないよう、メインループ側で一定世代ごとに
# clear() している(下記 CACHE_CLEAR_INTERVAL 参照)。
# ------------------------------------------------------------
var patternCache = initTable[seq[int], SeqPattern]()

const PATTERN_CACHE_MAX_ENTRIES = 120_000
proc compileSeqPatternCached*(pattern: seq[int]): SeqPattern =
  if patternCache.hasKey(pattern):
    return patternCache[pattern]
  # Cache eviction never invalidates the independently retained compiled
  # patterns. Bound the cache instead of allowing 2000 generations of unique
  # mutations to consume unbounded memory and GC time.
  if patternCache.len >= PATTERN_CACHE_MAX_ENTRIES:
    patternCache.clear()
  result = compileSeqPattern(pattern)
  patternCache[pattern] = result

# Revision identifiers are assigned once per changed pattern and preserved by
# shallow clones/crossovers. Checking a uint64 first saves re-hashing the
# entire sequence when population order changes between generations.
const REVISION_CACHE_MAX_ENTRIES = 600_000
var revisionPatternCache = initTable[uint64, SeqPattern]()

proc compileSeqPatternByRevision(pattern: seq[int], revision: uint64): SeqPattern =
  if revision != 0'u64 and revisionPatternCache.hasKey(revision):
    return revisionPatternCache[revision]
  result = compileSeqPatternCached(pattern)
  if revision != 0'u64:
    if revisionPatternCache.len >= REVISION_CACHE_MAX_ENTRIES:
      revisionPatternCache.clear()
    revisionPatternCache[revision] = result


# ------------------------------------------------------------
# 高速候補インデックス
# ------------------------------------------------------------
# 1-byte だけで候補化すると github-code のような通常文字列では
# ほとんど全ルールが候補に戻ってしまう。そこで先頭2リテラルを
# sparse pair key にして候補を大幅に絞る。
# 先頭リテラルが1要素しかない/空の場合だけ1-byte/alwaysへ落とす。

type
  # The old 256-token evaluator used a dense pair table and therefore did two
  # array reads in this extremely hot path.  A generic Table here made the
  # a widened-token build dramatically slower because EVERY adjacent input pair paid
  # hashing/generic-table overhead.  Keep the index sparse, but use a compact
  # fixed-load-factor open-addressed map specialized for int pair keys.
  PairHeadIndex = object
    keys: seq[int32]       # -1 = empty; pairKey is always non-negative
    heads: seq[int32]      # linked-list head (rule id + 1)
    usedSlots: seq[int32]  # clear only occupied slots on rebuild
    mask: int
    count: int

  # One-token buckets remain dense (513 entries is tiny), while two-token
  # buckets are sparse. A dense 513^2 table per genome would still waste memory across 450 genomes despite only ~3100 rule anchors being used.
  CandidateIndex = ref object
    head1: array[INDEX_TOKEN_COUNT, int32]
    head2: PairHeadIndex
    densePairHeads: seq[int32] # optional, generation-local direct lookup
    next1: seq[int32]
    next2: seq[int32]
    alwaysCandidates: seq[int32]
    candidateRuleCount: int

proc pairKey(a, b: int): int {.inline.} =
  a * INDEX_TOKEN_COUNT + b

proc initPairHeadIndex(expectedEntries: int): PairHeadIndex =
  # <= 0.5 load keeps lookup to ~1-2 probes for the normal 3100-rule model.
  var cap = 32
  let wanted = max(1, expectedEntries) * 2
  while cap < wanted: cap = cap shl 1
  result.keys = newSeq[int32](cap)
  result.keys.fill(-1'i32)
  result.heads = newSeq[int32](cap)
  result.usedSlots = newSeqOfCap[int32](max(1, expectedEntries))
  result.mask = cap - 1
  result.count = 0

proc pairHeadSlot(h: PairHeadIndex, key: int): int {.inline.} =
  # pairKey <= 263,168. Multiplicative hashing avoids the regular 513 stride
  # clustering that plain `key and mask` would create with a power-of-two table.
  int((uint64(key) * 2654435761'u64) and uint64(h.mask))

proc pairHeadGet(h: PairHeadIndex, key: int): int32 {.inline.} =
  if h.keys.len == 0: return 0'i32
  var slot = pairHeadSlot(h, key)
  while true:
    let stored = h.keys[slot]
    if stored == -1'i32: return 0'i32
    if stored == int32(key): return h.heads[slot]
    slot = (slot + 1) and h.mask

proc pairHeadPut(h: var PairHeadIndex, key: int, head: int32) {.inline.} =
  var slot = pairHeadSlot(h, key)
  while true:
    let stored = h.keys[slot]
    if stored == -1'i32:
      h.keys[slot] = int32(key)
      h.heads[slot] = head
      h.usedSlots.add(int32(slot))
      inc h.count
      return
    if stored == int32(key):
      h.heads[slot] = head
      return
    slot = (slot + 1) and h.mask

proc pairHeadClear(h: var PairHeadIndex) {.inline.} =
  for rawSlot in h.usedSlots:
    let slot = int(rawSlot)
    h.keys[slot] = -1'i32
    h.heads[slot] = 0'i32
  h.usedSlots.setLen(0)
  h.count = 0

proc buildCandidateIndex(patterns: seq[SeqPattern]): CandidateIndex {.gcsafe.} =
  var r: CandidateIndex
  new(r)
  result = r
  result.candidateRuleCount = patterns.len
  result.head2 = initPairHeadIndex(patterns.len)
  result.next1 = newSeq[int32](patterns.len)
  result.next2 = newSeq[int32](patterns.len)

  for pid, pat in patterns:
    if pat.parts.len == 0:
      continue

    # A leading wildcard does NOT make the rule unconditional. Any later
    # nonempty literal is a necessary substring of every successful match.
    # Index the first such anchor without changing ordered matching semantics.
    # An empty/all-wildcard pattern, or >16 captures, can never fire.
    if pat.allWildcard or pat.emptyPattern or pat.wildcardCount > 16:
      continue
    if pat.anchorLen <= 0:
      continue
    if pat.anchorLen == 1:
      let c = pat.anchorFirst
      if c >= 0 and c <= EMBEDDING_OOV:
        result.next1[pid] = result.head1[c]
        result.head1[c] = (pid + 1).int32
      else:
        result.alwaysCandidates.add(pid.int32)
    elif pat.anchorFirst >= 0 and pat.anchorFirst <= EMBEDDING_OOV and
         pat.anchorSecond >= 0 and pat.anchorSecond <= EMBEDDING_OOV:
      let key = pairKey(pat.anchorFirst, pat.anchorSecond)
      result.next2[pid] = pairHeadGet(result.head2, key)
      pairHeadPut(result.head2, key, (pid + 1).int32)
    else:
      result.alwaysCandidates.add(pid.int32)


proc rebuildCandidateIndex(
  result: var CandidateIndex,
  patterns: seq[SeqPattern]
) {.gcsafe.} =
  ## Reuse both linked-list arrays and the sparse two-token table. The table
  ## stores only anchors that actually occur in rules, independent of the
  ## internal-token vocabulary size.
  if result.isNil:
    var r: CandidateIndex
    new(r)
    r.head2 = initPairHeadIndex(patterns.len)
    result = r

  result.densePairHeads.setLen(0)
  result.candidateRuleCount = patterns.len
  result.head1.fill(0)
  pairHeadClear(result.head2)

  result.next1.setLen(patterns.len)
  result.next2.setLen(patterns.len)
  result.next1.fill(0)
  result.next2.fill(0)
  result.alwaysCandidates.setLen(0)

  for pid, pat in patterns:
    if pat.parts.len == 0:
      continue

    # A leading wildcard does NOT make the rule unconditional. Any later
    # nonempty literal is a necessary substring of every successful match.
    # Index the first such anchor without changing ordered matching semantics.
    # An empty/all-wildcard pattern, or >16 captures, can never fire.
    if pat.allWildcard or pat.emptyPattern or pat.wildcardCount > 16:
      continue
    if pat.anchorLen <= 0:
      continue
    if pat.anchorLen == 1:
      let c = pat.anchorFirst
      if c >= 0 and c <= EMBEDDING_OOV:
        result.next1[pid] = result.head1[c]
        result.head1[c] = (pid + 1).int32
      else:
        result.alwaysCandidates.add(pid.int32)
    elif pat.anchorFirst >= 0 and pat.anchorFirst <= EMBEDDING_OOV and
         pat.anchorSecond >= 0 and pat.anchorSecond <= EMBEDDING_OOV:
      let key = pairKey(pat.anchorFirst, pat.anchorSecond)
      result.next2[pid] = pairHeadGet(result.head2, key)
      pairHeadPut(result.head2, key, (pid + 1).int32)
    else:
      result.alwaysCandidates.add(pid.int32)


proc generationCandidateIndex(index: CandidateIndex): CandidateIndex {.gcsafe.} =
  # Only active generation workers pay ~1 MiB each, NOT all 450 models.
  # Immutable buckets are shared; the direct table is local to this model job.
  new(result)
  result.head1 = index.head1
  result.head2 = index.head2
  result.next1 = index.next1
  result.next2 = index.next2
  result.alwaysCandidates = index.alwaysCandidates
  result.candidateRuleCount = index.candidateRuleCount
  result.densePairHeads = newSeq[int32](INDEX_TOKEN_COUNT * INDEX_TOKEN_COUNT)
  for rawSlot in index.head2.usedSlots:
    let slot = int(rawSlot)
    result.densePairHeads[int(index.head2.keys[slot])] = index.head2.heads[slot]


type
  # candidateIds は「現在の入力配列」にのみ依存する。
  # sampleごとのrule適用で入力が実際に変更されるため、変更がなかった
  # ラウンドでは再計算する必要がない。
  CandidateScratch = object
    ## Deduplicate buckets as well as rules. Repeated bytes must not traverse
    ## the same linked list O(text.len) times. `pairSeenBits` is only ~129 KiB
    ## for 513^2 possible pairs and turns the previous hash-table lookup in the
    ## innermost text scan back into two shifts + one array access.
    ruleSeen: seq[int32]
    byteSeen: seq[int32]
    pairSeenBits: seq[uint64]
    pairTouchedWords: seq[int32]
    stamp: int32
    candidateValid: bool

proc markCandidate(
  pid, minRuleIndex: int,
  outIds: var seq[int],
  scratch: var CandidateScratch
) {.inline.} =
  # A rewrite at rule k cannot revisit rows before k on this ordered sweep.
  # Do not sort or mark candidates which are no longer eligible.
  if pid >= minRuleIndex and scratch.ruleSeen[pid] != scratch.stamp:
    scratch.ruleSeen[pid] = scratch.stamp
    outIds.add(pid)

proc scanCandidateIds(
  index: CandidateIndex,
  text: openArray[int],
  outIds: var seq[int],
  scratch: var CandidateScratch,
  minRuleIndex: int = 0
) {.gcsafe.} =
  ## 先頭1/2要素の候補を抽出するhot path。
  ## 直前要素をprevとして保持し、text[i] と text[i+1] の二重ロードを避ける。
  outIds.setLen(0)
  # Pair deduplication is per scan. Clear only 64-bit words that were touched
  # last time; never memset the full 513^2 bit domain and never hash pairs.
  for rawWord in scratch.pairTouchedWords:
    scratch.pairSeenBits[int(rawWord)] = 0'u64
  scratch.pairTouchedWords.setLen(0)
  if scratch.stamp == high(int32):
    scratch.ruleSeen.fill(0)
    scratch.byteSeen.fill(0)
    scratch.stamp = 0
  inc scratch.stamp

  for pid in index.alwaysCandidates:
    markCandidate(pid.int, minRuleIndex, outIds, scratch)

  if outIds.len >= index.candidateRuleCount - minRuleIndex:
    scratch.candidateValid = true
    return

  let n = text.len
  if n == 0:
    scratch.candidateValid = true
    return

  var i = 0
  var prev = text[0]
  while true:
    if prev >= 0 and prev <= EMBEDDING_OOV and index.head1[prev] != 0 and
       scratch.byteSeen[prev] != scratch.stamp:
      scratch.byteSeen[prev] = scratch.stamp
      var p = index.head1[prev]
      while p != 0:
        let pid = (p - 1).int
        # Buckets are built by ascending pid and linked at the head, so the
        # chain is strictly descending. Earlier rows cannot be eligible.
        if pid < minRuleIndex: break
        markCandidate(pid, minRuleIndex, outIds, scratch)
        if outIds.len >= index.candidateRuleCount - minRuleIndex:
          scratch.candidateValid = true
          return
        p = index.next1[pid]

    inc i
    if i >= n:
      break

    let curr = text[i]
    if prev >= 0 and prev <= EMBEDDING_OOV and curr >= 0 and curr <= EMBEDDING_OOV:
      let key = pairKey(prev, curr)
      # Deduplicate BEFORE the bucket lookup. The old order still paid the
      # sparse-hash lookup for every occurrence of a repeated pair, and pairs
      # with no bucket were never marked at all. Semantics are unchanged: one
      # occurrence is sufficient to prove a necessary pair exists in `text`.
      let word = key shr 6
      let bit = 1'u64 shl (key and 63)
      if (scratch.pairSeenBits[word] and bit) != 0'u64:
        prev = curr
        continue
      if scratch.pairSeenBits[word] == 0'u64:
        scratch.pairTouchedWords.add(int32(word))
      scratch.pairSeenBits[word] = scratch.pairSeenBits[word] or bit
      let head = if index.densePairHeads.len > 0: index.densePairHeads[key]
                 else: pairHeadGet(index.head2, key)
      if head != 0:
        var q = head
        while q != 0:
          let pid = (q - 1).int
          if pid < minRuleIndex: break
          markCandidate(pid, minRuleIndex, outIds, scratch)
          if outIds.len >= index.candidateRuleCount - minRuleIndex:
            scratch.candidateValid = true
            return
          q = index.next2[pid]

    prev = curr

  scratch.candidateValid = true


type
  # 各 literal の「次の一致位置」だけを遅延計算する。
  # 全 occurrence を seq に保存せず、match の進行に合わせて必要な分だけ探す。
  LiteralCursor = object
    lastMatch: int       # 直前に返した一致位置。pos 以下なら再利用できる。
    nextSearch: int     # 次の非重複一致を探し始める位置。
    exhausted: bool

  # 置換1回分だけ使う作業領域。GC対象の一時seqを作らない。
  ReplacementOp = object
    kind: uint8       # 0 literal, 1 capture, 2 sort, 3 reverse, 4..7 arithmetic
    arg: uint8        # capture index 0..15
    value: int        # literal token when kind == 0

  ReplacementPlan = object
    ops: array[64, ReplacementOp]
    len: uint8
    allLiteral: bool

  ReplaceScratch = object
    # wildcardCount <= 16, hence parts.len <= 17. Fixed storage removes a
    # per-rule seq growth/setLen path from the replacement hot loop.
    cursors: array[17, LiteralCursor]
    captures: array[16, Capture]
    sortBuf: seq[int]

  # 1個体の評価中、case/sampleをまたいで再利用する作業領域。
  # pair dedupe is a compact reusable bitset; allocating it per sample would still
  # be wasteful, so the whole scratch remains thread-local/reused across cases.
  EvalScratch = object
    candidateIds: seq[int]
    candidateScratch: CandidateScratch
    replaceScratch: ReplaceScratch
    replacementPlans: seq[ReplacementPlan]
    replacementPlanRevision: seq[uint64]
    bufA: seq[int]
    bufB: seq[int]
    outYs: seq[float]
    # Reused per worker. evaluateChunk used to allocate a new Table for every
    # case of every genome; clear() retains the backing storage.
    sampleMemo: Table[seq[int], float]
    seenAttPairs: seq[tuple[a: float64, b: float64]]
    rankIndices: seq[int]
    rankValues: seq[float]
    creditContrib: seq[float]
    creditTouched: seq[int]
    creditStamp: seq[int]
    creditEpoch: int

proc newEvalScratch(ruleCount: int, outCap: int = 32): EvalScratch =
  let initialCap = max(1037, outCap)
  result.candidateIds = newSeqOfCap[int](ruleCount)
  result.candidateScratch = CandidateScratch(
    ruleSeen: newSeq[int32](ruleCount),
    byteSeen: newSeq[int32](INDEX_TOKEN_COUNT),
    pairSeenBits: newSeq[uint64]((INDEX_TOKEN_COUNT * INDEX_TOKEN_COUNT + 63) shr 6),
    pairTouchedWords: newSeqOfCap[int32](256),
    stamp: 0,
    candidateValid: false
  )
  result.replaceScratch = ReplaceScratch(sortBuf: @[])
  result.replacementPlans = newSeq[ReplacementPlan](ruleCount)
  result.replacementPlanRevision = newSeq[uint64](ruleCount)
  result.replacementPlanRevision.fill(high(uint64))
  result.bufA = newSeqOfCap[int](initialCap)
  result.bufB = newSeqOfCap[int](initialCap)
  result.outYs = newSeqOfCap[float](max(8, outCap))
  result.sampleMemo = initTable[seq[int], float](max(16, outCap * 2))
  result.seenAttPairs = @[]
  result.rankIndices = newSeqOfCap[int](max(8, outCap))
  result.rankValues = newSeqOfCap[float](max(8, outCap))
  result.creditContrib = newSeqOfCap[float](ruleCount)
  result.creditTouched = newSeqOfCap[int](min(ruleCount, 256))
  # creditStamp は「容量」だけでなく ruleCount 個の要素が必要。
  # newSeqOfCap は len=0 のままなので、k番目へ直接アクセスすると
  # gen=0 の credit refresh で IndexDefect になる。
  result.creditStamp = newSeq[int](ruleCount)
  result.creditEpoch = 0

# ------------------------------------------------------------
# スレッドローカル EvalScratch プール
#
# 従来は evaluateIndividual / evaluateIndividualCases が呼ばれる
# たびに newEvalScratch(ruleCount=AAA) を新規確保していた。
# replacementPlans (ReplacementPlan[64要素] × ruleCount ≈ 数百KB~1MB超)や
# candidateScratch.ruleSeen / creditStamp 等をルール数分持つため、
# 世代あたり(FAST評価 pop_size回 + FULL評価 fullIds.len回)だけ
# 巨大なアロケーション/ゼロ初期化を繰り返すコストが無視できない。
#
# ここでの再利用が安全な理由:
#   getReplacementPlanPtr (682行目) は ruleIndex 位置ではなく
#   Rule.replacementRevision という「グローバル単調増加のuint64」で
#   キャッシュの有効/無効を判定している。rev==0(未採番)は明示的に
#   常に無効扱いにしているため、scratchを別個体の評価に使い回しても、
#   revisionが一致しない限り必ず compileReplacementPlan で作り直され、
#   古い個体のplanを取り違えて再利用することはない。
#   ruleCount(=AAA)も実行中は不変なので配列サイズの齟齬も起きない。
#
# Nimのstd/threadpoolはワーカーOSスレッドを使い捨てず常駐させて
# ジョブを流し込むため、{.threadvar.} はスレッドの生存期間だけ保持され、
# 以後spawnされてくる別個体のジョブでも自然に再利用される。
#
# ★注意: もし将来 ruleCount(AAA)を実行中に変更する、または
# 複数の異なる評価対象(ruleCountが異なるgenome)を混在させるように
# 変更した場合は、下の ruleCount 一致チェックにより自動的に
# 再確保されるので安全だが、そうでない限りこの前提を崩さないこと。
# ------------------------------------------------------------
var tlEvalScratch {.threadvar.}: EvalScratch
var tlEvalScratchReady {.threadvar.}: bool
var tlEvalScratchRuleCount {.threadvar.}: int

proc getThreadEvalScratch(ruleCount: int): var EvalScratch {.gcsafe.} =
  if not tlEvalScratchReady or tlEvalScratchRuleCount != ruleCount:
    tlEvalScratch = newEvalScratch(ruleCount, 2048)
    tlEvalScratchReady = true
    tlEvalScratchRuleCount = ruleCount
  tlEvalScratch

# ------------------------------------------------------------
# 評価用作業領域
#
# ★再変更: 個体ごとに newEvalScratch を都度確保する方式は
# 安全だが世代あたりのアロケーション量が大きいため、上の
# getThreadEvalScratch による {.threadvar.} 再利用へ戻した。
# 過去にこの方式でSIGSEGVが出た経緯があるが、その時点では
# getReplacementPlanPtr の revision ベース無効化(現在の実装)が
# 無かった/異なっていた可能性がある。再導入後は必ず、
# 十分な世代数(特にcheckpoint再開・mutation・crossoverで
# ruleが大量に入れ替わる序盤)を -d:release のまま(danger無しで)
# 走らせ、境界チェック/古いplanの誤再利用が起きていないか
# 確認してから -d:danger 本番ビルドへ切り替えること。
# ------------------------------------------------------------
proc findLiteralFrom(
  text: openArray[int],
  p: LiteralPart,
  start: int
): int {.inline.} =
  ## start 以降の最初の一致を返す。後続検索は重複した開始位置も
  ## 列挙する（置換自体の非重複性は呼び出し側の消費位置が保証）。
  let plen = p.data.len
  # A negative search cursor is a matcher defect, not a valid match position.
  # Test start before computing text.len - start to avoid invalid cursor reuse.
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

  # 元の lowerBound(occurrences, pos) と同じ意味を保つため、pos を
  # 飛び越してしまった一致も順に消費する。各一致は高々1回しか
  # 計算されないので、全 occurrence 配列を保持するよりメモリ負荷が小さい。
  var searchPos = cursor.nextSearch
  while true:
    let found = findLiteralFrom(text, p, searchPos)
    if found < 0:
      cursor.exhausted = true
      cursor.lastMatch = -1
      return -1

    cursor.lastMatch = found
    cursor.nextSearch = found + 1 # preserve overlapping occurrences for lower_bound
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
  ## parts[w] and parts[w+1] delimit capture w. Empty parts represent
  ## consecutive or boundary wildcards; nonempty parts ALSO delimit captures.
  ## Keep the literal anchor at start, use shortest interior matches, and
  ## assign the remaining suffix only to an actual trailing wildcard.
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


proc isAllWildcard(pat: SeqPattern): bool {.inline.} =
  pat.allWildcard

proc isEmptyPattern(pat: SeqPattern): bool {.inline.} =
  pat.emptyPattern


const MAX_REPLACE_OUTPUT = 1_048_576

proc appendIntsBulk(
  outp: var seq[int],
  input: openArray[int],
  start, len: int
) {.inline.} =
  if len <= 0:
    return
  # This is a copyMem fast path: Nim cannot bounds-check input[start..].
  # Enforce a complete half-open range before touching its pointer.
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

proc compileReplacementPlan(b: openArray[int], plan: var ReplacementPlan) {.inline.} =
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
      # Four contiguous banks of 16 captures, preserving all old opcodes:
      # -48..-63 +1; -64..-79 -1; -80..-95 *2; -96..-111 //2.
      plan.allLiteral = false
      let offset = -t - 48
      plan.ops[i] = ReplacementOp(
        kind: uint8(4 + offset div 16), arg: uint8(offset mod 16), value: 0)
    else:
      raise newException(
        SeqReplaceError,
        "Invalid replacement opcode: " & $t
      )

proc getReplacementPlanPtr(
  genome: Genome,
  ruleIndex: int,
  scratch: var EvalScratch
): ptr ReplacementPlan {.inline.} =
  let rev = genome[ruleIndex].replacementRevision
  # rev==0 は旧checkpointを読み込んだ直後などの未採番状態。
  # 0同士を有効なcache hitとみなすと、別ruleのplanを誤再利用し得る。
  if rev == 0'u64 or scratch.replacementPlanRevision[ruleIndex] != rev:
    compileReplacementPlan(genome[ruleIndex].b, scratch.replacementPlans[ruleIndex])
    scratch.replacementPlanRevision[ruleIndex] = rev
  addr scratch.replacementPlans[ruleIndex]

proc ensureReplacementPlan(
  replacement: openArray[int], plan: ptr ReplacementPlan,
  revision: uint64, cachedRevision: ptr uint64
) {.inline.} =
  ## An unmatched indexed rule never needs its replacement compiled. This is
  ## particularly useful in new offspring: most rule revisions have not fired.
  if cachedRevision != nil and
      (revision == 0'u64 or cachedRevision[] != revision):
    compileReplacementPlan(replacement, plan[])
    cachedRevision[] = revision

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
  # ReplacementPlan は最大64命令の固定配列を含むため、
  # `let plan = planPtr[]` で値コピーせずポインタ先を直接参照する。
  # このprocは置換の最内周なので、不要な構造体コピーを確実に排除する。
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
      # Captures are half-open slices of `input`. Validate BEFORE any pointer
      # arithmetic or direct indexing; this catches a bad matcher/cursor result
      # at its source instead of reporting a mysterious boundary index later.
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
      # Captures are half-open slices of `input`. Validate BEFORE any pointer
      # arithmetic or direct indexing; this catches a bad matcher/cursor result
      # at its source instead of reporting a mysterious boundary index later.
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
      # Captures are half-open slices of `input`. Validate BEFORE any pointer
      # arithmetic or direct indexing; this catches a bad matcher/cursor result
      # at its source instead of reporting a mysterious boundary index later.
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
      # Transform each token of one capture independently; no temporary seq.
      var n = int(op.arg)
      if wildcardCount == 0 or captures.len == 0:
        continue
      if n >= wildcardCount or n >= captures.len:
        n = 0
      let c = captures[n]
      # Captures are half-open slices of `input`. Validate BEFORE any pointer
      # arithmetic or direct indexing; this catches a bad matcher/cursor result
      # at its source instead of reporting a mysterious boundary index later.
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
      of 4'u8: # +1 modulo 512 within the internal-token domain
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] = (if v >= 0 and v < EMBEDDING_CODE_COUNT:
                            (v + 1) mod EMBEDDING_CODE_COUNT else: v)
      of 5'u8: # -1 modulo 512 within the internal-token domain
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] = (if v >= 0 and v < EMBEDDING_CODE_COUNT:
                            (v + EMBEDDING_CODE_COUNT - 1) mod EMBEDDING_CODE_COUNT else: v)
      of 6'u8: # *2 modulo 512 within the internal-token domain
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          outp[dst + j] = (if v >= 0 and v < EMBEDDING_CODE_COUNT:
                            (v * 2) mod EMBEDDING_CODE_COUNT else: v)
      of 7'u8: # floor division; high OOV/out-of-domain tokens stay untouched
        for j in 0 ..< c.len:
          let v = input[c.start + j]
          if v >= EMBEDDING_CODE_COUNT:
            outp[dst + j] = v
          else:
            let q = v div 2
            outp[dst + j] = (if v < 0 and v mod 2 != 0: q - 1 else: q)
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
  maxOutput: int = MAX_REPLACE_OUTPUT,
  replacementRevision: uint64 = 0'u64,
  cachedRevision: ptr uint64 = nil
): bool {.gcsafe.} =
  ## replaceSeq のコア処理。
  ##
  ## 戻り値は (changed, data) のタプル。置換が実際に発生したかどうかは
  ## この関数の内部で既にわかっているので、それをそのまま
  ## 返すことで呼び出し側の再比較を不要にする。
  ##
  ## {.gcsafe.}: 並列評価(spawn)から呼ばれるため明示。中身は引数のみを
  ## 使う純粋な処理でグローバル可変状態には触れないので安全。

  if isAllWildcard(pat) or isEmptyPattern(pat) or pat.wildcardCount > 16:
    return false

  var overflow = false
  if overflowed != nil: overflowed[] = false

  if pat.wildcardCount == 0:
    let literal = pat.parts[0]
    let plen = literal.data.len
    if plen == 0 or input.len < plen:
      return false

    # 通常リテラル用の最速経路。
    # マッチが無い間は allocation せず、最初のマッチを見つけた瞬間だけ
    # 出力バッファを確保。その後は入力を一度だけ左→右に走査する。
    # Reuse the length-specialized matcher: 1/2-token direct scan, <=8
    # short literal scan, and KMP for longer patterns. The previous duplicated
    # naive scan repeatedly compared long shared prefixes, causing O(n*m)
    # work on unproductive candidate rules.
    let firstMatch = findLiteralFrom(input, literal, 0)
    if firstMatch < 0:
      return false

    ensureReplacementPlan(replacement, replacementPlan,
      replacementRevision, cachedRevision)
    outp.setLen(0)
    var copyPos = 0
    var i = firstMatch

    while i >= 0:
      # 現在位置まで入力をコピーしてから置換。
      if copyPos < i:
        if outp.len + i - copyPos > maxOutput:
          if overflowed != nil: overflowed[] = true
          outp.setLen(0)
          return false
        appendIntsBulk(outp, input, copyPos, i - copyPos)

      if replacementPlan[].allLiteral:
        # Preserve bulk copy for ordinary literal replacement.
        if outp.len + replacement.len > maxOutput:
          if overflowed != nil: overflowed[] = true
          outp.setLen(0)
          return false
        appendIntValuesBulk(outp, replacement)
      else:
        # Literal patterns can still have capture/sort/reverse opcodes in b.
        # With no captures those instructions must NOT enter the output text.
        appendReplacementPlan(outp, replacementPlan, input,
          scratch.captures, pat.wildcardCount, scratch, overflow, maxOutput)
        if overflow:
          if overflowed != nil: overflowed[] = true
          outp.setLen(0)
          return false

      copyPos = i + plen
      i = copyPos

      # Nonoverlapping matches: resume at the end of the previous match.
      i = findLiteralFrom(input, literal, copyPos)

    if copyPos < input.len:
      if outp.len + input.len - copyPos > maxOutput:
        if overflowed != nil: overflowed[] = true
        outp.setLen(0)
        return false
      appendIntsBulk(outp, input, copyPos, input.len - copyPos)

    return true

  let hasLeadingLiteral = pat.hasLeadingLiteral

  # Fixed 17-slot cursor storage: max wildcardCount is 16, so parts <= 17.
  for i in 0 ..< pat.parts.len:
    scratch.cursors[i] = LiteralCursor(lastMatch: -1, nextSearch: 0, exhausted: false)

  # 出力バッファは呼び出し側から再利用する。
  # これにより「ruleが1回発火するたびに新しいseqを確保」を防ぐ。
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
        let nextStart = nextOccurrenceLazy(input, pat.parts[0], scratch.cursors[0], start + 1)
        if nextStart < 0:
          break

        let m2 = matchOccurrences(input, pat, scratch.cursors, nextStart, scratch.captures)

        if not m2.ok:
          pos = nextStart
          cachedStart = nextStart
          cachedFinish = m2.finish
          cachedOk = m2.ok
          hasCached = true
          continue

        if not outStarted:
          ensureReplacementPlan(replacement, replacementPlan,
            replacementRevision, cachedRevision)
          outp.setLen(0)
          outStarted = true

        if pos < nextStart:
          if outp.len + nextStart - pos > maxOutput:
            if overflowed != nil: overflowed[] = true
            outp.setLen(0)
            return false
          appendIntsBulk(outp, input, pos, nextStart - pos)

        appendReplacementPlan(
          outp, replacementPlan, input, scratch.captures, pat.wildcardCount,
          scratch, overflow, maxOutput
        )
        if overflow:
          if overflowed != nil: overflowed[] = true
          outp.setLen(0)
          return false

        changed = true
        pos = m2.finish
        hasCached = false
        continue

      else:
        break

    if not outStarted:
      ensureReplacementPlan(replacement, replacementPlan,
        replacementRevision, cachedRevision)
      outp.setLen(0)
      outStarted = true

    if pos < start:
      if outp.len + start - pos > maxOutput:
        if overflowed != nil: overflowed[] = true
        outp.setLen(0)
        return false
      appendIntsBulk(outp, input, pos, start - pos)

    appendReplacementPlan(
      outp, replacementPlan, input, scratch.captures, pat.wildcardCount,
      scratch, overflow, maxOutput
    )
    if overflow:
      if overflowed != nil: overflowed[] = true
      outp.setLen(0)
      return false

    changed = true
    pos = m.finish
    hasCached = false

  # A wildcard candidate may fail to match at every position. Previously
  # the full input was still copied into outp, then discarded by the caller.
  if not changed: return false
  if pos < input.len:
    if outp.len + input.len - pos > maxOutput:
      if overflowed != nil: overflowed[] = true
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
  let changed = replaceSeqCompiledInto(input, pat, replacement, outp, scratch, addr plan)
  if not changed:
    return (false, @[])
  (true, outp)


proc replaceSeq*(
  input: openArray[int],
  pattern: openArray[int],
  replacement: openArray[int]
): seq[int] =
  ## 従来通りの公開API。互換性のためそのまま残してある。
  let pat = compileSeqPattern(pattern)
  let r = replaceSeqCompiled(input, pat, replacement)
  r.data


proc replaceSeqInPlace*(
  input: var seq[int],
  pattern: openArray[int],
  replacement: openArray[int]
) =
  input = replaceSeq(input, pattern, replacement)

proc pickRandom(n, m: int): seq[int] =
  ## 0..<n から m 個を重複なしで選ぶ。
  if n <= 0 or m <= 0:
    return @[]

  let k = min(n, m)
  var a = newSeq[int](n)
  for i in 0 ..< n:
    a[i] = i

  result = newSeq[int](k)
  for i in 0 ..< k:
    let j = rand(i ..< n)
    swap(a[i], a[j])
    result[i] = a[i]

proc rank[T](x: openArray[T]): seq[float] =
  ## 同順位には平均順位を与える。入力値のコピーは作らず、
  ## index配列だけをsortする。長時間GAではこの経路が非常に頻繁に
  ## 呼ばれるため、不要な xs=@x allocation を削る。
  let n = x.len
  if n == 0:
    return @[]
  result = newSeq[float](n)

  var indices = newSeq[int](n)
  for i in 0 ..< n:
    indices[i] = i

  # NimではopenArrayをsort用クロージャから捕獲できないため、
  # 比較対象だけローカルのseqへコピーして安全に参照する。
  let values = @x
  indices.sort(proc(a, b: int): int =
    cmp(values[a], values[b])
  )

  var i = 0
  while i < n:
    var j = i + 1
    while j < n and x[indices[j]] == x[indices[i]]:
      inc j

    let avgRank = (float(i + 1) + float(j)) * 0.5
    for k in i ..< j:
      result[indices[k]] = avgRank
    i = j


proc spearman[T, U](x: openArray[T], y: openArray[U]): float {.gcsafe.} =
  ## Spearmanの順位相関係数。長時間GAでは assert に依存せず、
  ## 異常な入力は 0.0 として扱って評価スレッドを落とさない。
  if x.len != y.len or x.len < 2:
    return 0.0

  let rx = rank(x)
  let ry = rank(y)

  let n = rx.len.float

  var mx = 0.0
  var my = 0.0

  for i in 0 ..< rx.len:
    mx += rx[i]
    my += ry[i]

  mx /= n
  my /= n

  var numerator = 0.0
  var dx = 0.0
  var dy = 0.0

  for i in 0 ..< rx.len:
    let a = rx[i] - mx
    let b = ry[i] - my

    numerator += a * b
    dx += a * a
    dy += b * b

  if dx == 0.0 or dy == 0.0:
    return 0.0

  numerator / sqrt(dx * dy)


proc rankFloatInPlace(
  x: openArray[float],
  scratch: var EvalScratch
) {.inline.} =
  let n = x.len
  scratch.rankValues.setLen(n)
  scratch.rankIndices.setLen(n)
  if n == 0:
    return

  for i in 0 ..< n:
    scratch.rankIndices[i] = i

  let values = cast[ptr UncheckedArray[float]](unsafeAddr x[0])
  scratch.rankIndices.sort(proc(a, b: int): int =
    cmp(values[a], values[b])
  )

  var i = 0
  while i < n:
    var j = i + 1
    while j < n and values[scratch.rankIndices[j]] == values[scratch.rankIndices[i]]:
      inc j
    let avgRank = (float(i + 1) + float(j)) * 0.5
    for k in i ..< j:
      scratch.rankValues[scratch.rankIndices[k]] = avgRank
    i = j


proc spearmanWithRankedY(
  x: openArray[float],
  ry: openArray[float],
  scratch: var EvalScratch
): float {.inline, gcsafe.} =
  if x.len != ry.len or x.len < 2:
    return 0.0

  rankFloatInPlace(x, scratch)
  let n = x.len.float
  let meanRank = (n + 1.0) * 0.5
  var numerator = 0.0
  var dx = 0.0
  var dy = 0.0
  for i in 0 ..< x.len:
    let a = scratch.rankValues[i] - meanRank
    let b = ry[i] - meanRank
    numerator += a * b
    dx += a * a
    dy += b * b

  if dx <= 0.0 or dy <= 0.0:
    return 0.0
  numerator / sqrt(dx * dy)

proc cloneInts(x: openArray[int]): seq[int] =
  result = newSeq[int](x.len)
  if x.len > 0:
    copyMem(addr result[0], unsafeAddr x[0], x.len * sizeof(int))

proc sampleCorpusByte(): int {.inline.}

proc plusnoise_wariai*(
  input: seq[int],
  wariai: int
) : seq[int] =
  ## 指定箇所を入力中の別値へ置換する互換API。
  ## 新しいデータセット生成では下記の多様な拡散カーネルを使う。
  if input.len == 0 or wariai <= 0:
    return input

  result = cloneInts(input)
  let k = min(input.len, wariai)
  if k <= 0:
    return

  for i in 0 ..< k:
    let pos = rand(input.len - 1)
    var b = sampleCorpusByte()
    var guard = 0
    while b == result[pos] and guard < 16:
      b = sampleCorpusByte()
      inc guard
    if b != result[pos]:
      result[pos] = b


const
  DIFF_OP_SINGLE = 0
  DIFF_OP_BURST = 1
  DIFF_OP_LOCAL_WALK = 2
  DIFF_OP_GLOBAL = 3
  DIFF_OP_ALPHABET = 4
  DIFF_OP_COPY_BLOCK = 5
  DIFF_OP_REPLACE_BLOCK = 6
  DIFF_OP_PERMUTE_BLOCK = 7
  DIFF_OP_NEIGHBOR_COPY = 8

proc randomDiffByte(oldValue: int): int {.inline.} =
  var v = sampleCorpusByte()
  var guard = 0
  while v == oldValue and guard < 12:
    v = sampleCorpusByte()
    inc guard
  if v == oldValue:
    v = (oldValue + 1 + rand(254)) mod 256
  v

proc clampDiffScore(score, maxScore: int): int {.inline.} =
  if score <= 0:
    0
  elif score >= maxScore:
    maxScore
  else:
    score

proc estimateDiffusionDamage(
  base, state: openArray[int],
  previous: int
): int {.inline.} =
  ## Exact positional byte distance (not semantic quality or edit distance).
  ## A target must depend on the current input, never its trajectory history.
  discard previous  # Retained for source compatibility.
  var damage = abs(base.len - state.len)
  for i in 0 ..< min(base.len, state.len):
    if base[i] != state[i]: inc damage
  min(max(0, base.len - 1), damage)

proc applyDiffusionKernel(
  state: var seq[int],
  base: seq[int],
  op: int,
  progress: float
): int =
  ## Return an operation-size diagnostic only. Labels are recomputed from
  ## the resulting state; edits may undo earlier damage.
  if state.len == 0:
    return 0

  let n = state.len
  let baseLen = base.len
  let p = max(0.0, min(1.0, progress))
  let intensity = max(1, int(round(sqrt(max(1.0, p * p * float(max(1, baseLen)))))))

  case op
  of DIFF_OP_SINGLE:
    let pos = rand(n - 1)
    let old = state[pos]
    let v = randomDiffByte(old)
    if v != old:
      state[pos] = v
      return 1

  of DIFF_OP_BURST:
    let k = min(n, max(1, intensity * (1 + rand(2))))
    let start = rand(n - 1)
    var changed = 0
    for j in 0 ..< k:
      let pos = (start + j) mod n
      let old = state[pos]
      var v: int
      if rand(99) < 70:
        # 近傍値±小さな揺らぎ。ASCII/byte列の局所構造を壊しすぎない。
        let delta = rand(4) - 2
        v = max(0, min(255, old + delta))
      else:
        v = randomDiffByte(old)
      if v != old:
        state[pos] = v
        inc changed
    return changed

  of DIFF_OP_LOCAL_WALK:
    # 「隣接文字を少しずつノイズ化」に相当。ランダムな一点から左右へ
    # 歩き、各文字を元値の近傍へ少しだけ移動する。
    let k = min(n, max(1, intensity))
    var pos = rand(n - 1)
    var changed = 0
    for _ in 0 ..< k:
      let old = state[pos]
      let delta = rand(2) - 1
      let v = max(0, min(255, old + delta))
      if v != old:
        state[pos] = v
        inc changed
      if rand(1) == 0:
        if pos > 0 and rand(1) == 0:
          dec pos
        elif pos + 1 < n:
          inc pos
    return changed

  of DIFF_OP_GLOBAL:
    # 完全ランダム寄りの散布。replacementはコーパス分布から取る。
    let k = min(n, max(1, intensity * 2))
    var changed = 0
    for _ in 0 ..< k:
      let pos = rand(n - 1)
      let old = state[pos]
      let v = randomDiffByte(old)
      if v != old:
        state[pos] = v
        inc changed
    return changed

  of DIFF_OP_ALPHABET:
    # 「アルファベットAを別のアルファベットBへ」のglobal map。
    # 毎回1組だけ選ぶので一箇所ずつ壊すより大きな構造変化を作れる。
    let fromVal = state[rand(n - 1)]
    var toVal = randomDiffByte(fromVal)
    if toVal == fromVal:
      toVal = (fromVal + 1) mod 256
    var changed = 0
    for i in 0 ..< n:
      if state[i] == fromVal:
        state[i] = toVal
        inc changed
    return changed

  of DIFF_OP_COPY_BLOCK:
    # 任意の部分列を別位置へコピー。長さは短/中/長を抽選する。
    let maxLen = min(n, max(1, intensity * 2))
    let blockLen = 1 + rand(maxLen - 1)
    if blockLen > 0 and blockLen <= n:
      let src = rand(n - blockLen)
      let dst = rand(n - blockLen)
      let source = state[src ..< src + blockLen]
      var changed = 0
      for j in 0 ..< blockLen:
        let pos = dst + j
        let v = source[j]
        if state[pos] != v:
          state[pos] = v
          inc changed
      return changed

  of DIFF_OP_REPLACE_BLOCK:
    # 任意文字列を別文字列へ。長さは操作量で決まり、chunk長の固定上限は設けない。
    let oldLen = 1 + rand(min(n, max(1, intensity * 2)) - 1)
    let oldStart = rand(n - oldLen)
    let newLen = max(1, oldLen + (rand(2 * min(64, max(1, intensity))) - min(64, max(1, intensity))))
    let newBlock = block:
      var tmp = newSeq[int](newLen)
      for j in 0 ..< newLen:
        tmp[j] = randomDiffByte(if oldLen > 0: state[oldStart + min(j, oldLen - 1)] else: 0)
      tmp

    let outLen = n - oldLen + newLen
    if outLen < 1:
      return 0

    var next = newSeq[int](outLen)
    var w = 0
    for i in 0 ..< oldStart:
      next[w] = state[i]
      inc w
    for i in 0 ..< newLen:
      next[w] = newBlock[i]
      inc w
    let tailStart = oldStart + oldLen
    for i in tailStart ..< n:
      next[w] = state[i]
      inc w
    state.swap(next)
    return max(1, max(oldLen, newLen))

  of DIFF_OP_PERMUTE_BLOCK:
    if n < 2: return 0
    # ブロック内だけshuffle。文字頻度は維持しつつ順序だけ壊す。
    let blockLen = 2 + rand(min(n, max(2, intensity * 2)) - 2)
    if blockLen >= 2:
      let start = rand(n - blockLen)
      for i in countdown(start + blockLen - 1, start + 1):
        let j = start + rand(i - start)
        swap(state[i], state[j])
      return max(1, blockLen div 2)

  of DIFF_OP_NEIGHBOR_COPY:
    # 隣接文字を少しずつ別の隣接文字へ伝播させる。
    let k = min(n, max(1, intensity))
    var changed = 0
    for _ in 0 ..< k:
      let pos = rand(n - 1)
      if n > 1:
        let src = if pos == 0: 1 elif pos == n - 1: n - 2 else: pos + (if rand(1) == 0: -1 else: 1)
        if state[pos] != state[src]:
          state[pos] = state[src]
          inc changed
    return changed

  else:
    return 0

  0

proc datasetHasInvalidBytes(xs: seq[seq[seq[int]]]): bool =
  ## Checkpoint trajectories are external bytes, never latent model tokens.
  for trajectory in xs:
    for sample in trajectory:
      for value in sample:
        if value < 0 or value > 255: return true
  false

proc buildDiffusionTrajectory(
  base: seq[int],
  sampleCount: int
): tuple[x: seq[seq[int]], y: seq[int]] =
  ## 1チャンクにつき1本のランダム拡散軌道。
  ## 各stepで「どの壊し方を使うか」と「強度」を別々に抽選する。
  ## 単純な固定位置ノイズではなく、局所摂動・大域置換・文字置換・
  ## substring操作・順序撹乱を同一軌道へ混ぜる。
  result.x = @[]
  result.y = @[]
  if base.len < 2 or sampleCount < 2:
    return

  const OP_COUNT = 9
  let count = max(2, sampleCount)
  result.x = newSeqOfCap[seq[int]](count)
  result.y = newSeqOfCap[int](count)
  result.x.add(cloneInts(base))
  result.y.add(base.len)

  var state = cloneInts(base)
  var diffusionMass = 0
  for sidx in 1 ..< count:
    let progress = float(sidx) / float(count - 1)

    # 基本方針:
    #   「1文字だけをランダムノイズへ置換」を最頻出にする。
    # その次に、局所的で控えめな操作ほど高確率にする。
    # 大きなブロック置換・全体置換は低確率の例外として残す。
    #
    # 重み（大きいほど高確率）:
    #   SINGLE > LOCAL_WALK > NEIGHBOR_COPY > BURST >
    #   COPY_BLOCK > PERMUTE_BLOCK > GLOBAL > ALPHABET > REPLACE_BLOCK
    #
    # progress は「操作の種類」をひっくり返すためには使わず、
    # applyDiffusionKernel 側の intensity にだけ任せる。
    # これで時刻が進んでも、拡散の主役は常に穏やかな1文字置換になる。
    const
      W_SINGLE = 100
      W_LOCAL_WALK = 60
      W_NEIGHBOR_COPY = 45
      W_BURST = 30
      W_COPY_BLOCK = 20
      W_PERMUTE_BLOCK = 12
      W_GLOBAL = 8
      W_ALPHABET = 5
      W_REPLACE_BLOCK = 3
      W_TOTAL = W_SINGLE + W_LOCAL_WALK + W_NEIGHBOR_COPY +
                W_BURST + W_COPY_BLOCK + W_PERMUTE_BLOCK +
                W_GLOBAL + W_ALPHABET + W_REPLACE_BLOCK

    var roll = rand(W_TOTAL - 1)
    var op: int

    if roll < W_SINGLE:
      op = DIFF_OP_SINGLE
    else:
      roll -= W_SINGLE
      if roll < W_LOCAL_WALK:
        op = DIFF_OP_LOCAL_WALK
      else:
        roll -= W_LOCAL_WALK
        if roll < W_NEIGHBOR_COPY:
          op = DIFF_OP_NEIGHBOR_COPY
        else:
          roll -= W_NEIGHBOR_COPY
          if roll < W_BURST:
            op = DIFF_OP_BURST
          else:
            roll -= W_BURST
            if roll < W_COPY_BLOCK:
              op = DIFF_OP_COPY_BLOCK
            else:
              roll -= W_COPY_BLOCK
              if roll < W_PERMUTE_BLOCK:
                op = DIFF_OP_PERMUTE_BLOCK
              else:
                roll -= W_PERMUTE_BLOCK
                if roll < W_GLOBAL:
                  op = DIFF_OP_GLOBAL
                else:
                  roll -= W_GLOBAL
                  if roll < W_ALPHABET:
                    op = DIFF_OP_ALPHABET
                  else:
                    op = DIFF_OP_REPLACE_BLOCK

    discard applyDiffusionKernel(state, base, op, progress)
    # 操作回数deltaではなく、実際のstateとbaseの距離を推定して
    # diffusion target y を更新する。これにより、例えば
    # 「1回のalphabet置換で大量に変わる」ケースと
    # 「10回の無効に近い操作」の区別がつく。
    diffusionMass = clampDiffScore(
      estimateDiffusionDamage(base, state, diffusionMass),
      base.len - 1
    )

    result.x.add(cloneInts(state))
    result.y.add(max(1, base.len - diffusionMass))


randomize()

# ★追加: スレッドプールを論理コア数まで広げ、世代内の個体評価が
# 全コアに分散されるようにする。
#
# ★修正: 以前は min(countProcessors(), 8) でOSスレッド数そのものを
# 8本に固定していた一方、並列評価側(parallelEvaluate*Batch)は
# min(countProcessors(), 12) を前提にバッチサイズを計算しており、
# 8コアを超えるマシンでは実際に使えるスレッドが8本しかないのに
# 9〜12並列のつもりでジョブを積んでいた(実質そのコア数以上は完全に遊ぶ)。
# 両者を同じ上限に統一し、コメント通り「論理コア数まで」広げる。
# 極端に大きい論理コア数を報告する環境向けに緩い安全上限だけ残す。
const WORKER_POOL_CAP = 64
let workerCount = max(1, min(countProcessors(), WORKER_POOL_CAP))

const LEGACY_RULE_COUNT = 3100
var AAA = 3100

# ------------------------------------------------------------
# ★大改修: コーパス統計を使ったパターン初期化/突然変異
#
# 以前は a(マッチパターン)の各要素も b(置換)の直値も
# rand(256+1)-1 の完全一様乱数で生成していた。
# 実際のテキスト/コードは printable byte が大半を占めるので、
# 長さkのランダム列がコーパス中に一度でも出現する確率はkに対して
# 指数的にゼロへ近づく。つまり初期集団や突然変異で生まれる複数バイトの
# literalパターンは、その大半が「一生一度もマッチしない」死んだ遺伝子に
# なっていた。これがGAで学習が本質的に進みにくい最大の原因の一つと考え、
# ここでコーパスを先に読み込み、
#   1. バイト出現頻度に比例した重み付きサンプリング
#   2. 実際に頻出するn-gram(長さ2..6)を種として直接注入
# の2つを用意し、初期化・突然変異の両方で使う。
# ------------------------------------------------------------

var sourceChunks: seq[seq[int]] = @[]

proc loadTrainingCorpus() =
  var f = open("github-code.txt", fmRead)
  defer: f.close()
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

  if sourceChunks.len == 0:
    quit("No source chunks found.")

# データセットの読み取り位置。
# sourceChunks はファイル全体をメモリへ保持しているため、EOF に到達した後は
# ディスクを再オープンする代わりに先頭へ巻き戻して同じデータを再利用する。
var sourceChunkCursor = 0
# 世代単位でRNGを再seedすることで、checkpointから次世代の乱数系列を
# 再現可能にする。グローバルrand() APIの内部状態そのものではなく、
# 「次に実行する世代のseed」を保存する方式。
var runSeed: int64 = getTime().toUnix

proc reseedForGeneration(iter: int) {.inline.} =
  randomize(int(runSeed xor int64(iter) * 0x9E3779B9'i64))

# --- 1. バイト出現頻度の重み付きサンプリング ---
var corpusByteCount: array[EMBEDDING_ENTRY_COUNT, float]
var corpusByteCum: array[EMBEDDING_ENTRY_COUNT, float]
var corpusByteTotal: float

proc buildCorpusByteStats(chunks: seq[seq[int]]) =
  for c in chunks:
    for v in c:
      if v >= 0 and v < EMBEDDING_ENTRY_COUNT:
        corpusByteCount[v] += 1.0
  # ラプラススムージング: 出現しなかったバイトにも小さな探索確率を残す
  # (完全に0確率にすると、稀なバイトを使う正しいpatternを永久に
  #  発見できなくなるため)。
  for i in 0 ..< EMBEDDING_ENTRY_COUNT:
    corpusByteCount[i] += 1.0

  var acc = 0.0
  for i in 0 ..< EMBEDDING_ENTRY_COUNT:
    acc += corpusByteCount[i]
    corpusByteCum[i] = acc
  corpusByteTotal = acc

proc sampleEmbeddingDigit(): int {.inline.} =
  ## Any normal internal symbol, including latent/work tokens.
  rand(EMBEDDING_CODE_COUNT - 1)

proc sampleCorpusByte(): int {.inline.} =
  ## Corpus statistics remain over external bytes only.
  let r = rand(corpusByteTotal)
  var lo = 0
  var hi = INPUT_BYTE_COUNT - 1
  while lo < hi:
    let mid = (lo + hi) div 2
    if corpusByteCum[mid] < r:
      lo = mid + 1
    else:
      hi = mid
  lo

const LATENT_LITERAL_PERCENT = 8

proc sampleRuleLiteral(embedding: openArray[int]): int {.inline.} =
  ## Keep mutation corpus-grounded while still making the 256 free symbols
  ## reachable. Exactly 8% of literal draws are forced into the CURRENTLY
  ## unmapped latent set; the rest follow corpus byte frequency through E.
  if embedding.len == EMBEDDING_ENTRY_COUNT:
    if rand(99) < LATENT_LITERAL_PERCENT:
      return sampleUnusedEmbeddingCode(embedding)
    return embedding[sampleCorpusByte()]
  sampleEmbeddingDigit()


# --- 2. 頻出n-gramの抽出(種パターン) ---
const NGRAM_MIN_LEN = 2
const NGRAM_MAX_LEN = 6
const NGRAM_TOP_K = 500
const NGRAM_MIN_COUNT = 3
const NGRAM_SCAN_BYTE_BUDGET = 256_000  # 集計コストを抑えるための総スキャン量上限

# Multi-case / rolling-evaluation parameters.
# Three rolling cases: fast / medium / slow. Requested refresh widths are
# SLOW:MEDIUM:FAST = 2:3:11 generations. A deadline scheduler refreshes at most
# one group per generation, so the relatively dense 2/3 cadences do not collide.
# Each Spearman is computed within one trajectory, then aggregated.
const CASES_PER_TIMESCALE = 1
const MULTI_CASE_COUNT = 3 * CASES_PER_TIMESCALE
const RETAINED_SAMPLES_PER_CASE = 40
const FAST_ROLLING_SLOT = 0
const MEDIUM_ROLLING_SLOT = 1
const SLOW_ROLLING_SLOT = 2

# Multi-case selection parameters must be declared before the selection procs.
# epsilon-Lexicase uses an absolute fitness tolerance because case scores are
# Spearman correlations in [-1, 1].
const LEXICASE_EPSILON = 0.02
const BATCH_TOURNAMENT_SIZE = 5
const LEXICASE_RATE = 0.30

proc extractTopNgrams(chunks: seq[seq[int]]): seq[seq[int]] =
  ## コーパス中で実際に頻出する部分列(n-gram)を集計し、頻度上位を
  ## 種パターンとして取り出す。コーパス全体を舐めなくても頻出n-gramの
  ## 傾向は十分に捉えられるため、総スキャン量に上限を設けている。
  var counts = initTable[seq[int], int]()
  var scanned = 0
  for c in chunks:
    if scanned >= NGRAM_SCAN_BYTE_BUDGET:
      break
    for L in NGRAM_MIN_LEN .. NGRAM_MAX_LEN:
      if c.len < L:
        continue
      var i = 0
      while i + L <= c.len:
        let key = c[i ..< i + L]
        counts[key] = counts.getOrDefault(key, 0) + 1
        inc i
    scanned += c.len

  var pairs: seq[tuple[ng: seq[int], cnt: int]] = @[]
  for k, v in counts:
    if v >= NGRAM_MIN_COUNT:
      pairs.add((k, v))

  pairs.sort(proc(x, y: tuple[ng: seq[int], cnt: int]): int =
    if x.cnt > y.cnt: -1
    elif x.cnt < y.cnt: 1
    else: 0
  )

  result = @[]
  for i in 0 ..< min(NGRAM_TOP_K, pairs.len):
    result.add(pairs[i].ng)

var corpusNgrams: seq[seq[int]] = @[]

proc at(): int =
  ## Replacement-side opcodes:
  ## -1..-15  = direct backreferences ($1..$15)
  ## -16..-31 = sort($1)..sort($16)
  ## -32..-47 = reverse($1)..reverse($16)
  ## -48..-63 = ($1..$16) + 1 (mod 512 on internal tokens)
  ## -64..-79 = ($1..$16) - 1 (mod 512 on internal tokens)
  ## -80..-95 = ($1..$16) * 2 (mod 512 on internal tokens)
  ## -96..-111 = ($1..$16) // 2 (integer floor)
  ##
  ## ★変更: 直値(literal)は以前 0..256 の完全一様乱数だったが、
  ## コーパス出現頻度に比例したサンプリングへ変更した。
  if rand(257) == 0:
    # 元の分布とほぼ同じ比率(約1/258)でbackreferenceにする。
    return -(1 + rand(14))
  return sampleEmbeddingDigit()

proc makeRandomPatternSegment(len: int): seq[int] =
  ## 各要素をコーパスのバイト頻度分布からサンプリングする。
  result = newSeq[int](len)
  for i in 0 ..< len:
    result[i] = sampleEmbeddingDigit()

proc makeSeededPattern(targetLen: int, embedding: seq[int]): seq[int] =
  ## 70%の確率で、コーパスから抽出済みの頻出n-gramをそのまま
  ## (targetLenより長ければランダムな位置で切り出して)種として使う。
  ## 残り30%は頻度分布サンプリングによる合成(探索の多様性を残す)。
  if targetLen <= 0:
    return @[]
  if corpusNgrams.len > 0 and rand(99) < 70:
    let ng = corpusNgrams[rand(corpusNgrams.len - 1)]
    let encoded = encodeLiteralRuns(ng, embedding)
    if encoded.len <= targetLen:
      return encoded
    # Byte-aligned slice, so mutations can still move whole embedded tokens.
    let available = max(1, targetLen div EMBEDDING_WIDTH)
    let take = min(ng.len, available)
    let start = rand(ng.len - take)
    return encodeLiteralRuns(ng[start ..< start + take], embedding)
  var bytes = newSeqOfCap[int](max(1, targetLen div EMBEDDING_WIDTH))
  for _ in 0 ..< max(1, targetLen div EMBEDDING_WIDTH):
    bytes.add(min(EMBEDDING_ENTRY_COUNT - 1, sampleCorpusByte()))
  return encodeLiteralRuns(bytes, embedding)

var nextPatternRevision: uint64 = 1
var nextReplacementRevision: uint64 = 1


proc freshPatternRevision(): uint64 {.inline.} =
  result = nextPatternRevision
  inc nextPatternRevision
  if nextPatternRevision == 0:
    ## 実質的には到達不能だが、wrap時も0を予約値にしない。
    nextPatternRevision = 1

proc freshReplacementRevision(): uint64 {.inline.} =
  result = nextReplacementRevision
  inc nextReplacementRevision
  if nextReplacementRevision == 0:
    nextReplacementRevision = 1

proc countWildcards(x: openArray[int]): int {.inline.} =
  for v in x:
    if v <= -1:
      inc result


# The original direct-reference bank has 15 entries; all function banks
# have 16. Decode/re-encode in one place to avoid inconsistent validity
# checks (and accidentally discarding novel opcodes during crossover/load).
const MIN_REPLACEMENT_OPCODE = -111

proc replacementCaptureNumber(opcode: int): int {.inline.} =
  if opcode >= -15:
    -opcode
  else:
    1 + (-opcode - 16) mod 16

proc replacementOpcodeWithCapture(opcode, captureNumber: int): int {.inline.} =
  if opcode >= -15:
    -captureNumber
  else:
    -(16 + ((-opcode - 16) div 16) * 16 + captureNumber - 1)

proc replacementRefsNeedSanitize(r: Rule): bool {.inline.} =
  ## Check without changing COW-shared genome sequences.
  let wc = countWildcards(r.a)
  for t in r.b:
    if t < MIN_REPLACEMENT_OPCODE:
      return true
    if t < 0 and (wc == 0 or replacementCaptureNumber(t) > wc):
      return true
  false

proc sanitizePatternWildcards(r: var Rule) =
  ## Matcher scratch is fixed at 16 captures. Replacing a/b wholesale can
  ## otherwise make an irreversibly dead (>16 captures) rule.
  if countWildcards(r.a) <= 16: return
  var independent = newSeq[int](r.a.len)
  for i, v in r.a: independent[i] = v
  r.a = independent
  var n = 0
  for i in 0 ..< r.a.len:
    if r.a[i] < 0:
      inc n
      if n > 16:
        r.a[i] = sampleEmbeddingDigit()
  r.patternRevision = freshPatternRevision()


proc sanitizeReplacementRefs(r: var Rule) =
  ## Repair the capture index while keeping its operator family unchanged.
  ## A revision change is necessary only when b really changes.
  let wc = countWildcards(r.a)
  var changed = false
  for i in 0 ..< r.b.len:
    let t = r.b[i]
    if t < MIN_REPLACEMENT_OPCODE:
      r.b[i] = sampleEmbeddingDigit()
      changed = true
      continue
    if t >= 0:
      continue
    if wc <= 0:
      r.b[i] = sampleEmbeddingDigit()
      changed = true
    elif replacementCaptureNumber(t) > wc:
      # $16 does not exist in the legacy direct-reference bank.
      let available = (if t >= -15: min(15, wc) else: min(16, wc))
      let nn = 1 + rand(available - 1)
      r.b[i] = replacementOpcodeWithCapture(t, nn)
      changed = true
  if changed:
    r.replacementRevision = freshReplacementRevision()


proc makeSeededRule(embedding: seq[int] = @[]): Rule =
  ## Use the same corpus-seeded initialization for new and migrated genomes.
  var a: seq[int] = @[]
  var b: seq[int] = @[]
  let lenA = min(64, int(ceil(pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0))))
  let lenB = min(64, int(ceil(pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0) + pow(gauss(0, 1), 2.0))))
  let e = if embedding.len == EMBEDDING_ENTRY_COUNT: embedding else: defaultEmbedding()
  a = makeSeededPattern(lenA, e)
  if a.len >= EMBEDDING_WIDTH and rand(99) < 15:
    let bytePos = rand(a.len div EMBEDDING_WIDTH - 1) * EMBEDDING_WIDTH
    a.delete(bytePos, bytePos + EMBEDDING_WIDTH - 1)
    a.insert(-1, bytePos)
  # Seed output in one-token byte units. No multi-digit alignment required.
  for bi in 0 ..< min(64 div EMBEDDING_WIDTH,
                        max(1, (lenB + EMBEDDING_WIDTH - 1) div EMBEDDING_WIDTH)):
    b.add(e[min(EMBEDDING_ENTRY_COUNT - 1, sampleCorpusByte())])
  if b.len >= EMBEDDING_WIDTH and rand(99) < 12:
    let bytePos = rand(b.len div EMBEDDING_WIDTH - 1) * EMBEDDING_WIDTH
    b.delete(bytePos, bytePos + EMBEDDING_WIDTH - 1)
    b.insert(-(1 + rand(14)), bytePos)
  # Seed a small number of *valid* transform genes instead of requiring
  # a many-generation search to discover them by chance. No extra rows.
  let wc = countWildcards(a)
  if wc > 0 and b.len > 0 and rand(99) < 20:
    let family = rand(6)  # direct, sort, reverse, +1, -1, *2, //2
    let maxCapture = (if family == 0: min(wc, 15) else: min(wc, 16))
    let captureNumber = 1 + rand(maxCapture - 1)
    let opcode = (if family == 0: -captureNumber
       else: -(16 + (family - 1) * 16 + captureNumber - 1))
    if b.len >= EMBEDDING_WIDTH and b.len mod EMBEDDING_WIDTH == 0:
      let bytePos = rand(b.len div EMBEDDING_WIDTH - 1) * EMBEDDING_WIDTH
      b.delete(bytePos, bytePos + EMBEDDING_WIDTH - 1)
      b.insert(opcode, bytePos)
    else:
      b[rand(b.len - 1)] = opcode
  result = Rule(a: a, b: b, weight: gauss(0, 1),
    patternRevision: 0'u64,
    replacementRevision: 0'u64)
  sanitizeReplacementRefs(result)
  result.patternRevision = freshPatternRevision()
  result.replacementRevision = freshReplacementRevision()

proc makeObj(): Genome =
  result = newSeqOfCap[Rule](AAA)
  var mapping = defaultEmbedding()
  # Start mostly aligned for checkpoint/corpus continuity, but let some newborn
  # genomes claim a few of the 768 latent coordinates immediately.
  if rand(99) < 35:
    for _ in 0 ..< 1 + rand(3):
      let a = rand(EMBEDDING_ENTRY_COUNT - 1)
      if rand(99) < 70:
        mapping[a] = sampleUnusedEmbeddingCode(mapping)
      else:
        let b = rand(EMBEDDING_ENTRY_COUNT - 1)
        swap(mapping[a], mapping[b])
  for ruleIdx in 0 ..< AAA:
    result.add(makeSeededRule(mapping))
  if result.len > 0: result[0].embedding = mapping

proc expandLegacyGenome(g: var Genome): bool =
  ## Retain the original 3100 rules in order; append 3300 seeded rules for AAA=3100.
  ## Called only while loading, before any evaluator threads run.
  if g.len == AAA: return false
  if g.len != LEGACY_RULE_COUNT or AAA <= LEGACY_RULE_COUNT:
    raise newException(IOError, "Unsupported checkpoint genome length: " &
      $g.len & ", expected " & $AAA & " (or legacy " & $LEGACY_RULE_COUNT & ").")
  let oldLen = g.len
  for i in oldLen ..< AAA:
    g.add(makeSeededRule(if g.len > 0: g[0].embedding else: defaultEmbedding()))
  true

let pop_size = 450

var population : seq[Genome] = @[]


# ============================================================
# GA utilities
# ============================================================

type
  # 歴代の「新記録個体」を保存する永続エリート。
  # スコアだけでなく世代番号も持たせ、後から挙動を追跡できるようにする。
  EliteArchiveEntry = object
    genome: Genome
    score: float
    generation: int
    # ★追加: genomeFingerprint(genome)をキャッシュしておく。
    # archiveRecord/readEliteArchiveでgenomeを確定させた直後に必ず
    # 埋める。writeEliteArchive/readEliteArchiveのバイナリ形式には
    # 含めない(genomeから再計算できる派生値なので、チェックポイントの
    # 互換性を壊さないため)。
    fingerprint: Hash

  # Separately tracked generated-text champions. Only repeat winners are confirmed.
  # Stored in the SAME checkpoint stream as the GA state, not a sidecar.
  JevHallEntry = object
    genome: Genome
    fingerprint: Hash            # reconstructed on load, never serialized
    meanQuality: float           # per-candidate rubric mean, not GA fitness
    observations: int            # distinct successful Jev rounds
    wins: int                    # top-eight finishes with valid FULL fitness
    firstGeneration: int
    lastGeneration: int

  # ルール単位の「寄与」を軽量に追跡するための統計。
  # 各サンプルで、そのルールが出力へ足した値 x と
  # 目的値 y の共分散を蓄積し、後から相関係数を求める。
  RuleCreditStat = object
    xSum: float
    x2Sum: float
    xySum: float
    fired: int

  # 高寄与ルールが隣接している部分を「module」として保存する。
  # 実際のruleを丸ごと保持するので、別個体へ同位置/近傍へ移植できる。
  GuidedModule = object
    rules: seq[Rule]
    score: float
    startPos: int       # original location, retained for interface-aware transfer

# 寄与診断の更新間隔。毎世代やると評価を余計に増やすため、数世代に1回だけ行う。
const CREDIT_REFRESH_INTERVAL = 64
const CREDIT_STAGNANT_REFRESH_INTERVAL = 16
const CREDIT_STAGNATION_TRIGGER = 8
const MAX_GUIDED_MODULES = 16
const MODULE_MIN_LEN = 2
const MODULE_MAX_LEN = 32

var guidedModules: seq[GuidedModule] = @[]
var guidedCredit: seq[float] = @[]
# A credit is a property of the champion's actual rule, not of an array index
# in unrelated genomes. Keep its reference phenotype for identity checks.
var guidedReferenceGenome: Genome = @[]

# 新しい最高記録を出した個体は無条件でここへ保存する。
# いったん入った個体は通常の世代交代では捨てない。
var eliteOfElites: seq[EliteArchiveEntry] = @[]

const JEV_HALL_CAPACITY = 64
const JEV_HALL_INJECT_SLOTS = 4
const JEV_HALL_COMPARE_SLOTS = 12
const JEV_HALL_TOP_FINISH = 8
const JEV_HALL_CONFIRM_WINS = 3
const JEV_DONOR_POOL_CAP = 12
const JEV_DONOR_MIN_DISTANCE = 0.20
# Jev runs sparsely, so its genetic influence must remain much
# narrower than its observation cohort. Jev NEVER overwrites GA fitness. A
# Jev-only champion may live in the Hall, but may breed only after passing the
# Legacy Hall records remain readable; adaptive fusion below owns selection.
const JEV_DONOR_RATE = 5
const JEV_FRESH_DONOR_RATE = 8
var jevHall: seq[JevHallEntry] = @[]
var jevHallConfig = "" # same prompt/model/steps/cohort configuration as the quality proxy


# NOTE:
# Genome の a/b は seq なので参照共有される。
# 世代交代時に全3100行を深コピーすると非常に重い。
# そこで GA 側は shallow copy + 「変更直前だけ」copy-on-write にする。
proc cloneRuleShallow(r: Rule): Rule =
  Rule(
    a: r.a,
    b: r.b,
    weight: r.weight,
    patternRevision: r.patternRevision,
    replacementRevision: r.replacementRevision,
    embedding: r.embedding
  )

proc transplantRule(donor: Rule, donorMap, childMap: seq[int]): Rule =
  result = cloneRuleShallow(donor)
  # The embedding belongs only to the genome; never inherit a displaced row's map.
  result.embedding = @[]
  if donorMap.len == EMBEDDING_ENTRY_COUNT and childMap.len == EMBEDDING_ENTRY_COUNT and
      donorMap != childMap:
    # One 512-entry translation serves both a and b for this transplanted row.
    let translation = embeddingTranslation(donorMap, childMap)
    result.a = recodeLiteralRunsWithTranslation(donor.a, translation)
    result.b = recodeLiteralRunsWithTranslation(donor.b, translation)
    result.patternRevision = freshPatternRevision()
    result.replacementRevision = freshReplacementRevision()

proc cloneGenome(g: Genome): Genome =
  ## 構造だけコピー。a/b は共有する。
  ## mutateGenome は sequence を変更する直前に必ず clone するため安全。
  result = newSeq[Rule](g.len)
  for i in 0 ..< g.len:
    result[i] = cloneRuleShallow(g[i])

proc relocateGenomeBlock(
  g: var Genome,
  src, blockLen, dst: int,
  reverseBlock: bool = false
) =
  ## 固定長Genomeの「真の」block relocation。
  ## 以前の実装は source を残したまま destination を上書きしていたため、
  ## relocation のつもりで実際には duplicate + destructive overwrite になっていた。
  ## ここでは中間区間をrotateして全ruleを1回ずつ保つ。
  if blockLen <= 0 or src < 0 or dst < 0 or src == dst:
    return
  if src + blockLen > g.len or dst + blockLen > g.len:
    return

  var tmp = newSeq[Rule](blockLen)
  for j in 0 ..< blockLen:
    tmp[j] = g[src + j]
  if reverseBlock:
    tmp.reverse()

  if dst < src:
    # [dst ..< src) を右へ blockLen だけずらし、空いた dst へ挿入。
    for i in countdown(src - 1, dst):
      g[i + blockLen] = g[i]
  else:
    # (src+blockLen ..< dst+blockLen) を左へ詰め、空いた dst へ挿入。
    for i in src ..< dst:
      g[i] = g[i + blockLen]

  for j in 0 ..< blockLen:
    g[dst + j] = tmp[j]

# v20+ checkpoints serialize rules and one 256-entry embedding per genome; v23 uses a 512-token internal space.

# ------------------------------------------------------------
# weight -> rule score
# ------------------------------------------------------------
# weight は符号付き線形係数としてそのまま利用する。
# 正なら正の寄与、負なら負の寄与、0なら無効。
proc weightScore(w: float): float {.inline.} =
  w

proc weightScoreDerivative(w: float): float {.inline.} =
  1.0

proc randomCutPoints(n: int): tuple[l, r: int] =
  if n <= 1:
    return (0, 0)
  var l = rand(n - 1)
  var r = rand(n - 1)
  if l > r:
    swap(l, r)
  (l, r)

proc maxReplacementRef(x: openArray[int]): int {.inline.} =
  ## Maximum referenced capture over all seven opcode families.
  for v in x:
    if v >= MIN_REPLACEMENT_OPCODE and v < 0:
      result = max(result, replacementCaptureNumber(v))

proc compatibleRule(r: Rule): bool {.inline.} =
  ## Reject truly unknown opcodes; accept all four new transform families.
  for opcode in r.b:
    if opcode < MIN_REPLACEMENT_OPCODE: return false
  let wc = countWildcards(r.a)
  let mr = maxReplacementRef(r.b)
  mr == 0 or (wc > 0 and mr <= wc)


# Only blend weights of exactly homologous replacement rules.
proc blendHomologousWeights(dst: var Rule, donor: Rule) =
  if dst.a != donor.a or dst.b != donor.b: return
  let alpha = 0.20 + rand(60).float / 100.0
  let old = dst.weight
  let proposed =
    if oppositeNonZeroSign(old, donor.weight):
      (if old > 0.0: 1.0 else: -1.0) *
        (alpha * abs(old) + (1.0 - alpha) * abs(donor.weight))
    else:
      alpha * old + (1.0 - alpha) * donor.weight
  dst.weight = signPreservingWeight(old, proposed)

proc sameIntSeq(x, y: openArray[int]): bool {.inline.} =
  if x.len != y.len:
    return false
  for i in 0 ..< x.len:
    if x[i] != y[i]:
      return false
  true

proc sameRule(a, b: Rule): bool {.inline.} =
  sameIntSeq(a.a, b.a) and sameIntSeq(a.b, b.b)

# Complete genome identity includes the embedding; revisions are metadata.
proc genomeFingerprint(g: Genome): Hash =
  result = Hash(0)
  if g.len > 0: result = result !& hash(g[0].embedding)
  for r in g:
    result = result !& hash(r.a)
    result = result !& hash(r.b)
    result = result !& hash(r.weight)
  result = !$result

proc sameGenomeContent(a, b: Genome): bool {.inline.} =
  ## Complete phenotype equality: mapping, a/b and signed weights.
  ## フィンガープリントが衝突した場合の確認用フォールバックとして使う。
  if a.len != b.len:
    return false
  if a.len > 0 and a[0].embedding != b[0].embedding: return false
  for i in 0 ..< a.len:
    if not sameRule(a[i], b[i]) or a[i].weight != b[i].weight:
      return false
  true

proc sameGenomeRevisionState(a, b: Genome): bool {.inline.} =
  ## Fast exact-no-op check for parent -> child mutation. Revisions change on
  ## every structural edit, so no a/b sequence scan is needed in the common
  ## case; weights and the embedding are compared explicitly. This is not used
  ## as a persistence identity (sameGenomeContent remains authoritative there).
  if a.len != b.len: return false
  if a.len > 0 and a[0].embedding != b[0].embedding: return false
  for i in 0 ..< a.len:
    if a[i].patternRevision != b[i].patternRevision or
       a[i].replacementRevision != b[i].replacementRevision or
       a[i].weight != b[i].weight:
      return false
  true

# Four mutually exclusive operators. A *genome* is an ordered sequence:
# two-point crossover acts on the rule sequence, not merely inside a/b.
type CrossoverOp = enum
  coHomologous, coModule, coTwoPoint, coMacro

# Mutation scale is adapted on two levels. `feedbackMutationScale` remains the
# slow global 1/5-rule controller, while this three-arm bandit decides whether
# a child should exploit locally, use the balanced baseline, or explore more
# aggressively. The number of affected rows is STILL drawn from the exact same
# unrestricted log-uniform law; regimes change only event rates/magnitudes.
type MutationRegime = enum
  mrLocal, mrBalanced, mrExplore

const CROSSOVER_OP_COUNT = 4
const MUTATION_REGIME_COUNT = 3

proc mutationRegimeFactor(regime: MutationRegime): float {.inline.} =
  case regime
  of mrLocal: 0.72
  of mrBalanced: 1.0
  of mrExplore: 1.38

proc mutationScaleForRegime(baseScale: float, regime: MutationRegime): float {.inline.} =
  max(0.45, min(1.70, baseScale * mutationRegimeFactor(regime)))

type RuleTraceRow = object
  hits: int
  firstInput: Hash
  firstOutput: Hash

type RuleTrace = seq[RuleTraceRow]
type RuleAlignment = tuple[base: int, donor: int]

# Interface probe follows the same deterministic single trajectory as scoreRaw.
proc buildCrossoverTrace(
  g: Genome, compiled: seq[SeqPattern], candidates: CandidateIndex,
  samples: seq[seq[int]]
): RuleTrace =
  result = newSeq[RuleTraceRow](g.len)
  if g.len == 0: return
  # Traces run on the main thread after FULL workers have drained. Reuse the
  # same revision-checked scratch as scoring instead of allocating the large
  # replacement-plan array and sparse pair-stamp table for each parent trace.
  template scratch: var EvalScratch = getThreadEvalScratch(g.len)
  for sample in samples:
    if sample.len == 0: continue
    var state: seq[int] = @[]
    let embedded = g[0].embedding.len == EMBEDDING_ENTRY_COUNT
    if embedded:
      embedBytesInto(sample, g[0].embedding, state)
    else:
      state = cloneInts(sample)
    let maxTraceLen = min(32768,
      max(3100 * (if embedded: EMBEDDING_WIDTH else: 1), state.len * 32))
    scratch.candidateScratch.candidateValid = false
    for k in 0 ..< g.len:
      if not scratch.candidateScratch.candidateValid:
        scanCandidateIds(candidates, state, scratch.candidateIds,
          scratch.candidateScratch, k)
      if scratch.candidateScratch.ruleSeen[k] != scratch.candidateScratch.stamp:
        continue
      var overflow = false
      if not replaceSeqCompiledInto(state, compiled[k], g[k].b,
          scratch.bufB, scratch.replaceScratch,
          getReplacementPlanPtr(g, k, scratch), addr overflow,
          maxTraceLen):
        if overflow: break
        continue
      if result[k].hits == 0:
        result[k].firstInput = hash(state)
        result[k].firstOutput = hash(scratch.bufB)
      inc result[k].hits
      if scratch.bufB != state:
        swap(state, scratch.bufB)
        scratch.candidateScratch.candidateValid = false

proc sequenceSimilarity(a, b: openArray[int]): float {.inline.} =
  # The old implementation compared equal-length arrays once for identity and
  # a SECOND time to count matches whenever they differed. One pass computes
  # exactly the same result; both empty sequences must still score 1.0.
  if a.len == 0 or b.len == 0:
    return (if a.len == b.len: 1.0 else: 0.0)
  var matching = 0
  for i in 0 ..< min(a.len, b.len):
    if a[i] == b[i]: inc matching
  float(matching) / float(max(a.len, b.len))

proc ruleSimilarity(a, b: Rule, ta, tb: RuleTraceRow): float {.inline.} =
  result = 0.58 * sequenceSimilarity(a.a, b.a) +
           0.42 * sequenceSimilarity(a.b, b.b)
  if ta.hits > 0 and tb.hits > 0:
    if ta.firstInput == tb.firstInput: result += 0.08
    else: result -= 0.04
    if ta.firstOutput == tb.firstOutput: result += 0.08

proc traceAt(t: RuleTrace, i: int): RuleTraceRow {.inline.} =
  if i >= 0 and i < t.len: t[i] else: RuleTraceRow()

proc alignOrderedRules(
  base, donor: Genome, tb, td: RuleTrace
): seq[RuleAlignment] =
  ## Monotone greedy homology: any chosen donor index is strictly increasing.
  ## Structural resemblance is primary; shared runtime interfaces add evidence.
  let n = min(base.len, donor.len)
  var previousDonor = -1
  for i in 0 ..< n:
    # Local correspondence: narrow band is much cheaper than the old 129-rule
    # comparisons per base row. Module relocation handles long-range jumps.
    let lo = max(previousDonor + 1, i - 16)
    let hi = min(n - 1, i + 16)
    if lo > hi: continue
    var bestJ = -1
    var bestScore = 0.60
    for j in lo .. hi:
      let score = ruleSimilarity(base[i], donor[j], traceAt(tb, i),
        traceAt(td, j)) - 0.0008 * float(abs(i - j))
      if score > bestScore:
        bestScore = score
        bestJ = j
    if bestJ >= 0:
      result.add((base: i, donor: bestJ))
      previousDonor = bestJ

proc moduleInterfaceFitness(
  base, donor: Genome, tb, td: RuleTrace,
  dst, src, length: int
): float =
  ## Compare the entry and exit on two IDENTICAL profiling inputs. A matching
  ## observed hash is evidence, not a semantic proof. Missing hits fall back to
  ## pattern/replacement compatibility; there is no arbitrary random insertion.
  if dst < 0 or src < 0 or length <= 0 or
     dst + length > base.len or src + length > donor.len:
    return -Inf
  result = 0.35 * sequenceSimilarity(donor[src].a, base[dst].a)
  let leftA = traceAt(td, src)
  let leftB = traceAt(tb, dst)
  if leftA.hits > 0 and leftB.hits > 0:
    if leftA.firstInput == leftB.firstInput: result += 0.35
    else: result -= 0.12
  elif leftA.hits != leftB.hits:
    result -= 0.05
  if src + length < donor.len and dst + length < base.len:
    result += 0.25 * sequenceSimilarity(
      donor[src + length - 1].b, base[dst + length].a)
    let rightA = traceAt(td, src + length - 1)
    let rightB = traceAt(tb, dst + length)
    if rightA.hits > 0 and rightB.hits > 0 and
       rightA.firstOutput == rightB.firstInput:
      result += 0.25
  else:
    result += 0.15  # both modules can end at the genome boundary

proc sampleCrossoverLength(n: int): int =
  if n <= 1: return max(0, n)
  result = max(1, min(n, int(round(exp(rand(ln(float(n))))))))

proc sampleCrossoverSegment(n: int): tuple[l, r: int] =
  ## Usually import one log-uniform contiguous block; occasionally retain the
  ## classic uniformly selected two-point interval for genuine macro mixing.
  if n <= 0: return (0, 0)
  if rand(99) < 80:
    let length = sampleCrossoverLength(n)
    let l = rand(n - length)
    return (l, l + length)
  let l = rand(n - 1)
  let r = l + 1 + rand(n - l - 1)
  (l, r)

proc recombineHomologousRule(base, donor: Rule): Rule =
  result = cloneRuleShallow(base)
  if sameRule(base, donor): blendHomologousWeights(result, donor)

proc twoPointGenomeCrossover(base, donor: Genome, l, r: int): Genome =
  ## Exact [0,l) from base + [l,r) from donor + [r,n) from base.
  result = cloneGenome(base)
  let n = min(base.len, donor.len)
  if l < 0 or r > n or l >= r: return
  for i in l ..< r:
    if compatibleRule(donor[i]):
      result[i] = transplantRule(donor[i], donor[0].embedding, base[0].embedding)
  result[0].embedding = base[0].embedding
  for i in 1 ..< result.len: result[i].embedding = @[]

proc crossoverGenome(
  p1, p2: Genome, op: CrossoverOp,
  t1, t2: RuleTrace
): Genome =
  let n = min(p1.len, p2.len)
  result = cloneGenome(p1)
  if n == 0: return
  if p1[0].embedding != p2[0].embedding and
      op in {coHomologous, coModule}:
    # Trace hashes and structural homology are coordinate-system dependent.
    # Cross-map transfer stays valid via mapping-aware two-point transplant.
    let (l, r) = sampleCrossoverSegment(n)
    result = twoPointGenomeCrossover(p1, p2, l, r)
    return
  case op
  of coTwoPoint:
    let (l, r) = sampleCrossoverSegment(n)
    result = twoPointGenomeCrossover(p1, p2, l, r)
  of coHomologous:
    let matched = alignOrderedRules(p1, p2, t1, t2)
    if matched.len == 0:
      # Homology unavailable: use an ordinary sequence crossover, not a no-op.
      let (l, r) = sampleCrossoverSegment(n)
      result = twoPointGenomeCrossover(p1, p2, l, r)
    else:
      let count = sampleCrossoverLength(matched.len)
      let start = rand(matched.len - count)
      for mi in start ..< start + count:
        let a = matched[mi].base
        let b = matched[mi].donor
        if sameRule(p1[a], p2[b]) and rand(99) < 70:
          # Exact structural homology permits independent bit recombination.
          result[a] = recombineHomologousRule(p1[a], p2[b])
        else:
          # Different a/b: inherit the entire donor rule.
          if compatibleRule(p2[b]):
            result[a] = transplantRule(p2[b], p2[0].embedding, p1[0].embedding)
  of coModule:
    # Favor active source interfaces; do not assume index == function.
    var active: seq[int] = @[]
    for i in 0 ..< min(n, t2.len):
      if t2[i].hits > 0: active.add(i)
    # Prefer a champion-derived contiguous functional module when the same
    # a/b signature is present in this donor at its recorded source site.
    # Otherwise use a trace-active source and full-range log-uniform length.
    var guidedSources: seq[int] = @[]
    if guidedModules.len > 0 and guidedReferenceGenome.len > 0 and
        p1[0].embedding == p2[0].embedding and
        p2[0].embedding == guidedReferenceGenome[0].embedding:
      for mi, m in guidedModules:
        if m.rules.len < 1 or m.startPos < 0 or
           m.startPos + m.rules.len > n: continue
        var valid = true
        for j in 0 ..< m.rules.len:
          let donorRule = p2[m.startPos + j]
          if not sameRule(donorRule, m.rules[j]):
            valid = false
            break
        if valid: guidedSources.add(mi)
    var length = sampleCrossoverLength(n)
    var src = rand(n - length)
    if guidedSources.len > 0 and rand(99) < 40:
      let m = guidedModules[guidedSources[rand(guidedSources.len - 1)]]
      length = m.rules.len
      src = m.startPos
    elif active.len > 0 and rand(99) < 75:
      let anchor = active[rand(active.len - 1)]
      src = max(0, min(n - length, anchor - rand(length - 1)))
    var bestDst = src
    var bestFit = moduleInterfaceFitness(p1, p2, t1, t2,
      src, src, length)
    # Search nearby, aligned and exploratory slots. Never rotate or resize the
    # donor block; ordered internal dependencies and genome length are retained.
    for trial in 0 ..< 12:
      let dst = if trial < 8:
        max(0, min(n - length, src + rand(128) - 64))
      else:
        rand(n - length)
      let fit = moduleInterfaceFitness(p1, p2, t1, t2,
        dst, src, length)
      if fit > bestFit:
        bestFit = fit
        bestDst = dst
    for j in 0 ..< length:
      if compatibleRule(p2[src + j]):
        result[bestDst + j] = transplantRule(p2[src + j], p2[0].embedding, p1[0].embedding)
  of coMacro:
    # Non-homologous, full-range exploration: a single log-uniform block.
    let length = sampleCrossoverLength(n)
    let src = rand(n - length)
    let dst = if rand(99) < 85: src else: rand(n - length)
    let reversed = rand(99) < 2
    for j in 0 ..< length:
      let donorIdx = if reversed: src + length - j - 1 else: src + j
      if compatibleRule(p2[donorIdx]):
        result[dst + j] = transplantRule(p2[donorIdx], p2[0].embedding, p1[0].embedding)
  # The mapping is a separate allele; row-0 replacement must not overwrite it.
  result[0].embedding = p1[0].embedding
  for i in 1 ..< result.len: result[i].embedding = @[]

proc chooseCrossoverOp(preference: array[CrossoverOp, float]): CrossoverOp =
  ## 40% minimum uniform exploration + 60% softmax allocation.
  ## Runtime-only preferences reset on resume: no checkpoint format change.
  if rand(1.0) < 0.40:
    return CrossoverOp(rand(CROSSOVER_OP_COUNT - 1))
  var weights: array[CrossoverOp, float]
  var total = 0.0
  for op in CrossoverOp:
    weights[op] = exp(max(-4.0, min(4.0, 10.0 * preference[op])))
    total += weights[op]
  let draw = rand(total)
  var cumulative = 0.0
  for op in CrossoverOp:
    cumulative += weights[op]
    if draw <= cumulative: return op
  coTwoPoint

proc chooseMutationRegime(
  preference: array[MutationRegime, float],
  pressure: float
): MutationRegime =
  ## Keep a 30% uniform floor so a temporarily unlucky regime is never lost.
  ## The remaining 70% follows paired FULL evidence, with a mild state prior:
  ## healthy populations prefer exploitation; stagnation/diversity collapse
  ## shifts mass toward the exploratory arm. Preferences are runtime-only.
  if rand(1.0) < 0.30:
    return MutationRegime(rand(MUTATION_REGIME_COUNT - 1))
  let p = max(0.0, min(1.0, pressure))
  var weights: array[MutationRegime, float]
  var total = 0.0
  for regime in MutationRegime:
    var prior = 0.0
    case regime
    of mrLocal: prior = 0.55 * (1.0 - p)
    of mrBalanced: prior = 0.20
    of mrExplore: prior = 0.70 * p
    weights[regime] = exp(max(-4.0, min(4.0,
      12.0 * preference[regime] + prior)))
    total += weights[regime]
  let draw = rand(total)
  var cumulative = 0.0
  for regime in MutationRegime:
    cumulative += weights[regime]
    if draw <= cumulative: return regime
  mrBalanced

# Number of RULES touched by one triggered mutation operator.
# x = exp(U[0, ln(N)]), rounded to the nearest integer in [1, N].
# The triggering probability of each mutation operator is unchanged.
proc sampleLogUniformMutationRange(low, high: int): int {.inline.} =
  ## Draw logarithmically across the ENTIRE valid integer range, including
  ## operators requiring at least two elements to make a nontrivial edit.
  if high < low: return max(0, high)
  if high == low: return high
  let x = exp(ln(float(low)) + rand(ln(float(high) / float(low))))
  result = max(low, min(high, int(round(x))))

proc sampleLogUniformMutationCount(ruleCount: int): int {.inline.} =
  if ruleCount <= 0: return 0
  sampleLogUniformMutationRange(1, ruleCount)

proc swapRandomRows(g: var Genome) =
  ## いくつかの「行」同士を入れ替える。
  if g.len < 2:
    return

  # Currently unused helper; keep its operation count consistent with mutation policy.
  let count = sampleLogUniformMutationCount(g.len)
  for _ in 0 ..< count:
    let i = rand(g.len - 1)
    let j = rand(g.len - 1)
    if i != j:
      swap(g[i], g[j])

proc swapABRows(g: var Genome) =
  ## 「置換前(a)と置換後(b)を入れ替える」突然変異。
  ## ルールの意味そのものを反転させる。
  if g.len == 0:
    return

  let count = sampleLogUniformMutationCount(g.len)
  for _ in 0 ..< count:
    let i = rand(g.len - 1)
    swap(g[i].a, g[i].b)

proc randomReplacementOpcode(): int {.inline.} =
  ## Keep direct references common; expose every transform to mutation.
  ## rand(n) includes both endpoints, so bank sizes are exact.
  let r = rand(99)
  if r < 50:
    return -(1 + rand(14))          # $1..$15
  let family =
    (if r < 60: 0                  # sort
     elif r < 70: 1                # reverse
     elif r < 78: 2                # +1
     elif r < 86: 3                # -1
     elif r < 93: 4                # *2
     else: 5)                      # //2
  -(16 + family * 16 + rand(15))   # captures $1..$16


proc mutateSequence(x: var seq[int], makePattern: bool, embedding: seq[int] = @[]) =
  ## Exactly one model token per byte. Keep atomic capture/function opcodes.
  ## Log-uniform counts remain controlled by mutateGenome, unchanged.
  if x.len == 0:
    if makePattern and rand(99) < 60: x.add(-1)
    elif not makePattern and rand(99) < 20: x.add(randomReplacementOpcode())
    else: x.add(sampleRuleLiteral(embedding))
    return

  if makePattern and embedding.len == EMBEDDING_ENTRY_COUNT and
      corpusNgrams.len > 0 and rand(99) < 10:
    let raw = corpusNgrams[rand(corpusNgrams.len - 1)]
    let ng = encodeLiteralRuns(raw, embedding)
    let room = max(0, 64 - x.len)
    if room > 0:
      let n = min(room, ng.len)
      let p = rand(x.len)
      for j in 0 ..< n: x.insert(ng[j], p + j)
      return

  case rand(5)
  of 0: # Substitute a byte or an opcode.
    let p = rand(x.len - 1)
    if makePattern and rand(99) < 20:
      x[p] = -(1 + rand(14))
    elif not makePattern and rand(99) < 20:
      x[p] = randomReplacementOpcode()
    else:
      x[p] = sampleRuleLiteral(embedding)
  of 1: # Insert one complete byte/opcode.
    if x.len < 64:
      let p = rand(x.len)
      var v = sampleRuleLiteral(embedding)
      if makePattern and rand(99) < 15: v = -(1 + rand(14))
      elif not makePattern and rand(99) < 20: v = randomReplacementOpcode()
      x.insert(v, p)
  of 2: # Delete one token.
    if x.len > 1: x.delete(rand(x.len - 1))
  of 3: # Log-uniform segment reversal.
    if x.len >= 2:
      let length = sampleLogUniformMutationRange(2, x.len)
      let l = rand(x.len - length)
      var i = l
      var j = l + length - 1
      while i < j:
        swap(x[i], x[j])
        inc i
        dec j
  else: # Duplicate one token.
    if x.len < 64:
      let p = rand(x.len - 1)
      x.insert(x[p], rand(x.len))


const MAX_REASONABLE_WEIGHT = WEIGHT_ABS_LIMIT

proc gaussianDelta(scale: float): float =
  gauss(0.0, scale)

proc pickGuidedMutationPos(g: Genome): int {.inline.} =
  ## Credit belongs to the exact champion rule, NOT the index in every genome.
  ## In a different genome, unknown positions receive neutral priority (0).
  ## Pattern/replacement identity and weight sign must agree; magnitude can
  ## change without invalidating correlation of that feature with the label.
  if g.len <= 0:
    return 0
  if guidedCredit.len != g.len or guidedReferenceGenome.len != g.len or
      g[0].embedding != guidedReferenceGenome[0].embedding:
    return rand(g.len - 1)

  template rulePriority(pos: int): float =
    (if sameRule(g[pos], guidedReferenceGenome[pos]) and
        not oppositeNonZeroSign(g[pos].weight, guidedReferenceGenome[pos].weight) and
        (g[pos].weight == 0.0) == (guidedReferenceGenome[pos].weight == 0.0):
       guidedCredit[pos]
     else:
       0.0)

  var best = rand(g.len - 1)
  var bestPriority = rulePriority(best)
  for _ in 0 ..< 7:
    let p = rand(g.len - 1)
    let priority = rulePriority(p)
    if priority == priority and (bestPriority != bestPriority or priority < bestPriority):
      bestPriority = priority
      best = p
  best

proc pickMutationTargets(g: Genome): seq[int] =
  ## ALL affected-rule counts use the same unrestricted logarithmic sampler.
  ## Guidance chooses the first row, then unbiased sampling without replacement.
  let count = sampleLogUniformMutationCount(g.len)
  if count <= 0:
    return @[]
  result = newSeqOfCap[int](count)
  let first = pickGuidedMutationPos(g)
  result.add(first)
  if count == 1:
    return
  if g.len >= 128 and count <= g.len div 8:
    # Sparse virtual Fisher-Yates: preserve EXACTLY the old selection law and
    # order with O(K) storage instead of allocating/initializing N integers
    # even for a two-rule mutation. This path draws the same random numbers.
    var displaced = initTable[int, int]()
    let last = g.len - 1
    if first != last:
      displaced[first] = last
      displaced[last] = first
    for i in 1 ..< count:
      let remaining = g.len - i
      let j = rand(remaining - 1)
      let chosen = displaced.getOrDefault(j, j)
      let tail = displaced.getOrDefault(remaining - 1, remaining - 1)
      result.add(chosen)
      displaced[j] = tail
      displaced[remaining - 1] = chosen
    return
  # A dense partial shuffle is faster and leaner for a large sampled K.
  var available = newSeq[int](g.len)
  for i in 0 ..< g.len:
    available[i] = i
  swap(available[first], available[g.len - 1])
  for i in 1 ..< count:
    let remaining = g.len - i
    let j = rand(remaining - 1)
    result.add(available[j])
    swap(available[j], available[remaining - 1])

proc mutateWeightAffine(g: var Genome) =
  ## weight A + (B - C) * {0,1}
  ## A, B, C は「別の行の weight」を使う。
  # Three distinct rows are mandatory. At len=2, the old ci re-draw loop
  # can never terminate (ci must differ from both ai and bi).
  if g.len < 3:
    return

  var w = newSeq[float](g.len)
  for i in 0 ..< g.len:
    w[i] = g[i].weight

  for ai in pickMutationTargets(g):
    var bi = rand(g.len - 1)
    var ci = rand(g.len - 1)
    while bi == ai:
      bi = rand(g.len - 1)
    while ci == ai or ci == bi:
      ci = rand(g.len - 1)

    # {0,1} を確率的マスクとして扱う。
    let mask = (if rand(1) == 1: 1.0 else: 0.0)
    w[ai] = (w[ai] + (w[bi] - w[ci]) * mask)

  for i in 0 ..< g.len:
    g[i].weight = signPreservingWeight(g[i].weight, w[i])

# ------------------------------------------------------------
# 自前の反復FFT
# ------------------------------------------------------------

proc fftInPlace(a: var seq[Complex64], inverse: bool) =
  let n = a.len
  if n <= 1:
    return

  var j = 0
  for i in 1 ..< n:
    var bit = n shr 1
    while (j and bit) != 0:
      j = j xor bit
      bit = bit shr 1
    j = j xor bit

    if i < j:
      swap(a[i], a[j])

  var len = 2
  while len <= n:
    let angle = (if inverse: 2.0 * PI / float(len)
                 else: -2.0 * PI / float(len))
    let wLen = complex(cos(angle), sin(angle))
    let half = len shr 1

    var i = 0
    while i < n:
      var w = complex(1.0, 0.0)
      for k in 0 ..< half:
        let u = a[i + k]
        let v = a[i + k + half] * w
        a[i + k] = u + v
        a[i + k + half] = u - v
        w = w * wLen
      i += len

    len = len shl 1

  if inverse:
    let invN = 1.0 / float(n)
    for i in 0 ..< n:
      a[i] = a[i] * invN

proc nextPow2(n: int): int {.inline.} =
  ## fftInPlace は radix-2 Cooley-Tukey なので 2 の冪長でないと
  ## バタフライ演算が配列境界を越える(out-of-bounds)。
  ## そのため呼び出し側では常にこの長さへゼロ詰めしてから渡す。
  result = 1
  while result < n:
    result = result shl 1

proc ifftFftWeightTransform(
  A, B, C: openArray[float]
): seq[float] =
  ## ifft(fft(A) * fft(B) / fft(C))
  let n = min(A.len, min(B.len, C.len))
  result = newSeq[float](n)
  if n == 0:
    return

  # ★修正: fftInPlace は 2 の冪長を前提にしている。
  # 以前は n (= genome長。例えば3100) をそのまま渡していたため、
  # n が2の冪でない場合にバタフライ演算が配列境界外へ読み書きし、
  # release ビルド(境界チェック無効)ではヒープ破壊 → 断続的な
  # SIGSEGV の原因になっていた。ここで2の冪長へゼロ詰めしてから
  # FFT/IFFT を行い、結果は先頭 n 要素だけを使う。
  let m = nextPow2(n)

  var fa = newSeq[Complex64](m)
  var fb = newSeq[Complex64](m)
  var fc = newSeq[Complex64](m)

  for i in 0 ..< n:
    fa[i] = complex(A[i], 0.0)
    fb[i] = complex(B[i], 0.0)
    fc[i] = complex(C[i], 0.0)
  # 残り (m - n) 要素は newSeq のデフォルトである complex(0,0) のまま
  # (=ゼロパディング)。

  fftInPlace(fa, false)
  fftInPlace(fb, false)
  fftInPlace(fc, false)

  const EPS = 1.0e-8

  for i in 0 ..< m:
    let cr = fc[i].re
    let ci = fc[i].im
    let den2 = cr * cr + ci * ci

    if den2 < EPS:
      fa[i] = complex(0.0, 0.0)
    else:
      fa[i] = (fa[i] * fb[i]) / fc[i]

  fftInPlace(fa, true)

  for i in 0 ..< n:
    var v = fa[i].re
    if v != v:
      v = 0.0
    result[i] = (v)

proc mutateWeightFFT(g: var Genome) =
  if g.len < 4:
    return

  var A = newSeq[float](g.len)
  var B = newSeq[float](g.len)
  var C = newSeq[float](g.len)

  for i in 0 ..< g.len:
    A[i] = g[i].weight

  let shiftB = rand(g.len - 1)
  let shiftC = rand(g.len - 1)

  for i in 0 ..< g.len:
    B[i] = g[(i + shiftB) mod g.len].weight
    C[i] = g[(i + shiftC) mod g.len].weight

  let transformed = ifftFftWeightTransform(A, B, C)

  let alpha = 0.15 + rand(85).float / 100.0
  for i in pickMutationTargets(g):
    let oldWeight = g[i].weight
    # Ill-conditioned spectral division can yield enormous coefficients.
    # Apply it as a trust-region proposal instead of saturating at +/-32.
    let delta = max(-0.5, min(0.5, alpha * (transformed[i] - oldWeight)))
    let proposed = if oppositeNonZeroSign(oldWeight, oldWeight + delta):
                     0.5 * oldWeight
                   else:
                     oldWeight + delta
    g[i].weight = signPreservingWeight(oldWeight, proposed)

proc applyEmbeddingChange(g: var Genome, newMap: seq[int], preservePercent: int) =
  ## Recode coadapted rules through a FULL 512-coordinate permutation. Most
  ## latent symbols therefore keep identity; symbols displaced by a newly mapped
  ## byte are moved into the byte's vacated/free coordinate instead of colliding.
  if g.len == 0 or g[0].embedding == newMap: return
  doAssert validEmbedding(g[0].embedding) and validEmbedding(newMap)
  let translation = embeddingTranslation(g[0].embedding, newMap)
  for i in 0 ..< g.len:
    if rand(99) >= preservePercent: continue
    var changeA = false
    var changeB = false
    for v in g[i].a:
      if v >= 0 and v < EMBEDDING_CODE_COUNT and translation[v] != v:
        changeA = true
        break
    for v in g[i].b:
      if v >= 0 and v < EMBEDDING_CODE_COUNT and translation[v] != v:
        changeB = true
        break
    if changeA:
      var newA = cloneInts(g[i].a)
      for j in 0 ..< newA.len:
        let v = newA[j]
        if v >= 0 and v < EMBEDDING_CODE_COUNT: newA[j] = translation[v]
      g[i].a = newA
      g[i].patternRevision = freshPatternRevision()
    if changeB:
      var newB = cloneInts(g[i].b)
      for j in 0 ..< newB.len:
        let v = newB[j]
        if v >= 0 and v < EMBEDDING_CODE_COUNT: newB[j] = translation[v]
      g[i].b = newB
      g[i].replacementRevision = freshReplacementRevision()
  g[0].embedding = newMap
  doAssert validEmbedding(g[0].embedding)

proc mutateEmbedding(g: var Genome, mutationScale: float) =
  ## Injection-aware mutation. Most events move one byte into an unused latent
  ## coordinate; swaps/rotations retain local exploration among mapped bytes.
  ## Mutation counts remain log-uniform over the 256 dictionary entries.
  if g.len == 0 or g[0].embedding.len != EMBEDDING_ENTRY_COUNT: return
  if rand(999) >= int(min(160.0, 45.0 * mutationScale)): return
  var newMap = cloneEmbedding(g[0].embedding)
  let changed = sampleLogUniformMutationRange(1, EMBEDDING_ENTRY_COUNT)
  for _ in 0 ..< changed:
    let roll = rand(99)
    if roll < 65:
      let byte = rand(EMBEDDING_ENTRY_COUNT - 1)
      newMap[byte] = sampleUnusedEmbeddingCode(newMap)
    elif roll < 90:
      let a = rand(EMBEDDING_ENTRY_COUNT - 1)
      let b = rand(EMBEDDING_ENTRY_COUNT - 2)
      let other = if b >= a: b + 1 else: b
      swap(newMap[a], newMap[other])
    else:
      let length = sampleLogUniformMutationRange(2, EMBEDDING_ENTRY_COUNT)
      let start = rand(EMBEDDING_ENTRY_COUNT - length)
      let old = newMap[start]
      for j in start ..< start + length - 1:
        newMap[j] = newMap[j + 1]
      newMap[start + length - 1] = old
  applyEmbeddingChange(g, newMap, 90)

proc crossoverEmbedding(child: var Genome, donor: Genome, force = false) =
  ## Injection-preserving allele crossover. A donor target may be currently
  ## unused (direct move) or owned by another byte (swap/displacement).
  if child.len == 0 or donor.len == 0 or
      child[0].embedding.len != EMBEDDING_ENTRY_COUNT or
      donor[0].embedding.len != EMBEDDING_ENTRY_COUNT or
      child[0].embedding == donor[0].embedding: return
  if not force and rand(99) >= 12: return
  var newMap = cloneEmbedding(child[0].embedding)
  var owner: array[EMBEDDING_CODE_COUNT, int]
  owner.fill(-1)
  for byte, code in newMap: owner[code] = byte
  let k = sampleLogUniformMutationRange(1, EMBEDDING_ENTRY_COUNT)
  var indices = newSeq[int](EMBEDDING_ENTRY_COUNT)
  for i in 0 ..< EMBEDDING_ENTRY_COUNT: indices[i] = i
  for i in 0 ..< k:
    let j = i + rand(EMBEDDING_ENTRY_COUNT - 1 - i)
    swap(indices[i], indices[j])
    let byte = indices[i]
    let target = donor[0].embedding[byte]
    let previous = newMap[byte]
    if target == previous: continue
    let other = owner[target]
    if other >= 0:
      newMap[other] = previous
      owner[previous] = other
    else:
      owner[previous] = -1
    newMap[byte] = target
    owner[target] = byte
  applyEmbeddingChange(child, newMap, 90)

proc mutateGenome(
  g: var Genome,
  mutationScale: float = 1.0,
  stagnation: int = 0
) =
  ## Every count-bearing rule mutation draws its size log-uniformly from 1..N.
  ## Event probabilities remain independent of the sampled mutation size.
  if g.len == 0:
    return
  let inheritedEmbedding = g[0].embedding

  # No temporal or branch allele: only a/b/weight may mutate.

  # mutationScale changes event rates and perturbation magnitudes, not the
  # count distribution, which is always the same unrestricted log-uniform law.
  let s = max(0.35, min(1.7, mutationScale))

  # Keep the existing mutual exclusion of row/pattern/block categories.
  # This does not cap the size of the single category selected.
  var majorMutationUsed = false

  # Row swaps draw their affected-rule count from the full logarithmic range.
  if not majorMutationUsed and rand(100) < int(min(45.0, 14.0 * s)):
    if g.len >= 2:
      for i in pickMutationTargets(g):
        let offset = 1 + rand(g.len - 2)
        let j = (i + offset) mod g.len
        swap(g[i], g[j])
    majorMutationUsed = true

  # a <-> b: preserve its low activation probability; sample affected rows.
  if not majorMutationUsed and rand(100) < int(min(28.0, 6.0 * s)):
    for i in pickMutationTargets(g):
      swap(g[i].a, g[i].b)
      sanitizePatternWildcards(g[i])
      ## a/bは親とseq共有され得る。sanitizeが必要な場合だけbをCOW。
      if replacementRefsNeedSanitize(g[i]):
        g[i].b = cloneInts(g[i].b)
        sanitizeReplacementRefs(g[i])
      else:
        g[i].replacementRevision = freshReplacementRevision()
      g[i].patternRevision = freshPatternRevision()
    majorMutationUsed = true

  # Pattern/replacement mutation uses the same log-uniform target count.
  if not majorMutationUsed and rand(100) < int(min(60.0, 26.0 * s)):
    for p in pickMutationTargets(g):
      g[p].a = cloneInts(g[p].a)
      mutateSequence(g[p].a, true, inheritedEmbedding)
      sanitizePatternWildcards(g[p])
      g[p].patternRevision = freshPatternRevision()
      if rand(100) < 65:
        g[p].b = cloneInts(g[p].b)
        mutateSequence(g[p].b, false, inheritedEmbedding)
        g[p].replacementRevision = freshReplacementRevision()
        # Editing b can create an opcode referring to a nonexistent capture.
        if replacementRefsNeedSanitize(g[p]):
          sanitizeReplacementRefs(g[p])
      elif replacementRefsNeedSanitize(g[p]):
        ## aの変異でwildcard数が変わり、bのbackreferenceが不正になる場合。
        g[p].b = cloneInts(g[p].b)
        sanitizeReplacementRefs(g[p])
    majorMutationUsed = true

  # A+(B-C)*{0,1}: change the sampled distinct rule rows.
  if g.len >= 3 and rand(100) < int(min(55.0, 20.0 * s)):
    var changed = newSeq[(int, float)]()
    for ai in pickMutationTargets(g):
      var bi = rand(g.len - 1)
      var ci = rand(g.len - 1)
      while bi == ai:
        bi = rand(g.len - 1)
      while ci == ai or ci == bi:
        ci = rand(g.len - 1)
      # Differential evolution with continuous F; the old {0,1} mask made
      # half of these paid-for mutations exact no-ops. Bound extreme jumps.
      let f = (0.20 + rand(0.60)) * s
      let old = g[ai].weight
      let diff = g[bi].weight - g[ci].weight
      var delta = max(-1.0, min(1.0, f * diff))
      if abs(delta) < 1.0e-12:
        delta = gaussianDelta(0.015 * s)
      if oppositeNonZeroSign(old, old + delta):
        delta = -0.5 * old
      let nw = signPreservingWeight(old, old + delta)
      changed.add((ai, nw))
    for item in changed:
      g[item[0]].weight = item[1]

  # FFT変異は高コストなので、停滞時の脱出専用。
  if stagnation > 10 and s > 1.0 and rand(1000) < int(min(20.0, 4.0 * (s - 1.0) * 100.0 + 2.0)):
    mutateWeightFFT(g)

  # Ordinary weight fine-tuning: event probability and sigma unchanged.
  if rand(100) < int(min(60.0, 25.0 * s)):
    for p in pickMutationTargets(g):
      let oldWeight = g[p].weight
      g[p].weight = signPreservingWeight(
        oldWeight, oldWeight + gaussianDelta(0.05 * s)
      )

  # Building-block relocation/duplication remains an infrequent macro step.
  # Full-length reversal is still reachable by the unrestricted sampler.
  if not majorMutationUsed and g.len >= 16 and rand(1000) < 45:
    let blockLen = sampleLogUniformMutationCount(g.len)
    if blockLen == g.len:
      # A full-genome block has no alternative insertion position; reversing
      # the whole block keeps the sampled N-rule mutation meaningful.
      g.reverse()
    else:
      let src = rand(g.len - blockLen)
      var dst = rand(g.len - blockLen)
      if dst == src:
        dst = (src + 1 + rand(g.len - blockLen - 1)) mod (g.len - blockLen + 1)

      if rand(1) == 0:
        # block relocation: 全ruleを保ったまま区間を移動する。
        relocateGenomeBlock(g, src, blockLen, dst)
      else:
        # block duplication: 有効部品を別位置にも残し、局所的な
        # rule相互作用を増やす。コピー元そのものは消さない。
        # Snapshot overlapping source ranges before writing the destination.
        var copied = newSeq[Rule](blockLen)
        for j in 0 ..< blockLen:
          copied[j] = cloneRuleShallow(g[src + j])
        for j in 0 ..< blockLen:
          g[dst + j] = copied[j]
    majorMutationUsed = true

  # Independent fine-tuning insurance: unchanged activation probability.
  if rand(1000) < int(min(220.0, 120.0 + 140.0 * (s - 1.0))):
    for p in pickMutationTargets(g):
      let oldWeight = g[p].weight
      g[p].weight = signPreservingWeight(
        oldWeight, oldWeight + gaussianDelta(0.035 * s)
      )

  # Sign is deliberately a separate, rare mutation.  This keeps sign
  # evolvable without letting local/magnitude optimization flip it by accident.
  # Guided selection preferentially targets currently harmful rules.
  let signFlipPermille = int(min(18.0, 4.0 + 8.0 * (s - 1.0)))
  if rand(999) < signFlipPermille:
    for p in pickMutationTargets(g):
      if abs(g[p].weight) >= WEIGHT_SIGN_FLOOR:
        g[p].weight = clampWeight(-g[p].weight)


  # Row permutations can displace slot zero. Keep the mapping ONLY in row 0;
  # stale copies in relocated rows must not retain large seq references.
  g[0].embedding = inheritedEmbedding
  for i in 1 ..< g.len:
    g[i].embedding = @[]
  mutateEmbedding(g, s)

  # ここで全ルールを再sanitizeするのは非常に重い。
  # a/bを直接変える各mutation pathでは、そのruleだけ既にsanitize済み。
  # crossover側もcompatibleRuleを通したruleしか移植しないので全走査は不要。

proc randomImmigrant(): Genome =
  makeObj()

proc sortIndicesByScore(scores: openArray[float]): seq[int] =
  ## Nimのメモリ安全性のため、ソート用クロージャからopenArrayを捕捉しない。
  ## コピーはここで1回だけ行い、closureは所有seqを参照する。
  let s = @scores
  result = newSeq[int](s.len)
  for i in 0 ..< s.len:
    result[i] = i

  result.sort(proc(x, y: int): int =
    if s[x] > s[y]:
      -1
    elif s[x] < s[y]:
      1
    else:
      0
  )

# ------------------------------------------------------------
# ★追加: 個体1体分の評価を丸ごと切り出したプロシージャ。
#
# 世代内で pop_size 体それぞれを評価する処理は、個体間で
# 完全に独立している(他個体の情報を一切参照しない)ため、
# これを spawn して全コアに分散するのが最も効果的かつ安全な
# 並列化ポイントになる。
#
# 引数はすべて値型(seq/tuple/float/int)のみで、グローバルな
# 可変状態(patternCache, ファイルハンドル f, progress bar など)
# には一切触れない。これにより spawn / {.gcsafe.} の条件を満たす。
#
# 粒度は「個体1体まるごと」。各ラウンドで1文字列に対して
# 3100ルールを順次適用し、長さ超過でそのサンプルの処理を終了する。
# 仕事量・時間予算の超過は評価例外として扱う。
# ------------------------------------------------------------
proc seenAttPairContains(
  scratch: var EvalScratch, a, b: float64
): bool {.inline.} =
  for item in scratch.seenAttPairs:
    if item.a == a and item.b == b: return true
  false

proc rememberAttPair(
  scratch: var EvalScratch, a, b: float64
) {.inline.} =
  # Keep every accepted 2-gram for this sample. Capping at 64 would miss cycles
  # when sqrt(input.len) exceeds 64, which is exactly the expensive regime.
  scratch.seenAttPairs.add((a: a, b: b))

# Every input follows a single, deterministic sequence of ordered rewrites.
type
  EvaluationBudgetExceeded = object of CatchableError
  EvaluationBudget = object
    started: float
    ruleVisits: int
    lastClockCheck: int
    maxRuleVisits: int
    maxSampleSeconds: float
    clockCheckVisits: int

const
  EVAL_MAX_RULE_VISITS = 1_500_000
  EVAL_MAX_SAMPLE_SECONDS = 10.0
  # Clock reads are only a safety net; semantic/work cutoffs handle normal exit.
  # Checking every 128 *logical* visits was counterproductive after sparse skip:
  # skipped rows are cheap yet still triggered epochTime().  1024 retains a
  # prompt watchdog without taxing the normal sparse evaluator.
  EVAL_CLOCK_CHECK_VISITS = 1024

proc checkEvaluationBudget(budget: var EvaluationBudget) {.inline, gcsafe.} =
  inc budget.ruleVisits
  if budget.ruleVisits > budget.maxRuleVisits:
    raise newException(EvaluationBudgetExceeded,
      "single-trajectory evaluation exceeded its per-sample work budget")
  if budget.ruleVisits - budget.lastClockCheck >= budget.clockCheckVisits:
    budget.lastClockCheck = budget.ruleVisits
    if budget.maxSampleSeconds > 0.0 and
       epochTime() - budget.started > budget.maxSampleSeconds:
      raise newException(EvaluationBudgetExceeded,
        "single-trajectory evaluation exceeded its per-sample time budget")

proc advanceEvaluationBudget(budget: var EvaluationBudget, skippedVisits: int) {.inline, gcsafe.} =
  ## Skipping non-candidate rows changes no evaluation semantics or work ceiling.
  ## Count every logically visited row even though it is no longer iterated.
  if skippedVisits <= 0: return
  budget.ruleVisits += skippedVisits
  if budget.ruleVisits > budget.maxRuleVisits:
    raise newException(EvaluationBudgetExceeded,
      "single-trajectory evaluation exceeded its per-sample work budget")
  if budget.ruleVisits - budget.lastClockCheck >= budget.clockCheckVisits:
    budget.lastClockCheck = budget.ruleVisits
    if budget.maxSampleSeconds > 0.0 and
       epochTime() - budget.started > budget.maxSampleSeconds:
      raise newException(EvaluationBudgetExceeded,
        "single-trajectory evaluation exceeded its per-sample time budget")

proc scoreRaw(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  input: seq[int],
  evalScratch: var EvalScratch,
  maxRounds: int = SCORE_ROUNDS,
  recordCredit: bool = false,
  overflowed: ptr bool = nil,
  featureCoefficients: ptr seq[float] = nil,
  maxRuleVisits: int = EVAL_MAX_RULE_VISITS,
  maxSampleSeconds: float = EVAL_MAX_SAMPLE_SECONDS,
  clockCheckVisits: int = EVAL_CLOCK_CHECK_VISITS
): float {.gcsafe.} =
  ## One text, sequential rules, no parallel state or original-worldline.
  ## Each successful match earns its signed rule weight even if unchanged.
  ## An overlong replacement penalizes the sample by 1 and terminates it.
  ## Repeated adjacent pairs of round scores discard the duplicate round,
  ## including its features and credit (same policy as the old evaluator).
  if overflowed != nil: overflowed[] = false
  if featureCoefficients != nil:
    featureCoefficients[].setLen(genome.len)
    featureCoefficients[].fill(0.0)
  if maxRounds <= 0 or input.len == 0 or genome.len == 0: return 0.0
  let embedded = genome[0].embedding.len == EMBEDDING_ENTRY_COUNT
  let encodedLen = if embedded: input.len * EMBEDDING_WIDTH else: input.len
  let maxAggLen = min(32768,
    max(3100 * (if embedded: EMBEDDING_WIDTH else: 1), encodedLen * 32))
  # Maintain the original compute budget rather than doubling sqrt(N) rounds
  # merely because an external byte is represented by three digits.
  let rounds = min(maxRounds, max(1, int(ceil(2 * sqrt(float(input.len))))))
  # Reuse the per-worker input buffer. A fresh clone allocated and zero-filled
  # one seq for EVERY observation, even if no rule ever matched. move/defer
  # preserve distinct input/output buffers and restore ownership on exceptions.
  var state = move(evalScratch.bufA)
  if embedded:
    embedBytesInto(input, genome[0].embedding, state)
  else:
    state.setLen(input.len)
    copyMem(addr state[0], unsafeAddr input[0], input.len * sizeof(int))
  defer: evalScratch.bufA = move(state)
  var budget = EvaluationBudget(
    started: epochTime(),
    maxRuleVisits: max(1, maxRuleVisits),
    maxSampleSeconds: maxSampleSeconds,
    clockCheckVisits: max(1, clockCheckVisits)
  )
  let needFeatures = recordCredit or featureCoefficients != nil
  var roundFeatures: seq[float] = @[]
  if needFeatures: roundFeatures = newSeq[float](genome.len)
  evalScratch.seenAttPairs.setLen(0)
  var previousAtt = 0.0
  var havePreviousAtt = false
  for roundIdx in 0 ..< rounds:
    discard roundIdx
    var roundScore = 0.0
    var rewrote = false
    var terminated = false
    var candidateValid = false
    if needFeatures: roundFeatures.fill(0.0)
    var k = 0
    var candidateCursor = 0
    var sparseCandidates = false
    while k < genome.len:
      if not candidateValid:
        scanCandidateIds(candidates, state, evalScratch.candidateIds,
          evalScratch.candidateScratch, k)
        candidateValid = true
        # Sorting a short list pays off by avoiding thousands of missed rows.
        # A dense candidate set instead keeps the original O(N) bitmap walk.
        # No possible anchor means no rule can match in the remainder of this
        # ordered sweep. Stop immediately; the outer `not rewrote` guard then
        # terminates the temporal loop as well.
        if evalScratch.candidateIds.len == 0:
          advanceEvaluationBudget(budget, genome.len - k)
          break
        sparseCandidates = evalScratch.candidateIds.len * 3 < genome.len
        if sparseCandidates:
          evalScratch.candidateIds.sort()
          candidateCursor = 0
      if sparseCandidates:
        while candidateCursor < evalScratch.candidateIds.len and
              evalScratch.candidateIds[candidateCursor] < k:
          inc candidateCursor
        if candidateCursor == evalScratch.candidateIds.len:
          advanceEvaluationBudget(budget, genome.len - k)
          break
        let candidateK = evalScratch.candidateIds[candidateCursor]
        advanceEvaluationBudget(budget, candidateK - k)
        k = candidateK
        inc candidateCursor
      checkEvaluationBudget(budget)
      if not sparseCandidates and
          evalScratch.candidateScratch.ruleSeen[k] != evalScratch.candidateScratch.stamp:
        inc k
        continue
      var replacementOverflow = false
      var matched = false
      try:
        matched = replaceSeqCompiledInto(state, compiled[k], genome[k].b,
          evalScratch.bufB, evalScratch.replaceScratch,
          addr evalScratch.replacementPlans[k],
          addr replacementOverflow, maxAggLen,
          genome[k].replacementRevision,
          addr evalScratch.replacementPlanRevision[k])
      except Defect as e:
        raise newException(SeqReplaceError,
          "replacement crashed: rule=" & $k & " state.len=" & $state.len &
          " output.len=" & $evalScratch.bufB.len &
          " maxOutput=" & $maxAggLen & " pattern=" & $genome[k].a &
          " replacement=" & $genome[k].b & "\n" & e.msg &
          "\n" & getStackTrace(e))
      if replacementOverflow:
        roundScore -= 1.0
        if overflowed != nil: overflowed[] = true
        terminated = true
        break
      if not matched:
        inc k
        continue
      rewrote = true
      roundScore += genome[k].weight
      if needFeatures: roundFeatures[k] += 1.0
      if evalScratch.bufB != state:
        swap(state, evalScratch.bufB)
        candidateValid = false
      inc k
    if roundScore != roundScore or abs(roundScore) == Inf:
      raise newException(EvaluationBudgetExceeded,
        "non-finite accumulated round score")
    if havePreviousAtt:
      if seenAttPairContains(evalScratch, previousAtt, roundScore):
        break
      rememberAttPair(evalScratch, previousAtt, roundScore)
    previousAtt = roundScore
    havePreviousAtt = true
    result += roundScore
    if result != result or abs(result) == Inf:
      raise newException(EvaluationBudgetExceeded,
        "non-finite accumulated temporal score")
    if needFeatures:
      for k, coeff in roundFeatures:
        if coeff == 0.0: continue
        if featureCoefficients != nil:
          featureCoefficients[][k] += coeff
        if recordCredit:
          if evalScratch.creditStamp[k] != evalScratch.creditEpoch:
            evalScratch.creditStamp[k] = evalScratch.creditEpoch
            evalScratch.creditTouched.add(k)
          evalScratch.creditContrib[k] += coeff * genome[k].weight
    # Critical fixed-point cutoff: zero successful replacements in a round means
    # every later round would see the identical state and can do no new work.
    if terminated or not rewrote: break

proc evaluateChunk(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  aggsX: seq[seq[int]],
  rankedY: seq[float],
  evalScratch: var EvalScratch,
  maxRounds: int = SCORE_ROUNDS
): float {.gcsafe.} =
  evalScratch.outYs.setLen(0)
  # Exact keys: hash collisions are resolved by sequence equality.
  # No approximation or cross-genome reuse; retain duplicate rank observations.
  # The table itself is worker-local and reused to avoid allocator churn.
  evalScratch.sampleMemo.clear()
  for sampleId, input in aggsX:
    var value: float
    if evalScratch.sampleMemo.hasKey(input):
      value = evalScratch.sampleMemo[input]
    else:
      try:
        value = scoreRaw(genome, compiled, candidates, input, evalScratch, maxRounds)
      except EvaluationBudgetExceeded:
        # Preserve the existing whole-genome invalidation policy for budgets.
        # Only unexpected failures get diagnostic context.
        raise
      except CatchableError as e:
        raise newException(SeqReplaceError,
          "evaluation sample failure: sample=" & $sampleId & " input bytes=" & $input.len &
          " embedded tokens=" & $(input.len *
            (if genome.len > 0 and genome[0].embedding.len == EMBEDDING_ENTRY_COUNT:
              EMBEDDING_WIDTH
            else: 1)) & "\n" & e.msg)
      except Defect as e:
        raise newException(SeqReplaceError,
          "evaluation sample defect: sample=" & $sampleId & " input bytes=" & $input.len &
          "\n" & e.msg & "\n" & getStackTrace(e))
      evalScratch.sampleMemo[input] = value
    evalScratch.outYs.add(value)
  if evalScratch.outYs.len < 2:
    return 0.0
  result = spearmanWithRankedY(evalScratch.outYs, rankedY, evalScratch)

var miningGenome: Genome = @[]
var miningCompiled: seq[SeqPattern] = @[]
var miningCandidates: CandidateIndex

proc prepareScoreScratch(genome: Genome, scratch: var EvalScratch) =
  # Retained as a compatibility no-op; scoreRaw initializes temporal history.
  discard

proc mineCorruptions(
  base: seq[int],
  xs: var seq[seq[int]],
  ys: var seq[int],
  maxStages: int = 4
) =
  ## Bounded hard examples, generated only when a rolling case refreshes.
  ## Targets remain corruption distance, NOT assumed human quality labels.
  ## fast/medium/slowでadversarial mining強度を変え、2世代caseの急な
  ## teacher-driven distribution shiftが選択を振り回さないようにする。
  let stages = max(0, min(4, maxStages))
  if stages == 0 or miningGenome.len == 0 or base.len < 2:
    return
  var scratch = newEvalScratch(miningGenome.len)
  prepareScoreScratch(miningGenome, scratch)
  var state = cloneInts(base)
  for stage in 0 ..< stages:
    var chosen: seq[int] = @[]
    var best = -Inf
    for trial in 0 ..< 12:
      var candidate = cloneInts(state)
      # Fixed length, small byte edits; training already operates on byte noise.
      let edits = 1 + rand(min(7, base.len - 1))
      for k in 0 ..< edits:
        let at = rand(candidate.len - 1)
        candidate[at] = randomDiffByte(candidate[at])
      if candidate == base:
        continue
      try:
        let value = scoreRaw(miningGenome, miningCompiled, miningCandidates,
                             candidate, scratch)
        if value == value and value != Inf and value != -Inf and value > best:
          best = value
          chosen = candidate
      except CatchableError:
        discard
    if chosen.len > 0:
      state = chosen
      let damage = max(1, clampDiffScore(estimateDiffusionDamage(base, state, 0), base.len - 1))
      xs.add(cloneInts(state))
      ys.add(max(1, base.len - damage))


proc evaluateIndividualCases(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int = SCORE_ROUNDS,
  caseTimings: ptr seq[float] = nil,
  exceededCases: ptr int = nil
): seq[float] {.gcsafe.} =
  ## 各chunkを独立した「case」として返す。
  ## 平均を1本だけ返す旧APIとは別に、選抜時のBatch Tournament /
  ## epsilon-Lexicase が case ごとの得意不得意を見られるようにする。
  ##
  ## ★chunksX/chunksY/rankedYsはptr化されている。
  ## 世代内で全個体(400体)から共通して参照される不変データを
  ## spawnのたびに値コピー(ネストseqの深いコピー)させないため。
  ## 呼び出し側はこのデータの生存期間中(バッチ評価が完了するまで)
  ## 決してこの実体を書き換えたり破棄したりしないことを保証する。
  let n = min(chunksX[].len, chunksY[].len)
  result = newSeq[float](n)
  if caseTimings != nil: caseTimings[] = newSeq[float](n)
  if n == 0:
    return

  # weightScore(w) は現在恒等写像なので、3100要素のscore配列を
  # 個体ごとに構築せず evaluateChunk から genome[k].weight を直接読む。
  template evalScratch: var EvalScratch = getThreadEvalScratch(ruleCount)

  for chunkId in 0 ..< n:
    let began = epochTime()
    try:
      result[chunkId] = evaluateChunk(
        genome, compiled, candidates,
        chunksX[][chunkId], rankedYs[][chunkId],
        evalScratch, maxRounds
      )
    except EvaluationBudgetExceeded:
      # Invalidate the WHOLE genome evaluation, not one observation's raw
      # value: substituting one scalar could manufacture an attractive rank.
      for ci in 0 ..< n: result[ci] = -1.0
      if exceededCases != nil: inc exceededCases[]
      if caseTimings != nil:
        caseTimings[][chunkId] = max(1.0e-6, epochTime() - began)
      return
    except CatchableError as e:
      raise newException(SeqReplaceError,
        "case=" & $chunkId & " samples=" & $chunksX[][chunkId].len &
        " " & e.msg)
    except Defect as e:
      raise newException(SeqReplaceError,
        "case=" & $chunkId & " samples=" & $chunksX[][chunkId].len &
        " " & e.msg & "\n" & getStackTrace(e))
    if caseTimings != nil:
      caseTimings[][chunkId] = max(1.0e-6, epochTime() - began)

# Loose performance/fitness Pareto pressure. The absolute best fitness and
# case specialists retain their existing protected selection paths. Runtime
# influences ONLY a minority of parents and at most two extra elite slots.
const
  PARETO_FITNESS_SLACK = 0.012
  PARETO_FITNESS_NOISE = 0.002
  PARETO_TIME_ADVANTAGE = 1.6
  PARETO_PARENT_PERCENT = 30

proc looseRuntimeParetoPool(
  ids: openArray[int], fitness, seconds: openArray[float]
): seq[int] =
  if ids.len == 0: return @[]
  var bestId = ids[0]
  for id in ids:
    if fitness[id] > fitness[bestId]: bestId = id
  for id in ids:
    if fitness[id] < fitness[bestId] - PARETO_FITNESS_SLACK: continue
    var dominated = false
    for other in ids:
      if other == id or fitness[other] < fitness[bestId] - PARETO_FITNESS_SLACK:
        continue
      # Only a substantial, measured 1.6x advantage can justify trading
      # away <=0.012 Spearman. Tiny timing jitter is deliberately ignored.
      if fitness[other] >= fitness[id] - PARETO_FITNESS_NOISE and
         seconds[other] * PARETO_TIME_ADVANTAGE < seconds[id]:
        dominated = true
        break
      if fitness[other] >= fitness[id] + PARETO_FITNESS_NOISE and
         seconds[other] * 1.10 < seconds[id]:
        dominated = true
        break
    if not dominated: result.add(id)
  # Champion is always eligible irrespective of its runtime.
  if bestId notin result: result.add(bestId)

var rollingCaseAge: array[MULTI_CASE_COUNT, int]
for ci in 0 ..< MULTI_CASE_COUNT: rollingCaseAge[ci] = 64
# Collision-free refresh scheduler state. Credits are tiny bounded integers;
# resume reconstructs them deterministically from generation number.
var rollingRefreshCredit: array[3, int]

proc rebuildRollingRefreshCredit(nextIter: int) =
  rollingRefreshCredit.fill(0)
  const add = [6, 22, 33] # FAST=1/11, MEDIUM=1/3, SLOW=1/2 on denominator 66
  for generation in 1 ..< max(1, nextIter):
    discard generation
    for group in 0 .. 2: rollingRefreshCredit[group] += add[group]
    var chosen = -1
    var best = -1
    for group in 0 .. 2:
      if rollingRefreshCredit[group] >= 66 and
         (chosen < 0 or rollingRefreshCredit[group] > best):
        chosen = group
        best = rollingRefreshCredit[group]
    if chosen >= 0: rollingRefreshCredit[chosen] -= 66

# Checkpoint-v9 compatibility field. v4 used this as a chain-linked dataset
# difficulty offset for plotting, but that estimator was biased because the
# survivors used for the bridge had been selected on the outgoing case. Repeated
# refreshes therefore accumulated a spurious positive drift. v5 never uses these
# offsets for fitness or plotting; they are kept only so old v9 checkpoints remain
# byte-compatible.
var progressSigmaOffset: array[MULTI_CASE_COUNT, float]

proc rollingCaseBaseWeight(caseId, caseCount: int): float {.inline.} =
  ## slotそのものの長期的な重要度。これはdataset refresh直後でも変えない。
  ## 以前は refresh 直後に「slotの重みそのもの」を25%へ落としていたため、
  ## 新caseの影響を弱める代わりに残り2caseの相対重みが急上昇し、
  ## aggregate fitnessの分母まで毎世代動くという別のobjective shockを作っていた。
  if caseCount < 3:
    return 1.0
  # More stable (less frequently refreshed) cases receive more long-term
  # aggregate weight. With requested SLOW:MEDIUM:FAST = 2:3:11, FAST is now
  # actually the most stable historical label.
  case caseId mod 3
  of FAST_ROLLING_SLOT: result = 3.0
  of MEDIUM_ROLLING_SLOT: result = 2.0
  of SLOW_ROLLING_SLOT: result = 1.0
  else: result = 1.0

proc rollingCaseMaturity(caseId, caseCount: int): float {.inline.} =
  ## Parent-selection heuristic only: soften lexicase filtering on fresh cases.
  ## Never apply this age-dependent factor to reported/accepted fitness.
  if caseCount < 3:
    return 1.0
  var rampGenerations = 1
  case caseId mod 3
  of FAST_ROLLING_SLOT: rampGenerations = 5   # half of requested width 11
  of MEDIUM_ROLLING_SLOT: rampGenerations = 1 # half of requested width 3
  of SLOW_ROLLING_SLOT: rampGenerations = 1   # half of requested width 2
  else: discard
  let age = if caseId >= 0 and caseId < rollingCaseAge.len:
              rollingCaseAge[caseId]
            else:
              rampGenerations
  min(1.0, 0.25 + 0.75 * float(max(0, age)) / float(max(1, rampGenerations)))

proc rollingCaseSelectionWeight(caseId, caseCount: int): float {.inline.} =
  ## Lexicaseの先頭case抽選だけは新caseを控えめにする。
  ## aggregate側とは違い、ここは順序確率なので maturity を掛けても
  ## objectiveの分母は動かない。
  rollingCaseBaseWeight(caseId, caseCount) * rollingCaseMaturity(caseId, caseCount)

proc aggregateCaseScores(x: openArray[float]): float {.inline.} =
  ## A score must not change solely because a rolling case became older.
  ## Shrinking a case toward the others ALSO redistributes its effective weight,
  ## even with a constant denominator. Use the same fixed objective everywhere.
  ## Maturity remains only a parent-selection exploration heuristic.
  var total = 0.0
  var weightSum = 0.0
  for i, value in x:
    let weight = rollingCaseBaseWeight(i, x.len)
    total += weight * value
    weightSum += weight
  if weightSum <= 0.0: 0.0 else: total / weightSum

const BREEDING_DISPERSION_PENALTY = 0.05

proc stableBreedingScore(x: openArray[float]): float {.inline.} =
  ## Parent selection should prefer improvements that reproduce across rolling
  ## cases, but MUST NOT redefine the reported/FULL objective. Penalize only
  ## cross-case dispersion in the tournament lane. Fresh cases are downweighted
  ## by maturity so a just-refreshed trajectory cannot abruptly suppress a good
  ## lineage. Lexicase still preserves genuine specialists independently.
  let meanScore = aggregateCaseScores(x)
  if x.len < 2:
    return meanScore
  var varianceSum = 0.0
  var weightSum = 0.0
  for i, value in x:
    let weight = rollingCaseBaseWeight(i, x.len) * rollingCaseMaturity(i, x.len)
    let delta = value - meanScore
    varianceSum += weight * delta * delta
    weightSum += weight
  if weightSum <= 0.0:
    return meanScore
  meanScore - BREEDING_DISPERSION_PENALTY * sqrt(max(0.0, varianceSum / weightSum))


proc safeEvaluateIndividualCases(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int = SCORE_ROUNDS,
  caseTimings: ptr seq[float] = nil,
  exceededCases: ptr int = nil
): seq[float] {.gcsafe.} =
  try:
    result = evaluateIndividualCases(
      genome, compiled, candidates, chunksX, chunksY, rankedYs, ruleCount,
      maxRounds, caseTimings, exceededCases
    )
    if result.len == 0 or result.len != min(chunksX[].len, chunksY[].len):
      raise newException(SeqReplaceError, "evaluation returned missing case scores")
    for i in 0 ..< result.len:
      if result[i] != result[i] or result[i] < -1.0 or result[i] > 1.0:
        raise newException(SeqReplaceError,
          "invalid Spearman case=" & $i & " score=" & $result[i])
  except CatchableError:
    # Never turn a broken evaluator into apparently valid zero-fitness data.
    # The worker returns the exception to the main thread with the individual ID.
    raise
  # Defect is deliberately NOT converted to a score either.

var adaptiveLexicaseEpsilon = newSeq[float](MULTI_CASE_COUNT)
for i in 0 ..< adaptiveLexicaseEpsilon.len:
  adaptiveLexicaseEpsilon[i] = LEXICASE_EPSILON

proc medianFloat(values: openArray[float]): float =
  if values.len == 0:
    return 0.0
  var sortedValues = @values
  sortedValues.sort(proc(a, b: float): int = cmp(a, b))
  let mid = sortedValues.len div 2
  if sortedValues.len mod 2 == 1:
    sortedValues[mid]
  else:
    0.5 * (sortedValues[mid - 1] + sortedValues[mid])

proc refreshAdaptiveLexicaseEpsilon(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int]
) =
  ## 固定epsilonだと、caseが易しい/難しいだけでlexicaseの選択圧が変わる。
  ## 現世代FULL poolのMADからcaseごとの許容幅を作る標準的な
  ## epsilon-lexicase寄りの挙動にして、dataset refresh後の選択圧ショックを抑える。
  if candidateIds.len == 0:
    return
  let caseCount = min(MULTI_CASE_COUNT, caseScores[candidateIds[0]].len)
  for ci in 0 ..< caseCount:
    var values: seq[float] = @[]
    for id in candidateIds:
      if id >= 0 and id < caseScores.len and ci < caseScores[id].len:
        values.add(caseScores[id][ci])
    if values.len < 3:
      adaptiveLexicaseEpsilon[ci] = LEXICASE_EPSILON
      continue
    let med = medianFloat(values)
    var deviations = newSeq[float](values.len)
    for i, v in values:
      deviations[i] = abs(v - med)
    let mad = medianFloat(deviations)
    # Spearmanの[-1,1]スケール上で極端な緩さ/厳しさにはしない。
    adaptiveLexicaseEpsilon[ci] = max(0.004, min(0.035, 1.5 * mad))

proc epsilonLexicaseSelect(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int]
): int =
  ## 最大化版 epsilon-Lexicase。
  ## 各caseで「そのcaseの最良値から epsilon 以内」を残し、
  ## case順を毎回シャッフルすることで、総合平均だけでは残りにくい
  ## specialist も親として選ばれる。
  if candidateIds.len == 0:
    return -1
  if candidateIds.len == 1:
    return candidateIds[0]

  let caseCount = if caseScores[candidateIds[0]].len > 0:
                    caseScores[candidateIds[0]].len
                  else:
                    0
  if caseCount == 0:
    return candidateIds[rand(candidateIds.len - 1)]

  var pool = newSeq[int](candidateIds.len)
  for i, id in candidateIds:
    pool[i] = id

  var caseOrder = newSeq[int](caseCount)
  for i in 0 ..< caseCount:
    caseOrder[i] = i

  # Weighted shuffle without replacement. Lexicaseは最初のcaseの影響が特に強いので、
  # 2世代ごとに変わるfast caseが毎回1/3の確率で選択を支配しないようにする。
  # weightはaggregateと同じ1:2:3系だが、全caseは必ず1回ずつ評価される。
  for pos in 0 ..< caseCount:
    var totalWeight = 0.0
    for j in pos ..< caseCount:
      totalWeight += rollingCaseSelectionWeight(caseOrder[j], caseCount)
    if totalWeight <= 0.0:
      break
    let r = rand(totalWeight)
    var accWeight = 0.0
    var chosen = pos
    for j in pos ..< caseCount:
      accWeight += rollingCaseSelectionWeight(caseOrder[j], caseCount)
      if r <= accWeight:
        chosen = j
        break
    swap(caseOrder[pos], caseOrder[chosen])

  for ci in caseOrder:
    var best = -Inf
    for id in pool:
      best = max(best, caseScores[id][ci])

    var nextPool: seq[int] = @[]
    let baseEpsilon = if ci < adaptiveLexicaseEpsilon.len:
                        adaptiveLexicaseEpsilon[ci]
                      else:
                        LEXICASE_EPSILON
    # refresh直後のcaseはまだランダムdifficulty shockを多く含むため、
    # maturityが低い間だけepsilonを広げて「一発で全親候補を落とす」力を弱める。
    let maturity = rollingCaseMaturity(ci, caseCount)
    let epsilon = min(0.12, baseEpsilon / max(0.25, maturity))
    let threshold = best - epsilon
    for id in pool:
      if caseScores[id][ci] >= threshold:
        nextPool.add(id)

    if nextPool.len == 0:
      break
    pool = nextPool
    if pool.len == 1:
      return pool[0]

  pool[rand(pool.len - 1)]

proc batchTournamentSelect(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int]
): int =
  ## Lexicaseがcase別specialistを拾う一方、Tournament側は現在の全rolling case
  ## の固定重み平均を主軸に、case間分散へごく小さい保守ペナルティを掛ける。
  ## 2ケースしかないのに1ケースをランダム抽出すると、
  ## せっかくfast/slowの二時定数にした評価軸が親選択時に再び単一case化し、
  ## 選択圧が世代ごとに振れやすいため。
  if candidateIds.len == 0:
    return -1
  var bestId = candidateIds[rand(candidateIds.len - 1)]
  var bestScore = -Inf
  for _ in 0 ..< BATCH_TOURNAMENT_SIZE:
    let id = candidateIds[rand(candidateIds.len - 1)]
    if caseScores[id].len == 0:
      continue
    let s = stableBreedingScore(caseScores[id])
    if s > bestScore:
      bestScore = s
      bestId = id
  bestId

proc multiCaseSelect(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int]
): int =
  ## specialist重視のepsilon-Lexicaseを主軸にしつつ、
  ## 一部をBatch Tournamentにして平均性能も維持するハイブリッド。
  if rand(999) < int(LEXICASE_RATE * 1000.0):
    return epsilonLexicaseSelect(caseScores, candidateIds)
  batchTournamentSelect(caseScores, candidateIds)

# 親候補プールから完全ランダムに1体を選ぶ探索用経路。
# Lexicase/Tournamentだけでは同じ少数親へ遺伝子が集中しやすいため、
# 低確率で選択圧を意図的に外す。
const PARENT_RANDOM_RATE_BASE = 0.035
const PARENT_RANDOM_RATE_MAX = 0.12
# 世代内で1個体が過剰に繁殖するのを防ぐ。
const PARENT_MAX_USES = 16

proc selectParentId(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int],
  randomRate: float
): int {.inline.} =
  ## 親poolを「評価値だけでほぼ同じ少数個体へ収束」させないための
  ## 小さな探索経路。大半は従来のmulti-case選択を使い、一定確率だけ
  ## pool全体から一様抽出する。candidateIdsは全てFULL評価済み。
  if candidateIds.len == 0:
    return -1
  if rand(999) < int(max(0.0, min(PARENT_RANDOM_RATE_MAX, randomRate)) * 1000.0):
    return candidateIds[rand(candidateIds.len - 1)]
  multiCaseSelect(caseScores, candidateIds)

proc selectParentCapped(
  caseScores: seq[seq[float]],
  candidateIds: openArray[int],
  useCount: var seq[int],
  randomRate: float
): int {.inline.} =
  ## 同じ親が世代内に過剰繁殖しないよう、まず通常選択し、
  ## 上限へ到達した個体は数回だけ引き直す。全員が上限なら
  ## 最小使用回数の親へフォールバックする。
  if candidateIds.len == 0:
    return -1
  for _ in 0 ..< 6:
    let id = selectParentId(caseScores, candidateIds, randomRate)
    if id < 0:
      return -1
    if id >= useCount.len or useCount[id] < PARENT_MAX_USES:
      if id >= 0 and id < useCount.len:
        inc useCount[id]
      return id
  var bestId = candidateIds[0]
  var bestUses = high(int)
  var tied = 0
  for id in candidateIds:
    let u = if id < useCount.len: useCount[id] else: 0
    if u < bestUses:
      bestUses = u
      bestId = id
      tied = 1
    elif u == bestUses:
      inc tied
      if rand(tied - 1) == 0: bestId = id
  if bestId >= 0 and bestId < useCount.len:
    inc useCount[bestId]
  bestId

proc releaseParentUse(useCount: var seq[int], id: int) {.inline.} =
  if id >= 0 and id < useCount.len and useCount[id] > 0: dec useCount[id]

proc replaceParentUse(useCount: var seq[int], oldId, newId: int) {.inline.} =
  # newId already reserved by selectParentCapped. Undo only the old reservation.
  releaseParentUse(useCount, oldId)

proc evaluateIndividual(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int = SCORE_ROUNDS
): float {.gcsafe.} =
  ## Scalar/local-search evaluation must have EXACTLY the same objective as
  ## FULL selection. An unweighted mean can accept a weighted-score regression.
  let scores = evaluateIndividualCases(
    genome, compiled, candidates, chunksX, chunksY, rankedYs, ruleCount, maxRounds)
  aggregateCaseScores(scores)


proc pearsonFromStat(st: RuleCreditStat, ySum, y2Sum: float, n: int): float {.inline.} =
  if n < 2 or st.fired == 0:
    return 0.0
  let nf = float(n)
  let cov = st.xySum - st.xSum * ySum / nf
  let vx = st.x2Sum - st.xSum * st.xSum / nf
  let vy = y2Sum - ySum * ySum / nf
  if vx <= 1.0e-12 or vy <= 1.0e-12:
    return 0.0
  let den = sqrt(vx * vy)
  if den <= 1.0e-12:
    return 0.0
  max(-1.0, min(1.0, cov / den))

proc evaluateIndividualWithCredit(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: seq[seq[seq[int]]],
  chunksY: seq[seq[int]],
  rankedYs: seq[seq[float]],
  ruleCount: int,
  outCredit: var seq[float]
): float {.gcsafe.} =
  ## 通常評価と同じ単一路径の規則適用を辿りながら、
  ## 「各ruleが各sampleの出力へどれだけ寄与したか」を蓄積する。
  ## アブレーションのように3100回再評価する必要はなく、追加コストは
  ## 実際に発火したruleの本数にほぼ比例する。
  outCredit.setLen(ruleCount)
  outCredit.fill(0.0)
  if chunksX.len == 0:
    return 0.0

  var stats = newSeq[RuleCreditStat](ruleCount)
  ## statsは「このchunkで一度でも発火したrule」だけを初期化する。
  ## 毎sampleでruleCount全体をfillするのは、ruleCount≈3100では無視できない。
  var statTouched = newSeq[int](0)
  var creditSum = newSeq[float](ruleCount)
  # weightScore(w) == w のためruleScores配列は不要。
  var evalScratch = newEvalScratch(ruleCount, 2048)
  prepareScoreScratch(genome, evalScratch)

  var totalCorr = 0.0
  var validChunks = 0
  var scoredChunks = 0

  for chunkId in 0 ..< min(chunksX.len, chunksY.len):
    let aggsX = chunksX[chunkId]
    let aggsY = chunksY[chunkId]
    if aggsX.len < 2 or aggsY.len != aggsX.len:
      continue

    evalScratch.outYs.setLen(0)
    ## 前chunkで使われたstatsだけをzero化する。
    for k in statTouched:
      stats[k] = RuleCreditStat()
    statTouched.setLen(0)

    # ruleCount分の作業配列はEvalScratchに持たせ、chunk/sampleを跨いで再利用する。
    evalScratch.creditContrib.setLen(ruleCount)
    ## creditStampはsample単位で進める。これにより各sampleで実際に
    ## 寄与したruleだけを一度ずつcreditTouchedへ登録できる。
    evalScratch.creditTouched.setLen(0)

    var chunkYSum = 0.0
    var chunkY2Sum = 0.0
    var chunkN = 0

    for i in 0 ..< aggsX.len:
      inc evalScratch.creditEpoch
      if evalScratch.creditEpoch >= high(int):
        evalScratch.creditStamp.fill(0)
        evalScratch.creditEpoch = 1
      evalScratch.creditTouched.setLen(0)
      var blewUp = false
      let sampleScore = scoreRaw(genome, compiled, candidates, aggsX[i],
        evalScratch, SCORE_ROUNDS, true, addr blewUp)
      evalScratch.outYs.add(sampleScore)
      # Previous contributions survive an overlong replacement; this
      # trajectory cannot resume after a length-limit failure.
      let y = aggsY[i].float
      chunkYSum += y
      chunkY2Sum += y * y
      inc chunkN
      for k in evalScratch.creditTouched:
        let x = evalScratch.creditContrib[k]
        if stats[k].fired == 0:
          statTouched.add(k)
        stats[k].xSum += x
        stats[k].x2Sum += x * x
        stats[k].xySum += x * y
        inc stats[k].fired
        evalScratch.creditContrib[k] = 0.0
      evalScratch.creditTouched.setLen(0)

    let corr = spearmanWithRankedY(evalScratch.outYs, rankedYs[chunkId], evalScratch)
    totalCorr += corr
    inc scoredChunks
    if chunkN > 1:
      for k in statTouched:
        creditSum[k] += pearsonFromStat(stats[k], chunkYSum, chunkY2Sum, chunkN)
      inc validChunks

  if validChunks > 0:
    let invChunks = 1.0 / float(validChunks)
    for k in 0 ..< ruleCount:
      outCredit[k] = creditSum[k] * invChunks
  if scoredChunks > 0:
    let corr = totalCorr / float(scoredChunks)
    if corr != corr or corr < -1.0 or corr > 1.0:
      return 0.0
    return corr

  0.0

proc extractGuidedModules(
  genome: Genome,
  credits: openArray[float]
): seq[GuidedModule] =
  ## 高寄与の連続区間をmodule化する。
  ## 1本の巨大moduleにならないよう最大長を制限し、局所ピークから伸ばす。
  if genome.len == 0 or credits.len != genome.len:
    return @[]

  type Candidate = tuple[startPos: int, endPos: int, score: float]
  var candidates: seq[Candidate] = @[]

  for i in 0 ..< genome.len:
    if credits[i] < 0.10:
      continue
    var left = i
    var right = i
    var score = credits[i]
    while left > 0 and i - left < MODULE_MAX_LEN - 1 and credits[left - 1] >= 0.06:
      dec left
      score += credits[left]
    while right + 1 < genome.len and right - i < MODULE_MAX_LEN - 1 and credits[right + 1] >= 0.06:
      inc right
      score += credits[right]
    let len = right - left + 1
    if len >= MODULE_MIN_LEN:
      candidates.add((left, right, score / float(len)))

  candidates.sort(proc(a, b: Candidate): int =
    if a.score > b.score: -1
    elif a.score < b.score: 1
    else: 0
  )

  var pickedRanges: seq[tuple[l: int, r: int]] = @[]
  for c in candidates:
    if result.len >= MAX_GUIDED_MODULES:
      break
    if c.endPos < c.startPos:
      continue

    # 高得点候補同士がほぼ同じ場所を取り合うのを防ぐ。
    var overlap = false
    for pr in pickedRanges:
      let interL = max(pr.l, c.startPos)
      let interR = min(pr.r, c.endPos)
      if interL <= interR:
        let interLen = interR - interL + 1
        let cLen = c.endPos - c.startPos + 1
        if interLen * 2 >= min(cLen, pr.r - pr.l + 1):
          overlap = true
          break
    if overlap:
      continue

    let mlen = c.endPos - c.startPos + 1
    var rules = newSeq[Rule](mlen)
    for j in 0 ..< mlen:
      rules[j] = cloneRuleShallow(genome[c.startPos + j])
    result.add(GuidedModule(rules: rules, score: c.score,
      startPos: c.startPos))
    pickedRanges.add((c.startPos, c.endPos))

proc refreshGuidedModules(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: seq[seq[seq[int]]],
  chunksY: seq[seq[int]],
  rankedYs: seq[seq[float]]
) =
  if genome.len == 0:
    return
  var credits = newSeq[float](genome.len)
  discard evaluateIndividualWithCredit(
    genome, compiled, candidates, chunksX, chunksY, rankedYs, genome.len, credits
  )

  # Rule positions in different champion genomes are not the same features.
  # Never blend credit from unrelated rules occupying the same array index.
  guidedCredit = credits
  guidedReferenceGenome = cloneGenome(genome)

  guidedModules = extractGuidedModules(genome, guidedCredit)

proc safeEvaluateIndividual(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int = SCORE_ROUNDS
): float {.gcsafe.}

proc buildRankedY(ys: seq[seq[int]]): seq[seq[float]] =
  ## 評価データセット固定期間中は y の順位が変わらないので一度だけ求める。
  result = newSeq[seq[float]](ys.len)
  for i in 0 ..< ys.len:
    result[i] = rank(ys[i])

proc buildFastEvalDataset(
  srcX: seq[seq[seq[int]]],
  srcY: seq[seq[int]],
  stride: int = 2
): tuple[x: seq[seq[seq[int]]], y: seq[seq[int]]] =
  ## 予備評価用の薄いデータセット。
  ## 8点の元系列からstride=4なら3点（先頭・中点・末尾）程度が残る。
  ## 2点Spearmanにならないことを重視し、「粗いランキング → 上位だけ完全評価」に使う。
  result.x = newSeq[seq[seq[int]]](0)
  result.y = newSeq[seq[int]](0)
  let step = max(1, stride)
  for c in 0 ..< min(srcX.len, srcY.len):
    let n = min(srcX[c].len, srcY[c].len)
    if n < 2:
      continue
    var cx: seq[seq[int]] = @[]
    var cy: seq[int] = @[]
    cx.add(srcX[c][0])
    cy.add(srcY[c][0])
    var i = step
    while i < n - 1:
      cx.add(srcX[c][i])
      cy.add(srcY[c][i])
      i += step
    if n > 1:
      cx.add(srcX[c][n - 1])
      cy.add(srcY[c][n - 1])
    if cx.len >= 2:
      result.x.add(cx)
      result.y.add(cy)

# ONE-ROUND linear response for fixed a/b. Multi-round
# 2-gram termination depends on weights, so never use this cache for FULL
# selection or local search. Kept solely for the explicit one-round test.
type
  SparseRuleCoefficient = tuple[id: int, value: float]
  LinearWeightCache = object
    baseline: seq[seq[float]]
    features: seq[seq[seq[SparseRuleCoefficient]]]
    baseWeights: seq[float]

proc buildLinearWeightCache(
  genome: Genome, compiled: seq[SeqPattern], candidates: CandidateIndex,
  chunksX: seq[seq[seq[int]]], chunksY: seq[seq[int]]
): LinearWeightCache {.gcsafe.} =
  let n = min(chunksX.len, chunksY.len)
  result.baseline = newSeq[seq[float]](n)
  result.features = newSeq[seq[seq[SparseRuleCoefficient]]](n)
  result.baseWeights = newSeq[float](genome.len)
  for ri, r in genome: result.baseWeights[ri] = r.weight
  var scratch = newEvalScratch(genome.len, 2048)
  prepareScoreScratch(genome, scratch)
  var coefficients: seq[float] = @[]
  for ci in 0 ..< n:
    result.baseline[ci] = newSeq[float](chunksX[ci].len)
    result.features[ci] = newSeq[seq[SparseRuleCoefficient]](chunksX[ci].len)
    for si, x in chunksX[ci]:
      result.baseline[ci][si] = scoreRaw(genome, compiled, candidates, x,
        scratch, 1, false, nil, addr coefficients)
      for ri, coeff in coefficients:
        if coeff != 0.0:
          result.features[ci][si].add((id: ri, value: coeff))

proc scoreLinearProposal(
  cache: LinearWeightCache, proposal: Genome,
  rankedYs: seq[seq[float]]
): float {.gcsafe.} =
  var caseScores = newSeq[float](cache.baseline.len)
  # Reuse rank scratch; do not allocate replacementPlans/pairSeen per trial.
  template rankScratch: var EvalScratch = getThreadEvalScratch(proposal.len)
  for ci in 0 ..< cache.baseline.len:
    var outputs = newSeq[float](cache.baseline[ci].len)
    for si in 0 ..< outputs.len:
      var score = cache.baseline[ci][si]
      for term in cache.features[ci][si]:
        score += term.value * (proposal[term.id].weight -
          cache.baseWeights[term.id])
      if score != score or score == Inf or score == -Inf:
        return -Inf
      outputs[si] = score
    caseScores[ci] = spearmanWithRankedY(outputs, rankedYs[ci], rankScratch)
  aggregateCaseScores(caseScores)

proc optimizeWeightsByCredit(
  genome: var Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: seq[seq[seq[int]]],
  chunksY: seq[seq[int]],
  rankedYs: seq[seq[float]],
  currentScore: float
): float {.gcsafe.} =
  ## RuleCredit を「診断」で終わらせず、そのまま weight 局所探索へ接続する。
  ##
  ## credit は「現在の signed rule contribution」と目的値の Pearson 相関。
  ## そのまま勾配方向に使うと負weightで符号が逆になるため、更新時には
  ## current weight sign を掛け戻して raw firing feature の相関へ直す。
  ## Spearman目的そのものの厳密勾配ではないので、最終的には実測採用する。
  ##
  ## ★変更: 以前は「credit計算 → 3段階のstep幅を試す」を1回だけ行う
  ## one-shotな局所探索だった。
  ##
  ## この評価器では rewrite の発火系列は a/b だけで決まり、weight は
  ## 出力attへ線形に足されるだけで発火系列自体を変えない。さらに Pearson
  ## corr は正のscaleに不変なので、符号を固定した magnitude 最適化の途中で
  ## creditを再計算しても理論上ほぼ同じ値になる。そこでcreditは1回だけ計算し、
  ## 複数stepの「提案→FULL実測採用」だけを反復する。これで旧実装の無駄な
  ## credit再評価を消しつつ、Spearmanに対する安全なline-searchは維持する。
  if genome.len == 0 or chunksX.len == 0 or chunksY.len == 0:
    return currentScore

  # Spearmanは順位ベースなので、weightScoreDerivative をそのまま掛けても
  # 目的関数の勾配にはならない。creditは「方向情報」としてのみ使い、
  # boundedなstepで提案→実測採用する。
  const STEPS = [0.08, 0.20]
  const MAX_ROUNDS = 2

  var best = currentScore
  var stale = 0

  # One credit pass is enough while rule signs/structure stay fixed.
  var credits = newSeq[float](genome.len)
  discard evaluateIndividualWithCredit(
    genome, compiled, candidates, chunksX, chunksY, rankedYs, genome.len, credits
  )
  # A score 2-gram is an early-stop condition: changing a rule's weight can
  # change which round is the last accepted round. LinearWeightCache is not
  # an exact proposal evaluator here; use the authoritative FULL objective.
  var rawFeatureCredit = newSeq[float](genome.len)
  for i in 0 ..< genome.len:
    rawFeatureCredit[i] =
      if genome[i].weight < 0.0: -credits[i]
      elif genome[i].weight > 0.0: credits[i]
      else: 0.0

  for round in 0 ..< MAX_ROUNDS:
    var roundImproved = false
    for step in STEPS:
      var proposal = cloneGenome(genome)
      var touched = 0
      for i in 0 ..< proposal.len:
        # evaluateIndividualWithCredit が返す credit は
        #   corr(weight * firingFeature, y)
        # なので、負weightでは raw feature の相関符号が反転している。
        # その値をそのまま dScore/dWeight の向きとして使うと、負weightだけ
        # 更新方向が逆になる（例: w<0, raw corr>0 -> credit<0 -> 更に負へ）。
        # weightの符号を掛け戻し、
        #   corr(firingFeature, y)
        # 相当の方向へ戻してから局所最適化する。
        let c = rawFeatureCredit[i]
        let ac = abs(c)
        if ac < 0.05:
          continue
        let direction = if c > 0.0: 1.0 else: -1.0
        # |credit|が大きいほど強く、ただし1 stepを越えない。
        let scale = min(1.0, ac)
        let delta = direction * step * scale
        if abs(delta) > 1.0e-6:
          let oldWeight = proposal[i].weight
          proposal[i].weight = signPreservingWeight(
            oldWeight, oldWeight + delta
          )
          inc touched

      if touched == 0:
        continue

      let score = safeEvaluateIndividual(proposal, compiled, candidates,
        unsafeAddr chunksX, unsafeAddr chunksY, unsafeAddr rankedYs,
        genome.len, SCORE_ROUNDS)
      if score > best + 1.0e-9:
        genome = proposal
        best = score
        roundImproved = true

    if roundImproved:
      stale = 0
    else:
      inc stale
      if stale >= 2:
        break

  best

proc safeEvaluateIndividual(
  genome: Genome,
  compiled: seq[SeqPattern],
  candidates: CandidateIndex,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int
): float {.gcsafe.} =
  ## 1個体の評価失敗で全GAを終了させない防波堤.
  ## 通常系では例外は発生しないが、突然変異や壊れたcheckpoint等で
  ## 想定外の評価失敗が出ても、その個体だけ最低評価として扱う。
  try:
    let r = evaluateIndividual(
      genome, compiled, candidates, chunksX, chunksY, rankedYs, ruleCount, maxRounds
    )
    if r != r or r < -1.0 or r > 1.0:
      raise newException(SeqReplaceError,
        "invalid scalar Spearman=" & $r & " rounds=" & $maxRounds)
    r
  except CatchableError as e:
    # Local weight search must not accept a failed evaluation as fitness zero.
    raise newException(SeqReplaceError, "scalar evaluation failed: " & e.msg)
  # Defects propagate: they are implementation errors, not bad genomes.


# ------------------------------------------------------------
# 個体間並列評価
# ------------------------------------------------------------
# 評価関数は個体ごとに完全独立なので、「1個体=1ジョブ」で並列化する。
# workerから共有キャッシュ、progress bar、populationの再構築などは一切触らない。
# これにより、以前SIGSEGVの原因になった共有可変状態への並列アクセスを避ける。
# ------------------------------------------------------------
# spawn用のread-only参照ラッパー
#
# データセットだけでなく population / compiledPopulation も全ジョブ共通の
# 巨大な不変データである。個体seqをspawn引数へ直接渡さず、外側seqへの
# ptr + id だけを渡す。バッチ関数は全FlowVar回収まで戻らないので、
# ptrの生存期間はジョブ実行期間を完全に包含する。
# ------------------------------------------------------------
proc safeEvaluateIndividualById(
  population: ptr seq[Genome],
  compiledPopulation: ptr seq[seq[SeqPattern]],
  candidatePopulation: ptr seq[CandidateIndex],
  id: int,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int
): float {.gcsafe.} =
  safeEvaluateIndividual(
    population[][id],
    compiledPopulation[][id],
    candidatePopulation[][id],
    chunksX, chunksY, rankedYs, ruleCount, maxRounds
  )

type
  CaseEvaluationOutcome = object
    scores: seq[float]
    caseTimings: seq[float]
    exceededCases: int
    error: string

proc safeEvaluateIndividualCasesById(
  population: ptr seq[Genome],
  compiledPopulation: ptr seq[seq[SeqPattern]],
  candidatePopulation: ptr seq[CandidateIndex],
  id: int,
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int
): CaseEvaluationOutcome {.gcsafe.} =
  # Return failures as data only so ALL FlowVars can be drained before the
  # main thread raises; otherwise other workers might outlive stack pointers.
  try:
    result.scores = safeEvaluateIndividualCases(
      population[][id],
      compiledPopulation[][id],
      candidatePopulation[][id],
      chunksX, chunksY, rankedYs, ruleCount, maxRounds,
      addr result.caseTimings, addr result.exceededCases
    )
  except CatchableError as e:
    result.error = "CatchableError: " & e.msg & "\n" & getStackTrace(e)
  except Defect as e:
    result.error = "Defect: " & e.msg & "\n" & getStackTrace(e)


proc parallelEvaluateIndividualBatch(
  population: seq[Genome],
  compiledPopulation: seq[seq[SeqPattern]],
  candidatePopulation: seq[CandidateIndex],
  ids: seq[int],
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int
): seq[float] {.gcsafe.} =
  result = newSeq[float](ids.len)
  if ids.len == 0:
    return
  # ★変更: 以前は workerCount*2 件ずつの「バッチ」に区切り、
  # バッチ全体が完了してから次のバッチをspawnしていた。
  # 個体ごとの評価コストにばらつきがある(特にFULL評価で
  # 置換列が伸びやすい個体が混ざる)場合、バッチ内の1個体が
  # 重いだけで、既に処理を終えた他のワーカーがバッチの区切りまで
  # 完全に遊んでしまう問題があった。
  # Nimのstd/threadpoolはspawn自体がジョブキューを持ち、空いた
  # ワーカーへ自動的に次のジョブを割り当てる(spawnはキューが
  # 埋まっていればその場でブロックして空くのを待つ)ため、
  # ここでは対象全件を一括でspawnしてから順に回収すればよい。
  # これによりワーカー間の負荷分散をthreadpool任せにでき、
  # 「重い個体の待ち」で他コアが遊ぶ時間を減らせる。
  var jobs = newSeq[FlowVar[float]](ids.len)
  let populationPtr = unsafeAddr population
  let compiledPopulationPtr = unsafeAddr compiledPopulation
  let candidatePopulationPtr = unsafeAddr candidatePopulation
  for j in 0 ..< ids.len:
    let id = ids[j]
    jobs[j] = spawn safeEvaluateIndividualById(
      populationPtr, compiledPopulationPtr, candidatePopulationPtr, id,
      chunksX, chunksY, rankedYs, ruleCount, maxRounds
    )
  for j in 0 ..< ids.len:
    result[j] = ^jobs[j]


proc parallelEvaluateCasesBatch(
  population: seq[Genome],
  compiledPopulation: seq[seq[SeqPattern]],
  candidatePopulation: seq[CandidateIndex],
  ids: seq[int],
  chunksX: ptr seq[seq[seq[int]]],
  chunksY: ptr seq[seq[int]],
  rankedYs: ptr seq[seq[float]],
  ruleCount: int,
  maxRounds: int,
  phase: string = "FULL",
  measuredTimings: ptr seq[seq[float]] = nil
): seq[seq[float]] {.gcsafe.} =
  result = newSeq[seq[float]](ids.len)
  if measuredTimings != nil:
    measuredTimings[] = newSeq[seq[float]](ids.len)
  if ids.len == 0:
    return
  # parallelEvaluateIndividualBatch と同じ理由で、バッチ分割の
  # 同期バリアを外し、対象全件を一括spawnしてthreadpoolに
  # 負荷分散を任せる。
  var jobs = newSeq[FlowVar[CaseEvaluationOutcome]](ids.len)
  let populationPtr = unsafeAddr population
  let compiledPopulationPtr = unsafeAddr compiledPopulation
  let candidatePopulationPtr = unsafeAddr candidatePopulation
  for j in 0 ..< ids.len:
    let id = ids[j]
    jobs[j] = spawn safeEvaluateIndividualCasesById(
      populationPtr, compiledPopulationPtr, candidatePopulationPtr, id,
      chunksX, chunksY, rankedYs, ruleCount, maxRounds
    )
  var firstFailure = ""
  var totalBudgetFailures = 0
  for j in 0 ..< ids.len:
    let outcome = ^jobs[j]
    totalBudgetFailures += outcome.exceededCases
    if outcome.error.len > 0:
      let message = "evaluation failure: phase=" & phase &
        " id=" & $ids[j] & " rounds=" & $maxRounds & " " & outcome.error
      stderr.writeLine(message)
      if firstFailure.len == 0:
        firstFailure = message
    else:
      result[j] = outcome.scores
      if measuredTimings != nil:
        measuredTimings[][j] = outcome.caseTimings
  if totalBudgetFailures > 0:
    echo "  evaluation_budget: phase=", phase,
         " invalidated_genomes=", totalBudgetFailures,
         " (fitness=-1, no partial rank scores)"
  if firstFailure.len > 0:
    # All worker jobs have finished. Never return a partly scored population.
    raise newException(SeqReplaceError, firstFailure)

# ------------------------------------------------------------
# 世代ごとの進捗をCSVに記録し、gnuplotでPNGにプロットする。
#
# 生のspearman(-1..1)は1に近づくほど差が見えにくくなるので、
# 各世代を -log2(1 - spearman) に変換してから、その系列に
# 移動平均をかけてグラフ化する。つまり「対数化 → 移動平均」。
#
# gnuplot が入っていない環境でも GA 自体は止まらないよう、
# execCmd の失敗は無視する。
# ------------------------------------------------------------
import std/strformat
import std/osproc

# CSV is appended every generation; only PNG rendering is throttled.
# Gnuplot's two stats passes and plot pass reread the full history each run.
const PLOT_INTERVAL = 1

proc erfinv(x: float): float =
  ## Inverse error function (Winitzki + Newton refinement).
  if x < -1.0 or x > 1.0:
    raise newException(ValueError, "erfinv domain is [-1, 1]")
  if x == -1.0:
    return -Inf
  if x == 1.0:
    return Inf
  if x == 0.0:
    return 0.0

  let sign = if x < 0.0: -1.0 else: 1.0
  let ax = abs(x)
  const a = 0.147
  let lnTerm = ln(1.0 - ax * ax)
  let first = 2.0 / (PI * a) + lnTerm / 2.0
  let second = lnTerm / a

  var y = sign * sqrt(sqrt(first * first - second) - first)
  for _ in 0 ..< 5:
    let err = erf(y) - x
    let deriv = 2.0 / sqrt(PI) * exp(-y * y)
    if deriv < 1.0e-300:
      break
    let delta = err / deriv
    y -= delta
    if abs(delta) < 1.0e-15 * max(1.0, abs(y)):
      break
  y

proc spearmanToSigma(x: float): float {.inline.} =
  ## Spearman=1.0 で -log2(0)=Inf になり、移動平均や分散を
  ## NaN/Infへ汚染するのを防ぐ。
  const EPS = 1.0e-12
  let p = max(-1.0 + EPS, min(1.0 - EPS, x))
  -log2(1.0 - p)

proc sigmaToProbability(z: float): float {.inline.} =
  if z != z:
    return 0.0
  if z <= 0.0:
    return 0.0
  let p = 1.0 - pow(2.0, -z)
  max(0.0, min(1.0 - 1.0e-12, p))

const MA_WINDOW = 75  # Elite-of-Elites 判定用の移動平均窓(世代数)
const PLOT_MA_SHORT_WINDOW = 200   # グラフ用の短期移動平均窓
const PLOT_MA_LONG_WINDOW = 200  # グラフ用の長期移動平均窓（CSV互換・内部計算用。描画はしない）

# Display completed generations, not the zero-based iter stored in CSV.
# Both axes span the same full drawing height but have independent data ranges.
const PLOT_START_GENERATION = 200
const PLOT_AXIS_PADDING = 0.08
const PLOT_LEFT_MIN_SPAN = 0.01
const PLOT_RIGHT_MIN_SPAN = 0.002

const PLOT_SCRIPT = """
set datafile separator ","
set terminal pngcairo size 1000,600
set output 'progress.png.pending'
set title "GA progress: moving-dataset training fitness (not held-out validation)"
set xlabel "generation (completed; starts at 200)"
set ylabel "mean[-log2(1 - best Spearman)]"
set grid
set key left top
set ytics nomirror
set xrange [__XMIN__:__XMAX__]
set yrange [__LEFT_LO__:__LEFT_HI__]
__JEV_AXIS__

# The saved CSV uses zero-based iter. Display iter 199 as generation 200.
# Never connect lines to Jev observations from before completed generation 200.
plot 'progress.csv' using (($1+1 >= 200) ? $1+1 : 1/0):4 axes x1y1 \
       with linespoints pt 7 ps 0.35 lw 2 title 'Spearman MA200'__JEV_SERIES__
"""

type PlotSeriesRange = object
  valid: bool
  lo, hi: float
  maxGeneration: int

proc scanPlotSeries(path: string, valueColumn: int, maxZeroBasedGeneration = high(int)): PlotSeriesRange =
  ## CSV generation is zero based: the first complete 200-generation
  ## Spearman window is iter 199; its displayed x coordinate is 200.
  if not fileExists(path): return
  try:
    for line in lines(path):
      let fields = line.split(',')
      if fields.len <= valueColumn: continue
      try:
        let observedIter = parseInt(fields[0])
        if observedIter >= maxZeroBasedGeneration: continue
        let completed = observedIter + 1
        if completed < PLOT_START_GENERATION: continue
        let value = parseFloat(fields[valueColumn])
        if value != value or value == Inf or value == -Inf: continue
        if not result.valid:
          result.valid = true
          result.lo = value
          result.hi = value
        else:
          result.lo = min(result.lo, value)
          result.hi = max(result.hi, value)
        result.maxGeneration = max(result.maxGeneration, completed)
      except ValueError:
        discard # Header, partial write, or malformed row: never invent a point.
  except OSError:
    result = PlotSeriesRange() # Plotting must not terminate the GA.

proc paddedPlotRange(data: PlotSeriesRange, minimumSpan: float): tuple[lo, hi: float] =
  ## Equal 8% fractional padding on both independently autoscaled axes.
  ## A constant/single-point series receives a nonzero minimum height.
  let observedSpan = max(data.hi - data.lo, minimumSpan)
  let pad = observedSpan * PLOT_AXIS_PADDING
  result.lo = data.lo - pad - max(0.0, minimumSpan - (data.hi - data.lo)) / 2.0
  result.hi = data.hi + pad + max(0.0, minimumSpan - (data.hi - data.lo)) / 2.0

const PLOT_PNG_PATH = "progress.png"
const PLOT_PNG_PENDING = "progress.png.pending"
var plotUnavailableReported = false
var jevProgressPath = "jev_progress.csv" # assigned from checkpointPath before training
var previousJevProgressPath = "jev_progress.previous.csv" # assigned alongside jevProgressPath
var archivedJevUpperGeneration = -1 # disable old-configuration plot for new runs

proc plotProgress() =
  if findExe("gnuplot").len == 0:
    if not plotUnavailableReported:
      stderr.writeLine("gnuplot not found: continuing training with CSV output only.")
      plotUnavailableReported = true
    return
  # Render synchronously: an old gnuplot process must never overwrite a newer
  # plot after checkpoint restoration. Keep the old image until a valid new PNG
  # is fully rendered, then replace it in one same-directory rename.

  # Compute actual extrema ONLY for rows with completed generation >= 200.
  # This avoids hard-coding Jev to 0..1 (flattening its visible variation),
  # and gives both independent axes equally tall plot regions.
  let left = scanPlotSeries("progress.csv", 3)
  if not left.valid:
    return  # Never overwrite a valid PNG with an empty range.
  let leftAxis = paddedPlotRange(left, PLOT_LEFT_MIN_SPAN)
  let right = scanPlotSeries(jevProgressPath, 2)
  # A new Jev evaluation budget/prompt/model is intentionally NOT mixed into
  # its old moving average. Display the archived history as a separate,
  # clearly labelled historical line instead of making it vanish on resume.
  var previous: PlotSeriesRange
  if archivedJevUpperGeneration >= 0:
    previous = scanPlotSeries(previousJevProgressPath, 2, archivedJevUpperGeneration)
  let xmax = max(PLOT_START_GENERATION + 1,
                 max(left.maxGeneration, max(right.maxGeneration, previous.maxGeneration)))
  var jevAxis = "unset y2tics\nunset y2label"
  var jevSeries = ""
  if right.valid or previous.valid:
    var bounds = if right.valid: right else: previous
    if right.valid and previous.valid:
      bounds.lo = min(right.lo, previous.lo)
      bounds.hi = max(right.hi, previous.hi)
    let rightAxis = paddedPlotRange(bounds, PLOT_RIGHT_MIN_SPAN)
    jevAxis = "set y2label \"Jev rank-weighted quality MA20 (not accuracy)\"\n" &
      "set y2tics\nset y2range [" & $rightAxis.lo & ":" & $rightAxis.hi & "]"
    if previous.valid:
      let oldPath = "\"" & previousJevProgressPath.replace("\\", "\\\\").replace("\"", "\\\"") & "\""
      jevSeries.add(", " & "\\" & "\n     " & oldPath &
        " using (($1+1 >= 200 && $1 < " & $archivedJevUpperGeneration &
        ") ? $1+1 : 1/0):3 axes x1y2" &
        " with linespoints dt 2 pt 6 ps 0.4 lw 1 title 'Jev previous settings (not comparable)'")
    if right.valid:
      let quotedPath = "\"" & jevProgressPath.replace("\\", "\\\\").replace("\"", "\\\"") & "\""
      jevSeries.add(", " & "\\" & "\n     " & quotedPath &
        " using (($1+1 >= 200) ? $1+1 : 1/0):3 axes x1y2" &
        " with linespoints pt 5 ps 0.45 lw 2 title 'Jev rank-weighted MA20 (current settings)'")
  var script = PLOT_SCRIPT.replace("__XMIN__", $PLOT_START_GENERATION)
  script = script.replace("__XMAX__", $xmax)
  script = script.replace("__LEFT_LO__", $leftAxis.lo)
  script = script.replace("__LEFT_HI__", $leftAxis.hi)
  script = script.replace("__JEV_AXIS__", jevAxis)
  script = script.replace("__JEV_SERIES__", jevSeries)
  var procHandle: Process = nil
  try:
    # The previous progress.png remains intact through rendering/failure.
    if fileExists(PLOT_PNG_PENDING): removeFile(PLOT_PNG_PENDING)
    writeFile("plot.gnuplot", script)
    procHandle = startProcess("gnuplot", args = @["plot.gnuplot"],
                              options = {poUsePath, poStdErrToStdOut})
    let status = waitForExit(procHandle)
    let diagnostics = outputStream(procHandle).readAll()
    close(procHandle)
    procHandle = nil
    if status != 0:
      stderr.writeLine("gnuplot failed (exit ", status, "): ", diagnostics)
      return
    if not fileExists(PLOT_PNG_PENDING):
      stderr.writeLine("gnuplot produced no PNG; old progress.png was preserved. ", diagnostics)
      return
    let bytes = readFile(PLOT_PNG_PENDING)
    if bytes.len < 45 or bytes[0 ..< 8] != "\x89PNG\r\n\x1a\n" or
       bytes[bytes.len - 12 ..< bytes.len] != "\x00\x00\x00\x00IEND\xae\x42\x60\x82":
      stderr.writeLine("gnuplot PNG is incomplete/invalid; old progress.png was preserved.")
      return
    moveFile(PLOT_PNG_PENDING, PLOT_PNG_PATH)
  except CatchableError as e:
    stderr.writeLine("gnuplot refresh failed; old progress.png preserved: ", e.msg)
  finally:
    if not procHandle.isNil:
      try: close(procHandle)
      except CatchableError: discard
    if fileExists(PLOT_PNG_PENDING):
      try: removeFile(PLOT_PNG_PENDING)
      except CatchableError: discard

# ------------------------------------------------------------
# GA 本体
# ------------------------------------------------------------

let eliteCount = 40

# Elite-of-Elites injection budget. Keep it a sparse restart mechanism rather
# than a second population that is re-measured every generation.
# Archive injection is deliberately sparse during productive search. Rechecking
# 40 historical genomes every generation spent a large fraction of the FULL
# budget measuring old lineages instead of new offspring. During stagnation we
# temporarily reopen the archive more aggressively as an explicit restart lane.
const ARCHIVE_INJECT_BASE = 8
const ARCHIVE_INJECT_STAGNANT = 20
const ARCHIVE_INJECT_INTERVAL = 4
const ARCHIVE_STAGNATION_TRIGGER = 12
# 長時間運転で「新記録」のたびに3100-rule genomeを無制限に保持すると
# アーカイブだけでメモリを食い潰す可能性があるため上限を設ける。
const ARCHIVE_MAX_ENTRIES = 2048

# Elite-of-Elites は population へ再注入して「現在の rolling dataset」で
# 再評価させる。過去の別dataset上のscoreを根拠に親へ直接バイパスしない。

# 粗密二段階評価。全個体を薄いサンプルで一次選抜し、
# 上位だけ完全評価する。個体数が増えても計算量が素直に増えにくい。
# FAST sample thinning. Higher stride = fewer observations = faster screening.
# Start substantially thinner than the old 4, but retain a 4-point safety floor
# that the rejection audit can fall back to. Clean high-correlation generations
# may climb as far as 8 (roughly half the observations of stride=4 on 40-point cases).
const FAST_EVAL_STRIDE = 6
const FAST_EVAL_STRIDE_MIN = 4
# Six-to-seven retained points per ~40-sample case is the sparsest screening
# allowed. stride=8 made rank estimates noticeably quantized near the cutoff.
const FAST_EVAL_STRIDE_MAX = 7
var adaptiveFastEvalStride = FAST_EVAL_STRIDE
const FULL_EVAL_TOP = 42
# Temporal depth is now elastic. Start at the previous 16-round cost, but if
# paired FAST/FULL evidence degrades, allow the screener to deepen to 24 rather
# than remaining permanently biased by an artificial 16-round ceiling.
const FAST_EVAL_MIN_ROUNDS = 12
const FAST_EVAL_INITIAL_ROUNDS = 16
const FAST_EVAL_MAX_ROUNDS = 24
const ADAPTIVE_FULL_EXTRA_MAX = 16
var adaptiveFastEvalRounds = FAST_EVAL_INITIAL_ROUNDS
# FULL評価は上位だけに閉じず、親候補の遺伝的多様性を確保するため
# 毎世代ランダムな個体も追加で完全評価する。
const FULL_EVAL_DIVERSITY_EXTRA = 8
# Uniform audit of candidates excluded by FAST/merit/diversity.
# FULL-evaluated audit candidates are eligible parents, not discarded probes.
# Six independent rejection probes give the FAST gate enough evidence to
# detect false negatives without returning to the old eight-probe overhead.
# Audit discoveries are rescued into the genuine FULL parent/survivor pool.
const FAST_REJECT_AUDIT_COUNT = 6
const FAST_AUDIT_MISS_MARGIN = 0.005
# A single small merit miss is noisy evidence. Only a best miss, repeated merit
# misses, or a clearly large margin should force a denser FAST dataset.
const FAST_AUDIT_SEVERE_MARGIN = 0.030
const WEIGHT_LOCAL_SEARCH_INTERVAL = 128
const WEIGHT_LOCAL_SEARCH_STAGNANT_INTERVAL = 64
const WEIGHT_LOCAL_SEARCH_STAGNATION_TRIGGER = 12

const IMMIGRANT_RATE = 0.005
const STAGNATION_IMMIGRANT_BONUS = 0.03

# パターンキャッシュを定期的にクリアする間隔。
# 無制限に貯め続けるとユニークな .a 列の総数に比例してメモリが
# 増え続けるため、ある程度で捨てて作り直す。世代内の再利用効果は
# この間隔内で十分得られる。
const CACHE_CLEAR_INTERVAL = 2000

# ★追加: futureのポーリング間隔と、詰まっている個体を
# ログに出すまでの待機回数。
const POLL_INTERVAL_MS = 10
const STUCK_LOG_EVERY_N_POLLS = 500  # 10ms * 500 = 5秒ごと

# 学習用chunkは===SPLIT===間の全バイト列をそのまま使用する。
# 長いchunkは時間・メモリを大幅に消費し得るが、長さによる切り出しはしない。
# 評価の既存の実行予算・異常終了処理は変更しない。
const MIN_EVAL_CHUNK_LEN = 96

var bestEver = -Inf
var stagnation = 0
# 現population先頭の「前世代からそのまま継いだ個体」数。
# fresh runでは0。checkpoint再開直後だけ安全側の上限を使い、次世代から実数へ更新する。
var protectedPrefixCount = 0

proc randomArchiveIndex(): int {.inline.} =
  ## rolling datasetでは何百世代も前の個体ほど現在の探索分布から外れやすい。
  ## 70%は最近256件、30%は全履歴から取り、再利用性と長期多様性を両立する。
  if eliteOfElites.len == 0:
    return -1
  if eliteOfElites.len <= 256 or rand(99) >= 70:
    return rand(eliteOfElites.len - 1)
  let firstRecent = max(0, eliteOfElites.len - 256)
  firstRecent + rand(eliteOfElites.len - 1 - firstRecent)

proc archiveRecord(score: float, genome: Genome, generation: int) =
  ## 非有限値は保存しない。
  ## 重複を除き、上限超過時は最古の個体を退避する。
  ## 異なるrolling dataset上の過去スコアで生存を決めない。
  if score != score or score == Inf or score == -Inf:
    return
  let fp = genomeFingerprint(genome)
  for i in 0 ..< eliteOfElites.len:
    if eliteOfElites[i].fingerprint == fp and
       sameGenomeContent(eliteOfElites[i].genome, genome):
      # Exact duplicate: refresh metadata; distinct weight solutions remain separate.
      let refreshed = cloneGenome(genome)
      eliteOfElites[i].genome = refreshed
      eliteOfElites[i].score = score
      eliteOfElites[i].generation = generation
      eliteOfElites[i].fingerprint = genomeFingerprint(refreshed)
      return

  let storedGenome = cloneGenome(genome)
  eliteOfElites.add(EliteArchiveEntry(
    genome: storedGenome,
    score: score,
    generation: generation,
    fingerprint: genomeFingerprint(storedGenome)
  ))

  if eliteOfElites.len > ARCHIVE_MAX_ENTRIES:
    var worst = 0
    for i in 1 ..< eliteOfElites.len:
      if eliteOfElites[i].generation < eliteOfElites[worst].generation:
        worst = i
    eliteOfElites.delete(worst)

# Hall identity is the complete genome (including its byte embedding/weights).
# A new entry is provisional. Only three DIFFERENT successful Jev rounds can
# confirm it; historical rubric scores are never used as GA fitness.
proc jevHallIndex(genome: Genome): int =
  let fp = genomeFingerprint(genome)
  for i in 0 ..< jevHall.len:
    if jevHall[i].fingerprint == fp and
       sameGenomeContent(jevHall[i].genome, genome):
      return i
  -1

proc confirmedJevHallMatch(genome: Genome, fingerprint: Hash): bool =
  ## Fingerprint is calculated once by the caller, and exact equality resolves
  ## collisions. Never use historical Spearman or Jev quality as GA fitness.
  for entry in jevHall:
    if entry.wins >= JEV_HALL_CONFIRM_WINS and
       entry.fingerprint == fingerprint and
       sameGenomeContent(entry.genome, genome):
      return true
  false

proc jevHallObserve(genome: Genome, quality: float, generation: int,
                    topFinish: bool) =
  if quality != quality or quality < 0.0 or quality > 1.0:
    return
  let existing = jevHallIndex(genome)
  if existing >= 0:
    # One observation per genome per round, including a non-winning recheck.
    if jevHall[existing].lastGeneration == generation: return
    let oldN = jevHall[existing].observations
    jevHall[existing].meanQuality =
      (jevHall[existing].meanQuality * float(oldN) + quality) / float(oldN + 1)
    inc jevHall[existing].observations
    if topFinish: inc jevHall[existing].wins
    jevHall[existing].lastGeneration = generation
    return
  if not topFinish or quality <= 0.0: return
  if jevHall.len >= JEV_HALL_CAPACITY:
    # A one-off winner must never evict a previously confirmed champion.
    var evict = -1
    for i in 0 ..< jevHall.len:
      if jevHall[i].wins < JEV_HALL_CONFIRM_WINS and
         (evict < 0 or jevHall[i].lastGeneration < jevHall[evict].lastGeneration):
        evict = i
    if evict < 0: return
    jevHall.delete(evict)
  let saved = cloneGenome(genome)
  jevHall.add(JevHallEntry(genome: saved,
    fingerprint: genomeFingerprint(saved), meanQuality: quality,
    observations: 1, wins: 1,
    firstGeneration: generation, lastGeneration: generation))

proc jevHallInjectionOrder(): seq[int] =
  ## Balance provisional rechecks with confirmed long-term champions: a full
  ## provisional queue must never starve confirmed entries (or vice versa).
  var provisional: seq[int] = @[]
  var confirmed: seq[int] = @[]
  for i in 0 ..< jevHall.len:
    if jevHall[i].wins >= JEV_HALL_CONFIRM_WINS: confirmed.add(i)
    else: provisional.add(i)
  proc olderFirst(a, b: int): int =
    let byAge = cmp(jevHall[a].lastGeneration, jevHall[b].lastGeneration)
    if byAge != 0: return byAge
    cmp(jevHall[b].meanQuality, jevHall[a].meanQuality)
  provisional.sort(olderFirst)
  confirmed.sort(olderFirst)
  let firstP = min(JEV_HALL_INJECT_SLOTS div 2, provisional.len)
  let firstC = min(JEV_HALL_INJECT_SLOTS div 2, confirmed.len)
  for i in 0 ..< firstP: result.add(provisional[i])
  for i in 0 ..< firstC: result.add(confirmed[i])
  var p = firstP
  var c = firstC
  while p < provisional.len or c < confirmed.len:
    if p < provisional.len:
      result.add(provisional[p])
      inc p
    if c < confirmed.len:
      result.add(confirmed[c])
      inc c

# Adaptive second objective: current FULL G, observed J, exact genome identity.
# No inherited child score and no comparison of scores across unlike cohorts.
type
  FusionMember = object
    genome: Genome
    fingerprint: Hash
    quality: float
    generation: int
  FusionPending = object
    genome: Genome
    due: int
  FusionPoint = object
    id: int
    g, j: float
var fusionArchive: seq[FusionMember] = @[]
var fusionAnchors: seq[FusionMember] = @[]
var fusionPending: seq[FusionPending] = @[]
var fusionReliability = 0.0
var fusionEvidence = 0
var fusionNoise = 0.0
const FUSION_ANCHORS = 8
const FUSION_CAP = 16
const FUSION_NURSERY = 8
const FUSION_INTERVAL = 16

proc resetFusion() =
  fusionArchive.setLen(0)
  fusionAnchors.setLen(0)
  fusionPending.setLen(0)
  fusionReliability = 0.0
  fusionEvidence = 0
  fusionNoise = 0.0

proc fusionMatch(g: Genome, members: seq[FusionMember]): int =
  if members.len == 0: return -1
  let fp = genomeFingerprint(g)
  for i, member in members:
    if fp == member.fingerprint and sameGenomeContent(g, member.genome): return i
  -1

proc uniqueGenomeIds(population: seq[Genome], ids: seq[int],
                     fingerprints: var seq[Hash]): seq[int] =
  fingerprints.setLen(population.len)
  var buckets = initTable[Hash, seq[int]]()
  for id in ids:
    let fp = genomeFingerprint(population[id])
    fingerprints[id] = fp
    var duplicate = false
    for previous in buckets.getOrDefault(fp):
      if sameGenomeContent(population[id], population[previous]):
        duplicate = true
        break
    if not duplicate:
      buckets.mgetOrPut(fp, @[]).add(id)
      result.add(id)

proc fusionSnapshot(g: Genome, q: float, generation: int): FusionMember =
  result.genome = cloneGenome(g)
  result.fingerprint = genomeFingerprint(result.genome)
  result.quality = q
  result.generation = generation

proc fusionRepeatSignal(a, b: seq[float]): tuple[signal, noise: float] =
  # Conservative heuristic, NOT a calibrated confidence/probability. Compare
  # the same anchors on both occasions; remove a common rubric level shift.
  if a.len < 6 or a.len != b.len: return
  let ma = a.foldl(a + b, 0.0) / float(a.len)
  let mb = b.foldl(a + b, 0.0) / float(b.len)
  var va, vb, vd: float
  for i in 0 ..< a.len:
    let x = a[i] - ma
    let y = b[i] - mb
    va += x*x
    vb += y*y
    vd += (x-y)*(x-y)
  if va <= 1e-18 or vb <= 1e-18: return
  let spread = (va + vb) / (2.0 * float(a.len))
  let noiseVar = vd / (2.0 * float(a.len))
  result.noise = sqrt(noiseVar)
  let rho = max(-0.999, min(0.999, spearman(a, b)))
  # Small panels must not confer full authority after one lucky ordering.
  let z = 0.5 * ln((1.0 + rho) / (1.0 - rho))
  let lowerZ = z - 1.64 / sqrt(float(a.len - 3))
  let lowerR = max(0.0, (exp(2.0*lowerZ)-1.0)/(exp(2.0*lowerZ)+1.0))
  result.signal = lowerR * max(0.0, 1.0 - noiseVar / spread)

proc fusionUpdateTrust(a, b: seq[float]) =
  let measure = fusionRepeatSignal(a, b)
  if a.len >= 6: inc fusionEvidence
  let target = measure.signal * min(1.0, float(fusionEvidence)/4.0)
  let alpha = if target < fusionReliability: 0.5 else: 0.25
  fusionReliability += alpha * (target - fusionReliability)
  fusionNoise = measure.noise

proc fusionFront(points: seq[FusionPoint], cap: int): seq[int] =
  # Transitive epsilon-box dominance. The earlier proposed pairwise +/- eps
  # tolerance is not a partial order and must NOT be used for nondominated sort.
  if cap <= 0: return
  let eg = 0.002
  let ej = max(0.005, min(0.10, fusionNoise))
  var valid: seq[int] = @[]
  var gxBox, jyBox = newSeq[float](points.len)
  for i, p in points:
    if p.g != p.g or p.j != p.j or abs(p.g) == Inf or abs(p.j) == Inf: continue
    valid.add(i)
    gxBox[i] = floor(p.g/eg)
    jyBox[i] = floor(p.j/ej)
  if valid.len == 0: return
  for i in valid:
    let p = points[i]
    var dominated = false
    let gx = gxBox[i]
    let jy = jyBox[i]
    for k in valid:
      let q = points[k]
      if i == k: continue
      let qg = gxBox[k]
      let qj = jyBox[k]
      if (qg >= gx and qj >= jy and (qg > gx or qj > jy)) or
         (qg == gx and qj == jy and
          (q.g > p.g or (q.g == p.g and (q.j > p.j or
           (q.j == p.j and k < i))))):
        dominated = true
        break
    if not dominated: result.add(i)
  # Explicitly retain both true extrema; HV alone does not guarantee this.
  var bestG = valid[0]
  var bestJ = valid[0]
  for i in valid:
    if points[i].g > points[bestG].g or
       (points[i].g == points[bestG].g and points[i].j > points[bestG].j): bestG = i
    if points[i].j > points[bestJ].j or
       (points[i].j == points[bestJ].j and points[i].g > points[bestJ].g): bestJ = i
  for i in [bestG, bestJ]:
    if i notin result: result.add(i)
  if cap == 1: return @[bestG]
  # Sort once; removals preserve this order. HV deletion scans allocate nothing.
  result.sort(proc(a,b: int): int = cmp(points[b].g, points[a].g))
  proc area(ids: seq[int], omit: int): float =
    var height = 0.0
    for id in ids:
      if id == omit: continue
      # Fixed reference (-1.01,-0.01); never normalize by a changing cohort.
      let y = points[id].j + 0.01
      if y > height:
        result += max(0.0, points[id].g + 1.01) * (y-height)
        height = y
  while result.len > cap:
    let total = area(result, -1)
    var victim = -1
    var least = Inf
    for pos, id in result:
      if id == bestG or id == bestJ: continue
      let contribution = max(0.0, total - area(result, id))
      if contribution < least or (contribution == least and
         (victim < 0 or id < result[victim])):
        least = contribution
        victim = pos
    if victim < 0: break
    result.delete(victim)
  result.sort(proc(a,b: int): int = cmp(points[a].g, points[b].g))

proc fusionInject(dest: var seq[Genome], generation, limit: int) =
  var fps: seq[Hash] = @[]
  for g in dest: fps.add(genomeFingerprint(g))
  proc addUnique(dest: var seq[Genome], fps: var seq[Hash], g: Genome, limit: int) =
    if dest.len >= limit: return
    let fp = genomeFingerprint(g)
    for i, old in dest:
      if fps[i] == fp and sameGenomeContent(old, g): return
    dest.add(cloneGenome(g))
    fps.add(fp)
  # Only inject calibration models immediately before an actual Jev round.
  if (generation + 1) mod FUSION_INTERVAL == 0:
    for anchor in fusionAnchors: addUnique(dest, fps, anchor.genome, limit)
  for pending in fusionPending:
    if generation <= pending.due: addUnique(dest, fps, pending.genome, limit)
  let count = if (generation + 1) mod FUSION_INTERVAL == 0: fusionArchive.len
              else: min(fusionArchive.len, int(round(FUSION_CAP.float*fusionReliability)))
  # Evenly spaced along the front, including both extremes when count >= 2.
  for i in 0 ..< count:
    let pos = if count <= 1: fusionArchive.len-1
              else: i*(fusionArchive.len-1) div (count-1)
    addUnique(dest, fps, fusionArchive[pos].genome, limit)

proc bestArchiveEntry(): int =
  if eliteOfElites.len == 0:
    return -1
  var best = 0
  for i in 1 ..< eliteOfElites.len:
    if eliteOfElites[i].score > eliteOfElites[best].score:
      best = i
  best

# ★追加: -ln(1-best) の単純移動平均を計算するための履歴バッファ。
var maHistory: seq[float] = @[]
var maHead = 0
var maCount = 0
var maSum = 0.0

proc maReset() =
  maHistory.setLen(MA_WINDOW)
  maHead = 0
  maCount = 0
  maSum = 0.0

proc maPush(v: float) {.inline.} =
  if maHistory.len != MA_WINDOW:
    maHistory.setLen(MA_WINDOW)
  if maCount < MA_WINDOW:
    let idx = (maHead + maCount) mod MA_WINDOW
    maHistory[idx] = v
    inc maCount
    maSum += v
  else:
    maSum -= maHistory[maHead]
    maHistory[maHead] = v
    maSum += v
    maHead = (maHead + 1) mod MA_WINDOW

proc maValueAt(i: int): float {.inline.} =
  maHistory[(maHead + i) mod MA_WINDOW]

proc maValuesChronological(): seq[float] =
  result = newSeqOfCap[float](maCount)
  for i in 0 ..< maCount:
    result.add(maValueAt(i))

proc archiveLogThreshold(): tuple[ready: bool, value: float, mean: float, stddev: float] =
  ## Compare the new champion ONLY against preceding generations. Inserting the
  ## current best first pulls the threshold towards the value being tested.
  ## At least two historical observations are needed for a useful dispersion.
  if maCount < 2:
    return (false, Inf, 0.0, 0.0)
  let mean = maSum / float(maCount)
  var varianceSum = 0.0
  for i in 0 ..< maCount:
    let delta = maValueAt(i) - mean
    varianceSum += delta * delta
  let stddev = sqrt(max(0.0, varianceSum / float(maCount)))
  (true, mean + stddev, mean, stddev)

# ------------------------------------------------------------
# セーブ / ロード
#
# 「世代をまたいで続きから再開」するため、populationだけでなく
# 次の世代番号、bestEver/stagnation、移動平均、グラフ履歴、
# Elite-of-Elites、現在の評価データセットまで保存する。
#
# 使い方:
#   ./at                         # 新規開始 + 毎世代保存
#   ./at --load                  # ga_checkpoint.bin から再開
#   ./at --load=foo.bin          # 指定チェックポイントから再開
#   ./at --save=foo.bin          # 保存先だけ変更
# ------------------------------------------------------------

type
  ProgressPoint = object
    iter: int
    best: float
    transformed: float
    sqrtMovingAvg: float
    fixedMovingAvg: float

const CHECKPOINT_MAGIC = "KLEISMIC_GA_CHECKPOINT"
const CHECKPOINT_VERSION = 24
const CHECKPOINT_OBJECTIVE_VERSION = 23 # 512-token latent-space objective
var migratedRuleCount = false
var loadedCheckpointVersion = 0
const CHECKPOINT_INTERVAL = 50
const TARGET_GENERATIONS = 1000000

var progressHistory: seq[ProgressPoint] = @[]

proc writeIntSeq(s: Stream, xs: seq[int]) =
  s.write(xs.len.int64)
  for x in xs:
    s.write(x.int64)

proc readIntSeq(s: Stream): seq[int] =
  let n64 = s.readInt64()
  if n64 < 0 or n64 > 100_000_000:
    raise newException(IOError, "Invalid int sequence length in checkpoint.")
  let n = int(n64)
  result = newSeq[int](n)
  for i in 0 ..< n:
    result[i] = int(s.readInt64())

proc writeRule(s: Stream, r: Rule) =
  writeIntSeq(s, r.a)
  writeIntSeq(s, r.b)
  s.write(r.weight)

proc readRule(s: Stream, version: int): Rule =
  result.a = readIntSeq(s)
  result.b = readIntSeq(s)
  result.weight = s.readFloat64()
  if version >= 8 and version <= 15:
    discard s.readFloat64() # historical reserved temporal allele
  if version == 15:
    let obsoleteWeight = s.readFloat64()
    let obsoleteMode = uint8(s.readInt8())
    if obsoleteWeight != obsoleteWeight or abs(obsoleteWeight) > 2.0 or
       obsoleteMode > 0b111'u8:
      raise newException(IOError, "Invalid legacy branch allele in checkpoint/model.")
    # All old branching alleles are deliberately dropped; only a/b/w survive.
  result.patternRevision = freshPatternRevision()
  result.replacementRevision = freshReplacementRevision()

proc writeGenome(s: Stream, g: Genome) =
  s.write(g.len.int64)
  for r in g:
    writeRule(s, r)
  if g.len == 0 or g[0].embedding.len != EMBEDDING_ENTRY_COUNT:
    writeIntSeq(s, defaultEmbedding())
  else:
    if not validEmbedding(g[0].embedding):
      raise newException(IOError, "Invalid embedding before checkpoint/model save.")
    writeIntSeq(s, g[0].embedding)

proc readGenome(s: Stream, version: int): Genome =
  let n64 = s.readInt64()
  if n64 < 0 or n64 > 1_000_000:
    raise newException(IOError, "Invalid genome length in checkpoint.")
  let n = int(n64)
  result = newSeq[Rule](n)
  for i in 0 ..< n:
    result[i] = readRule(s, version)
    ## 古い/手動編集されたcheckpointに不正opcodeが残っていても、
    ## replacement compilerへ渡す前に安全なliteralへ正規化する。
    sanitizePatternWildcards(result[i])
    sanitizeReplacementRefs(result[i])
  if version >= 20:
    let storedMapping = readIntSeq(s)
    let mappingValid =
      if version <= 21: validLegacyBytePermutation(storedMapping)
      elif version == 22: validV22Embedding(storedMapping)
      else: validEmbedding(storedMapping)
    if not mappingValid:
      raise newException(IOError, "Invalid or duplicate embedding in checkpoint/model.")
    if n > 0:
      if version <= 21:
        # v20/v21 reserved positive literal 256 as OOV. In the current model
        # 256 is a genuine latent coordinate; move only the old sentinel.
        for i in 0 ..< n:
          var changedA = false
          var changedB = false
          for j in 0 ..< result[i].a.len:
            if result[i].a[j] == INPUT_BYTE_COUNT:
              result[i].a[j] = EMBEDDING_OOV
              changedA = true
          for j in 0 ..< result[i].b.len:
            if result[i].b[j] == INPUT_BYTE_COUNT:
              result[i].b[j] = EMBEDDING_OOV
              changedB = true
          if changedA: result[i].patternRevision = freshPatternRevision()
          if changedB: result[i].replacementRevision = freshReplacementRevision()
        result[0].embedding = storedMapping
      elif version == 22:
        let migrated = migrateV22Embedding(storedMapping)
        for i in 0 ..< n:
          var changedA = false
          var changedB = false
          for j in 0 ..< result[i].a.len:
            let v = result[i].a[j]
            if v >= 0:
              result[i].a[j] = if v <= V22_EMBEDDING_OOV: migrated.translation[v]
                               else: EMBEDDING_OOV
              changedA = true
          for j in 0 ..< result[i].b.len:
            let v = result[i].b[j]
            if v >= 0:
              result[i].b[j] = if v <= V22_EMBEDDING_OOV: migrated.translation[v]
                               else: EMBEDDING_OOV
              changedB = true
          if changedA: result[i].patternRevision = freshPatternRevision()
          if changedB: result[i].replacementRevision = freshReplacementRevision()
          sanitizeReplacementRefs(result[i])
        result[0].embedding = migrated.mapping
      else:
        result[0].embedding = storedMapping
  elif version in [18, 19]:
    let oldMap = readIntSeq(s)
    let codeCount = (if version == 18: LEGACY_OCTAL4_CODE_COUNT
                     else: LEGACY_OCTAL3_CODE_COUNT)
    let mapping = migrateLegacyMap(oldMap, codeCount)
    if n > 0:
      for i in 0 ..< n:
        result[i].a = migrateLegacyEncodedLiterals(result[i].a,
          oldMap, mapping, (if version == 18: 4 else: 3), codeCount)
        result[i].b = migrateLegacyEncodedLiterals(result[i].b,
          oldMap, mapping, (if version == 18: 4 else: 3), codeCount)
        result[i].patternRevision = freshPatternRevision()
        result[i].replacementRevision = freshReplacementRevision()
        sanitizeReplacementRefs(result[i])
      result[0].embedding = mapping
  elif n > 0:
    let mapping = defaultEmbedding()
    for i in 0 ..< n:
      result[i].a = encodeLiteralRuns(result[i].a, mapping)
      result[i].b = encodeLiteralRuns(result[i].b, mapping)
      result[i].patternRevision = freshPatternRevision()
      result[i].replacementRevision = freshReplacementRevision()
      sanitizeReplacementRefs(result[i])
    result[0].embedding = mapping


proc writeGraph(s: Stream, xs: seq[ProgressPoint]) =
  s.write(xs.len.int64)
  for x in xs:
    s.write(x.iter.int64)
    s.write(x.best)
    s.write(x.transformed)
    s.write(x.sqrtMovingAvg)
    s.write(x.fixedMovingAvg)

proc readGraph(s: Stream, version: int): seq[ProgressPoint] =
  let n64 = s.readInt64()
  if n64 < 0 or n64 > 10_000_000:
    raise newException(IOError, "Invalid graph history length in checkpoint.")
  let n = int(n64)
  result = newSeq[ProgressPoint](n)
  for i in 0 ..< n:
    result[i].iter = int(s.readInt64())
    result[i].best = s.readFloat64()
    result[i].transformed = s.readFloat64()

    # IMPORTANT: v5 writes BOTH moving-average values:
    #   sqrtMovingAvg, fixedMovingAvg
    #
    # The previous loader consumed only one float here. That shifted the
    # stream by 8 bytes per graph point, so the next read (population/genome
    # length) eventually interpreted a floating-point value as an integer and
    # produced "Invalid genome length in checkpoint".
    discard s.readFloat64() # sqrtMovingAvg
    if version >= 5:
      discard s.readFloat64() # fixedMovingAvg

    # Always reconstruct with the current MA definition after loading.
    result[i].sqrtMovingAvg = 0.0
    result[i].fixedMovingAvg = 0.0

proc trailingSigmaMean(xs: seq[ProgressPoint], endIdx: int, window: int): float {.inline.} =
  ## 直近 window 世代について、各世代の -log2(1-rho) を先に計算し、
  ## その値を平均する。graph = MA(-log2(1 - Spearman))。
  let startIdx = max(0, endIdx - window + 1)
  var sum = 0.0
  for j in startIdx .. endIdx:
    sum += xs[j].transformed
  let count = endIdx - startIdx + 1
  sum / float(count)

proc updatePlotMovingAverages(xs: var seq[ProgressPoint]) =
  ## 表示は「各世代のSpearmanを対数変換 → 後方200世代平均」。
  ## 200点そろう前はCSVへ出さないので、ここでは完全窓だけ確定する。
  let n = xs.len
  if n < PLOT_MA_SHORT_WINDOW:
    return
  let i = n - 1
  xs[i].sqrtMovingAvg = trailingSigmaMean(xs, i, PLOT_MA_SHORT_WINDOW)
  xs[i].fixedMovingAvg = trailingSigmaMean(xs, i, PLOT_MA_LONG_WINDOW)

proc rebuildPlotMovingAverages(xs: var seq[ProgressPoint]) =
  ## checkpoint内の旧MA値は信用せず、best_spearmanから各世代を対数化し、
  ## その系列の移動平均を全履歴について再構築する。
  if xs.len == 0:
    return

  var prefix = newSeq[float](xs.len + 1)
  for i in 0 ..< xs.len:
    xs[i].transformed = spearmanToSigma(xs[i].best)
    prefix[i + 1] = prefix[i] + xs[i].transformed
    xs[i].sqrtMovingAvg = 0.0
    xs[i].fixedMovingAvg = 0.0

  for i in 0 ..< xs.len:
    if i + 1 >= PLOT_MA_SHORT_WINDOW:
      let lo = i + 1 - PLOT_MA_SHORT_WINDOW
      xs[i].sqrtMovingAvg = (prefix[i + 1] - prefix[lo]) / float(PLOT_MA_SHORT_WINDOW)
    if i + 1 >= PLOT_MA_LONG_WINDOW:
      let lo = i + 1 - PLOT_MA_LONG_WINDOW
      xs[i].fixedMovingAvg = (prefix[i + 1] - prefix[lo]) / float(PLOT_MA_LONG_WINDOW)

proc writePopulation(s: Stream, p: seq[Genome]) =
  s.write(p.len.int64)
  for g in p:
    writeGenome(s, g)

proc readPopulation(s: Stream, version: int): seq[Genome] =
  let n64 = s.readInt64()
  if n64 < 0 or n64 > 10_000:
    raise newException(IOError, "Invalid population size in checkpoint.")
  let n = int(n64)
  # Older v20/v21 runs contain 400 genomes; a past release used 500.
  # Always CONSUME all stored genomes before resizing, otherwise subsequent
  # archive, dataset, Hall and end-marker offsets will be corrupted.
  if n != pop_size and n != 400 and n != 500:
    raise newException(IOError, "Checkpoint population size is " & $n &
      ", expected " & $pop_size & " (or legacy 400/500).")
  result = newSeq[Genome](n)
  for i in 0 ..< n:
    result[i] = readGenome(s, version)
    if expandLegacyGenome(result[i]):
      migratedRuleCount = true

  if n > pop_size:
    result.setLen(pop_size)
    echo "Migrated population: ", n, " -> ", pop_size,
      " (survivor prefix retained; trailing legacy entries dropped)"

proc expandLoadedPopulation(p: var seq[Genome]) =
  ## Called ONLY after the entire checkpoint (including Hall/end marker) was
  ## validated and surviving genomes were checked. Keep old indices/survivors
  ## unchanged and place new independently initialized genomes at the tail.
  if p.len == pop_size: return
  if p.len != 400 or pop_size <= p.len:
    raise newException(IOError, "Unsupported checkpoint population expansion: " &
      $p.len & " -> " & $pop_size)
  let previous = p.len
  for i in previous ..< pop_size:
    p.add(makeObj())
  echo "[checkpoint] population expanded: ", previous, " -> ", p.len,
    " (original genomes and their indices preserved; ", p.len - previous,
    " new genomes appended for FAST evaluation)"

proc writeEliteArchive(s: Stream, xs: seq[EliteArchiveEntry]) =
  s.write(xs.len.int64)
  for e in xs:
    writeGenome(s, e.genome)
    s.write(e.score)
    s.write(e.generation.int64)

proc readEliteArchive(s: Stream, version: int): seq[EliteArchiveEntry] =
  let n64 = s.readInt64()
  if n64 < 0 or n64 > 1_000_000:
    raise newException(IOError, "Invalid elite archive length in checkpoint.")
  let n = int(n64)
  result = newSeq[EliteArchiveEntry](n)
  for i in 0 ..< n:
    result[i].genome = readGenome(s, version)
    if expandLegacyGenome(result[i].genome):
      migratedRuleCount = true
    result[i].score = s.readFloat64()
    result[i].generation = int(s.readInt64())
    # fingerprintはバイナリに含めていないので、ロード直後に必ず再計算する。
    result[i].fingerprint = genomeFingerprint(result[i].genome)

proc writeJevHall(s: Stream) =
  ## v21+ extension is INSIDE ga_checkpoint.bin, immediately before end marker.
  s.write(jevHallConfig.len.int64)
  s.write(jevHallConfig)
  s.write(jevHall.len.int64)
  for e in jevHall:
    writeGenome(s, e.genome)
    s.write(e.meanQuality)
    s.write(e.observations.int64)
    s.write(e.wins.int64)
    s.write(e.firstGeneration.int64)
    s.write(e.lastGeneration.int64)

proc readJevHall(s: Stream, version, nextIter: int) =
  jevHall = @[]
  jevHallConfig = ""
  if version < 21: return # v20 and earlier had no Hall section.
  let configLen = s.readInt64()
  if configLen < 0 or configLen > 4096:
    raise newException(IOError, "Invalid Jev Hall config length in checkpoint.")
  jevHallConfig = s.readStr(int(configLen))
  if jevHallConfig.len != int(configLen):
    raise newException(IOError, "Truncated Jev Hall config in checkpoint.")
  let count = s.readInt64()
  if count < 0 or count > JEV_HALL_CAPACITY or
     (count > 0 and jevHallConfig.len == 0):
    raise newException(IOError, "Invalid Jev Hall size/config in checkpoint.")
  for i in 0 ..< int(count):
    var e: JevHallEntry
    e.genome = readGenome(s, version)
    e.meanQuality = s.readFloat64()
    e.observations = int(s.readInt64())
    e.wins = int(s.readInt64())
    e.firstGeneration = int(s.readInt64())
    e.lastGeneration = int(s.readInt64())
    if e.genome.len != AAA or not validEmbedding(e.genome[0].embedding) or
       e.meanQuality != e.meanQuality or e.meanQuality < 0.0 or
       e.meanQuality > 1.0 or e.observations < 1 or e.wins < 1 or
       e.wins > e.observations or e.firstGeneration < 0 or
       e.firstGeneration > e.lastGeneration or e.lastGeneration >= nextIter:
      raise newException(IOError, "Invalid Jev Hall entry in checkpoint.")
    e.fingerprint = genomeFingerprint(e.genome)
    for previous in jevHall:
      if e.fingerprint == previous.fingerprint and
         sameGenomeContent(e.genome, previous.genome):
        raise newException(IOError, "Duplicate Jev Hall entry in checkpoint.")
    jevHall.add(e)

proc writeFusion(s: Stream) =
  s.write(fusionReliability)
  s.write(fusionEvidence.int64)
  s.write(fusionNoise)
  for members in [fusionArchive, fusionAnchors]:
    s.write(members.len.int64)
    for m in members:
      writeGenome(s, m.genome)
      s.write(m.quality)
      s.write(m.generation.int64)
  s.write(fusionPending.len.int64)
  for p in fusionPending:
    writeGenome(s, p.genome)
    s.write(p.due.int64)

proc readFusion(s: Stream, version, nextIter: int) =
  resetFusion()
  if version < 24: return
  fusionReliability = s.readFloat64()
  fusionEvidence = int(s.readInt64())
  fusionNoise = s.readFloat64()
  if fusionReliability != fusionReliability or fusionReliability < 0 or
     fusionReliability > 1 or fusionNoise != fusionNoise or fusionNoise < 0 or
     fusionNoise > 1 or fusionEvidence < 0:
    raise newException(IOError, "Invalid adaptive Jev reliability state")
  for lane in 0..1:
    let n = s.readInt64()
    let cap = if lane == 0: FUSION_CAP else: FUSION_ANCHORS
    if n < 0 or n > cap:
      raise newException(IOError, "Invalid adaptive Jev archive length")
    var members: seq[FusionMember] = @[]
    for i in 0..<int(n):
      let g = readGenome(s, version)
      let q = s.readFloat64()
      let generation = int(s.readInt64())
      if g.len != AAA or q != q or q < 0 or q > 1 or
         generation < 0 or generation >= nextIter:
        raise newException(IOError, "Invalid adaptive Jev observation")
      if fusionMatch(g, members) >= 0:
        raise newException(IOError, "Duplicate adaptive Jev genome")
      members.add(fusionSnapshot(g, q, generation))
    if lane == 0: fusionArchive = members
    else: fusionAnchors = members
  let count = s.readInt64()
  if count < 0 or count > FUSION_NURSERY:
    raise newException(IOError, "Invalid adaptive Jev nursery length")
  for i in 0..<int(count):
    let g = readGenome(s, version)
    let due = int(s.readInt64())
    if g.len != AAA or due < nextIter or due >= nextIter + FUSION_INTERVAL:
      raise newException(IOError, "Invalid adaptive Jev nursery expiry")
    fusionPending.add(FusionPending(genome: g, due: due))

proc writeDataset(s: Stream, xs: seq[seq[seq[int]]], ys: seq[seq[int]]) =
  let chunks = min(xs.len, ys.len)
  s.write(chunks.int64)
  for c in 0 ..< chunks:
    s.write(xs[c].len.int64)
    for x in xs[c]:
      writeIntSeq(s, x)
    s.write(ys[c].len.int64)
    for y in ys[c]:
      s.write(y.int64)

proc readDataset(s: Stream, xs: var seq[seq[seq[int]]], ys: var seq[seq[int]]) =
  let nc64 = s.readInt64()
  if nc64 < 0 or nc64 > 10_000:
    raise newException(IOError, "Invalid evaluation chunk count in checkpoint.")
  let nc = int(nc64)
  xs = newSeq[seq[seq[int]]](nc)
  ys = newSeq[seq[int]](nc)

  for c in 0 ..< nc:
    let nx64 = s.readInt64()
    if nx64 < 0 or nx64 > 100_000:
      raise newException(IOError, "Invalid evaluation X size in checkpoint.")
    let nx = int(nx64)
    xs[c] = newSeq[seq[int]](nx)
    for i in 0 ..< nx:
      xs[c][i] = readIntSeq(s)

    let ny64 = s.readInt64()
    if ny64 < 0 or ny64 > 100_000:
      raise newException(IOError, "Invalid evaluation Y size in checkpoint.")
    let ny = int(ny64)
    ys[c] = newSeq[int](ny)
    for i in 0 ..< ny:
      ys[c][i] = int(s.readInt64())

proc validateCheckpointState(
  population: seq[Genome],
  eliteOfElites: seq[EliteArchiveEntry]
) =
  ## 保存直前に壊れた参照・異常な genome を検出し、
  ## 不正データを Stream に流す前に中断する。
  if population.len != pop_size:
    raise newException(IOError, "Invalid population size before checkpoint save.")

  for gi, g in population:
    if g.len != AAA:
      raise newException(IOError, "Invalid genome length before checkpoint save: population[" & $gi & "]=" & $g.len)
    if not validEmbedding(g[0].embedding):
      raise newException(IOError, "Missing/invalid embedding in population checkpoint.")
    for ri, r in g:
      if r.a.len > 64 or r.b.len > 64:
        raise newException(IOError, "Oversized rule sequence before checkpoint save: " & $gi & "/" & $ri)
      if r.weight != r.weight or abs(r.weight) > 1.0e6:
        raise newException(IOError, "Invalid rule weight before checkpoint save: " & $gi & "/" & $ri)

  for ei, e in eliteOfElites:
    if e.genome.len != AAA:
      raise newException(IOError, "Invalid elite genome length before checkpoint save: " & $ei)
    if not validEmbedding(e.genome[0].embedding):
      raise newException(IOError, "Missing/invalid embedding in elite checkpoint.")
    if e.score != e.score or abs(e.score) > 1.0e6:
      raise newException(IOError, "Invalid elite score before checkpoint save: " & $ei)

proc saveCheckpoint(
  path: string,
  nextIter: int,
  population: seq[Genome],
  bestEver: float,
  stagnation: int,
  maHistory: seq[float],
  progressHistory: seq[ProgressPoint],
  eliteOfElites: seq[EliteArchiveEntry],
  evalAggsX: seq[seq[seq[int]]],
  evalAggsY: seq[seq[int]],
  sourceChunkCursor: int,
  runSeed: int64,
  progressSigmaOffset: array[MULTI_CASE_COUNT, float]
) =
  validateCheckpointState(population, eliteOfElites)
  if jevHall.len > JEV_HALL_CAPACITY or jevHallConfig.len > 4096:
    raise newException(IOError, "Invalid Jev Hall metadata before checkpoint save.")
  for e in jevHall:
    if e.genome.len != AAA or not validEmbedding(e.genome[0].embedding) or
       e.meanQuality != e.meanQuality or e.meanQuality < 0.0 or
       e.meanQuality > 1.0 or e.observations < 1 or e.wins < 1 or
       e.wins > e.observations:
      raise newException(IOError, "Invalid Jev Hall entry before checkpoint save.")

  let tmp = path & ".tmp"
  var fs = newFileStream(tmp, fmWrite)
  if fs.isNil:
    raise newException(IOError, "Could not open checkpoint for writing: " & tmp)

  fs.write(CHECKPOINT_MAGIC)
  fs.write(CHECKPOINT_VERSION.int64)
  fs.write(nextIter.int64)
  fs.write(bestEver)
  fs.write(stagnation.int64)
  fs.write(sourceChunkCursor.int64)
  fs.write(runSeed)
  for ci in 0 ..< MULTI_CASE_COUNT:
    fs.write(progressSigmaOffset[ci])

  let savedMa = maValuesChronological()
  fs.write(savedMa.len.int64)
  for x in savedMa:
    fs.write(x)

  # v4 では評価データセットそのものを checkpoint に含めない。
  # evalAggsX/evalAggsY は sourceChunkCursor から完全に再構築でき、
  # 長時間運転時の checkpoint サイズと保存時メモリ負荷を大幅に減らせる。
  writeGraph(fs, progressHistory)
  writePopulation(fs, population)
  writeEliteArchive(fs, eliteOfElites)
  # v6: 固定6ケース+rollingケースをそのまま保存する。
  # これによりcheckpoint再開後も「固定評価」が本当に同一データになる。
  writeDataset(fs, evalAggsX, evalAggsY)
  writeJevHall(fs) # legacy Hall retained for migration
  writeFusion(fs) # v24: reliability, exact anchors/front, pending offspring
  fs.write(0x4B4C4D56434B5036'i64) # KLMVCP6 / checkpoint end marker
  fs.close()

  # close 済みの tmp を本体へ差し替える。
  # 先に旧本体を .bak へ退避する方式は、新本体への移動に失敗すると
  # 本体が一時的に消えるため、失敗時には .bak から必ずロールバックする。
  let backup = path & ".bak"
  var oldMoved = false
  try:
    if fileExists(path):
      if fileExists(backup):
        removeFile(backup)
      moveFile(path, backup)
      oldMoved = true

    moveFile(tmp, path)
    echo "[checkpoint] saved: ", path, " (next_generation=", nextIter, ")"
  except CatchableError:
    if oldMoved and not fileExists(path) and fileExists(backup):
      try:
        moveFile(backup, path)
      except CatchableError as restoreErr:
        echo "[checkpoint] WARNING: failed to restore previous checkpoint: ", restoreErr.msg
    raise

proc loadCheckpoint(
  path: string,
  nextIter: var int,
  population: var seq[Genome],
  bestEver: var float,
  stagnation: var int,
  maHistory: var seq[float],
  progressHistory: var seq[ProgressPoint],
  eliteOfElites: var seq[EliteArchiveEntry],
  evalAggsX: var seq[seq[seq[int]]],
  evalAggsY: var seq[seq[int]],
  sourceChunkCursor: var int,
  runSeed: var int64,
  progressSigmaOffset: var array[MULTI_CASE_COUNT, float]
): bool =
  if not fileExists(path):
    return false

  migratedRuleCount = false
  # すべての状態を先に空にしておく。壊れたcheckpointを読んだ後に
  # 部分的な状態だけが残り、次の復旧処理へ混入するのを防ぐ。
  population = @[]
  eliteOfElites = @[]
  resetFusion()
  jevHall = @[]
  jevHallConfig = ""
  progressHistory = @[]
  maHistory = @[]
  evalAggsX = @[]
  evalAggsY = @[]

  var fs = newFileStream(path, fmRead)
  if fs.isNil:
    raise newException(IOError, "Could not open checkpoint: " & path)

  try:
    let magic = fs.readStr(CHECKPOINT_MAGIC.len)
    if magic != CHECKPOINT_MAGIC:
      raise newException(IOError, "Invalid checkpoint magic: " & path)

    let version = int(fs.readInt64())
    if version < 3 or (version > 9 and version notin [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, CHECKPOINT_VERSION]):
      raise newException(IOError, "Unsupported checkpoint version: " & $version)

    nextIter = int(fs.readInt64())
    bestEver = fs.readFloat64()
    stagnation = int(fs.readInt64())
    sourceChunkCursor = int(fs.readInt64())
    runSeed = fs.readInt64()
    for ci in 0 ..< MULTI_CASE_COUNT: progressSigmaOffset[ci] = 0.0
    if version >= 9:
      # Both the original 3-case v16 checkpoint and the subsequently released
      # 9-case v16 checkpoint exist. v17 records nine offsets unambiguously.
      # Inspect (but do not consume) the six extra zero-valued legacy offsets;
      # the 3-case layout instead has MA length/history/population at that spot.
      var offsetCount = MULTI_CASE_COUNT
      if version == 9:
        offsetCount = 3
      elif version == 16:
        let offsetStart = fs.getPosition()
        fs.setPosition(offsetStart + 3 * sizeof(float64))
        var sixZeroOffsets = true
        for i in 0 ..< 6:
          if fs.readFloat64() != 0.0: sixZeroOffsets = false
        let followingMaLength = fs.readInt64()
        offsetCount = if sixZeroOffsets and followingMaLength >= 0 and
                         followingMaLength <= MA_WINDOW: 9 else: 3
        fs.setPosition(offsetStart)
        echo "[checkpoint] detected v16 layout: case_offsets=", offsetCount
      for ci in 0 ..< offsetCount:
        progressSigmaOffset[ci] = fs.readFloat64()

    let maLen64 = fs.readInt64()
    if maLen64 < 0 or maLen64 > 10_000_000:
      raise newException(IOError, "Invalid moving-average history length.")
    let maLen = int(maLen64)
    maHistory.setLen(0)
    let loadCount = min(maLen, MA_WINDOW)
    let skipCount = maLen - loadCount
    for i in 0 ..< maLen:
      let v = fs.readFloat64()
      if i >= skipCount:
        maHistory.add(v)

    progressHistory = readGraph(fs, version)
    # v4のbridge-normalized履歴は選択バイアスで累積ドリフトするため廃止。
    # checkpoint versionに関係なく、生のbest Spearmanから表示履歴を再構築する。
    for i in 0 ..< progressHistory.len:
      progressHistory[i].transformed = spearmanToSigma(progressHistory[i].best)
    for ci in 0 ..< MULTI_CASE_COUNT:
      progressSigmaOffset[ci] = 0.0
    population = readPopulation(fs, version)
    eliteOfElites = readEliteArchive(fs, version)

    if version == 3:
      # 旧 checkpoint は従来通り評価データセットまで持っている。
      readDataset(fs, evalAggsX, evalAggsY)
    elif version >= 6:
      # v6以降は固定/rollingの現在データセットを保存している。
      readDataset(fs, evalAggsX, evalAggsY)
      readJevHall(fs, version, nextIter) # v20 reads zero extra bytes.
      readFusion(fs, version, nextIter)
      let marker = fs.readInt64()
      if marker != 0x4B4C4D56434B5036'i64:
        raise newException(IOError, "Invalid checkpoint end marker.")
      if not fs.atEnd():
        raise newException(IOError, "Trailing data detected in checkpoint.")
    else:
      # v4/v5 は保存しない。後段で sourceChunkCursor から再構築する。
      evalAggsX = @[]
      evalAggsY = @[]
      let marker = fs.readInt64()
      if marker != 0x4B4C4D56434B5034'i64:
        raise newException(IOError, "Invalid checkpoint end marker.")
      if not fs.atEnd():
        raise newException(IOError, "Trailing data detected in checkpoint.")

    # 実行時のMAリングはグローバル管理なので、checkpointの
    # chronological historyを一旦コピーしてから復元する。
    let restoredMa = @maHistory
    maReset()
    for v in restoredMa:
      maPush(v)

    # 旧チェックポイントでは bestEver が指数移動平均になっていた。
    # アーカイブに保存されている実スコアから、本当の歴代最高値へ復元する。
    if eliteOfElites.len > 0:
      var archiveBest = eliteOfElites[0].score
      for i in 1 ..< eliteOfElites.len:
        if eliteOfElites[i].score > archiveBest:
          archiveBest = eliteOfElites[i].score
      bestEver = max(bestEver, archiveBest)

    loadedCheckpointVersion = version
    result = true
  finally:
    fs.close()

proc loadCheckpointWithRecovery(
  path: string,
  nextIter: var int,
  population: var seq[Genome],
  bestEver: var float,
  stagnation: var int,
  maHistory: var seq[float],
  progressHistory: var seq[ProgressPoint],
  eliteOfElites: var seq[EliteArchiveEntry],
  evalAggsX: var seq[seq[seq[int]]],
  evalAggsY: var seq[seq[int]],
  sourceChunkCursor: var int,
  runSeed: var int64,
  progressSigmaOffset: var array[MULTI_CASE_COUNT, float]
): bool =
  ## メインcheckpointが壊れている場合、直前の正常世代を保持した .bak を
  ## 自動的に試す。長時間運転では「保存中の一時障害で全履歴を失う」
  ## ことを絶対に避ける。
  let candidates = @[path, path & ".bak"]
  for idx, candidate in candidates:
    if not fileExists(candidate):
      continue
    try:
      if loadCheckpoint(candidate, nextIter, population, bestEver, stagnation,
                        maHistory, progressHistory, eliteOfElites, evalAggsX,
                        evalAggsY, sourceChunkCursor, runSeed, progressSigmaOffset):
        if idx == 1:
          echo "[checkpoint] primary checkpoint was invalid; recovered from .bak"
        return true
    except CatchableError as e:
      if idx == 0:
        echo "[checkpoint] failed to load primary: " & e.msg
        if fileExists(path & ".bak"):
          echo "[checkpoint] trying backup: " & path & ".bak"
      else:
        echo "[checkpoint] backup is also invalid: " & e.msg

  # 両方とも壊れている場合は、新規開始へ移行する。
  # ここで例外を上へ投げないので、放置運転がcheckpoint破損一発で終了しない。
  population = @[]
  eliteOfElites = @[]
  resetFusion()
  jevHall = @[]
  jevHallConfig = ""
  progressHistory = @[]
  maHistory = @[]
  for ci in 0 ..< MULTI_CASE_COUNT: progressSigmaOffset[ci] = 0.0
  evalAggsX = @[]
  evalAggsY = @[]
  nextIter = 0
  return false

# CSV is a derived artifact, not a checkpoint: rebuild it on first use or
# after a history reset, then append only the new points (avoids O(N^2) I/O).
var progressCsvLastPath = ""
var progressCsvLastWrittenIndex = -1

proc writeProgressCsv(path: string, xs: seq[ProgressPoint]) =
  let rebuild = path != progressCsvLastPath or not fileExists(path) or
    xs.len <= progressCsvLastWrittenIndex
  let target = if rebuild: path & ".tmp" else: path
  var f = open(target, if rebuild: fmWrite else: fmAppend)
  if rebuild:
    f.writeLine("iter,best_spearman,raw_sigma,ma200_of_log_spearman,ma200_of_log_spearman_dup")

  # Only complete trailing windows are output; previous rows are final and
  # cannot change when later generations are appended.
  let first = max(PLOT_MA_SHORT_WINDOW - 1,
    (if rebuild: 0 else: progressCsvLastWrittenIndex + 1))
  if first < xs.len:
    for i in first ..< xs.len:
      let x = xs[i]
      f.writeLine(&"{x.iter},{x.best:.6f},{x.transformed:.6f},{x.sqrtMovingAvg:.6f},{x.fixedMovingAvg:.6f}")

  f.flushFile()
  f.close()
  if rebuild: moveFile(target, path)
  progressCsvLastPath = path
  progressCsvLastWrittenIndex = xs.len - 1


# Requested rolling widths, in the user's literal SLOW:MEDIUM:FAST order = 2:3:11.
# The labels are historical; scheduling is deadline-based below. Since
# 1/2 + 1/3 + 1/11 < 1, at most ONE case group is refreshed per generation,
# eliminating simultaneous cache invalidation/objective shocks while keeping
# the requested average cadences.
const DATASET_CHUNKS = MULTI_CASE_COUNT
const SLOW_DATASET_REFRESH_INTERVAL = 2
const MEDIUM_DATASET_REFRESH_INTERVAL = 3
const FAST_DATASET_REFRESH_INTERVAL = 11

proc rollingAgeAtResume(nextIter, firstRefresh, interval: int): int {.inline.} =
  ## checkpointはrollingCaseAgeを保存していない旧形式とも互換にする。
  ## nextIter直前までに最後にrefreshされた世代からの経過世代数を再構築する。
  let lastCompleted = nextIter - 1
  if lastCompleted < firstRefresh:
    return 64
  let lastRefresh = firstRefresh + ((lastCompleted - firstRefresh) div interval) * interval
  max(0, lastCompleted - lastRefresh)

# (sourceChunks / sourceChunkCursor / コーパス統計は、makeObj の
#  パターン初期化で使うため、ファイル先頭側で既に読み込み済み。)

proc hardNegativeStagesForSlot(slot: int): int {.inline.} =
  ## teacher-mined hard negativeは有用だが、teacher更新も評価分布を内生的に
  ## 変化させる。Spearmanでは少数のrank violationでも効くので、
  ## 旧0/2/4から0/1/1へ抑え、teacher更新は32世代周期にして
  ## curriculum shockを評価datasetのrolling shockより小さく保つ。
  case slot mod 3
  of FAST_ROLLING_SLOT: 0
  of MEDIUM_ROLLING_SLOT: 1
  of SLOW_ROLLING_SLOT: 1
  else: 0

proc chooseRollingChunkId(
  chunks: seq[seq[int]],
  cursor: var int,
  avoid: seq[int] = @[]
): int =
  ## github-code.txt 内の並び順をそのまま学習時系列にしない。
  ## repo/language/sizeなどでchunkが固まっていると、sequential cursorは
  ## 数十世代単位のdifficulty driftを作り、200世代MAまで波打たせる。
  ## まずランダム抽出し、極端に短いchunk/重複だけ避ける。
  if chunks.len == 0:
    return -1

  let randomTries = min(64, max(8, chunks.len * 2))
  for _ in 0 ..< randomTries:
    let id = rand(chunks.len - 1)
    if chunks[id].len < MIN_EVAL_CHUNK_LEN:
      continue
    var duplicate = false
    for x in avoid:
      if x == id:
        duplicate = true
        break
    if duplicate:
      continue
    cursor = (cursor + 1) mod chunks.len  # checkpoint互換用の進行カウンタ
    return id

  # まず情報量の十分なchunkを決定的に探す。極端に短いchunkは
  # diffusion damageが早く飽和し、tieだらけのSpearmanになりやすい。
  if cursor < 0 or cursor >= chunks.len:
    cursor = 0
  for off in 0 ..< chunks.len:
    let id = (cursor + off) mod chunks.len
    if chunks[id].len < MIN_EVAL_CHUNK_LEN:
      continue
    var duplicate = false
    for x in avoid:
      if x == id:
        duplicate = true
        break
    if not duplicate:
      cursor = (id + 1) mod chunks.len
      return id

  # コーパスが短片だけの場合は互換性のため最終的に2byte以上へfallback。
  for off in 0 ..< chunks.len:
    let id = (cursor + off) mod chunks.len
    if chunks[id].len < 2:
      continue
    var duplicate = false
    for x in avoid:
      if x == id:
        duplicate = true
        break
    if not duplicate:
      cursor = (id + 1) mod chunks.len
      return id
  -1

proc retainedTrajectoryIndices(n, keep: int): seq[int] =
  let count = min(n, keep)
  if count <= 0: return @[]
  if count == 1: return @[0]
  for i in 0 ..< count:
    result.add((i * (n - 1)) div (count - 1))

proc trainingTrajectory(base: seq[int]): tuple[x: seq[seq[int]], y: seq[int]] =
  let full = buildDiffusionTrajectory(base, RETAINED_SAMPLES_PER_CASE)
  #for index in retainedTrajectoryIndices(full.x.len, RETAINED_SAMPLES_PER_CASE):
  result = full

proc rebuildDataset(
  chunks: seq[seq[int]],
  cursor: var int,
  outX: var seq[seq[seq[int]]],
  outY: var seq[seq[int]]
) =
  outX.setLen(0)
  outY.setLen(0)
  if chunks.len == 0:
    return

  var chosenChunkIds: seq[int] = @[]
  while outX.len < DATASET_CHUNKS:
    var chunkId = chooseRollingChunkId(chunks, cursor, chosenChunkIds)
    if chunkId < 0 and chunks.len < DATASET_CHUNKS:
      chunkId = chooseRollingChunkId(chunks, cursor)
    if chunkId < 0:
      break
    chosenChunkIds.add(chunkId)

    let base = chunks[chunkId]

    var trajectory = trainingTrajectory(base)
    let slot = outX.len
    mineCorruptions(base, trajectory.x, trajectory.y, hardNegativeStagesForSlot(slot))
    if trajectory.x.len >= 2 and trajectory.y.len == trajectory.x.len:
      outX.add(trajectory.x)
      outY.add(trajectory.y)

proc refreshDatasetSlot(
  chunks: seq[seq[int]],
  cursor: var int,
  ioX: var seq[seq[seq[int]]],
  ioY: var seq[seq[int]],
  slot: int
): bool =
  ## 指定した1本のrolling caseだけを新しいdiffusion trajectoryへ交換する。
  ## 全slotはrollingのまま。chunkはcorpus順ではなくランダム抽出する。
  if chunks.len == 0:
    return false

  let n = min(ioX.len, ioY.len)
  if n == 0:
    rebuildDataset(chunks, cursor, ioX, ioY)
    return ioX.len > 0 and ioY.len > 0
  if slot < 0 or slot >= n:
    return false

  var attempts = 0
  while attempts < min(chunks.len, 64):
    inc attempts
    let chunkId = chooseRollingChunkId(chunks, cursor)
    if chunkId < 0:
      break
    let base = chunks[chunkId]

    # 他のactive caseと完全に同じbaseだけは避ける。
    var duplicateBase = false
    for ci in 0 ..< n:
      if ci != slot and ioX[ci].len > 0 and ioX[ci][0] == base:
        duplicateBase = true
        break
    if duplicateBase:
      continue

    var trajectory = trainingTrajectory(base)
    mineCorruptions(base, trajectory.x, trajectory.y, hardNegativeStagesForSlot(slot))
    if trajectory.x.len >= 2 and trajectory.y.len == trajectory.x.len:
      ioX[slot] = trajectory.x
      ioY[slot] = trajectory.y
      return true

  false


proc refreshDatasetGroup(
  chunks: seq[seq[int]], cursor: var int,
  ioX: var seq[seq[seq[int]]], ioY: var seq[seq[int]],
  group: int, refreshedCases: var array[MULTI_CASE_COUNT, bool]
): bool =
  for ci in countup(group, MULTI_CASE_COUNT - 1, 3):
    let changed = refreshDatasetSlot(chunks, cursor, ioX, ioY, ci)
    refreshedCases[ci] = changed
    if changed:
      rollingCaseAge[ci] = 0
      result = true

# Standalone inference V2: the training winner, Unicode alphabet, smoothed corpus
# character frequencies and short corpus n-grams. Old V1 .model files still load.
# A generational multi-island GA replaces the previous 4-chain annealer.
# Every candidate is selected using the same multi-round raw evaluator used
# in training. --steps is the total *candidate proposal* budget (including the
# initial population); memo hits do not consume another exact score evaluation.
const ENERGY_MODEL_MAGIC_V1 = "KLEISMIC_ENERGY_V1"
const ENERGY_MODEL_MAGIC_V2 = "KLEISMIC_ENERGY_V2"
const ENERGY_MODEL_MAGIC_V3 = "KLEISMIC_ENERGY_V3"
const ENERGY_MODEL_MAGIC_V4 = "KLEISMIC_ENERGY_V4"
const ENERGY_MODEL_MAGIC_V5 = "KLEISMIC_ENERGY_V5"
const ENERGY_MODEL_MAGIC_V6 = "KLEISMIC_ENERGY_V6"
const ENERGY_MODEL_MAGIC_V7 = "KLEISMIC_ENERGY_V7"
const ENERGY_MODEL_MAGIC_V8 = "KLEISMIC_ENERGY_V8"
const ENERGY_MODEL_MAGIC = "KLEISMIC_ENERGY_V9"
const GENERATION_ISLANDS = 4
const GENERATION_NGRAM_LIMIT = 512
# Text search must never inherit the 10-second TRAINING watchdog per proposal.
# A handful of pathological generated strings used to make one generation take
# tens of seconds (e.g. 4 candidates x ~10 s). Normal candidates are far below
# these limits; only pathological rewrite explosions are rejected as -Inf.
const TEXT_GENERATION_SCORE_MAX_SECONDS = 0.50
const TEXT_GENERATION_SCORE_MAX_RULE_VISITS = 250_000
const TEXT_GENERATION_CLOCK_CHECK_VISITS = 128

type GenerationGram = object
  tokens: seq[int]
  frequency: int

type TextIndividual = object
  suffix: seq[int]   # only the generated suffix is mutable; the prompt is fixed
  # Cached UTF-8 suffix. Generation previously rebuilt runeText(suffix) several
  # times per proposal (seen-set checks + memo lookup + scoring), which made
  # long --length runs hit allocator/GC cliffs. Keep one exact representation.
  key: string
  energy: float      # the ONLY selection/output fitness: multi-round scoreRaw

proc runeText(tokens: seq[int]): string =
  for v in tokens: result.add($(unicode.Rune(v)))

proc textBytes(text: string): seq[int] =
  result = newSeq[int](text.len)
  for i, c in text: result[i] = ord(c)

proc appendRuneUtf8ByteInts(outp: var seq[int], v: int) {.inline.} =
  ## Direct UTF-8 encoder for generation hot paths. `generationAlphabet` only
  ## admits Unicode scalar values, so invalid/surrogate values are not expected.
  if v <= 0x7F:
    outp.add(v)
  elif v <= 0x7FF:
    outp.add(0xC0 or (v shr 6))
    outp.add(0x80 or (v and 0x3F))
  elif v <= 0xFFFF:
    outp.add(0xE0 or (v shr 12))
    outp.add(0x80 or ((v shr 6) and 0x3F))
    outp.add(0x80 or (v and 0x3F))
  else:
    outp.add(0xF0 or (v shr 18))
    outp.add(0x80 or ((v shr 12) and 0x3F))
    outp.add(0x80 or ((v shr 6) and 0x3F))
    outp.add(0x80 or (v and 0x3F))

proc buildPromptSuffixBytes(
    promptBytes: openArray[int], suffix: openArray[int], outp: var seq[int]) {.inline.} =
  outp.setLen(promptBytes.len)
  if promptBytes.len > 0:
    copyMem(addr outp[0], unsafeAddr promptBytes[0], promptBytes.len * sizeof(int))
  for v in suffix:
    appendRuneUtf8ByteInts(outp, v)

proc generationAlphabet(): seq[int] =
  var seen = initHashSet[int]()
  for v in 32 .. 126:
    seen.incl(v)
    result.add(v)
  for v in [9, 10]:
    seen.incl(v)
    result.add(v)
  var scanned = 0
  for chunk in sourceChunks:
    if scanned >= 256_000: break
    var text = newString(chunk.len)
    for i, v in chunk: text[i] = char(v)
    scanned += chunk.len
    if unicode.validateUtf8(text) != -1: continue
    for rune in unicode.toRunes(text):
      let v = int(rune)
      if v >= 32 and v != 127 and v notin seen:
        seen.incl(v)
        result.add(v)
        if result.len >= 4096: return

proc generationCorpusPriors(alphabet: seq[int]): tuple[
    frequencies: seq[int], grams: seq[GenerationGram]] =
  ## Runs only when exporting, never when generating; the inference model is
  ## self-contained. Short 2..5-codepoint grams avoid retaining long corpus text.
  var characterCounts = initTable[int, int]()
  var ngramCounts = initTable[seq[int], int]()
  var scanned = 0
  var gramScanned = 0
  for chunk in sourceChunks:
    if scanned >= 256_000: break
    var text = newString(chunk.len)
    for i, v in chunk: text[i] = char(v)
    scanned += chunk.len
    if unicode.validateUtf8(text) != -1: continue
    var runes: seq[int] = @[]
    for r in unicode.toRunes(text):
      let v = int(r)
      if v >= 32 or v == 9 or v == 10:
        characterCounts[v] = characterCounts.getOrDefault(v) + 1
      runes.add(v)
    if gramScanned < 96_000:
      gramScanned += chunk.len
      for i in 0 ..< runes.len:
        for n in 2 .. 5:
          if i + n > runes.len: break
          var valid = true
          for j in i ..< i + n:
            if runes[j] < 32 and runes[j] != 9 and runes[j] != 10:
              valid = false
              break
          if not valid: continue
          let key = runes[i ..< i + n]
          if ngramCounts.hasKey(key):
            inc ngramCounts[key]
          elif ngramCounts.len < 120_000:
            ngramCounts[key] = 1
  result.frequencies = newSeq[int](alphabet.len)
  for i, v in alphabet:
    result.frequencies[i] = max(1, characterCounts.getOrDefault(v) + 1)
  var ranked: seq[GenerationGram] = @[]
  for tokens, frequency in ngramCounts:
    if frequency >= 2:
      ranked.add(GenerationGram(tokens: tokens, frequency: frequency))
  ranked.sort(proc(a, b: GenerationGram): int =
    if a.frequency > b.frequency: -1
    elif a.frequency < b.frequency: 1
    else: cmp($a.tokens, $b.tokens)
  )
  for i in 0 ..< min(GENERATION_NGRAM_LIMIT, ranked.len):
    result.grams.add(ranked[i])

proc exportEnergyModel(path: string, genome: Genome) =
  let alphabet = generationAlphabet()
  let prior = generationCorpusPriors(alphabet)
  let temporary = path & ".tmp"
  var stream = newFileStream(temporary, fmWrite)
  if stream.isNil: raise newException(IOError, "Cannot write model: " & path)
  try:
    stream.write(ENERGY_MODEL_MAGIC)
    writeGenome(stream, genome)
    writeIntSeq(stream, alphabet)
    writeIntSeq(stream, prior.frequencies)
    stream.write(prior.grams.len.int64)
    for gram in prior.grams:
      writeIntSeq(stream, gram.tokens)
      stream.write(gram.frequency.int64)
    stream.flush()
  finally:
    stream.close()
  moveFile(temporary, path)

proc pickGenerationToken(alphabet: seq[int], cumulative: seq[float]): int =
  ## 12% uniform exploration; otherwise sqrt-frequency sampling so common
  ## tokens dominate initialization without losing rare Unicode tokens.
  if alphabet.len == 1 or rand(99) < 12:
    return alphabet[rand(alphabet.len - 1)]
  let draw = rand(cumulative[^1])
  var lo = 0
  var hi = alphabet.len - 1
  while lo < hi:
    let mid = (lo + hi) div 2
    if cumulative[mid] < draw: lo = mid + 1
    else: hi = mid
  alphabet[lo]

proc initialGenerationSuffix(
    count: int, alphabet: seq[int], cumulative: seq[float],
    grams: seq[GenerationGram]): seq[int] =
  result = newSeqOfCap[int](count)
  let mode = rand(99)
  while result.len < count:
    if grams.len > 0 and mode < 75 and rand(99) < 83:
      # Build from corpus-supported short spans instead of isolated noise.
      let gram = grams[rand(grams.len - 1)]
      for v in gram.tokens:
        if result.len >= count: break
        result.add(v)
    elif mode >= 95:
      result.add(alphabet[rand(alphabet.len - 1)])
    else:
      result.add(pickGenerationToken(alphabet, cumulative))

proc generationDistance(a, b: seq[int]): float =
  if a.len == 0: return 0.0
  let stride = max(1, a.len div 32)
  var total = 0
  var diff = 0
  for i in countup(0, a.len - 1, stride):
    inc total
    if a[i] != b[i]: inc diff
  float(diff) / float(max(1, total))

proc generationParent(pop: seq[TextIndividual], novelty: bool): int =
  ## Fitness tournament most of the time; a minority of tournaments favor a
  ## distant parent to avoid the whole island turning into an elite clone.
  result = rand(pop.len - 1)
  for _ in 0 ..< 2:
    let next = rand(pop.len - 1)
    if novelty:
      if generationDistance(pop[next].suffix, pop[0].suffix) >
          generationDistance(pop[result].suffix, pop[0].suffix):
        result = next
    elif pop[next].energy > pop[result].energy:
      result = next

proc generationMutation(
    genes: var seq[int], alphabet: seq[int], cumulative: seq[float],
    grams: seq[GenerationGram], intensity: int) =
  ## Sparse independent edits plus occasional contiguous segment repair.
  let n = genes.len
  if n == 0: return
  let edits = sampleLogUniformMutationCount(min(n, max(1, intensity)))
  for _ in 0 ..< edits:
    let pos = rand(n - 1)
    if grams.len > 0 and rand(99) < 32:
      let gram = grams[rand(grams.len - 1)]
      let beginAt = min(pos, max(0, n - gram.tokens.len))
      for j, v in gram.tokens:
        if beginAt + j < n: genes[beginAt + j] = v
    elif n > 1 and rand(99) < 12:
      genes[pos] = genes[rand(n - 1)]
    else:
      genes[pos] = pickGenerationToken(alphabet, cumulative)

proc generationOperator(
    trials, wins: array[6, int], stagnation: int): int =
  ## Mild success-based adaptation, with an exploration floor for every op.
  ## Do not make a lucky operator monopolize reproduction permanently.
  const base: array[6, float] = [8.0, 6.0, 9.0, 7.0, 4.0, 2.0]
  var cumulative: array[6, float]
  var total = 0.0
  for i in 0 .. 5:
    let success = float(wins[i] + 1) / float(trials[i] + 4)
    var weight = base[i] * (0.75 + 1.5 * success)
    if i == 5: weight *= 1.0 + float(min(12, stagnation)) / 6.0
    total += weight
    cumulative[i] = total
  let draw = rand(total)
  for i in 0 .. 5:
    if draw <= cumulative[i]: return i
  5

proc generationOffspring(
    p1, p2: seq[int], op, intensity: int,
    alphabet: seq[int], cumulative: seq[float],
    grams: seq[GenerationGram]): seq[int] =
  let n = p1.len
  if op == 5:
    return initialGenerationSuffix(n, alphabet, cumulative, grams)
  result = cloneInts(p1)
  case op
  of 0: # One contiguous block crossover (including arbitrary UTF-8 widths).
    let a = rand(n - 1)
    let b = rand(n - 1)
    for i in min(a, b) .. max(a, b): result[i] = p2[i]
  of 1: # Alternating multi-point crossover; blocks preserve local dependencies.
    var at = 0
    var useSecond = rand(1) == 0
    while at < n:
      let stop = min(n, at + 1 + rand(min(15, n - at - 1)))
      if useSecond:
        for i in at ..< stop: result[i] = p2[i]
      useSecond = not useSecond
      at = stop
  of 2: # Sparse exploitation around a selected parent.
    generationMutation(result, alphabet, cumulative, grams, intensity)
  of 3: # Corpus n-gram transplant, or a sparse mutation for legacy V1 models.
    if grams.len > 0:
      let gram = grams[rand(grams.len - 1)]
      let start = rand(max(0, n - min(n, gram.tokens.len)))
      for j in 0 ..< min(n, gram.tokens.len): result[start + j] = gram.tokens[j]
    else:
      generationMutation(result, alphabet, cumulative, grams, intensity)
  of 4: # Segment rearrangement; keeps a compatible suffix length.
    if n > 1:
      let start = rand(n - 2)
      let maxSize = min(14, n - start)
      let selectedSize = sampleLogUniformMutationRange(2, maxSize)
      if rand(1) == 0:
        for k in 0 ..< selectedSize div 2:
          swap(result[start + k], result[start + selectedSize - 1 - k])
      else:
        let src = rand(n - selectedSize)
        let saved = p1[src ..< src + selectedSize]
        for k in 0 ..< selectedSize: result[start + k] = saved[k]
    else:
      generationMutation(result, alphabet, cumulative, grams, intensity)
  else: discard
  # Mutation after crossover keeps exploration alive when parents converge.
  if op != 2 and rand(99) < 65:
    generationMutation(result, alphabet, cumulative, grams, intensity)
  if result == p1 and alphabet.len > 1:
    let pos = rand(n - 1)
    var alternative = pickGenerationToken(alphabet, cumulative)
    if alternative == result[pos]:
      alternative = alphabet[(rand(alphabet.len - 2) + 1) mod alphabet.len]
    result[pos] = alternative

proc optimizeText(
    modelPath, prompt, outputPath: string,
    count, steps, seed, requestedPopulation: int) =
  if count < 1 or count > 3100 or steps < 1:
    quit("--length must be 1..3100 appended Unicode characters; --steps must be positive.")
  if requestedPopulation < 8 or requestedPopulation > 256:
    quit("--population must be 8..256.")
  if unicode.validateUtf8(prompt) != -1:
    quit("Prompt must be valid UTF-8.")
  randomize(seed)
  var stream = newFileStream(modelPath, fmRead)
  if stream.isNil: quit("Cannot open energy model: " & modelPath)
  var genome: Genome
  var alphabet: seq[int]
  var frequencies: seq[int]
  var grams: seq[GenerationGram] = @[]
  try:
    let header = stream.readStr(ENERGY_MODEL_MAGIC.len)
    if header notin [ENERGY_MODEL_MAGIC, ENERGY_MODEL_MAGIC_V8, ENERGY_MODEL_MAGIC_V7,
                     ENERGY_MODEL_MAGIC_V6, ENERGY_MODEL_MAGIC_V5, ENERGY_MODEL_MAGIC_V4,
                     ENERGY_MODEL_MAGIC_V3, ENERGY_MODEL_MAGIC_V2, ENERGY_MODEL_MAGIC_V1]:
      quit("Expected a V1..V9 .model file, not a GA checkpoint.")
    let modelVersion =
      if header == ENERGY_MODEL_MAGIC: CHECKPOINT_VERSION
      elif header == ENERGY_MODEL_MAGIC_V8: 22
      elif header == ENERGY_MODEL_MAGIC_V7: 21
      elif header == ENERGY_MODEL_MAGIC_V6: 19
      elif header == ENERGY_MODEL_MAGIC_V5: 18
      elif header == ENERGY_MODEL_MAGIC_V4: 17
      elif header == ENERGY_MODEL_MAGIC_V3: 15
      else: 14
    genome = readGenome(stream, modelVersion)
    alphabet = readIntSeq(stream)
    if header != ENERGY_MODEL_MAGIC_V1:
      frequencies = readIntSeq(stream)
      let gramCount = stream.readInt64()
      if gramCount < 0 or gramCount > 1024:
        quit("Invalid generation gram count.")
      for _ in 0 ..< int(gramCount):
        let tokens = readIntSeq(stream)
        let frequency = stream.readInt64()
        if tokens.len < 2 or tokens.len > 8 or frequency < 1 or
            frequency > 1_000_000_000:
          quit("Invalid generation n-gram.")
        grams.add(GenerationGram(tokens: tokens, frequency: int(frequency)))
    else:
      frequencies = newSeq[int](alphabet.len)
      for i in 0 ..< alphabet.len: frequencies[i] = 1
    if not stream.atEnd(): quit("Trailing energy model data.")
  finally:
    stream.close()
  if genome.len < 1 or genome.len > 100000 or alphabet.len == 0 or
      alphabet.len > 4096 or frequencies.len != alphabet.len:
    quit("Invalid energy model.")
  for r in genome:
    if r.weight != r.weight or abs(r.weight) > 1.0e6:
      quit("Invalid model weights.")
  var allowed = initHashSet[int]()
  var cumulative = newSeq[float](alphabet.len)
  var total = 0.0
  for i, v in alphabet:
    if v < 0 or v > 0x10FFFF or (v >= 0xD800 and v <= 0xDFFF) or
        frequencies[i] <= 0 or frequencies[i] > 1_000_000_000 or v in allowed:
      quit("Invalid model Unicode alphabet/frequencies.")
    allowed.incl(v)
    total += sqrt(float(frequencies[i]))
    cumulative[i] = total
  for gram in grams:
    for v in gram.tokens:
      if v notin allowed: quit("Generation n-gram not in alphabet.")
  var compiled = newSeq[SeqPattern](genome.len)
  for i, rule in genome: compiled[i] = compileSeqPattern(rule.a)
  let candidates = buildCandidateIndex(compiled)
  var scratch = newEvalScratch(genome.len)
  prepareScoreScratch(genome, scratch)
  # Bounded rolling memo. The old code did `memo.clear()` exactly at 32,768
  # entries. Crossing --steps~=32768 therefore destroyed the entire accumulated
  # cache in one operation and abruptly turned converged duplicates back into
  # expensive scoreRaw calls. Evict one oldest key at a time instead.
  # Bound cached UTF-8 key bytes too: a 32,768-entry cache is cheap at length 64
  # but huge at length 3100, causing a second allocator/GC cliff.
  const GENERATION_MEMO_MAX_ENTRIES = 32768
  const GENERATION_MEMO_KEY_BYTE_BUDGET = 16 * 1024 * 1024
  var memo = initTable[string, float]()
  var memoOrder = newSeq[string](GENERATION_MEMO_MAX_ENTRIES)
  var memoHead = 0
  var memoCount = 0
  var memoKeyBytes = 0
  var evaluations = 0
  var cacheHits = 0
  var attempted = 0
  var bestEnergy = -Inf
  var bestSuffix: seq[int] = @[]
  var exactScoreSeconds = 0.0
  var generationBudgetCutoffs = 0
  var maxExactScoreSeconds = 0.0
  var minObservedInputBytes = high(int)
  var maxObservedInputBytes = 0
  var minObservedRounds = high(int)
  var maxObservedRounds = 0
  var minObservedMaxAgg = high(int)
  var maxObservedMaxAgg = 0

  proc memoInsertRolling(key: string, value: float) =
    # `key` is known absent. Keep the table bounded without a global flush.
    while memoCount > 0 and
          (memoCount >= GENERATION_MEMO_MAX_ENTRIES or
           memoKeyBytes + key.len > GENERATION_MEMO_KEY_BYTE_BUDGET):
      let oldKey = memoOrder[memoHead]
      if oldKey.len > 0:
        memo.del(oldKey)
        memoKeyBytes = max(0, memoKeyBytes - oldKey.len)
      memoOrder[memoHead].setLen(0)
      memoHead = (memoHead + 1) mod GENERATION_MEMO_MAX_ENTRIES
      dec memoCount
    let slot = (memoHead + memoCount) mod GENERATION_MEMO_MAX_ENTRIES
    memoOrder[slot] = key
    memo[key] = value
    memoKeyBytes += key.len
    inc memoCount

  proc score(suffix: seq[int], key: string): float =
    if memo.hasKey(key):
      inc cacheHits
      return memo[key]
    let whole = prompt & key
    let rawBytes = whole.len
    let expectedRounds = max(1, int(ceil(2 * sqrt(float(max(1, rawBytes))))))
    let expectedMaxAgg = min(32768, max(3100, rawBytes * 32))
    minObservedInputBytes = min(minObservedInputBytes, rawBytes)
    maxObservedInputBytes = max(maxObservedInputBytes, rawBytes)
    minObservedRounds = min(minObservedRounds, expectedRounds)
    maxObservedRounds = max(maxObservedRounds, expectedRounds)
    minObservedMaxAgg = min(minObservedMaxAgg, expectedMaxAgg)
    maxObservedMaxAgg = max(maxObservedMaxAgg, expectedMaxAgg)
    result = -Inf
    let scoreStarted = epochTime()
    try:
      result = scoreRaw(genome, compiled, candidates, textBytes(whole), scratch,
        maxRuleVisits = TEXT_GENERATION_SCORE_MAX_RULE_VISITS,
        maxSampleSeconds = TEXT_GENERATION_SCORE_MAX_SECONDS,
        clockCheckVisits = TEXT_GENERATION_CLOCK_CHECK_VISITS)
      if result != result or result == Inf: result = -Inf
    except EvaluationBudgetExceeded:
      inc generationBudgetCutoffs
      result = -Inf
    except CatchableError:
      result = -Inf
    let scoreElapsed = epochTime() - scoreStarted
    exactScoreSeconds += scoreElapsed
    maxExactScoreSeconds = max(maxExactScoreSeconds, scoreElapsed)
    memoInsertRolling(key, result)
    inc evaluations

  proc measure(suffix: seq[int]): TextIndividual =
    result.suffix = suffix
    result.key = runeText(suffix)
    result.energy = score(suffix, result.key)
    inc attempted
    if result.energy > bestEnergy:
      bestEnergy = result.energy
      bestSuffix = cloneInts(suffix)
  proc sortIsland(pop: var seq[TextIndividual]) =
    pop.sort(proc(a, b: TextIndividual): int =
      if a.energy > b.energy: -1
      elif a.energy < b.energy: 1
      else: 0
    )

  let actualPopulation = min(requestedPopulation, steps)
  let islandCount = min(GENERATION_ISLANDS, actualPopulation)
  var islands = newSeq[seq[TextIndividual]](islandCount)
  for i in 0 ..< actualPopulation:
    let proposed = initialGenerationSuffix(count, alphabet, cumulative, grams)
    islands[i mod islandCount].add(measure(proposed))
  for island in 0 ..< islandCount: sortIsland(islands[island])
  let initialBest = bestEnergy
  # Keep the requested .txt continuously useful: it always contains ONLY the
  # best complete text seen so far.  This is one tiny write per text-GA
  # generation, never one file/write per candidate.
  if bestSuffix.len > 0:
    writeFile(outputPath, prompt & runeText(bestSuffix))
  var localStagnation = newSeq[int](islandCount)
  var trials: array[6, int]
  var wins: array[6, int]
  var generation = 0
  while attempted < steps:
    inc generation
    for island in 0 ..< islandCount:
      let old = islands[island]
      let target = old.len
      let before = old[0].energy
      var next: seq[TextIndividual] = @[]
      var seen = initHashSet[string]()
      # Always reserve a proposal slot, including singleton/two-member islands.
      # Otherwise population=8 retains every member and attempted never advances.
      for i in 0 ..< min(2, max(0, target - 1)):
        let key = old[i].key
        if key notin seen:
          next.add(old[i])
          seen.incl(key)
      var diversity = 0.0
      for item in old:
        diversity += generationDistance(item.suffix, old[0].suffix)
      diversity /= float(old.len)
      let extra = if diversity < 0.15: 2 else: 0
      let intensity = min(10, 1 + localStagnation[island] div 5 + extra)
      while next.len < target and attempted < steps:
        let p1 = generationParent(old, false)
        let p2 = generationParent(old, rand(99) < 35)
        var op = generationOperator(trials, wins, localStagnation[island])
        if diversity < 0.08 and rand(99) < 25: op = 5
        var proposal = generationOffspring(old[p1].suffix, old[p2].suffix,
          op, intensity, alphabet, cumulative, grams)
        # Suppress duplicate evaluations; keep retries strictly bounded.
        # Build UTF-8 only once per actual proposal version.
        var proposalKey = runeText(proposal)
        for retry in 0 ..< 3:
          if proposalKey notin seen: break
          generationMutation(proposal, alphabet, cumulative, grams, intensity + retry)
          proposalKey = runeText(proposal)
        var child = TextIndividual(suffix: proposal, key: proposalKey, energy: 0.0)
        child.energy = score(child.suffix, child.key)
        inc attempted
        if child.energy > bestEnergy:
          bestEnergy = child.energy
          bestSuffix = cloneInts(child.suffix)
        inc trials[op]
        let benchmark = if op == 5: before
                        elif op == 0 or op == 1: max(old[p1].energy, old[p2].energy)
                        else: old[p1].energy
        if child.energy > benchmark: inc wins[op]
        if child.energy != -Inf and child.key notin seen:
          next.add(child)
          seen.incl(child.key)
      # If the requested proposal budget ends partway through an island,
      # restore the unfilled slots with the old individuals.
      if next.len < target:
        for item in old:
          if next.len >= target: break
          if item.key notin seen:
            next.add(item)
            seen.incl(item.key)
      if next.len == 0: next.add(old[0])
      sortIsland(next)
      islands[island] = next
      if next[0].energy > before:
        localStagnation[island] = 0
      else:
        inc localStagnation[island]
      if attempted >= steps: break
    # Sparse ring migration and an immutable global archive; migration does not
    # alter scoring or admit stale scores from another dataset/model.
    if islandCount > 1 and generation mod 8 == 0:
      var migrants = newSeq[TextIndividual](islandCount)
      for i in 0 ..< islandCount: migrants[i] = islands[i][0]
      for i in 0 ..< islandCount:
        let incoming = migrants[(i + islandCount - 1) mod islandCount]
        var found = false
        for item in islands[i]:
          if item.suffix == incoming.suffix:
            found = true
            break
        if not found and incoming.energy > islands[i][^1].energy:
          islands[i][^1] = incoming
          sortIsland(islands[i])
    # Persist the current global champion after EVERY text-GA generation.
    # Overwrite the same output path so disk use stays constant and only the
    # best individual is exposed, as requested.
    if bestSuffix.len > 0:
      writeFile(outputPath, prompt & runeText(bestSuffix))

    if generation mod 5 == 0 or attempted >= steps:
      let avgExactSeconds = if evaluations > 0:
                              exactScoreSeconds / float(evaluations)
                            else: 0.0
      echo "generate gen=", generation, " candidates=", attempted, "/", steps,
           " exact_evaluations=", evaluations, " cache_hits=", cacheHits,
           " memo_entries=", memo.len,
           " memo_key_MiB=", formatFloat(float(memoKeyBytes) / (1024.0 * 1024.0), ffDecimal, 2),
           " exact_score_s=", formatFloat(exactScoreSeconds, ffDecimal, 3),
           " avg_exact_ms=", formatFloat(avgExactSeconds * 1000.0, ffDecimal, 3),
           " max_exact_ms=", formatFloat(maxExactScoreSeconds * 1000.0, ffDecimal, 3),
           " budget_cutoffs=", generationBudgetCutoffs,
           " input_bytes=", minObservedInputBytes, "..", maxObservedInputBytes,
           " rounds=", minObservedRounds, "..", maxObservedRounds,
           " max_state=", minObservedMaxAgg, "..", maxObservedMaxAgg,
           " initial_best=", initialBest, " best=", bestEnergy,
           " island_stagnation=", localStagnation
  if bestEnergy == -Inf: quit("All proposals failed evaluation; no text written.")
  writeFile(outputPath, prompt & runeText(bestSuffix))
  echo "Generated: ", outputPath, " initial_score=", initialBest,
       " best_score=", bestEnergy, " exact_evaluations=", evaluations,
       " cache_hits=", cacheHits, " candidates=", attempted,
       " memo_entries=", memo.len,
       " memo_key_MiB=", formatFloat(float(memoKeyBytes) / (1024.0 * 1024.0), ffDecimal, 2),
       " exact_score_s=", formatFloat(exactScoreSeconds, ffDecimal, 3),
       " max_exact_ms=", formatFloat(maxExactScoreSeconds * 1000.0, ffDecimal, 3),
       " budget_cutoffs=", generationBudgetCutoffs,
       " generations=", generation, " population=", actualPopulation,
       " islands=", islandCount
  echo prompt & runeText(bestSuffix)

# ---------------------------------------------------------------------------
# Jev generation lane (opt-in when a server-side API key is available).
# One fixed prompt, up to 64 FULL-evaluated genomes (default 64), 6400
# logical candidate evaluations per genome, with a 16-member inner elitist text GA.
# RNG and scoring scratch are local; GA random state is untouched.
# The external Jev process receives texts, never genomes or training data.
# ---------------------------------------------------------------------------
const JEV_BRIDGE_PROTOCOL = 1
const JEV_MAX_MODELS = 64
const JEV_DEFAULT_MODELS = 64
const JEV_GENERATION_INTERVAL = FUSION_INTERVAL  # one shared cadence (16 generations)
const JEV_STEPS = 7200
const JEV_SUFFIX_LENGTH = 64
# Inner TEXT GA, independent for each outer Genome. Evaluations include the
# initial 16 seeds and every offspring; reusing an elite costs no evaluation.
const JEV_INNER_POPULATION = 20
const JEV_INNER_ELITES = 3
const JEV_INNER_TOURNAMENT = 4
const JEV_INNER_CROSSOVER_PERCENT = 45
const JEV_INNER_IMMIGRANT_PERCENT = 3
static:
  doAssert JEV_INNER_POPULATION <= JEV_STEPS
  doAssert JEV_INNER_ELITES > 0 and JEV_INNER_ELITES < JEV_INNER_POPULATION
  doAssert JEV_INNER_TOURNAMENT >= 2
const JEV_SURVIVOR_SLOTS = 8
const JEV_MA_WINDOW = 20

type
  JevGenerated = object
    id: int
    text: string
    energy: float
    evaluations: int
    actualScoreCalls: int
    cacheHits: int
    innerGenerations: int
    budgetCutoffs: int
    scoreSeconds: float
    maxScoreSeconds: float
    error: string
  JevRound = object
    valid: bool
    ranked: seq[int]
    qualities: seq[tuple[id: int, quality: float]]
    meanQuality: float
    pairComparisons: int
    apiRequests: int
    localEvaluations: int
    actualScoreCalls: int
    cacheHits: int
    budgetCutoffs: int
    scoreSeconds: float
    maxScoreSeconds: float
  JevPoint = object
    generation: int
    quality: float
    average: float

var jevHistory: seq[JevPoint] = @[]

proc jevSampleToken(rng: var Rand, alphabet: seq[int], cumulative: seq[float]): int {.gcsafe.} =
  if alphabet.len == 1 or rand(rng, 99) < 12:
    return alphabet[rand(rng, alphabet.len - 1)]
  let draw = rand(rng, cumulative[^1])
  var lo = 0
  var hi = alphabet.len - 1
  while lo < hi:
    let mid = (lo + hi) div 2
    if cumulative[mid] < draw: lo = mid + 1
    else: hi = mid
  alphabet[lo]

proc jevInitialSuffix(rng: var Rand, length: int,
                      alphabet: seq[int], cumulative: seq[float],
                      grams: seq[GenerationGram]): seq[int] {.gcsafe.} =
  result = newSeqOfCap[int](length)
  while result.len < length:
    if grams.len > 0 and rand(rng, 99) < 65:
      let gram = grams[rand(rng, grams.len - 1)]
      for token in gram.tokens:
        if result.len >= length: break
        result.add(token)
    else:
      result.add(jevSampleToken(rng, alphabet, cumulative))

# The ordinary GA evolves Genomes (replacement rules); this miniature GA
# evolves TEXT suffixes while scoring every candidate with one immutable Genome.
type JevInnerIndividual = object
  suffix: seq[int]
  energy: float

proc jevInnerSort(pop: var seq[JevInnerIndividual]) {.inline.} =
  pop.sort(proc(a, b: JevInnerIndividual): int =
    if a.energy > b.energy: -1
    elif a.energy < b.energy: 1
    else: 0
  )

proc jevInnerTournament(pop: seq[JevInnerIndividual], rng: var Rand): int {.inline, gcsafe.} =
  result = rand(rng, pop.len - 1)
  for _ in 1 ..< JEV_INNER_TOURNAMENT:
    let candidate = rand(rng, pop.len - 1)
    if pop[candidate].energy > pop[result].energy:
      result = candidate

proc jevGenerate(g: Genome, compiled: seq[SeqPattern],
                 index: CandidateIndex, id, generation: int,
                 prompt: string, alphabet: seq[int], cumulative: seq[float],
                 grams: seq[GenerationGram],
                 editUpperSchedule: ptr seq[int]): JevGenerated {.gcsafe.} =
  ## EXACTLY 2,500 logical candidate evaluations per outer model:
  ## 16 initial seeds + 2,484 offspring. Elites/parents are NEVER rescored.
  ## Cache hits count as logical evaluations as in the previous hill climber;
  ## actualScoreCalls separately counts expensive scoreRaw invocations.
  result.id = id
  # Common random initialization within a round; moving a genome to another
  # population slot must not change its generated text. Rotate across rounds.
  var rng = initRand(int64(generation + 1) * 1_000_003'i64 + 7_919'i64)
  template scratch: var EvalScratch = getThreadEvalScratch(g.len)
  prepareScoreScratch(g, scratch)
  let generationIndex = generationCandidateIndex(index)
  # The prompt is fixed for all 16k proposals. Avoid allocating a UTF-8 string
  # and then converting that string back to byte integers on every score.
  let promptBytes = textBytes(prompt)
  var scoreInput = newSeqOfCap[int](promptBytes.len + JEV_SUFFIX_LENGTH * 4)
  # Cache by immutable rune suffix, not by the repeatedly rebuilt whole string.
  var memo = initTable[seq[int], float](JEV_STEPS)
  var actualScoreCalls = 0
  var cacheHits = 0
  var budgetCutoffs = 0
  var scoreSeconds = 0.0
  var maxScoreSeconds = 0.0
  proc tryEnergy(suffix: seq[int]): float =
    let cached = memo.getOrDefault(suffix, NaN)
    if cached == cached:
      inc cacheHits
      return cached
    inc actualScoreCalls
    buildPromptSuffixBytes(promptBytes, suffix, scoreInput)
    let scoreStarted = epochTime()
    try:
      result = scoreRaw(g, compiled, generationIndex, scoreInput, scratch,
        maxRuleVisits = TEXT_GENERATION_SCORE_MAX_RULE_VISITS,
        maxSampleSeconds = TEXT_GENERATION_SCORE_MAX_SECONDS,
        clockCheckVisits = TEXT_GENERATION_CLOCK_CHECK_VISITS)
      if result != result or result == Inf: result = -Inf
    except EvaluationBudgetExceeded:
      inc budgetCutoffs
      result = -Inf
    except CatchableError:
      result = -Inf
    let elapsed = epochTime() - scoreStarted
    scoreSeconds += elapsed
    maxScoreSeconds = max(maxScoreSeconds, elapsed)
    # All GA parents/offspring are immutable after scoring; clone defensively so
    # a future mutation refactor cannot invalidate a hash-table key in place.
    memo[cloneInts(suffix)] = result

  # Maintain 16 diverse text lineages rather than mutating one greedy incumbent.
  # Sorting ensures the best two survive each GENERATION without reevaluation.
  var innerPop = newSeqOfCap[JevInnerIndividual](JEV_INNER_POPULATION)
  var bestScore = -Inf
  var bestSuffix: seq[int] = @[]
  for _ in 0 ..< JEV_INNER_POPULATION:
    let seed = jevInitialSuffix(rng, JEV_SUFFIX_LENGTH, alphabet, cumulative, grams)
    let score = tryEnergy(seed)
    inc result.evaluations
    innerPop.add(JevInnerIndividual(suffix: seed, energy: score))
    if bestSuffix.len == 0 or score > bestScore:
      bestScore = score
      bestSuffix = cloneInts(seed)
  jevInnerSort(innerPop)
  var stagnation = 0
  # Reversible partial Fisher-Yates: only O(edits), not O(64) initialization
  # on every offspring. All state is per-worker/per-model, never shared.
  var positions = newSeq[int](JEV_SUFFIX_LENGTH)
  var swapTargets = newSeq[int](JEV_SUFFIX_LENGTH)
  for i in 0 ..< positions.len: positions[i] = i

  while result.evaluations < JEV_STEPS:
    let before = innerPop[0].energy
    var next = newSeqOfCap[JevInnerIndividual](JEV_INNER_POPULATION)
    for i in 0 ..< JEV_INNER_ELITES:
      next.add(innerPop[i]) # immutable survivor; no scoreRaw call
    while next.len < JEV_INNER_POPULATION and result.evaluations < JEV_STEPS:
      var child: seq[int]
      let immigrantRate = if stagnation >= 32: 12 else: JEV_INNER_IMMIGRANT_PERCENT
      if rand(rng, 99) < immigrantRate:
        # Fresh grammar/corpus initialization keeps the search from collapsing.
        child = jevInitialSuffix(rng, JEV_SUFFIX_LENGTH, alphabet, cumulative, grams)
      else:
        let a = jevInnerTournament(innerPop, rng)
        child = cloneInts(innerPop[a].suffix)
        if rand(rng, 99) < JEV_INNER_CROSSOVER_PERCENT:
          var b = jevInnerTournament(innerPop, rng)
          for _ in 0 ..< 3:
            if b != a: break
            b = jevInnerTournament(innerPop, rng)
          if b != a:
            # Two-point crossover exchanges a contiguous UTF-8 codepoint span.
            var left = rand(rng, JEV_SUFFIX_LENGTH - 1)
            var right = rand(rng, JEV_SUFFIX_LENGTH - 1)
            if left > right: swap(left, right)
            for pos in left .. right:
              child[pos] = innerPop[b].suffix[pos]
        # A shared schedule eliminates 32 repeated pow() loops. Keep log-uniform
        # edits, but reopen broader exploration after sustained stagnation.
        var upper = editUpperSchedule[][result.evaluations]
        if stagnation >= 48: upper = max(upper, 16)
        elif stagnation >= 24: upper = max(upper, 8)
        upper = max(1, min(child.len, upper))
        let edits = max(1, min(upper,
          int(round(exp(rand(rng, ln(float(upper))))))))
        for j in 0 ..< edits:
          let k = j + rand(rng, positions.len - j - 1)
          swapTargets[j] = k
          swap(positions[j], positions[k])
          let at = positions[j]
          var token = jevSampleToken(rng, alphabet, cumulative)
          if token == child[at] and alphabet.len > 1:
            token = alphabet[rand(rng, alphabet.len - 1)]
          if token == child[at] and alphabet.len > 1:
            token = if alphabet[0] != child[at]: alphabet[0] else: alphabet[1]
          if token != child[at]: child[at] = token
        for j in countdown(edits - 1, 0):
          swap(positions[j], positions[swapTargets[j]])
      # Avoid spending the budget on copies of the surviving population.
      # The repair is bounded; failure falls back to the exact memoized score.
      for _ in 0 ..< 4:
        var duplicate = false
        for item in next:
          if item.suffix == child:
            duplicate = true
            break
        if not duplicate:
          for item in innerPop:
            if item.suffix == child:
              duplicate = true
              break
        if not duplicate: break
        let pos = rand(rng, child.len - 1)
        let oldToken = child[pos]
        var token = jevSampleToken(rng, alphabet, cumulative)
        if token == oldToken and alphabet.len > 1:
          token = if alphabet[0] != oldToken: alphabet[0] else: alphabet[1]
        child[pos] = token

      let score = tryEnergy(child)
      inc result.evaluations
      if bestSuffix.len == 0 or score > bestScore + 1.0e-10:
        bestScore = score
        bestSuffix = cloneInts(child)
      next.add(JevInnerIndividual(suffix: child, energy: score))
    # The final generation may be partial; its best was already tracked.
    jevInnerSort(next)
    if next[0].energy > before + 1.0e-10:
      stagnation = 0
    else:
      inc stagnation
    innerPop = move(next)
    inc result.innerGenerations

  if bestScore == -Inf or bestScore == Inf or bestScore != bestScore:
    raise newException(ValueError, "All Jev generation proposals failed evaluation")
  result.text = runeText(bestSuffix)
  result.energy = bestScore
  result.actualScoreCalls = actualScoreCalls
  result.cacheHits = cacheHits
  result.budgetCutoffs = budgetCutoffs
  result.scoreSeconds = scoreSeconds
  result.maxScoreSeconds = maxScoreSeconds
  # An invariant failure must not silently bias the external Jev ranking.
  if result.evaluations != JEV_STEPS or
     result.actualScoreCalls + result.cacheHits != JEV_STEPS:
    raise newException(ValueError, "Jev inner evaluation budget mismatch")

proc jevGenerateById(population: ptr seq[Genome],
                     compiled: ptr seq[seq[SeqPattern]],
                     indices: ptr seq[CandidateIndex],
                     id, generation: int, prompt: ptr string,
                     alphabet: ptr seq[int], cumulative: ptr seq[float],
                     grams: ptr seq[GenerationGram],
                     editUpperSchedule: ptr seq[int]): JevGenerated {.gcsafe.} =
  ## Immutable parent arrays; all RNG, scratch, and memo state are worker-local.
  ## Return errors as data so the caller can drain every FlowVar before exiting.
  try:
    result = jevGenerate(population[][id], compiled[][id], indices[][id],
      id, generation, prompt[], alphabet[], cumulative[], grams[],
      editUpperSchedule)
  except CatchableError as e:
    result.id = id
    result.error = e.msg
  except Defect as e:
    result.id = id
    result.error = e.msg

proc jevRun(generation: int, prompt, model, bridge: string,
            candidateIds: seq[int], population: seq[Genome],
            compiled: seq[seq[SeqPattern]], indices: seq[CandidateIndex],
            alphabet: seq[int], cumulative: seq[float],
            grams: seq[GenerationGram], generationWorkers: int): JevRound =
  if candidateIds.len < 2 or not fileExists(bridge): return
  var request = %*{"version": JEV_BRIDGE_PROTOCOL,
                   "generation": generation, "prompt": prompt,
                   "population_size": population.len,
                   "candidates": []}
  var generatedIds = initHashSet[int]()
  for id in candidateIds:
    if id in generatedIds or id < 0 or id >= population.len: return
    generatedIds.incl(id)
  let generationStarted = epochTime()
  let concurrency = max(1, min(candidateIds.len, generationWorkers))
  # The mathematical edit envelope depends ONLY on (step, suffix length), so
  # computing pow() separately for every model repeats identical work 64x.
  var editUpperSchedule = newSeq[int](JEV_STEPS)
  for step in 1 ..< JEV_STEPS:
    let progress = float(step) / float(JEV_STEPS - 1)
    editUpperSchedule[step] = max(1, min(JEV_SUFFIX_LENGTH,
      int(round(pow(float(JEV_SUFFIX_LENGTH), 1.0 - progress)))))
  echo "  jev: generating ", candidateIds.len, " texts; workers=", concurrency,
       " per_model_steps=", JEV_STEPS, " model_seconds=uncapped"
  # Bounded batches prevent starting 64 large EvalScratch allocations at once.
  # Every batch is fully joined before its stack-backed immutable pointers die.
  let populationPtr = unsafeAddr population
  let compiledPtr = unsafeAddr compiled
  let indicesPtr = unsafeAddr indices
  let promptPtr = unsafeAddr prompt
  let alphabetPtr = unsafeAddr alphabet
  let cumulativePtr = unsafeAddr cumulative
  let gramsPtr = unsafeAddr grams
  var generated = 0
  var launched = 0
  var active = 0
  var failed = false
  var jobs = newSeq[FlowVar[JevGenerated]](concurrency)
  var busy = newSeq[bool](concurrency)
  var jobStarted = newSeq[float](concurrency)
  var maxModelSeconds = 0.0
  while generated < candidateIds.len:
    # Refill each freed slot immediately: no barrier waiting for the slowest
    # model in a batch of eight. All in-flight jobs are drained on failure.
    for slot in 0..<concurrency:
      if not busy[slot] and launched < candidateIds.len and not failed:
        jobStarted[slot] = epochTime()
        jobs[slot] = spawn jevGenerateById(populationPtr, compiledPtr, indicesPtr,
          candidateIds[launched], generation, promptPtr,
          alphabetPtr, cumulativePtr, gramsPtr, unsafeAddr editUpperSchedule)
        busy[slot] = true
        inc launched
        inc active
    var completedAny = false
    for slot in 0..<concurrency:
      if not busy[slot] or not isReady(jobs[slot]): continue
      let sample = ^jobs[slot]
      busy[slot] = false
      dec active
      inc generated
      completedAny = true
      let modelElapsed = epochTime() - jobStarted[slot]
      maxModelSeconds = max(maxModelSeconds, modelElapsed)
      if sample.error.len > 0 or sample.text.len == 0 or
         sample.energy != sample.energy or abs(sample.energy) == Inf or
         unicode.validateUtf8(sample.text) != -1:
        stderr.writeLine("  jev: generation failed for id=", sample.id,
          " (", sample.error, "; utf8_bytes=", sample.text.len, ")")
        failed = true
      else:
        result.localEvaluations += sample.evaluations
        result.actualScoreCalls += sample.actualScoreCalls
        result.cacheHits += sample.cacheHits
        result.budgetCutoffs += sample.budgetCutoffs
        result.scoreSeconds += sample.scoreSeconds
        result.maxScoreSeconds = max(result.maxScoreSeconds, sample.maxScoreSeconds)
        request["candidates"].add(%*{"id": sample.id, "text": sample.text})
      if generated mod concurrency == 0 or generated == candidateIds.len:
        echo "  jev: generated=", generated, "/", candidateIds.len,
             " logical_evaluations=", result.localEvaluations,
             " actual_score_calls=", result.actualScoreCalls,
             " cache_hits=", result.cacheHits,
             " budget_cutoffs=", result.budgetCutoffs,
             " score_cpu_s_sum=", formatFloat(result.scoreSeconds, ffDecimal, 2),
             " avg_score_ms=", formatFloat(1000.0*result.scoreSeconds / float(max(1,result.actualScoreCalls)), ffDecimal, 3),
             " max_score_ms=", formatFloat(result.maxScoreSeconds*1000.0, ffDecimal, 1),
             " max_model_s=", formatFloat(maxModelSeconds, ffDecimal, 2),
             " generation_seconds=", formatFloat(epochTime()-generationStarted, ffDecimal, 2)
    if failed and active == 0: return
    if not completedAny: sleep(1)
  # Completion order must not perturb the bridge's ranking/tie-break order.
  var canonicalCandidates = newJArray()
  for id in candidateIds:
    for candidate in request["candidates"]:
      if candidate["id"].getInt() == id: canonicalCandidates.add(candidate)
  request["candidates"] = canonicalCandidates
  echo "  jev: generation complete; starting API, seconds=",
       formatFloat(epochTime() - generationStarted, ffDecimal, 3)
  # Files are namespaced by generation and process, and always removed.
  let base = getTempDir() / ("at_jev_" & $getCurrentProcessId() & "_" & $generation)
  let inputPath = base & ".input.json"
  let outputPath = base & ".output.json"
  var process: Process = nil
  try:
    if fileExists(outputPath): removeFile(outputPath)
    writeFile(inputPath, $request)
    process = startProcess("python3", args = @[bridge,
      "--input", inputPath, "--output", outputPath, "--model", model,
      "--workers", "8", "--pairwise-top", $candidateIds.len],
      options = {poUsePath, poParentStreams})
    let apiStarted = epochTime()
    let status = waitForExit(process, 90_000)
    if status == -1:
      terminate(process)
      discard waitForExit(process, 5_000)
      stderr.writeLine("  jev: bridge timeout (90s); skipping this round")
      return
    if status != 0 or not fileExists(outputPath):
      stderr.writeLine("  jev: API/bridge failed; no ranking or score admitted")
      return
    echo "  jev: API completed in ",
      formatFloat(epochTime() - apiStarted, ffDecimal, 3), " seconds"
    let data = parseFile(outputPath)
    if data["version"].getInt() != JEV_BRIDGE_PROTOCOL or
       data["generation"].getInt() != generation or
       data["ranked_ids"].len != candidateIds.len or
       data["scores"].len != candidateIds.len:
      raise newException(ValueError, "invalid bridge result sizes/version")
    var seenRank = initHashSet[int]()
    for item in data["ranked_ids"]:
      let id = item.getInt()
      if id notin generatedIds or id in seenRank:
        raise newException(ValueError, "invalid Jev rank identity")
      seenRank.incl(id)
      result.ranked.add(id)
    var seenScore = initHashSet[int]()
    var totalQuality = 0.0
    for item in data["scores"]:
      let id = item["id"].getInt()
      let quality = item["score"].getFloat()
      if id notin generatedIds or id in seenScore or quality != quality or
         quality < 0 or quality > 1.0:
        raise newException(ValueError, "invalid Jev score identity/value")
      seenScore.incl(id)
      result.qualities.add((id: id, quality: quality))
      totalQuality += quality
    result.meanQuality = totalQuality / float(candidateIds.len)
    let reportedMean = data["mean_quality"].getFloat()
    if reportedMean != reportedMean or abs(reportedMean) == Inf or
       abs(reportedMean - result.meanQuality) > 1.0e-6:
      raise newException(ValueError, "inconsistent Jev mean")
    result.pairComparisons = data["comparisons"].getInt()
    result.apiRequests = data["requests"].getInt()
    if result.apiRequests < 1 or result.pairComparisons < 0:
      raise newException(ValueError, "invalid Jev usage")
    result.valid = true
  except CatchableError as e:
    stderr.writeLine("  jev: no selection/metric admitted (", e.msg, ")")
    result = JevRound()
  finally:
    if not process.isNil:
      try:
        close(process)
      except OSError:
        discard
    for temporary in [inputPath, outputPath, outputPath & ".tmp"]:
      try:
        if fileExists(temporary): removeFile(temporary)
      except OSError:
        discard

proc jevAverage(points: seq[JevPoint], newest: float): float =
  var sum = newest
  var count = 1
  for i in countdown(points.len - 1, max(0, points.len - JEV_MA_WINDOW + 1)):
    if i < 0: break
    sum += points[i].quality
    inc count
  sum / float(count)

proc writeJevProgress(path: string, points: seq[JevPoint]) =
  ## Atomically replace one row per successful Jev evaluation.
  var lines = "generation,jev_rank_weighted_quality,jev_rank_weighted_ma20\n"
  for p in points:
    lines.add($p.generation & "," & $p.quality & "," & $p.average & "\n")
  writeFile(path & ".tmp", lines)
  moveFile(path & ".tmp", path)

proc loadJevProgress(path: string, resumeGeneration: int,
                     compatible: bool): seq[JevPoint] =
  if not compatible or not fileExists(path):
    writeJevProgress(path, result)
    return
  # Refuse silently reinterpreting a legacy equal-weight CSV as weighted data.
  # The configuration check above normally prevents this; validate its header
  # as a second line of defense before parsing old observations.
  let expectedHeader = "generation,jev_rank_weighted_quality,jev_rank_weighted_ma20"
  if not readFile(path).startsWith(expectedHeader & "\n"):
    quit("[jev-plot] incompatible Jev CSV header; refusing to overwrite " & path)
  var previous = -1
  for line in readFile(path).splitLines():
    let fields = line.split(',')
    if fields.len != 3 or fields[0] == "generation": continue
    try:
      let generation = parseInt(fields[0])
      let quality = parseFloat(fields[1])
      if generation <= previous or generation >= resumeGeneration or
         generation < 0 or quality != quality or quality < 0 or quality > 1:
        continue
      result.add(JevPoint(generation: generation, quality: quality,
                          average: jevAverage(result, quality)))
      previous = generation
    except ValueError:
      discard
  writeJevProgress(path, result)

var generatePath = ""
var generatePrompt = ""
var generateOutput = "generated.txt"
var generateLength = 128
var generateSteps = 2000
var generateSeed = 42
var generatePopulation = 48
for arg in commandLineParams():
  if arg.startsWith("--generate="): generatePath = arg.split("=", 1)[1]
  elif arg.startsWith("--prompt="): generatePrompt = arg.split("=", 1)[1]
  elif arg.startsWith("--out="): generateOutput = arg.split("=", 1)[1]
  elif arg.startsWith("--length="): generateLength = parseInt(arg.split("=", 1)[1])
  elif arg.startsWith("--steps="): generateSteps = parseInt(arg.split("=", 1)[1])
  elif arg.startsWith("--seed="): generateSeed = parseInt(arg.split("=", 1)[1])
  elif arg.startsWith("--population="): generatePopulation = parseInt(arg.split("=", 1)[1])
if "--process-audit-test" in commandLineParams():
  randomize(123456)
  doAssert datasetHasInvalidBytes(@[@[@[0, 256]]])
  doAssert datasetHasInvalidBytes(@[@[@[-1, 255]]])
  doAssert not datasetHasInvalidBytes(@[@[@[0, 255]]])
  buildCorpusByteStats(@[@[0, 65, 66, 255]])
  block:
    for op in 0 .. 8:
      for seed in 0 ..< 500:
        randomize(seed)
        var state = @[255, 255, 255, 255, 255, 255, 255, 255]
        discard applyDiffusionKernel(state, state, op, 1.0)
        for v in state: doAssert v >= 0 and v <= 255
    var shortState = @[65]
    discard applyDiffusionKernel(shortState, @[65], DIFF_OP_PERMUTE_BLOCK, 1.0)
    doAssert shortState == @[65]
    echo "PASS: all diffusion kernels preserve bytes; singleton shuffle is safe"
  block:
    let base = @[10, 20, 30, 40, 50, 60, 70, 80]
    for seed in 0 ..< 500:
      randomize(seed)
      let length = 1 + rand(5)
      let src = rand(base.len - length)
      let dst = rand(base.len - length)
      var expected = cloneInts(base)
      for j in 0 ..< length: expected[dst+j] = base[src+j]
      randomize(seed)
      var actual = cloneInts(base)
      discard applyDiffusionKernel(actual, base, DIFF_OP_COPY_BLOCK, 1.0)
      doAssert actual == expected
    echo "PASS: overlapping diffusion copies agree with snapshot reference"
  block:
    let base = @[1, 2, 3, 4]
    for seed in 0 ..< 50:
      randomize(seed)
      let trajectory = buildDiffusionTrajectory(base, 20)
      for i, state in trajectory.x:
        doAssert trajectory.y[i] == max(1, base.len - estimateDiffusionDamage(base, state, 0))
      doAssert trajectory.x[0] == base
    echo "PASS: trajectory targets describe the actual resulting state"
  block:
    let cases = @[@[0.5], @[0.5], @[0.5], @[0.5]]
    var observed: array[4, int]
    randomize(728)
    for _ in 0 ..< 1000:
      var uses = @[PARENT_MAX_USES, PARENT_MAX_USES, PARENT_MAX_USES, PARENT_MAX_USES]
      inc observed[selectParentCapped(cases, [0,1,2,3], uses, 0.0)]
    for count in observed: doAssert count > 150 and count < 350
    echo "PASS: exhausted parent-cap ties do not favor first candidate"
  block:
    var g = newSeq[Rule](16)
    for i in 0 ..< g.len:
      g[i] = Rule(a: @[65, -1], b: @[-1], weight: 0.2,
        patternRevision: freshPatternRevision(), replacementRevision: freshReplacementRevision())
    g[0].embedding = defaultEmbedding()
    let snapshot = genomeFingerprint(g)
    for _ in 0 ..< 250:
      var child = cloneGenome(g)
      mutateGenome(child, 1.7, 15)
      doAssert genomeFingerprint(g) == snapshot
      doAssert validEmbedding(child[0].embedding)
      for rule in child:
        doAssert compatibleRule(rule) and countWildcards(rule.a) <= 16
      for op in CrossoverOp:
        let crossed = crossoverGenome(g, child, op, @[], @[])
        doAssert validEmbedding(crossed[0].embedding)
        for rule in crossed: doAssert compatibleRule(rule)
      doAssert genomeFingerprint(g) == snapshot
    echo "PASS: mutation/crossover keep parents immutable and captures/mappings valid"
  block:
    let g = @[Rule(a: @[65], b: @[65], weight: 0.0,
      embedding: defaultEmbedding())]
    let compiled = @[compileSeqPattern(g[0].a)]
    let index = buildCandidateIndex(compiled)
    var schedule = newSeq[int](JEV_STEPS)
    for i in 0 ..< schedule.len: schedule[i] = 4
    let a = jevGenerate(g, compiled, index, 0, 17, "", @[65,66], @[1.0,2.0], @[], addr schedule)
    let b = jevGenerate(g, compiled, index, 91, 17, "", @[65,66], @[1.0,2.0], @[], addr schedule)
    doAssert a.text == b.text and a.energy == b.energy
    doAssert a.evaluations == JEV_STEPS and b.evaluations == JEV_STEPS
    var invalid = cloneGenome(g)
    invalid[0].weight = -Inf
    var rejected = false
    try:
      discard jevGenerate(invalid, compiled, index, 0, 17, "", @[65], @[1.0], @[], addr schedule)
    except ValueError: rejected = true
    doAssert rejected
    exportEnergyModel("audit.model", g)
    echo "PASS: Jev generation is ID-independent, respects budget, rejects total failure"
  quit(0)

if generatePath.len > 0:
  optimizeText(generatePath, generatePrompt, generateOutput,
               generateLength, generateSteps, generateSeed, generatePopulation)
  quit(0)

if "--refinement-test" in commandLineParams():
  proc fusionFrontReference(points: seq[FusionPoint], cap: int): seq[int] =
    # Transitive epsilon-box dominance. The earlier proposed pairwise +/- eps
    # tolerance is not a partial order and must NOT be used for nondominated sort.
    if cap <= 0: return
    let eg = 0.002
    let ej = max(0.005, min(0.10, fusionNoise))
    for i, p in points:
      if p.g != p.g or p.j != p.j or p.g <= -Inf: continue
      var dominated = false
      let gx = floor(p.g/eg)
      let jy = floor(p.j/ej)
      for k, q in points:
        if i == k: continue
        let qg = floor(q.g/eg)
        let qj = floor(q.j/ej)
        if (qg >= gx and qj >= jy and (qg > gx or qj > jy)) or
           (qg == gx and qj == jy and
            (q.g > p.g or (q.g == p.g and (q.j > p.j or
             (q.j == p.j and k < i))))):
          dominated = true
          break
      if not dominated: result.add(i)
    if points.len == 0: return
    # Explicitly retain both true extrema; HV alone does not guarantee this.
    var bestG = 0
    var bestJ = 0
    for i in 1 ..< points.len:
      if points[i].g > points[bestG].g: bestG = i
      if points[i].j > points[bestJ].j or
         (points[i].j == points[bestJ].j and points[i].g > points[bestJ].g): bestJ = i
    for i in [bestG, bestJ]:
      if i notin result: result.add(i)
    if cap == 1: return @[bestG]
    proc area(ids: seq[int], omit: int): float =
      var ordered: seq[int] = @[]
      for id in ids:
        if id != omit: ordered.add(id)
      ordered.sort(proc(a,b: int): int = cmp(points[b].g, points[a].g))
      var height = 0.0
      for id in ordered:
        # Fixed reference (-1.01,-0.01); never normalize by a changing cohort.
        let y = points[id].j + 0.01
        if y > height:
          result += max(0.0, points[id].g + 1.01) * (y-height)
          height = y
    while result.len > cap:
      let total = area(result, -1)
      var victim = -1
      var least = Inf
      for pos, id in result:
        if id == bestG or id == bestJ: continue
        let contribution = max(0.0, total - area(result, id))
        if contribution < least:
          least = contribution
          victim = pos
      if victim < 0: break
      result.delete(victim)
    result.sort(proc(a,b: int): int = cmp(points[a].g, points[b].g))
  randomize(62817)
  for trial in 0..<300:
    var pts: seq[FusionPoint] = @[]
    for i in 0..<64:
      pts.add(FusionPoint(id:i,g:rand(2.0)-1.0,j:rand(1.0)))
    let capacity = 2 + trial mod 15
    doAssert fusionFront(pts,capacity) == fusionFrontReference(pts,capacity)
  let tied = @[FusionPoint(id:0,g:0.9,j:0.1),FusionPoint(id:1,g:0.9,j:0.8)]
  doAssert fusionFront(tied,16) == @[1]
  let invalid = @[FusionPoint(id:0,g:NaN,j:0.2),FusionPoint(id:1,g:Inf,j:0.9),
    FusionPoint(id:2,g:0.7,j:0.5),FusionPoint(id:3,g:0.8,j:NaN)]
  doAssert fusionFront(invalid,16) == @[2]
  doAssert fusionFront(@[invalid[0],invalid[1]],16).len == 0
  var uses = @[2,0,0]
  inc uses[1] # new parent is reserved by selection
  replaceParentUse(uses,0,1)
  doAssert uses == @[1,1,0]
  inc uses[0] # rejected retry
  releaseParentUse(uses,0)
  doAssert uses == @[1,1,0]
  releaseParentUse(uses,1) # asexual offspring did not use second parent
  doAssert uses == @[1,0,0]
  let g = @[Rule(a: @[97],b: @[98],weight:1.0,embedding:defaultEmbedding())]
  var different = cloneGenome(g)
  different[0].weight = 2.0
  let pop = @[g,cloneGenome(g),different]
  var fps: seq[Hash] = @[]
  doAssert uniqueGenomeIds(pop,@[0,1,2],fps) == @[0,2]
  doAssert fps[0] == fps[1] and sameGenomeContent(pop[0],pop[1])
  var tradeoff: seq[FusionPoint] = @[]
  for i in 0..<64:
    tradeoff.add(FusionPoint(id:i,g:float(i)/64.0,j:1.0-float(i)/64.0))
  var oldSeconds, newSeconds: float
  for repeatId in 0..<8:
    for lane in 0..1:
      let old = (lane+repeatId) mod 2 == 0
      let started = cpuTime()
      for i in 0..<8:
        let ids = if old: fusionFrontReference(tradeoff,16) else: fusionFront(tradeoff,16)
        doAssert ids.len == 16
      if old: oldSeconds += cpuTime()-started
      else: newSeconds += cpuTime()-started
  echo "PASS: 300 Pareto/HV equivalence cases, finite/tied extrema, parent accounting, exact cohort deduplication"
  echo "pareto_benchmark before_s=",oldSeconds," after_s=",newSeconds,
       " speedup=",oldSeconds/max(1e-9,newSeconds)
  quit(0)

if "--generation-index-test" in commandLineParams():
  var g: Genome = @[]
  randomize(83129)
  for i in 0..<3100:
    let a = @[rand(511), rand(511)]
    let b = @[rand(511)]
    g.add(Rule(a:a,b:b,weight:float(i mod 7)-3.0,
      patternRevision:freshPatternRevision(),replacementRevision:freshReplacementRevision()))
  var compiled: seq[SeqPattern] = @[]
  for r in g: compiled.add(compileSeqPattern(r.a))
  let sparse = buildCandidateIndex(compiled)
  let dense = generationCandidateIndex(sparse)
  var s1 = newEvalScratch(g.len)
  var s2 = newEvalScratch(g.len)
  var samples: seq[seq[int]] = @[]
  for k in 0..<128:
    var sample: seq[int] = @[]
    for i in 0..<256: sample.add(rand(511))
    samples.add(sample)
    scanCandidateIds(sparse,sample,s1.candidateIds,s1.candidateScratch,k)
    scanCandidateIds(dense,sample,s2.candidateIds,s2.candidateScratch,k)
    doAssert s1.candidateIds == s2.candidateIds
    doAssert scoreRaw(g,compiled,sparse,sample,s1) == scoreRaw(g,compiled,dense,sample,s2)
  var sparseTime, denseTime: float
  for repeatId in 0..<6:
    for lane in 0..1:
      let useDense = (lane + repeatId) mod 2 == 1
      let index = if useDense: dense else: sparse
      let started = cpuTime()
      for j in 0..<4096:
        scanCandidateIds(index,samples[j mod samples.len],s1.candidateIds,s1.candidateScratch,j mod 128)
      let elapsed = cpuTime()-started
      if useDense: denseTime += elapsed
      else: sparseTime += elapsed
  echo "PASS: 128 dense/sparse candidate and exact-score equivalence cases"
  echo "candidate_scan_benchmark sparse_s=",sparseTime," dense_s=",denseTime,
       " speedup=",sparseTime/max(1e-9,denseTime)
  quit(0)

if "--fusion-test" in commandLineParams():
  resetFusion()
  let rising = @[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
  let falling = @[0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
  doAssert fusionRepeatSignal(rising, falling).signal == 0.0
  doAssert fusionRepeatSignal(@[0.1,0.1,0.1,0.1,0.1,0.1], @[0.2,0.2,0.2,0.2,0.2,0.2]).signal == 0.0
  doAssert fusionRepeatSignal(@[0.1,0.2], @[0.1,0.2]).signal == 0.0
  fusionUpdateTrust(rising, rising)
  doAssert fusionReliability > 0 and fusionReliability < 0.07
  for _ in 0..<20: fusionUpdateTrust(rising, rising)
  doAssert fusionReliability > 0.9
  fusionUpdateTrust(rising, falling)
  doAssert fusionReliability < 0.5
  resetFusion()
  let ps = @[FusionPoint(id:0,g:0.94,j:0.55), FusionPoint(id:1,g:0.89,j:0.82),
    FusionPoint(id:2,g:0.91,j:0.70), FusionPoint(id:3,g:0.80,j:0.40)]
  let front = fusionFront(ps, 16)
  doAssert front.len == 3 and 0 in front and 1 in front and 2 in front
  let ends = fusionFront(ps, 2)
  doAssert ends.len == 2 and 0 in ends and 1 in ends
  doAssert fusionFront(@[], 2).len == 0
  doAssert fusionFront(ps, 0).len == 0
  doAssert fusionFront(ps, 1) == @[0]
  doAssert fusionFront(@[ps[0], ps[0]], 16).len == 1
  let oldAAA = AAA
  AAA = 1
  let g = @[Rule(a: @[97], b: @[98], weight: 1.0,
    embedding: defaultEmbedding(), patternRevision: freshPatternRevision(),
    replacementRevision: freshReplacementRevision())]
  var changed = cloneGenome(g)
  changed[0].weight = 2.0
  fusionArchive = @[fusionSnapshot(g,0.7,4)]
  fusionAnchors = @[fusionSnapshot(g,0.7,4)]
  doAssert fusionMatch(changed, fusionArchive) == -1
  fusionPending = @[FusionPending(genome:cloneGenome(changed),due:9)]
  var next: seq[Genome] = @[]
  fusionInject(next, 5, 450)
  doAssert next.len == 1 and sameGenomeContent(next[0], changed)
  next.setLen(0)
  fusionInject(next, FUSION_INTERVAL - 1, 450)
  doAssert next.len == 1 and sameGenomeContent(next[0], g) # calibration; pending expired
  next.setLen(0)
  fusionInject(next, 10, 450)
  doAssert next.len == 0 # expired; zero trust archive cannot breed/survive
  fusionReliability = 0.6
  fusionEvidence = 8
  fusionNoise = 0.03
  let stream = newStringStream()
  writeFusion(stream)
  stream.write(12345'i64)
  stream.setPosition(0)
  resetFusion()
  readFusion(stream,24,5)
  doAssert abs(fusionReliability-0.6) < 1e-12 and fusionEvidence == 8
  doAssert fusionArchive.len == 1 and fusionAnchors.len == 1 and fusionPending.len == 1
  doAssert fusionPending[0].due == 9 and sameGenomeContent(fusionPending[0].genome,changed)
  doAssert stream.readInt64() == 12345'i64
  stream.close()
  let legacy = newStringStream()
  legacy.write(54321'i64)
  legacy.setPosition(0)
  readFusion(legacy,23,5)
  doAssert fusionReliability == 0.0 and fusionArchive.len == 0
  doAssert legacy.readInt64() == 54321'i64
  legacy.close()
  let invalid = newStringStream()
  invalid.write(2.0)
  invalid.write(0'i64)
  invalid.write(0.0)
  invalid.setPosition(0)
  var rejected = false
  try: readFusion(invalid,24,5)
  except IOError: rejected = true
  doAssert rejected
  invalid.close()
  resetFusion()
  AAA = oldAAA
  echo "PASS: adaptive Jev trust, Pareto extremes, nursery expiry, genome identity, v24 roundtrip and v23 migration"
  quit(0)

if "--regression-test" in commandLineParams():
  doAssert INPUT_BYTE_COUNT == 256
  doAssert EMBEDDING_ENTRY_COUNT == 256
  doAssert EMBEDDING_SIZE == 512
  doAssert EMBEDDING_CODE_COUNT == 512
  doAssert EMBEDDING_OOV == 512
  doAssert INDEX_TOKEN_COUNT == 513
  proc testRule(a, b: seq[int], w: float): Rule =
    Rule(a: a, b: b, weight: w,
      patternRevision: freshPatternRevision(),
      replacementRevision: freshReplacementRevision())
  proc compileTest(g: Genome): seq[SeqPattern] =
    for r in g: result.add(compileSeqPattern(r.a))
  let middleCapture = replaceSeq(@[97,120,120,98], @[97,-1,98], @[-1])
  let leadingCapture = replaceSeq(@[122,122,97,113], @[-1,97], @[-1])
  echo "middle wildcard: expected=@[120, 120] actual=", middleCapture
  echo "leading wildcard: expected=@[122, 122, 113] actual=", leadingCapture
  doAssert replaceSeq(@[97,49,98,50,99], @[97,-1,98,-2,99], @[-2,-1]) == @[50,49]
  doAssert replaceSeq(@[97,49,50], @[97,-1], @[-1]) == @[49,50]
  doAssert replaceSeq(@[49,50,97], @[-1,-2,97], @[-1,99,-2]) == @[99,49,50]
  doAssert replaceSeq(@[97,49,98,97,50,98], @[97,-1,98], @[-1]) == @[49,50]
  # Boundary regression: 1,500 bytes must remain exactly 1,500 tokens.
  block:
    let original = repeat(65, 3100)
    let mapping = defaultEmbedding()
    var encoded: seq[int] = @[]
    embedBytesInto(original, mapping, encoded)
    doAssert encoded.len == 3100 and encoded[1499] == 65
    var edgeInput = repeat(1, 3100)
    edgeInput[0] = 0
    let edgePattern = compileSeqPattern(@[0, -1])
    var edgeScratch = ReplaceScratch(sortBuf: @[])
    var edgePlan: ReplacementPlan
    var edgeOutput: seq[int] = @[]
    var edgeOverflow = false
    compileReplacementPlan(@[-1, 0], edgePlan)
    doAssert replaceSeqCompiledInto(edgeInput, edgePattern, @[-1, 0],
      edgeOutput, edgeScratch, addr edgePlan, addr edgeOverflow, 3100)
    doAssert not edgeOverflow and edgeOutput.len == 3100
    doAssert edgeOutput[0] == 1 and edgeOutput[^1] == 0
    compileReplacementPlan(@[-1, 0, 0], edgePlan)
    doAssert not replaceSeqCompiledInto(edgeInput, edgePattern, @[-1, 0, 0],
      edgeOutput, edgeScratch, addr edgePlan, addr edgeOverflow, 3100)
    doAssert edgeOverflow and edgeOutput.len == 0
    var embeddedGenome = @[testRule(@[65], @[65], 1.0)]
    embeddedGenome[0].embedding = mapping
    let embeddedCompiled = compileTest(embeddedGenome)
    let embeddedIndex = buildCandidateIndex(embeddedCompiled)
    var embeddedScratch = newEvalScratch(embeddedGenome.len)
    doAssert scoreRaw(embeddedGenome, embeddedCompiled, embeddedIndex,
      original, embeddedScratch, 1) == 1.0
    echo "PASS: 3100-byte / 3100-token embedding and exact output boundary"
    # An invalid capture must be rejected before arithmetic/sort/reverse/copy
    # accesses input[3100], or before the unchecked bulk copy corrupts memory.
    for opcode in [-1, -16, -32, -48, -64, -80, -96]:
      var badPlan: ReplacementPlan
      compileReplacementPlan(@[opcode], badPlan)
      var badCaptures = [Capture(start: 3100, len: 1)]
      var badOutput: seq[int] = @[]
      var badOverflow = false
      var rejected = false
      try:
        appendReplacementPlan(badOutput, addr badPlan, edgeInput,
          badCaptures, 1, edgeScratch, badOverflow, 3100)
      except SeqReplaceError:
        rejected = true
      doAssert rejected
    var copyRejected = false
    try:
      var invalidCopy: seq[int] = @[]
      appendIntsBulk(invalidCopy, edgeInput, 3100, 1)
    except SeqReplaceError:
      copyRejected = true
    doAssert copyRejected
    echo "PASS: invalid end-of-buffer captures/copies rejected before access"
  # Element-wise arithmetic on a captured sequence; existing opcodes unchanged.
  let arithmeticInput = @[97, 2, 4, 8, 98]
  let arithmeticPattern = @[97, -1, 98]
  doAssert replaceSeq(arithmeticInput, arithmeticPattern, @[-48]) == @[3, 5, 9]
  doAssert replaceSeq(arithmeticInput, arithmeticPattern, @[-64]) == @[1, 3, 7]
  doAssert replaceSeq(arithmeticInput, arithmeticPattern, @[-80]) == @[4, 8, 16]
  doAssert replaceSeq(arithmeticInput, arithmeticPattern, @[-96]) == @[1, 2, 4]
  doAssert replaceSeq(@[97, 1, 98, 6, 99], @[97, -1, 98, -2, 99],
    @[-49, -48]) == @[7, 2] # +1($2), +1($1)
  doAssert replaceSeq(arithmeticInput, arithmeticPattern,
    @[-16, -32, -48, -64, -80, -96]) ==
      @[2, 4, 8, 8, 4, 2, 3, 5, 9, 1, 3, 7, 4, 8, 16, 1, 2, 4]
  doAssert replaceSeq(@[97, -3, -2, -1, 98], arithmeticPattern,
    @[-96]) == @[-2, -1, -1] # True floor division on negative tokens
  # Internal arithmetic lives in Z/512Z. 256..511 are ordinary latent
  # symbols; only 512 is OOV and must pass through unchanged.
  doAssert replaceSeq(@[97, 0, 511, 256, EMBEDDING_OOV, 98], arithmeticPattern,
    @[-48]) == @[1, 0, 257, EMBEDDING_OOV]
  doAssert replaceSeq(@[97, 0, 511, 256, EMBEDDING_OOV, 98], arithmeticPattern,
    @[-64]) == @[511, 510, 255, EMBEDDING_OOV]
  doAssert replaceSeq(@[97, 0, 511, 256, EMBEDDING_OOV, 98], arithmeticPattern,
    @[-80]) == @[0, 510, 0, EMBEDDING_OOV]
  doAssert replaceSeq(@[97, 0, 511, 256, EMBEDDING_OOV, 98], arithmeticPattern,
    @[-96]) == @[0, 255, 128, EMBEDDING_OOV]
  var allInternalTokens: seq[int] = @[2000]
  for token in 0 ..< EMBEDDING_CODE_COUNT: allInternalTokens.add(token)
  allInternalTokens.add(2001)
  let internalPattern = @[2000, -1, 2001]
  let plusResult = replaceSeq(allInternalTokens, internalPattern, @[-48])
  let minusResult = replaceSeq(allInternalTokens, internalPattern, @[-64])
  let twiceResult = replaceSeq(allInternalTokens, internalPattern, @[-80])
  doAssert plusResult.len == EMBEDDING_CODE_COUNT
  doAssert minusResult.len == EMBEDDING_CODE_COUNT
  doAssert twiceResult.len == EMBEDDING_CODE_COUNT
  for token in 0 ..< EMBEDDING_CODE_COUNT:
    doAssert plusResult[token] == (token + 1) mod EMBEDDING_CODE_COUNT
    doAssert minusResult[token] == (token + EMBEDDING_CODE_COUNT - 1) mod EMBEDDING_CODE_COUNT
    doAssert twiceResult[token] == (token * 2) mod EMBEDDING_CODE_COUNT
  # Corrupt/out-of-domain inputs never overflow or impersonate an OOV token.
  doAssert replaceSeq(@[97, high(int), low(int), 98], arithmeticPattern,
    @[-48, -64, -80]) ==
      @[high(int), low(int), high(int), low(int),
        high(int), low(int)]
  echo "PASS: modulo-512 capture arithmetic, latent tokens and OOV safety"
  # Log-uniform event sizes are sampled from the actual genome length,
  # including 1 and N; distinct mutation targets cannot silently collapse.
  doAssert sampleLogUniformMutationCount(0) == 0
  doAssert sampleLogUniformMutationCount(1) == 1
  randomize(72631)
  var smallCount = 0
  var largeCount = 0
  for _ in 0 ..< 20000:
    let count = sampleLogUniformMutationCount(AAA)
    doAssert count >= 1 and count <= AAA
    if count <= 2: inc smallCount
    if count >= AAA div 2: inc largeCount
  doAssert smallCount > 1000 and largeCount > 1000
  var targetGenome = newSeq[Rule](AAA)
  for trial in 0 ..< 100:
    let positions = pickMutationTargets(targetGenome)
    let uniquePositions = toHashSet(positions)
    doAssert positions.len >= 1 and positions.len <= AAA
    doAssert uniquePositions.len == positions.len
  # Every count-bearing rule mutation uses this unrestricted log-uniform law.
  var localSizedDraws = 0
  var macroSizedDraws = 0
  for trial in 0 ..< 1000:
    let positions = pickMutationTargets(targetGenome)
    doAssert toHashSet(positions).len == positions.len
    if positions.len <= 3: inc localSizedDraws
    if positions.len >= 100: inc macroSizedDraws
  doAssert localSizedDraws > 0 and macroSizedDraws > 0
  # Compare the sparse/dense target sampler with the original full-array
  # partial Fisher-Yates on the SAME RNG stream, including the guided first
  # choice. Target ordering and the log-uniform count must both be identical.
  var sparseCases = 0
  for trial in 0 ..< 128:
    let seed = 94721 + trial
    randomize(seed)
    let expectedCount = sampleLogUniformMutationCount(targetGenome.len)
    let first = pickGuidedMutationPos(targetGenome)
    var expected = @[first]
    if expectedCount > 1:
      var available = newSeq[int](targetGenome.len)
      for i in 0 ..< targetGenome.len:
        available[i] = i
      swap(available[first], available[targetGenome.len - 1])
      for i in 1 ..< expectedCount:
        let remaining = targetGenome.len - i
        let j = rand(remaining - 1)
        expected.add(available[j])
        swap(available[j], available[remaining - 1])
    randomize(seed)
    let observed = pickMutationTargets(targetGenome)
    doAssert observed == expected
    if observed.len > 1 and observed.len <= targetGenome.len div 8:
      inc sparseCases
  doAssert sparseCases > 0
  # Similarity must be bitwise-equivalent for empty/equal/unequal sequences.
  block:
    doAssert sequenceSimilarity(newSeq[int](0), newSeq[int](0)) == 1.0
    doAssert sequenceSimilarity(@[1, 2], @[1, 3]) == 0.5
    doAssert sequenceSimilarity(@[1, 2], @[1, 2, 3, 4]) == 0.5
    doAssert sequenceSimilarity(@[1], newSeq[int](0)) == 0.0
  echo "PASS: single-pass similarity retains empty/equal/mismatch semantics"
  # Leading wildcard rules must not be treated as unconditional, and
  # searching a later literal must never filter out a possible match.
  block:
    let patterns = @[
      compileSeqPattern(@[-1, 97, 98]),
      compileSeqPattern(@[-1, 120]),
      compileSeqPattern(@[-1]),
      compileSeqPattern(@[97, -1, 98]),
      compileSeqPattern(@[97, 98])
    ]
    let indexed = buildCandidateIndex(patterns)
    var scratch = newEvalScratch(patterns.len)
    scanCandidateIds(indexed, @[55, 97, 98, 77],
      scratch.candidateIds, scratch.candidateScratch)
    doAssert 0 in scratch.candidateIds
    doAssert 1 notin scratch.candidateIds
    doAssert 2 notin scratch.candidateIds
    doAssert 3 in scratch.candidateIds
    doAssert 4 in scratch.candidateIds
    scanCandidateIds(indexed, @[55, 120, 77],
      scratch.candidateIds, scratch.candidateScratch)
    doAssert 1 in scratch.candidateIds
    doAssert 0 notin scratch.candidateIds
  echo "PASS: later literal anchored wildcard indexing preserves candidates"
  block:
    # Sparse index must handle latent IDs near the top of the 512-token space
    # without allocating a 513^2 dense table.
    let latentPatterns = @[
      compileSeqPattern(@[500]),
      compileSeqPattern(@[500, 511]),
      compileSeqPattern(@[511, 500]),
      compileSeqPattern(@[EMBEDDING_OOV])
    ]
    let latentIndex = buildCandidateIndex(latentPatterns)
    var latentScratch = newEvalScratch(latentPatterns.len)
    scanCandidateIds(latentIndex, @[500, 511], latentScratch.candidateIds,
      latentScratch.candidateScratch)
    doAssert 0 in latentScratch.candidateIds
    doAssert 1 in latentScratch.candidateIds
    doAssert 2 notin latentScratch.candidateIds
    doAssert latentIndex.head2.count <= latentPatterns.len
    scanCandidateIds(latentIndex, @[EMBEDDING_OOV], latentScratch.candidateIds,
      latentScratch.candidateScratch)
    doAssert 3 in latentScratch.candidateIds
  echo "PASS: sparse candidate index covers 512 internal tokens plus OOV"
  block:
    let literal = makeLiteral(@[97, 97])
    var cursor = LiteralCursor(lastMatch: -1, nextSearch: 0, exhausted: false)
    doAssert nextOccurrenceLazy(@[97, 97, 97], literal, cursor, 0) == 0
    doAssert nextOccurrenceLazy(@[97, 97, 97], literal, cursor, 1) == 1
    # The consuming replacement sweep still advances by the full match length.
    doAssert replaceSeq(@[97, 97, 97], @[97, 97], @[98]) == @[98, 97]
  echo "PASS: overlapping occurrence lookup without overlapping replacements"
  block:
    var rounds = FAST_EVAL_MAX_ROUNDS
    rounds = max(FAST_EVAL_MIN_ROUNDS, rounds - 1)
    doAssert rounds == FAST_EVAL_MAX_ROUNDS - 1
    rounds = min(FAST_EVAL_MAX_ROUNDS, rounds + 2)
    doAssert rounds == FAST_EVAL_MAX_ROUNDS
    rounds = FAST_EVAL_MIN_ROUNDS
    rounds = max(FAST_EVAL_MIN_ROUNDS, rounds - 1)
    doAssert rounds == FAST_EVAL_MIN_ROUNDS
    # This test doesn't require loading a corpus or initializing global RNG.
    doAssert corpusByteCount.len == EMBEDDING_ENTRY_COUNT
    doAssert corpusByteCum.len == EMBEDDING_ENTRY_COUNT
  echo "PASS: FAST controller bounds and sentinel-free byte sampling domain"
  echo "PASS: log-uniform sparse selection exactly matches legacy Fisher-Yates"
  echo "PASS: unrestricted sampler preserved for macro mutation exploration"
  # Invalid wildcard candidates must not allocate/copy the unmatched input;
  # growth beyond the evaluator cap must stop BEFORE expanding the buffer.
  block:
    var replacementScratch = ReplaceScratch(sortBuf: @[])
    var plan: ReplacementPlan
    var replacementOutput = @[123]
    var overflow = false
    let wildcardPattern = compileSeqPattern(@[97, -1, 98])
    compileReplacementPlan(@[-1], plan)
    doAssert not replaceSeqCompiledInto(repeat(99, 32), wildcardPattern,
      @[-1], replacementOutput, replacementScratch, addr plan, addr overflow, 4)
    doAssert not overflow
    let literalPattern = compileSeqPattern(@[97])
    compileReplacementPlan(@[97, 98], plan)
    doAssert not replaceSeqCompiledInto(@[97, 97], literalPattern,
      @[97, 98], replacementOutput, replacementScratch, addr plan, addr overflow, 3)
    doAssert overflow
    overflow = false
    # A literal pattern has no capture: the opcode should be skipped, not
    # emitted as a negative text token. Same for a reverse opcode.
    compileReplacementPlan(@[-1, 98], plan)
    doAssert replaceSeqCompiledInto(@[97], literalPattern,
      @[-1, 98], replacementOutput, replacementScratch, addr plan, addr overflow, 3)
    doAssert not overflow and replacementOutput == @[98]
    compileReplacementPlan(@[-32, 99], plan)
    doAssert replaceSeqCompiledInto(@[97], literalPattern,
      @[-32, 99], replacementOutput, replacementScratch, addr plan, addr overflow, 3)
    doAssert not overflow and replacementOutput == @[99]
    compileReplacementPlan(@[-48, -64, -80, -96, 99], plan)
    doAssert replaceSeqCompiledInto(@[97], literalPattern,
      @[-48, -64, -80, -96, 99], replacementOutput,
      replacementScratch, addr plan, addr overflow, 3)
    doAssert not overflow and replacementOutput == @[99]
    # An arithmetic reference cannot bypass the output cap.
    let arithmeticPat = compileSeqPattern(@[97, -1, 98])
    compileReplacementPlan(@[-48], plan)
    overflow = false
    doAssert not replaceSeqCompiledInto(@[97, 1, 2, 3, 98],
      arithmeticPat, @[-48], replacementOutput, replacementScratch,
      addr plan, addr overflow, 2)
    doAssert overflow and replacementOutput.len == 0
    overflow = false
    echo "PASS: unmatched wildcard, output cap and literal opcode handling"
    # KMP must retain leftmost/non-overlapping matches after replacing the
    # duplicate naive scan in the ordinary literal fast path.
    var longPattern = repeat(97, 60)
    longPattern.add(98)
    var longInput = repeat(97, 256)
    longInput.add(98)
    longInput.add(repeat(97, 60))
    longInput.add(98)
    var longExpected = repeat(97, 196)
    longExpected.add(99)
    longExpected.add(99)
    doAssert replaceSeq(longInput, longPattern, @[99]) == longExpected
    doAssert replaceSeq(repeat(97, 512), longPattern, @[99]).len == 0
    echo "PASS: long literal KMP retains leftmost non-overlapping semantics"

  proc referenceRaw(g: Genome, input: seq[int], rounds: int): float =
    let compiled = compileTest(g)
    var state = cloneInts(input)
    var scratch = newEvalScratch(g.len)
    let limit = min(32768, max(3100, input.len * 32))
    let nRounds = min(rounds, max(1, int(ceil(2 * sqrt(float(input.len))))))
    var attPairs: seq[(float, float)] = @[]
    var prev = 0.0
    var havePrev = false
    for j in 0 ..< nRounds:
      var subtotal = 0.0
      var changedAny = false
      var terminated = false
      for k in 0 ..< g.len:
        if replaceSeqCompiledInto(state, compiled[k], g[k].b,
            scratch.bufB, scratch.replaceScratch,
            getReplacementPlanPtr(g, k, scratch)):
          if scratch.bufB.len > limit:
            subtotal -= 1.0
            terminated = true
            break
          changedAny = true
          subtotal += g[k].weight
          swap(state, scratch.bufB)
      if havePrev:
        if (prev, subtotal) in attPairs: break
        attPairs.add((prev, subtotal))
      prev = subtotal
      havePrev = true
      result += subtotal
      if terminated or not changedAny: break
  proc actualRaw(g: Genome, input: seq[int], rounds: int): float =
    let compiled = compileTest(g)
    let index = buildCandidateIndex(compiled)
    var scratch = newEvalScratch(g.len)
    prepareScoreScratch(g, scratch)
    scoreRaw(g, compiled, index, input, scratch, rounds)

  # Independent ordered reference: random genomes with no alternate paths.
  randomize(81251)
  for trial in 0 ..< 128:
    var testGenome: Genome = @[]
    for i in 0 ..< 2 + rand(1):
      testGenome.add(testRule(@[97 + rand(1)], @[97 + rand(1)],
        float(rand(6) - 3)))
    var text: seq[int] = @[]
    for i in 0 ..< 4: text.add(97 + rand(1))
    doAssert abs(actualRaw(testGenome, text, SCORE_ROUNDS) -
      referenceRaw(testGenome, text, SCORE_ROUNDS)) <= 1.0e-10
  # Exercise arbitrary internal anchors, wildcard-leading and suffix parts,
  # and lazy replacement plans against the independent full-rule evaluator.
  randomize(82341)
  for trial in 0 ..< 192:
    var g: Genome = @[]
    for r in 0 ..< 2 + rand(2):
      var a = @[97 + rand(2)]
      if rand(1) == 1:
        a.insert(-1, 0)
      if rand(1) == 1:
        a.add(-1)
        a.add(97 + rand(2))
        if rand(1) == 1:
          a.add(97 + rand(2))
      var b = @[97 + rand(2)]
      if rand(2) == 0:
        b = @[-1]
      g.add(testRule(a, b, float(rand(6) - 3)))
    var x: seq[int] = @[]
    for i in 0 ..< 1 + rand(19): x.add(97 + rand(2))
    doAssert abs(actualRaw(g, x, SCORE_ROUNDS) -
      referenceRaw(g, x, SCORE_ROUNDS)) <= 1.0e-10
  echo "PASS: 192 wildcard/internal-anchor cases match independent full scan"
  echo "PASS: 128 random deterministic trajectories match independent reference"
  # Reusing a worker's input/output seqs must not leak the previous sample,
  # even after many expand/shrink rewrites with changing input lengths.
  block:
    let reuseGenome = @[
      testRule(@[97], @[98, 97], 1.0),
      testRule(@[98], @[99], -0.5),
      testRule(@[99], @[97], 0.75)
    ]
    let reuseCompiled = compileTest(reuseGenome)
    let reuseIndex = buildCandidateIndex(reuseCompiled)
    var reuseScratch = newEvalScratch(reuseGenome.len)
    randomize(81927)
    for trial in 0 ..< 250:
      var sample = newSeq[int](1 + rand(63))
      for i in 0 ..< sample.len:
        sample[i] = 97 + rand(2)
      let expected = referenceRaw(reuseGenome, sample, 4)
      let observed = scoreRaw(reuseGenome, reuseCompiled, reuseIndex,
        sample, reuseScratch, 4)
      doAssert abs(expected - observed) <= 1.0e-10
    echo "PASS: reused input buffers match independent reference across 250 samples"

  let competing = @[testRule(@[97], @[98], 1.0), testRule(@[97], @[99], 10.0)]
  let chain = @[testRule(@[97], @[98], 1.0), testRule(@[98], @[99], 2.0)]
  echo "competing rules: expected=1 actual=", actualRaw(competing, @[97], 1)
  echo "same-round chain: expected=3 actual=", actualRaw(chain, @[97], 1)
  # Input length one has ceil(2*sqrt(len)) = 2; each sweep scores 3.
  let cycling = @[testRule(@[97], @[98], 1.0), testRule(@[98], @[97], 2.0)]
  doAssert actualRaw(cycling, @[97], 37) == 6.0
  doAssert actualRaw(cycling, @[97], 1) == 3.0
  echo "PASS: ordered single-sweep override and sqrt-length temporal ceiling"

  # Temporal scheduling and cyclic 2-gram termination on one trajectory.
  doAssert actualRaw(cycling, repeat(97, 4), 37) == 6.0
  doAssert actualRaw(cycling, repeat(97, 16), 37) == 6.0
  doAssert actualRaw(cycling, repeat(97, 16), 1) == 3.0
  var fixed = @[testRule(@[97], @[97], 5.0)]
  doAssert actualRaw(fixed, repeat(97, 16), SCORE_ROUNDS) == 10.0
  let once = @[testRule(@[97], @[98], 5.0)]
  doAssert actualRaw(once, repeat(97, 16), SCORE_ROUNDS) == 5.0
  block:
    let cf = compileTest(fixed)
    let ix = buildCandidateIndex(cf)
    var sc = newEvalScratch(fixed.len)
    sc.creditContrib.setLen(fixed.len)
    inc sc.creditEpoch
    var feats: seq[float]
    doAssert scoreRaw(fixed, cf, ix, repeat(97, 16), sc, SCORE_ROUNDS,
      true, nil, addr feats) == 10.0
    doAssert feats == @[2.0]
    doAssert sc.creditContrib[0] == 10.0
    doAssert sc.creditTouched == @[0]
  echo "PASS: rounds, sqrt ceiling, 2-gram discard and rule credit"

  block:
    var huge = @[testRule(@[97], @[97], 4.0),
      testRule(@[97, -1, 98], repeat(-1, 64), 5.0),
      testRule(@[97], @[97], 7.0)]
    var text = @[97]
    text.add(repeat(99, 16382))
    text.add(98)
    let hc = compileTest(huge)
    let hi = buildCandidateIndex(hc)
    var hs = newEvalScratch(huge.len)
    var overflowed = false
    let value = scoreRaw(huge, hc, hi, text, hs, 1, false,
      addr overflowed)
    doAssert overflowed and value == 3.0 # 4 - 1; no forked original continuation
    echo "PASS: overflow penalized once and terminates single trajectory sweep"

  block:
    let g = @[testRule(@[97], @[97], 2.0),
      testRule(@[97], @[98], 3.0), testRule(@[98], @[99], -4.0)]
    let gc = compileTest(g)
    let gi = buildCandidateIndex(gc)
    var gs = newEvalScratch(g.len)
    var coefs: seq[float]
    let base = scoreRaw(g, gc, gi, @[97], gs, 1, false, nil, addr coefs)
    doAssert base == 1.0 and coefs == @[1.0, 1.0, 1.0]
    var changed = cloneGenome(g)
    changed[0].weight += 0.4
    changed[1].weight -= 0.2
    doAssert abs(scoreRaw(changed, gc, gi, @[97], gs, 1) -
      (base + 0.4 - 0.2)) < 1.0e-10
    let sampleX = @[@[@[97], @[98], @[97,97]]]
    let sampleY = @[@[3,2,1]]
    let ranks = buildRankedY(sampleY)
    let lc = buildLinearWeightCache(g, gc, gi, sampleX, sampleY)
    doAssert abs(scoreLinearProposal(lc, changed, ranks) -
      evaluateIndividual(changed, gc, gi, unsafeAddr sampleX,
        unsafeAddr sampleY, unsafeAddr ranks, changed.len, 1)) < 1.0e-10
    echo "PASS: one-round weight feature cache matches exact evaluator"

  # The first 32,768 rules do not match; the final rule MUST still be reached.
  # The candidate bitmap is built once for this unchanged state.
  var longSweep: Genome = @[]
  for i in 0 ..< 32768:
    longSweep.add(testRule(@[98], @[99], 1.0))
  longSweep.add(testRule(@[97], @[97], 7.0))
  doAssert actualRaw(longSweep, @[97], 1) == 7.0
  echo "PASS: rule visit 32,769 is evaluated (within normal work budget)"
  # Thousands of misses must not stop an ordered rule chain.
  var longChain: Genome = @[]
  for i in 0 ..< 3099:
    longChain.add(testRule(@[2000 + i], @[2001 + i], 1.0))
  longChain.add(testRule(@[2000 + 3099], @[2000 + 3100], 1.0))
  doAssert longChain.len == 3100
  doAssert actualRaw(longChain, @[2000], 1) == 3100.0
  echo "PASS: 3100 ordered matches without recursive state expansion"
  # Force sparse sorted candidate iteration, a miss stretch, then state mutation
  # followed by a NEW candidate scan. The later pattern was absent initially.
  var sparseGenome: Genome = @[]
  for i in 0 ..< 1497:
    sparseGenome.add(testRule(@[5000 + i], @[6000 + i], 0.0))
  sparseGenome.add(testRule(@[97], @[98], 1.0))
  sparseGenome.add(testRule(@[98], @[99], 2.0))
  sparseGenome.add(testRule(@[99], @[100], 4.0))
  doAssert actualRaw(sparseGenome, @[97], 1) == 7.0
  doAssert actualRaw(sparseGenome, @[100], 1) == 0.0
  echo "PASS: sparse evaluator skips misses and rescans after each ordered rewrite"

  var mismatches = 0
  randomize(415927)
  for trial in 0 ..< 1000:
    var g: Genome = @[]
    for k in 0 ..< 1 + rand(12):
      var a, b: seq[int]
      for i in 0 .. rand(2): a.add(97 + rand(2))
      for i in 0 .. rand(2): b.add(97 + rand(2))
      if trial mod 4 == 0 and k mod 3 == 0:
        a = @[97, -1, 98]
        b = @[-1, 99]
      g.add(testRule(a, b, float(rand(8) - 4)))
    var input: seq[int]
    for i in 0 ..< 2 + rand(12): input.add(97 + rand(2))
    let rounds = 1 + rand(5)
    if abs(actualRaw(g, input, rounds) - referenceRaw(g, input, rounds)) > 1.0e-10:
      inc mismatches
  doAssert mismatches == 0
  echo "PASS: 1000 random ordered evaluations match independent reference"

  # Credit and ordinary scoring must agree on the same rewrite sequence.
  let creditGenome = @[testRule(@[97,-1,98], @[-1], 2.0),
    testRule(@[120], @[121], -1.0), testRule(@[121], @[122], 3.0)]
  let cx = @[@[@[97,120,98], @[97,120,120,98], @[120], @[97,98], @[122]]]
  let cy = @[@[5,4,3,2,1]]
  let cyRank = buildRankedY(cy)
  let cc = compileTest(creditGenome)
  let ci = buildCandidateIndex(cc)
  var credit: seq[float]
  var refValues: seq[float]
  for x in cx[0]: refValues.add(referenceRaw(creditGenome, x, SCORE_ROUNDS))
  var rankScratch = newEvalScratch(creditGenome.len)
  let expectedCreditScore = spearmanWithRankedY(refValues, cyRank[0], rankScratch)
  let creditScore = evaluateIndividualWithCredit(creditGenome, cc, ci,
    cx, cy, cyRank, creditGenome.len, credit)
  doAssert abs(creditScore - expectedCreditScore) < 1.0e-12
  echo "credit evaluator agrees with ordered reference"

  # Two-point means an actual [left,right) cut on the ordered RULE SEQUENCE.
  # The unchanged prefix/suffix must be inherited from p1 exactly.
  var crossA: Genome = @[]
  var crossB: Genome = @[]
  for i in 0 ..< 12:
    var ruleA = testRule(@[100 + i], @[200 + i], float(i + 1))
    var ruleB = testRule(@[300 + i], @[400 + i], float(i + 11))
    crossA.add(ruleA)
    crossB.add(ruleB)
  let cutChild = twoPointGenomeCrossover(crossA, crossB, 3, 9)
  for i in 0 ..< 12:
    if i >= 3 and i < 9:
      doAssert sameGenomeContent(@[cutChild[i]], @[crossB[i]])
    else:
      doAssert sameGenomeContent(@[cutChild[i]], @[crossA[i]])
  doAssert cutChild.len == crossA.len
  echo "PASS: rule-sequence crossover preserves donor and outside segments"
  # Exact a/b permits sign-preserving homologous numeric blending.
  let bitA = testRule(@[97, -1], @[-1, 98], 1.5)
  let bitB = testRule(@[97, -1], @[-1, 98], 2.5)
  randomize(59371)
  for trial in 0 ..< 512:
    let offspring = recombineHomologousRule(bitA, bitB)
    doAssert offspring.a == bitA.a and offspring.b == bitA.b
    doAssert offspring.weight >= bitA.weight and offspring.weight <= bitB.weight
  echo "PASS: homologous numeric weight blending without branch alleles"
  # Mixed whole-rule crossover never creates invalid capture references.
  var invalidDonor = bitB
  invalidDonor.b = @[-15]
  doAssert not compatibleRule(invalidDonor)
  let safeChild = twoPointGenomeCrossover(@[bitA], @[invalidDonor], 0, 1)
  doAssert sameGenomeContent(safeChild, @[bitA])
  echo "PASS: whole-rule crossover rejects invalid donor references"
  # Verify the actual mutation path uses the unrestricted sampler too.
  randomize(59372)
  var nLocal = 0
  var nMacro = 0
  for trial in 0 ..< 2000:
    let positions = pickMutationTargets(targetGenome)
    doAssert toHashSet(positions).len == positions.len
    if positions.len <= 8: inc nLocal
    if positions.len >= 100: inc nMacro
  doAssert nLocal > 200 and nLocal < 1400 and nMacro > 400
  echo "PASS: all mutation target counts follow the unrestricted logarithmic law"
  # Replacement mutation can emit an invalid reference, which must be fixed
  # on the modified row rather than silently poisoning an offspring.
  var sanitationProbe = testRule(@[-1, 97], @[-15], 0.5)
  doAssert not compatibleRule(sanitationProbe)
  sanitizeReplacementRefs(sanitationProbe)
  doAssert compatibleRule(sanitationProbe) and sanitationProbe.b == @[-1]
  var extendedOpcode = testRule(@[-1, 97], @[-48, -64, -80, -96], 0.5)
  doAssert compatibleRule(extendedOpcode)
  var ref16 = testRule(@[-1, 97], @[-63, -79, -95, -111], 0.5)
  doAssert not compatibleRule(ref16)
  sanitizeReplacementRefs(ref16)
  doAssert compatibleRule(ref16) and ref16.b == @[-48, -64, -80, -96]
  var badOpcode = testRule(@[97], @[-112], 0.5)
  doAssert not compatibleRule(badOpcode)
  badOpcode.b = @[98]
  doAssert compatibleRule(badOpcode)
  doAssert not replacementRefsNeedSanitize(extendedOpcode)
  doAssert replacementRefsNeedSanitize(badOpcode) == false
  echo "PASS: replacement-reference sanitation and invalid-opcode rejection"
  # Exercise the complete mutation entrypoint repeatedly, not merely its
  # helper, including random b edits and preserved parent COW invariants.
  let mutationParent = cloneGenome(crossA)
  randomize(59373)
  for trial in 0 ..< 350:
    var mutated = cloneGenome(mutationParent)
    mutateGenome(mutated, 1.0, 0)
    doAssert mutated.len == mutationParent.len
    for rule in mutated:
      doAssert compatibleRule(rule)
  doAssert sameGenomeContent(crossA, mutationParent)
  echo "PASS: mutation never corrupts parent or capture references"
  # The four operators are exclusive, keep the genome length, and never mutate
  # the parents. Alignments cannot go backwards even on shifted homologs.
  var shifted = cloneGenome(crossB)
  shifted[4] = cloneRuleShallow(crossA[2])
  shifted[7] = cloneRuleShallow(crossA[5])
  let matched = alignOrderedRules(crossA, shifted, newSeq[RuleTraceRow](0), newSeq[RuleTraceRow](0))
  var prevA = -1
  var prevB = -1
  for pair in matched:
    doAssert pair.base > prevA and pair.donor > prevB
    prevA = pair.base
    prevB = pair.donor
  doAssert matched.len >= 2
  let crossAOriginal = cloneGenome(crossA)
  let crossBOriginal = cloneGenome(crossB)
  randomize(12345)
  for op in CrossoverOp:
    for trial in 0 ..< 25:
      let offspring = crossoverGenome(crossA, crossB, op,
        newSeq[RuleTraceRow](0), newSeq[RuleTraceRow](0))
      doAssert offspring.len == crossA.len
      for r in offspring:
        doAssert compatibleRule(r)
    echo "PASS: crossover operator ", $op
  doAssert sameGenomeContent(crossA, crossAOriginal)
  doAssert sameGenomeContent(crossB, crossBOriginal)
  echo "PASS: ordered homology and immutable crossover parents"

  var left: Genome = @[]
  for k in 0 ..< 32: left.add(testRule(@[97], @[98], 1.0))
  var right = cloneGenome(left)
  doAssert sameGenomeContent(left, right)
  echo "PASS: genomic identity includes pattern/replacement/weight and embedding"

  var many: Genome = @[]
  for k in 0 ..< 200: many.add(testRule(@[97], @[98], 1.0))
  for k in 0 ..< 200: many.add(testRule(@[97,97], @[98], 1.0))
  many.add(testRule(@[122], @[121], 1.0))
  let index = buildCandidateIndex(compileTest(many))
  var scratch = newEvalScratch(many.len)
  let repeated = repeat(97, 3100)
  let began = cpuTime()
  for i in 0 ..< 1000:
    scanCandidateIds(index, repeated, scratch.candidateIds, scratch.candidateScratch)
    doAssert scratch.candidateIds.len == 400
  echo "candidate scan 1000 x 3100 repeated bytes CPU seconds: ", cpuTime() - began
  scratch.candidateScratch.stamp = high(int32)
  scanCandidateIds(index, repeated, scratch.candidateIds, scratch.candidateScratch)
  doAssert scratch.candidateScratch.stamp == 1 and scratch.candidateIds.len == 400
  # A rewritten state is rescanned only for rules at/after the next row.
  # Membership and order must agree with full scanning filtered by that row.
  scanCandidateIds(index, repeated, scratch.candidateIds,
    scratch.candidateScratch)
  var originalSuffix: seq[int] = @[]
  for pid in scratch.candidateIds:
    if pid >= 250: originalSuffix.add(pid)
  originalSuffix.sort()
  scanCandidateIds(index, repeated, scratch.candidateIds,
    scratch.candidateScratch, 250)
  scratch.candidateIds.sort()
  doAssert scratch.candidateIds == originalSuffix
  echo "PASS: suffix-only candidate scan retains exactly the eligible rules"
  doAssert middleCapture == @[120,120]
  doAssert leadingCapture == @[122,122,113]
  doAssert mismatches == 0
  # A smaller population must not leave unread genomes in a checkpoint stream.
  let originalRuleCount = AAA
  AAA = 2
  var saved = newSeq[Genome](500)
  for i in 0 ..< saved.len:
    saved[i] = @[testRule(@[97], @[98], float(i)), testRule(@[98], @[99], 1.0)]
  let stream = newStringStream()
  writePopulation(stream, saved)
  stream.write(123456789'i64)
  stream.setPosition(0)
  let restored = readPopulation(stream, CHECKPOINT_VERSION)
  doAssert restored.len == pop_size
  doAssert restored[0][0].weight == 0.0 and
    restored[^1][0].weight == float(pop_size - 1)
  doAssert stream.readInt64() == 123456789'i64
  stream.close()
  # An actual 400-genome legacy stream must stay aligned before adding 50.
  let old400Stream = newStringStream()
  writePopulation(old400Stream, saved[0 ..< 400])
  old400Stream.write(4242424242'i64)
  old400Stream.setPosition(0)
  var expanded400 = readPopulation(old400Stream, CHECKPOINT_VERSION)
  doAssert expanded400.len == 400 and old400Stream.readInt64() == 4242424242'i64
  old400Stream.close()
  let survivorFirst = cloneGenome(expanded400[0])
  let survivorLast = cloneGenome(expanded400[399])
  expandLoadedPopulation(expanded400)
  doAssert expanded400.len == 450 and
    sameGenomeContent(expanded400[0], survivorFirst) and
    sameGenomeContent(expanded400[399], survivorLast)
  doAssert expanded400[400].len == AAA and expanded400[449].len == AAA
  AAA = originalRuleCount
  # v21 Jev Hall is appended BEFORE the unchanged checkpoint end marker.
  # v20 must consume no Hall bytes, leaving its original end marker aligned.
  block:
    AAA = 2
    jevHall = @[]
    jevHallConfig = "jev-regression-config"
    var specimen = cloneGenome(saved[0])
    specimen[0].embedding = defaultEmbedding()
    jevHallObserve(specimen, 0.25, 15, true)
    doAssert jevHall.len == 1 and jevHall[0].wins == 1
    jevHallObserve(specimen, 0.55, 15, true)
    doAssert jevHall[0].observations == 1 # same round never counts twice
    jevHallObserve(specimen, 0.35, 31, false)
    doAssert jevHall[0].wins == 1 and jevHall[0].observations == 2
    doAssert not confirmedJevHallMatch(specimen, genomeFingerprint(specimen))
    jevHallObserve(specimen, 0.45, 47, true)
    doAssert jevHall[0].wins == 2 and jevHall[0].observations == 3
    doAssert not confirmedJevHallMatch(specimen, genomeFingerprint(specimen))
    let hallStream = newStringStream()
    writeJevHall(hallStream)
    hallStream.write(0x4B4C4D56434B5036'i64)
    hallStream.setPosition(0)
    jevHall = @[]
    jevHallConfig = ""
    readJevHall(hallStream, 21, 48)
    doAssert jevHall.len == 1 and jevHall[0].wins == 2
    doAssert jevHall[0].observations == 3 and
      jevHallConfig == "jev-regression-config" and
      sameGenomeContent(specimen, jevHall[0].genome)
    doAssert hallStream.readInt64() == 0x4B4C4D56434B5036'i64
    hallStream.close()
    let legacyStream = newStringStream()
    legacyStream.write(0x4B4C4D56434B5036'i64)
    legacyStream.setPosition(0)
    readJevHall(legacyStream, 20, 48)
    doAssert jevHall.len == 0 and jevHallConfig.len == 0
    doAssert legacyStream.readInt64() == 0x4B4C4D56434B5036'i64
    legacyStream.close()
    # Complete synthetic v20 checkpoint, including a real 400-genome population
    # and one historical fitness point. The NEW reader must retain them exactly.
    let legacyPath = getTempDir() / ("at_jev_v20_fixture_" &
      $getCurrentProcessId() & ".bin")
    let legacyCheckpoint = newFileStream(legacyPath, fmWrite)
    doAssert not legacyCheckpoint.isNil
    legacyCheckpoint.write(CHECKPOINT_MAGIC)
    legacyCheckpoint.write(20'i64)
    legacyCheckpoint.write(200'i64) # next generation
    legacyCheckpoint.write(0.8)
    legacyCheckpoint.write(0'i64) # stagnation
    legacyCheckpoint.write(0'i64) # chunk cursor
    legacyCheckpoint.write(123'i64) # run seed
    for ci in 0 ..< MULTI_CASE_COUNT: legacyCheckpoint.write(0.0)
    legacyCheckpoint.write(0'i64) # MA ring length
    writeGraph(legacyCheckpoint, @[ProgressPoint(iter: 199,
      best: 0.8, transformed: spearmanToSigma(0.8))])
    var legacyPop = newSeq[Genome](400)
    for i in 0 ..< legacyPop.len: legacyPop[i] = cloneGenome(specimen)
    writePopulation(legacyCheckpoint, legacyPop)
    writeEliteArchive(legacyCheckpoint, newSeq[EliteArchiveEntry](0))
    writeDataset(legacyCheckpoint, newSeq[seq[seq[int]]](0),
      newSeq[seq[int]](0))
    legacyCheckpoint.write(0x4B4C4D56434B5036'i64)
    legacyCheckpoint.close()
    var restoredPop: seq[Genome] = @[]
    var restoredMa: seq[float] = @[]
    var restoredPlot: seq[ProgressPoint] = @[]
    var restoredElite: seq[EliteArchiveEntry] = @[]
    var restoredX: seq[seq[seq[int]]] = @[]
    var restoredY: seq[seq[int]] = @[]
    var restoredNext = 0
    var restoredBest = 0.0
    var restoredStagnation = 0
    var restoredCursor = 0
    var restoredSeed = 0'i64
    var restoredOffsets: array[MULTI_CASE_COUNT, float]
    doAssert loadCheckpoint(legacyPath, restoredNext, restoredPop,
      restoredBest, restoredStagnation, restoredMa, restoredPlot,
      restoredElite, restoredX, restoredY, restoredCursor,
      restoredSeed, restoredOffsets)
    doAssert loadedCheckpointVersion == 20 and restoredNext == 200 and
      restoredPop.len == 400 and restoredPlot.len == 1 and
      restoredPlot[0].best == 0.8 and jevHall.len == 0
    expandLoadedPopulation(restoredPop)
    doAssert restoredPop.len == 450 and restoredPop[0].len == AAA and
      restoredPop[399].len == AAA and restoredPop[449].len == AAA and
      restoredPlot.len == 1 and restoredPlot[0].best == 0.8
    removeFile(legacyPath)
    jevHall = @[]
    # Full v21 checkpoint with an old 400-genome population AND an embedded
    # Jev Hall: expansion must not shift Hall or discard ranking history.
    let oldV21Path = getTempDir() / ("at_jev_v21_pop400_fixture_" &
      $getCurrentProcessId() & ".bin")
    jevHallConfig = "compatible-old-jev-settings"
    jevHallObserve(specimen, 0.5, 15, true)
    let v21Stream = newFileStream(oldV21Path, fmWrite)
    doAssert not v21Stream.isNil
    v21Stream.write(CHECKPOINT_MAGIC)
    v21Stream.write(21'i64)
    v21Stream.write(200'i64)
    v21Stream.write(0.8)
    v21Stream.write(0'i64)
    v21Stream.write(0'i64)
    v21Stream.write(123'i64)
    for ci in 0 ..< MULTI_CASE_COUNT: v21Stream.write(0.0)
    v21Stream.write(0'i64)
    writeGraph(v21Stream, @[ProgressPoint(iter: 199,
      best: 0.8, transformed: spearmanToSigma(0.8))])
    writePopulation(v21Stream, legacyPop)
    writeEliteArchive(v21Stream, newSeq[EliteArchiveEntry](0))
    writeDataset(v21Stream, newSeq[seq[seq[int]]](0), newSeq[seq[int]](0))
    writeJevHall(v21Stream)
    v21Stream.write(0x4B4C4D56434B5036'i64)
    v21Stream.close()
    doAssert loadCheckpoint(oldV21Path, restoredNext, restoredPop,
      restoredBest, restoredStagnation, restoredMa, restoredPlot,
      restoredElite, restoredX, restoredY, restoredCursor,
      restoredSeed, restoredOffsets)
    doAssert loadedCheckpointVersion == 21 and restoredPop.len == 400 and
      restoredPlot.len == 1 and restoredPlot[0].best == 0.8 and
      jevHall.len == 1 and jevHallConfig == "compatible-old-jev-settings" and
      jevHall[0].wins == 1
    expandLoadedPopulation(restoredPop)
    doAssert restoredPop.len == 450 and jevHall.len == 1 and
      restoredPop[399].len == AAA and restoredPop[449].len == AAA
    removeFile(oldV21Path)
    jevHall = @[]
  AAA = originalRuleCount
  echo "PASS: full v20/v21 checkpoint restores under v22 before objective reset and 400 -> 450 expansion"
  echo "PASS: 400 -> 450 expansion and 500 -> 450 trimming; original genome/stream alignment preserved"

  # All legacy row layouts must be consumed exactly; v16 writes only a/b/w.
  let legacyRuleStream = newStringStream()
  writeIntSeq(legacyRuleStream, @[97])
  writeIntSeq(legacyRuleStream, @[98])
  legacyRuleStream.write(12.5)
  legacyRuleStream.write(0.0) # historical temporal slot
  legacyRuleStream.write(987654321'i64)
  legacyRuleStream.setPosition(0)
  let legacyRule = readRule(legacyRuleStream, 14)
  doAssert legacyRule.weight == 12.5
  doAssert legacyRuleStream.readInt64() == 987654321'i64
  legacyRuleStream.close()
  let branchRuleStream = newStringStream()
  writeIntSeq(branchRuleStream, @[97])
  writeIntSeq(branchRuleStream, @[98])
  branchRuleStream.write(5.0)
  branchRuleStream.write(0.0)
  branchRuleStream.write(-1.5)
  branchRuleStream.write(0b110'u8)
  branchRuleStream.write(987654321'i64)
  branchRuleStream.setPosition(0)
  let legacyBranchRule = readRule(branchRuleStream, 15)
  doAssert legacyBranchRule.weight == 5.0
  doAssert branchRuleStream.readInt64() == 987654321'i64
  branchRuleStream.close()
  let newRuleStream = newStringStream()
  writeRule(newRuleStream, testRule(@[97], @[98], 5.0))
  newRuleStream.write(987654321'i64)
  newRuleStream.setPosition(0)
  let restoredRule = readRule(newRuleStream, CHECKPOINT_VERSION)
  doAssert restoredRule.weight == 5.0
  doAssert newRuleStream.readInt64() == 987654321'i64
  newRuleStream.close()
  echo "PASS: legacy checkpoint layouts read and v22 rule round-trip aligned"
  let baseMapping = defaultEmbedding()
  doAssert validEmbedding(baseMapping)
  var encoded: seq[int] = @[]
  embedBytesInto(@[65, 66], baseMapping, encoded)
  doAssert encoded == @[65, 66]
  var changedMapping = cloneEmbedding(baseMapping)
  swap(changedMapping[65], changedMapping[66])
  doAssert validEmbedding(changedMapping)
  var rawNoise: seq[int] = @[]
  embedBytesInto(@[256], changedMapping, rawNoise)
  doAssert rawNoise == @[EMBEDDING_OOV]
  block:
    var allBytes: seq[int] = @[]
    for b in 0 ..< EMBEDDING_ENTRY_COUNT: allBytes.add(b)
    var allCodes: seq[int] = @[]
    embedBytesInto(allBytes, changedMapping, allCodes)
    doAssert allCodes.len == EMBEDDING_ENTRY_COUNT
    var seenCodes = initHashSet[int]()
    for code in allCodes:
      doAssert code >= 0 and code < EMBEDDING_CODE_COUNT and code notin seenCodes
      seenCodes.incl(code)
    doAssert seenCodes.len == EMBEDDING_ENTRY_COUNT
  var expandedMap = cloneEmbedding(baseMapping)
  expandedMap[65] = 400
  doAssert validEmbedding(expandedMap)
  var collisionMap = cloneEmbedding(expandedMap)
  collisionMap[65] = collisionMap[66]
  doAssert not validEmbedding(collisionMap)
  let remapped = recodeLiteralRuns(encoded, baseMapping, changedMapping)
  doAssert remapped == @[66, 65]
  doAssert recodeLiteralRuns(remapped, changedMapping, baseMapping) == encoded
  # Moving byte 65 into latent coordinate 400 swaps/displaces that latent
  # coordinate through the full 512-token translation, preserving bijection.
  let expandedRecode = recodeLiteralRuns(@[65, 400], baseMapping, expandedMap)
  doAssert expandedRecode == @[400, 65]
  doAssert recodeLiteralRuns(expandedRecode, expandedMap, baseMapping) == @[65, 400]
  doAssert encodeLiteralRuns(@[65, -1, 66], baseMapping) == @[65, -1, 66]
  echo "PASS: injective 256->512 embedding and reversible full-space recoding"
  var embedGenome: Genome = @[testRule(@[65], @[66], 1.0)]
  embedGenome[0].embedding = baseMapping
  let originalMapSnapshot = cloneEmbedding(embedGenome[0].embedding)
  let transplanted = transplantRule(embedGenome[0], baseMapping, changedMapping)
  doAssert transplanted.a == @[66] and transplanted.b == @[65]
  doAssert embedGenome[0].embedding == originalMapSnapshot
  var embeddedScratch = newEvalScratch(1)
  let embeddedCompiled = compileTest(embedGenome)
  let embeddedIndex = buildCandidateIndex(embeddedCompiled)
  let embeddedScore = scoreRaw(embedGenome, embeddedCompiled, embeddedIndex,
    @[65], embeddedScratch, 1)
  doAssert embeddedScore == 1.0
  embedGenome[0] = transplanted
  embedGenome[0].embedding = changedMapping
  let remappedCompiled = compileTest(embedGenome)
  let remappedIndex = buildCandidateIndex(remappedCompiled)
  doAssert scoreRaw(embedGenome, remappedCompiled, remappedIndex,
    @[65], embeddedScratch, 1) == embeddedScore
  echo "PASS: mapping-aware score, donor transplant and parent immutability"
  var crossMapChild = cloneGenome(embedGenome)
  var crossMapDonor = cloneGenome(embedGenome)
  crossMapDonor[0].embedding = baseMapping
  for i in 0 ..< 100:
    crossoverEmbedding(crossMapChild, crossMapDonor, true)
    doAssert validEmbedding(crossMapChild[0].embedding)
    mutateEmbedding(crossMapChild, 1000.0)
    doAssert validEmbedding(crossMapChild[0].embedding)
  doAssert validEmbedding(crossMapDonor[0].embedding)
  echo "PASS: injection mutation and crossover remain collision-free"
  let older = newStringStream()
  older.write(1'i64)
  writeRule(older, testRule(@[65, -1, 66], @[66, -1], 2.0))
  older.setPosition(0)
  let migrated = readGenome(older, 17)
  older.close()
  doAssert validEmbedding(migrated[0].embedding)
  doAssert migrated[0].a == @[65, -1, 66]
  doAssert migrated[0].b == @[66, -1]
  echo "PASS: v17 raw-byte checkpoint migration preserves token length"
  block:
    var oldV22Map = newSeq[int](EMBEDDING_ENTRY_COUNT)
    for i in 0 ..< EMBEDDING_ENTRY_COUNT:
      oldV22Map[i] = if i mod 2 == 0: i else: i + 512
    doAssert validV22Embedding(oldV22Map)
    let migratedV22 = migrateV22Embedding(oldV22Map)
    doAssert validEmbedding(migratedV22.mapping)
    var byteTargets = initHashSet[int]()
    for v in migratedV22.mapping: byteTargets.incl(v)
    doAssert byteTargets.len == EMBEDDING_ENTRY_COUNT
    for oldToken in 0 ..< V22_EMBEDDING_CODE_COUNT:
      doAssert migratedV22.translation[oldToken] >= 0 and
               migratedV22.translation[oldToken] < EMBEDDING_CODE_COUNT
    doAssert migratedV22.translation[V22_EMBEDDING_OOV] == EMBEDDING_OOV
  echo "PASS: v22 1024-token embedding migrates to collision-free 512-byte map"
  block:
    var oldV22Map = newSeq[int](EMBEDDING_ENTRY_COUNT)
    for i in 0 ..< EMBEDDING_ENTRY_COUNT:
      oldV22Map[i] = if i mod 2 == 0: i else: i + 512
    let oldV22Stream = newStringStream()
    oldV22Stream.write(1'i64)
    writeRule(oldV22Stream, testRule(
      @[oldV22Map[65], 700, V22_EMBEDDING_OOV],
      @[oldV22Map[66], 900], 1.0))
    writeIntSeq(oldV22Stream, oldV22Map)
    oldV22Stream.setPosition(0)
    let migratedGenome = readGenome(oldV22Stream, 22)
    oldV22Stream.close()
    doAssert migratedGenome.len == 1 and validEmbedding(migratedGenome[0].embedding)
    doAssert migratedGenome[0].a[0] == migratedGenome[0].embedding[65]
    doAssert migratedGenome[0].a[2] == EMBEDDING_OOV
    for v in migratedGenome[0].a:
      if v >= 0: doAssert v <= EMBEDDING_OOV
    for v in migratedGenome[0].b:
      if v >= 0: doAssert v <= EMBEDDING_OOV
  echo "PASS: v22 serialized genome is migrated through readGenome(version=22)"
  block:
    var legacyMap = newSeq[int](EMBEDDING_ENTRY_COUNT)
    for i in 0 ..< EMBEDDING_ENTRY_COUNT: legacyMap[i] = i + 1024
    let old = newStringStream()
    old.write(1'i64)
    writeRule(old, testRule(@[2, 1, 0, 1, -1, 2, 1, 0, 2],
                            @[2, 1, 0, 2, -48], 2.0))
    writeIntSeq(old, legacyMap)
    old.write(424242'i64)
    old.setPosition(0)
    let converted = readGenome(old, 18)
    doAssert validEmbedding(converted[0].embedding)
    doAssert converted[0].a == @[converted[0].embedding[65], -1,
                                  converted[0].embedding[66]]
    doAssert converted[0].b == @[converted[0].embedding[66], -48]
    doAssert migrateLegacyEncodedLiterals(@[7, 7, 7, 7], legacyMap,
      converted[0].embedding, 4, LEGACY_OCTAL4_CODE_COUNT) == @[EMBEDDING_OOV]
    doAssert old.readInt64() == 424242'i64
    old.close()
    let roundTrip = newStringStream()
    writeGenome(roundTrip, converted)
    roundTrip.write(123456'i64)
    roundTrip.setPosition(0)
    let recovered = readGenome(roundTrip, CHECKPOINT_VERSION)
    doAssert sameGenomeContent(converted, recovered)
    doAssert roundTrip.readInt64() == 123456'i64
    roundTrip.close()
  echo "PASS: v18 four-octal checkpoint migration and v22 round-trip"
  block:
    let oldMap = defaultEmbedding()
    let octal3 = newStringStream()
    octal3.write(1'i64)
    writeRule(octal3, testRule(@[1, 0, 1, -1, 1, 0, 2],
                               @[1, 0, 2, -48], 2.0))
    writeIntSeq(octal3, oldMap)
    octal3.setPosition(0)
    let converted = readGenome(octal3, 19)
    doAssert converted[0].a == @[65, -1, 66]
    doAssert converted[0].b == @[66, -48]
    octal3.close()
  echo "PASS: v19 three-octal checkpoint migration to single-byte tokens"


  quit(0)

if "--self-test" in commandLineParams():
  # Archive admission: strict > mean + 1 population standard deviation in
  # log space, with the current observation excluded and a 75-point ring.
  maReset()
  doAssert not archiveLogThreshold().ready
  maPush(1.0)
  doAssert not archiveLogThreshold().ready
  maPush(1.0)
  doAssert archiveLogThreshold().ready
  doAssert abs(archiveLogThreshold().value - 1.0) < 1.0e-12
  doAssert not (1.0 > archiveLogThreshold().value) # strict comparison
  maPush(3.0)
  let gate = archiveLogThreshold()
  doAssert abs(gate.mean - (5.0 / 3.0)) < 1.0e-12
  doAssert abs(gate.stddev - sqrt(8.0 / 9.0)) < 1.0e-12
  doAssert not (2.6 > gate.value) and 2.7 > gate.value
  let savedWindow = maValuesChronological()
  maReset()
  for score in savedWindow: maPush(score)
  doAssert abs(archiveLogThreshold().value - gate.value) < 1.0e-12
  maReset()
  for i in 0 ..< MA_WINDOW: maPush(1.0)
  maPush(3.0)
  doAssert maCount == MA_WINDOW
  doAssert abs(archiveLogThreshold().mean - (77.0 / 75.0)) < 1.0e-12
  maReset()
  echo "PASS: elite log-threshold admission, strict boundary and ring restoration"

  var scores = newSeq[float](MULTI_CASE_COUNT)
  scores.fill(1.0)
  doAssert abs(aggregateCaseScores(scores) - 1.0) < 1.0e-12
  for ci in 0 ..< MULTI_CASE_COUNT:
    scores[ci] = 0.0
    let expectedDrop = float(3 - ci mod 3) / float(6 * CASES_PER_TIMESCALE)
    doAssert abs(1.0 - aggregateCaseScores(scores) - expectedDrop) < 1.0e-12
    scores[ci] = 1.0
  let indices = retainedTrajectoryIndices(RETAINED_SAMPLES_PER_CASE, 18)
  doAssert indices.len == 18 and indices[0] == 0 and indices[^1] == RETAINED_SAMPLES_PER_CASE-1
  for i in 1 ..< indices.len: doAssert indices[i] > indices[i - 1]
  echo "PASS: configured case weights and full-range trajectory sampling"
  block:
    let flatCases = @[0.80, 0.80, 0.80]
    let dispersedCases = @[0.84, 0.80, 0.76]
    doAssert abs(stableBreedingScore(flatCases) - aggregateCaseScores(flatCases)) < 1.0e-12
    doAssert stableBreedingScore(dispersedCases) < aggregateCaseScores(dispersedCases)
    let localScale = mutationScaleForRegime(1.0, mrLocal)
    let balancedScale = mutationScaleForRegime(1.0, mrBalanced)
    let exploreScale = mutationScaleForRegime(1.0, mrExplore)
    doAssert localScale < balancedScale and balancedScale < exploreScale
    doAssert mutationScaleForRegime(10.0, mrExplore) <= 1.70
    doAssert mutationScaleForRegime(0.01, mrLocal) >= 0.45
  echo "PASS: conservative breeding score and bounded mutation regimes"
  doAssert AAA == 3100
  let syntheticScores = @[0.800, 0.798, 0.700, 0.795]
  let syntheticTimes = @[10.0, 1.0, 0.5, 5.0]
  let pareto = looseRuntimeParetoPool(@[0, 1, 2, 3],
    syntheticScores, syntheticTimes)
  doAssert 0 in pareto and 1 in pareto and 2 notin pareto and 3 notin pareto
  var forcedBudget = EvaluationBudget(started: epochTime() - 10.0)
  var caughtBudget = false
  try:
    for i in 0 ..< 1024: checkEvaluationBudget(forcedBudget)
  except EvaluationBudgetExceeded:
    caughtBudget = true
  doAssert caughtBudget
  echo "PASS: AAA=3100, loose Pareto champion preservation and budget guard"
  quit(0)

setMaxPoolSize(workerCount)
setMinPoolSize(workerCount)
loadTrainingCorpus()
buildCorpusByteStats(sourceChunks)
corpusNgrams = extractTopNgrams(sourceChunks)
echo "corpus stats: ngram_seeds=", corpusNgrams.len, " chunks=", sourceChunks.len

var evalAggsX : seq[seq[seq[int]]] = @[]
var evalAggsY : seq[seq[int]] = @[]
var evalRankedY : seq[seq[float]] = @[]
var fastAggsX : seq[seq[seq[int]]] = @[]
var fastAggsY : seq[seq[int]] = @[]
var fastRankedY : seq[seq[float]] = @[]

let CHECKPOINT_DEFAULT = "ga_checkpoint.bin"
var checkpointPath = CHECKPOINT_DEFAULT
var loadRequested = false
var resetEliteArchiveRequested = false

for arg in commandLineParams():
  if arg == "--load":
    loadRequested = true
  elif arg.startsWith("--load="):
    loadRequested = true
    checkpointPath = arg.split("=", 1)[1]
  elif arg.startsWith("--save="):
    checkpointPath = arg.split("=", 1)[1]
  elif arg == "--reset-elite-archive":
    resetEliteArchiveRequested = true

var startIter = 0
var loadedCheckpoint = false

# 前回の異常終了で .tmp だけ残った場合、壊れた一時ファイルを
# 次回起動のcheckpointと誤認しない。
let staleTmp = checkpointPath & ".tmp"
if fileExists(staleTmp):
  try:
    removeFile(staleTmp)
  except CatchableError:
    discard

if loadRequested and fileExists(checkpointPath):
  if loadCheckpointWithRecovery(
    checkpointPath,
    startIter,
    population,
    bestEver,
    stagnation,
    maHistory,
    progressHistory,
    eliteOfElites,
    evalAggsX,
    evalAggsY,
    sourceChunkCursor,
    runSeed,
    progressSigmaOffset
  ):
    loadedCheckpoint = true

if loadedCheckpoint:
  # Optional one-time purge for an archive from a previous training run.
  # The on-disk checkpoint remains untouched until the normal save cadence.
  if resetEliteArchiveRequested:
    echo "[archive] cleared ", eliteOfElites.len,
         " legacy entries (--reset-elite-archive)"
    eliteOfElites.setLen(0)
  if migratedRuleCount:
    # New rules AND a different scoring objective invalidate historical scores.
    progressHistory.setLen(0)
    echo "[migration] legacy rule count expanded (population/archive); plot history restarted"
  if loadedCheckpointVersion < CHECKPOINT_OBJECTIVE_VERSION:
    # v23 shrinks the internal alphabet 1024 -> 512. v22 byte identities are
    # preserved injectively; historical latent symbols are folded only into the
    # new latent half. Scores/Halls are not comparable and are reset.
    progressHistory.setLen(0)
    eliteOfElites.setLen(0)
    jevHall.setLen(0)
    jevHallConfig = ""
    maReset()
    bestEver = -Inf
    stagnation = 0
    echo "[migration] v23 512-token objective: old scores/archive/Jev Hall discarded; genomes migrated"
  if loadedCheckpointVersion < 16:
    # Remove stale history and old-objective archives; preserve population.
    # Archival genomes were selected by incompatible branch scores.
    progressHistory.setLen(0)
    eliteOfElites.setLen(0)
    echo "[migration] v16 single trajectory: branch alleles discarded; old score/archive history reset"
  # A checkpoint may have been produced by v24, which injected up to 40 archive
  # genomes. Keep the first resumed generation conservative; subsequent
  # generations store the exact protected prefix length again.
  protectedPrefixCount = min(eliteCount + 40, pop_size)
  for ci in 0 ..< MULTI_CASE_COUNT:
    case ci mod 3
    of FAST_ROLLING_SLOT: rollingCaseAge[ci] = startIter mod FAST_DATASET_REFRESH_INTERVAL
    of MEDIUM_ROLLING_SLOT: rollingCaseAge[ci] = startIter mod MEDIUM_DATASET_REFRESH_INTERVAL
    else: rollingCaseAge[ci] = startIter mod SLOW_DATASET_REFRESH_INTERVAL
  rebuildRollingRefreshCredit(startIter)
  rebuildPlotMovingAverages(progressHistory)
  if population.len != pop_size and population.len != 400:
    quit("Checkpoint population size (" & $population.len &
         ") cannot migrate to pop_size (" & $pop_size & ").")
  for gi, genome in population:
    if genome.len != AAA:
      quit("Checkpoint genome[" & $gi & "] rule count (" & $genome.len &
           ") does not match AAA (" & $AAA & ").")
    for ri, rule in genome:
      if rule.a.len > 64 or rule.b.len > 64:
        quit("Checkpoint genome[" & $gi & "][" & $ri & "] contains an oversized sequence.")
      if rule.weight != rule.weight or abs(rule.weight) > 1.0e6:
        quit("Checkpoint genome[" & $gi & "][" & $ri & "] contains an invalid weight.")
  # All checkpoint sections have now been decoded, and existing genomes were
  # validated. Expand legacy 400 -> 450 *after* confirming stream alignment;
  # the first 400 indices (including protected survivors) remain unchanged.
  expandLoadedPopulation(population)

  # Migrate older case counts while preserving genomes and archive entries.
  # Reset plotted history because 3-case and 9-case objectives differ.
  if evalAggsX.len != DATASET_CHUNKS or evalAggsY.len != DATASET_CHUNKS or
      datasetHasInvalidBytes(evalAggsX):
    # 旧case数checkpointだけruntime評価集合を作り直す。現在はcorpus順を
    # 学習時系列へ持ち込まないランダムrollingなので、cursorの巻き戻しはしない。
    rebuildDataset(sourceChunks, sourceChunkCursor, evalAggsX, evalAggsY)
    progressHistory.setLen(0)
    eliteOfElites.setLen(0) # Different 3/9-case objective: old archive scores cannot be reused.
    maReset()
    maHistory.setLen(0)
    bestEver = -Inf
    stagnation = 0
    for ci in 0 ..< MULTI_CASE_COUNT: rollingCaseAge[ci] = 64
    echo "[migration] genomes retained; incompatible/invalid-byte dataset and fitness history cleared; nine cases rebuilt"

  # Old checkpoints contain sampled/history-dependent damage labels.
  # The first observation is the original base; relabel without discarding genomes.
  for ci in 0 ..< evalAggsX.len:
    if evalAggsX[ci].len == 0: continue
    let base = evalAggsX[ci][0]
    evalAggsY[ci].setLen(evalAggsX[ci].len)
    for si in 0 ..< evalAggsX[ci].len:
      evalAggsY[ci][si] = max(1, base.len -
        estimateDiffusionDamage(base, evalAggsX[ci][si], 0))
  # Rebuild the admission window from the checkpoint's comparable plot history.
  # The old code unconditionally reset this ring on EVERY resume, silently
  # disabling the 75-generation admission baseline after restarting training.
  # Migration paths above clear progressHistory if the evaluator changed.
  maReset()
  for hi in max(0, progressHistory.len - MA_WINDOW) ..< progressHistory.len:
    maPush(spearmanToSigma(progressHistory[hi].best))
  bestEver = -Inf
  stagnation = 0

  let fast = buildFastEvalDataset(evalAggsX, evalAggsY, adaptiveFastEvalStride)
  fastAggsX = fast.x
  fastAggsY = fast.y
  evalRankedY = buildRankedY(evalAggsY)
  fastRankedY = buildRankedY(fastAggsY)

  # グラフ用MAは表示窓変更や末尾データ追加によって先頭側も遡及補正される。
  # checkpointから復元した履歴も、現在の200/200窓で再構築する。
  rebuildPlotMovingAverages(progressHistory)
  # A checkpoint migrated to a new scoring objective can reset progressHistory.
  # Do not overwrite the historical CSV/PNG without an accessible backup.
  if fileExists("progress.csv"):
    let previousCsv = scanPlotSeries("progress.csv", 3)
    if previousCsv.valid and previousCsv.maxGeneration > progressHistory.len:
      try:
        copyFile("progress.csv", "progress.pre-resume.csv")
        if fileExists(PLOT_PNG_PATH):
          copyFile(PLOT_PNG_PATH, "progress.pre-resume.png")
        echo "[plot] old progress preserved: progress.pre-resume.csv/png"
      except CatchableError as e:
        quit("Cannot preserve pre-resume progress; original CSV untouched: " & e.msg)
  writeProgressCsv("progress.csv", progressHistory)

  # Plot after Jev CSV has been validated and truncated to checkpoint generation.
  # (Calling plotProgress here could accidentally show future/stale Jev scores.)

  echo "Loaded checkpoint: ", checkpointPath,
       " next_generation=", startIter,
       " elite_archive=", eliteOfElites.len,
       " graph_points=", progressHistory.len
else:
  protectedPrefixCount = 0
  if loadRequested:
    echo "Checkpoint not found: ", checkpointPath, " -> starting a new run."
  for i in 0 ..< pop_size:
    population.add(makeObj())
  rebuildDataset(sourceChunks, sourceChunkCursor, evalAggsX, evalAggsY)
  let fast = buildFastEvalDataset(evalAggsX, evalAggsY, adaptiveFastEvalStride)
  fastAggsX = fast.x
  fastAggsY = fast.y
  evalRankedY = buildRankedY(evalAggsY)
  fastRankedY = buildRankedY(fastAggsY)
  bestEver = -Inf
  stagnation = 0
  for ci in 0 ..< MULTI_CASE_COUNT: progressSigmaOffset[ci] = 0.0
  maReset()
  progressHistory.setLen(0)
  eliteOfElites.setLen(0)

echo "evaluation mode: parallel individual evaluation (--threads:on)"
echo "rolling evaluation schedule: fast=11, medium=3, slow=2; credit scheduler, at most one refresh per generation"

# stagnation判定は、32世代だけ保持されるslow rolling case上の改善で測る。
# case交換そのものを「悪化」扱いしないため、slow case交換直後はbaselineだけ張り直す。
# checkpointからの再開時も saved stagnation は保持し、baselineだけ再初期化する。
var slowEpochBest = -Inf
var slowEpochInitialized = false

var compiledPopulation = newSeq[seq[SeqPattern]](pop_size)
# 各ruleのpattern revision。
# a の内容が変化したときだけrevisionが更新されるので、
# 毎世代pop_size*AAA回のseq hashを行わず、O(1)でcompiled patternの
# 有効性を判定できる。ポインタ再利用や同一長への変更にも安全。
var compiledPatternRevision = newSeq[seq[uint64]](pop_size)
var candidatePopulation = newSeq[CandidateIndex](pop_size)
for jp in 0 ..< pop_size:
  compiledPopulation[jp] = newSeq[SeqPattern](AAA)
  compiledPatternRevision[jp] = newSeq[uint64](AAA)
  compiledPatternRevision[jp].fill(0'u64)
  new(candidatePopulation[jp])
  candidatePopulation[jp].head2 = initPairHeadIndex(AAA)
  candidatePopulation[jp].next1 = newSeq[int32](AAA)
  candidatePopulation[jp].next2 = newSeq[int32](AAA)

# Scores correspond exactly to the unchanged survivor prefix of population.
# Runtime-only: an empty cache on resume simply forces a normal evaluation.
var carriedCaseScores: seq[seq[float]] = @[]
# Per-case FULL seconds align with carriedCaseScores; both are runtime-only.
var carriedCaseTimings: seq[seq[float]] = @[]
# Probe parents are unchanged survivor-prefix indices in the NEXT population.
# Probe children bypass FAST screening to avoid success-selection bias.
var mutationProbes: seq[tuple[parent, child: int, regime: MutationRegime]] = @[]
# These children receive NO mutation: their difference from the anchor is
# attributable to this one crossover operator on identical FULL cases.
var crossoverProbes: seq[tuple[parent, child: int, op: CrossoverOp, seconds: float]] = @[]
var crossoverPreference: array[CrossoverOp, float]
var crossoverLastCounts: array[CrossoverOp, int]
var crossoverLastSeconds: array[CrossoverOp, float]
# Probe overhead used to consume 4 crossover + 6 mutation mandatory FULL slots
# every generation. Rotate one crossover operator per generation and keep three
# mutation probes; the EMAs integrate evidence across generations.
const CROSSOVER_PROBE_COUNT = 1

var mutationSuccessEma = 0.20
var feedbackMutationScale = 1.0
var mutationRegimePreference: array[MutationRegime, float]
var mutationRegimeLastCounts: array[MutationRegime, int]
const MUTATION_PROBE_COUNT = MUTATION_REGIME_COUNT

# Keep Jev metric files tied to their checkpoint, not to a shared CWD filename.
jevProgressPath = checkpointPath & ".jev_progress.csv"
previousJevProgressPath = jevProgressPath & ".previous.csv"
if loadedCheckpoint: archivedJevUpperGeneration = startIter
# Jev is optional but never silently simulated: no key => pure GA, no fake data.
var jevPrompt = "def "
var jevModel = getEnv("JEV_MODEL", "jev-1.13.0")
var jevDisabled = false
var jevGenerationWorkers = max(1, min(16, workerCount))
var jevModelCount = JEV_DEFAULT_MODELS
for arg in commandLineParams():
  if arg == "--no-jev": jevDisabled = true
  elif arg.startsWith("--jev-prompt="): jevPrompt = arg.split("=", 1)[1]
  elif arg.startsWith("--jev-model="): jevModel = arg.split("=", 1)[1]
  elif arg.startsWith("--jev-max-models="):
    try:
      jevModelCount = parseInt(arg.split("=", 1)[1])
    except ValueError:
      quit("--jev-max-models must be an integer between 2 and 64")
  elif arg.startsWith("--jev-generation-workers="):
    try:
      jevGenerationWorkers = parseInt(arg.split("=", 1)[1])
    except ValueError:
      quit("--jev-generation-workers must be an integer between 1 and 16")
if jevGenerationWorkers < 1 or jevGenerationWorkers > min(16, workerCount):
  quit("--jev-generation-workers must be 1..min(16, CPU worker count)")
if jevModelCount < 2 or jevModelCount > JEV_MAX_MODELS:
  quit("--jev-max-models must be 2..64")
if unicode.validateUtf8(jevPrompt) != -1 or jevPrompt.len > 2048:
  quit("--jev-prompt must be valid UTF-8 and at most 2048 bytes")
let jevBridgeApp = getAppDir() / "jev_bridge.py"
var jevBridge = getCurrentDir() / "jev_bridge.py"
if fileExists(jevBridgeApp): jevBridge = jevBridgeApp
let jevKeyAvailable = getEnv("TYPESAFE_API_KEY").len > 0 or
                      getEnv("JEV_API_KEY").len > 0
let jevEnabled = not jevDisabled and jevKeyAvailable and fileExists(jevBridge)
if not jevEnabled:
  echo "Jev lane OFF: set TYPESAFE_API_KEY (or JEV_API_KEY), put jev_bridge.py next to binary, and omit --no-jev"
else:
  echo "Jev lane ON: every ", JEV_GENERATION_INTERVAL, " generations, ", jevModelCount,
       " FULL genomes x <=", JEV_STEPS, " scoring proposals; generation workers=",
       jevGenerationWorkers, "; provider charges may apply"
var jevAlphabet: seq[int] = @[]
var jevCumulative: seq[float] = @[]
var jevGrams: seq[GenerationGram] = @[]
if jevEnabled:
  jevAlphabet = generationAlphabet()
  let priors = generationCorpusPriors(jevAlphabet)
  var total = 0.0
  for frequency in priors.frequencies:
    total += sqrt(float(frequency))
    jevCumulative.add(total)
  jevGrams = priors.grams
# A different Jev scoring configuration OR graph metric starts a new MA20 series.
# Preserve the old CSV separately: legacy equal-weight MA20 cannot be converted
# into rank-weighted history without archived per-candidate scores/rankings.
let jevConfigPath = jevProgressPath & ".config.json"
let jevConfig = $(%*{"prompt": jevPrompt, "model": jevModel,
                    "steps": JEV_STEPS, "suffix": JEV_SUFFIX_LENGTH,
                    "candidates": jevModelCount, "generation_algorithm": 7,
                    "internal_tokens": EMBEDDING_CODE_COUNT,
                    "quality_metric": "rank_linear_weighted_v1",
                    "version": JEV_BRIDGE_PROTOCOL})
# v20 checkpoints have no Hall and remain loadable. A Jev configuration change
# invalidates only Hall rankings, never GA history, genomes or conventional EoE.
if jevHallConfig != jevConfig:
  resetFusion()
if jevHall.len > 0 and jevHallConfig != jevConfig:
  echo "[jev-hall] prompt/model/generation settings changed; clearing incompatible Hall (", jevHall.len, ")"
  jevHall.setLen(0)
jevHallConfig = jevConfig
let jevCompatible = loadedCheckpoint and fileExists(jevConfigPath) and
                    readFile(jevConfigPath) == jevConfig
if loadedCheckpoint and not jevCompatible and fileExists(jevProgressPath) and
   readFile(jevProgressPath).splitLines().len > 2:
  # Keep the previous curve and settings (do not splice unlike MA20 windows).
  # A failed archive must NOT be followed by destructive CSV initialization.
  try:
    if fileExists(previousJevProgressPath):
      var backup = previousJevProgressPath & "." & $getTime().toUnix & ".bak"
      var serial = 1
      while fileExists(backup):
        backup = previousJevProgressPath & "." & $getTime().toUnix & "." & $serial & ".bak"
        inc serial
      copyFile(previousJevProgressPath, backup)
    copyFile(jevProgressPath, previousJevProgressPath)
    if fileExists(jevConfigPath):
      copyFile(jevConfigPath, previousJevProgressPath & ".config.json")
    if fileExists(PLOT_PNG_PATH):
      copyFile(PLOT_PNG_PATH, "progress.previous.png")
    echo "[plot] previous Jev curve archived: ", previousJevProgressPath,
         " (different scoring settings; plotted separately)"
  except CatchableError as e:
    quit("Jev curve archive failed; existing CSV was NOT overwritten: " & e.msg)
jevHistory = loadJevProgress(jevProgressPath, startIter, jevCompatible)
writeFile(jevConfigPath & ".tmp", jevConfig)
moveFile(jevConfigPath & ".tmp", jevConfigPath)
if loadedCheckpoint and progressHistory.len >= PLOT_MA_SHORT_WINDOW:
  plotProgress()

rebuildRollingRefreshCredit(startIter)

for iter in startIter ..< TARGET_GENERATIONS:
  let tGenStart = epochTime()
  # The previous champion is always survivor 0. Capture BEFORE refreshed scores
  # are written and BEFORE local search modifies any genome. Empty on resume.
  let previousChampionScore = if carriedCaseScores.len > 0:
                               aggregateCaseScores(carriedCaseScores[0])
                             else: -Inf
  reseedForGeneration(iter)
  for ci in 0 ..< MULTI_CASE_COUNT:
    rollingCaseAge[ci] = min(1000000, rollingCaseAge[ci] + 1)
  # Rolling evaluation with requested SLOW:MEDIUM:FAST widths 2:3:11.
  # A naive modulo schedule creates periodic double/triple refreshes. Integer
  # credits preserve those average rates while allowing at most one refresh per
  # generation, preventing refresh-induced FULL-evaluation spikes.
  var refreshedCases: array[MULTI_CASE_COUNT, bool]
  var datasetChanged = false
  var refreshedFast = false
  var refreshedMedium = false
  var refreshedSlow = false
  if iter > 0:
    # 66 is lcm(2,3,11). Add exact rational credits 6/66, 22/66, 33/66
    # each generation and spend at most one 66-credit refresh. Total offered
    # load is 61/66 < 1, so no long-run backlog is required and average rates
    # converge exactly to 1/11, 1/3, 1/2 without double-refresh generations.
    const refreshCreditAdd = [6, 22, 33] # FAST, MEDIUM, SLOW
    for group in 0 .. 2: rollingRefreshCredit[group] += refreshCreditAdd[group]
    var chosenGroup = -1
    var bestCredit = -1
    for group in 0 .. 2:
      if rollingRefreshCredit[group] >= 66 and
         (chosenGroup < 0 or rollingRefreshCredit[group] > bestCredit):
        chosenGroup = group
        bestCredit = rollingRefreshCredit[group]
    if chosenGroup >= 0:
      rollingRefreshCredit[chosenGroup] -= 66
      let changed = refreshDatasetGroup(
        sourceChunks, sourceChunkCursor, evalAggsX, evalAggsY, chosenGroup,
        refreshedCases
      )
      datasetChanged = changed
      if changed:
        rollingCaseAge[chosenGroup] = 0
        case chosenGroup
        of FAST_ROLLING_SLOT: refreshedFast = true
        of MEDIUM_ROLLING_SLOT: refreshedMedium = true
        of SLOW_ROLLING_SLOT: refreshedSlow = true
        else: discard
      # 新しいslow caseと古いslow caseの絶対scoreは比較しない。
      # case交換そのものは悪化ではないのでbaselineは張り直す。
      # ただし stagnation を毎32世代ゼロへ戻すと、長期停滞でも探索圧が
      # 永遠に育たない。半減だけして「過去の停滞証拠」を穏やかに残す。
      slowEpochInitialized = false
      slowEpochBest = -Inf
      # dataset境界だけで探索圧を急落させない。25%だけ減衰させて連続化する。
      stagnation = (stagnation * 3) div 4

  if datasetChanged:
    let fast = buildFastEvalDataset(evalAggsX, evalAggsY, adaptiveFastEvalStride)
    fastAggsX = fast.x
    fastAggsY = fast.y
    evalRankedY = buildRankedY(evalAggsY)
    fastRankedY = buildRankedY(fastAggsY)
    var totalSamples = 0
    for c in evalAggsX:
      totalSamples += c.len
    echo "gen=", iter, " dataset rolling refresh: fast=", refreshedFast,
         " medium=", refreshedMedium, " slow=", refreshedSlow, " total_cases=", evalAggsX.len,
         " total_samples=", totalSamples

  if iter > 0 and iter mod CACHE_CLEAR_INTERVAL == 0:
    # 100世代ごとの全消去は再コンパイルを大量発生させ、
    # キャッシュの恩恵を捨ててしまう。まずは長い間隔だけで世代を切る。
    patternCache.clear()
    revisionPatternCache.clear()

  var patternRevisionUpdates = 0
  var candidateIndexRebuilds = 0
  for jp in 0 ..< pop_size:
    var patternChanged = false
    for i in 0 ..< AAA:
      let rule = population[jp][i]
      let rev = rule.patternRevision

      # patternRevision が一致している限り、a の内容は不変。
      # b/weightだけ変更された個体では compiled pattern を完全再利用する。
      if rev != 0'u64 and compiledPatternRevision[jp][i] == rev:
        continue

      patternChanged = true
      inc patternRevisionUpdates
      compiledPopulation[jp][i] = compileSeqPatternByRevision(rule.a, rev)
      compiledPatternRevision[jp][i] = rev

    # .aが1つも変わっていなければcandidate indexも完全に再利用できる。
    # これもelite系統で毎世代の3100-rule走査を丸ごと省略する。
    if patternChanged:
      inc candidateIndexRebuilds
      rebuildCandidateIndex(candidatePopulation[jp], compiledPopulation[jp])

  let tAfterCompile = epochTime()

  let aggs_x = evalAggsX
  let aggs_y = evalAggsY
  if iter mod 25 == 0:
    var totalSamples = 0
    var sampleSummary = ""
    for c in aggs_x:
      totalSamples += c.len
      if sampleSummary.len > 0:
        sampleSummary.add(",")
      sampleSummary.add($c.len)
    echo "gen=", iter, " eval_chunks=", aggs_x.len,
         " total_samples=", totalSamples,
         " samples_per_chunk=[", sampleSummary, "]"

  var accs = newSeq[float](pop_size)
  var bar = newProgressBar(total = pop_size)
  bar.start()

  # ------------------------------------------------------------
  # Stage 1: 薄い評価で全個体を高速スクリーニング。
  # 個体間だけを並列化し、共有データはread-onlyで渡す。
  # ------------------------------------------------------------
  var allIds = newSeq[int](pop_size)
  for jp in 0 ..< pop_size:
    allIds[jp] = jp

  # The inherited survivor/archive prefix and mutation-probe pairs MUST pass
  # FULL evaluation regardless of their FAST rank. Running FAST on them is
  # redundant: their screening score cannot change that decision. Screening
  # only genuinely optional offspring both saves work and frees merit slots.
  let protectedCount = min(protectedPrefixCount, pop_size)
  var fullIds: seq[int] = @[]
  var fullSeen = newSeq[bool](pop_size)
  for jp in 0 ..< protectedCount:
    fullSeen[jp] = true
    fullIds.add(jp)
  for probe in mutationProbes:
    for jp in [probe.parent, probe.child]:
      if jp >= 0 and jp < pop_size and not fullSeen[jp]:
        fullSeen[jp] = true
        fullIds.add(jp)
  for probe in crossoverProbes:
    for jp in [probe.parent, probe.child]:
      if jp >= 0 and jp < pop_size and not fullSeen[jp]:
        fullSeen[jp] = true
        fullIds.add(jp)

  if jevEnabled:
    for jp in allIds:
      if not fullSeen[jp] and
         (fusionMatch(population[jp], fusionArchive) >= 0 or
          ((iter+1) mod FUSION_INTERVAL == 0 and fusionMatch(population[jp], fusionAnchors) >= 0)):
        fullSeen[jp] = true
        fullIds.add(jp)
    for jp in allIds:
      if fullSeen[jp]: continue
      for pending in fusionPending:
        if iter <= pending.due and sameGenomeContent(population[jp], pending.genome):
          fullSeen[jp] = true
          fullIds.add(jp)
          break
  var fastIds: seq[int] = @[]
  for jp in allIds:
    if not fullSeen[jp]:
      fastIds.add(jp)
  let screenedResults = parallelEvaluateCasesBatch(
    population,
    compiledPopulation,
    candidatePopulation,
    fastIds,
    unsafeAddr fastAggsX,
    unsafeAddr fastAggsY,
    unsafeAddr fastRankedY,
    AAA,
    adaptiveFastEvalRounds,
    "FAST"
  )
  var fastCaseScores = newSeq[seq[float]](pop_size)
  var fastScores = newSeq[float](fastIds.len)
  for i, jp in fastIds:
    fastCaseScores[jp] = screenedResults[i]
    fastScores[i] = aggregateCaseScores(screenedResults[i])
    accs[jp] = fastScores[i]

  let tAfterFastEval = epochTime()

  let screeningOrderLocal = sortIndicesByScore(fastScores)
  var screeningOrder = newSeq[int](screeningOrderLocal.len)
  for i, position in screeningOrderLocal:
    screeningOrder[i] = fastIds[position]
  # Previously FULL_EVAL_TOP=42 included incumbents that were going to be FULL
  # evaluated anyway. Preserve a similar total FULL budget but guarantee a
  # useful number of distinct, screened newcomers in each generation.
  let fullCount = min(max(16, FULL_EVAL_TOP - protectedCount), screeningOrder.len)

  # ------------------------------------------------------------
  # ★重要: elitism を FAST 評価に依存させない。
  #
  # 前世代の survivor elite は先頭、その直後にarchive注入個体が置かれる。
  # これらが FAST の順位だけで FULL_EVAL_TOP 外へ落ちると、粗い評価値のまま
  # 捨てられる。特にarchiveは過去rolling datasetのscoreを信用できないので、
  # CURRENT FULL評価を必ず通してから生存・親選択を決める。
  #
  # よって FULL 評価対象は
  #   1. 前世代 survivor elite + archive注入prefix
  #   2. FAST 上位
  # の和集合にする。
  # ------------------------------------------------------------
  # The mandatory FULL set is already assembled before FAST screening.
  # Do not re-add or re-screen these individuals here.

  for oi in 0 ..< fullCount:
    let jp = screeningOrder[oi]
    if not fullSeen[jp]:
      fullSeen[jp] = true
      fullIds.add(jp)

  # Rescue case specialists hidden by mean-only screening. Their true FULL
  # scores, never their FAST estimates, still determine survival and parenthood.
  const SPECIALISTS_PER_CASE = 1
  for ci in 0 ..< fastAggsX.len:
    var scores = newSeq[float](fastIds.len)
    for i, jp in fastIds:
      scores[i] = fastCaseScores[jp][ci]
    let specialists = sortIndicesByScore(scores)
    for oi in 0 ..< min(SPECIALISTS_PER_CASE, specialists.len):
      let jp = fastIds[specialists[oi]]
      if not fullSeen[jp]:
        fullSeen[jp] = true
        fullIds.add(jp)

  # ------------------------------------------------------------
  # ★遺伝的多様性のためのFULL評価枠。
  # 単純なランダム抽出ではなく、既に選ばれた個体から遺伝的に遠い
  # FAST上位個体をgreedy farthest-pointで追加する。
  # これにより「低fitnessだから捨てる」ではなく、「異なるbuilding blockを
  # 持っているので一度は真面目に測る」というGAらしい選別になる。
  # ------------------------------------------------------------
  proc localRuleDistance(a, b: Rule): float {.inline.} =
    var d = abs(a.a.len - b.a.len).float * 0.10
    d += abs(a.b.len - b.b.len).float * 0.10
    let na = min(4, min(a.a.len, b.a.len))
    for k in 0 ..< na:
      if a.a[k] != b.a[k]: d += 0.18
    let nb = min(4, min(a.b.len, b.b.len))
    for k in 0 ..< nb:
      if a.b[k] != b.b[k]: d += 0.18
    d += min(1.0, abs(a.weight - b.weight) / 4.0) * 0.08
    d

  proc localGenomeDistance(a, b: Genome): float {.inline.} =
    if a.len == 0 or b.len == 0: return 1.0
    let probes = min(37, min(a.len, b.len))
    if probes <= 0: return 0.0
    var d = 0.0
    for k in 0 ..< probes:
      let ia = (k * (a.len - 1)) div max(1, probes - 1)
      let ib = (k * (b.len - 1)) div max(1, probes - 1)
      d += localRuleDistance(a[ia], b[ib])
    d = d / probes.float
    # Different byte dictionaries are also genuine genetic diversity.
    if a[0].embedding.len == EMBEDDING_ENTRY_COUNT and
        b[0].embedding.len == EMBEDDING_ENTRY_COUNT:
      var mapDiff = 0
      for byte in countup(0, EMBEDDING_ENTRY_COUNT - 1, 8):
        if a[0].embedding[byte] != b[0].embedding[byte]: inc mapDiff
      d += 0.50 * float(mapDiff) / 32.0
    d

  proc selectDistantParentId(
    anchorId: int,
    candidateIds: openArray[int],
    useCount: openArray[int]
  ): int {.inline.} =
    if candidateIds.len == 0 or anchorId < 0:
      return -1
    var bestId = -1
    var bestValue = -Inf
    let samples = min(10, candidateIds.len)
    # 候補を調べるだけでparentUseCountを消費してはいけない。
    # 実際に選んだ1体だけは呼び出し側で通常の上限管理を通す。
    for _ in 0 ..< samples:
      let id = candidateIds[rand(candidateIds.len - 1)]
      if id == anchorId:
        continue
      if id < useCount.len and useCount[id] >= PARENT_MAX_USES:
        continue
      let d = localGenomeDistance(population[anchorId], population[id])
      let value = d + 0.15 * max(-1.0, min(1.0, accs[id]))
      if value > bestValue:
        bestValue = value
        bestId = id
    bestId

  # ここは全pop_size体を毎回総当たりするとかなり重い。
  # 一世代につき候補を一度だけランダム抽出し、その候補内でfarthest-pointを行う。
  const DIVERSITY_CANDIDATE_POOL = 64
  var diversityCandidates = newSeq[int](0)
  let diversityPoolTarget = min(DIVERSITY_CANDIDATE_POOL, pop_size)
  if diversityPoolTarget > 0:
    diversityCandidates = pickRandom(pop_size, diversityPoolTarget)

  var diversityAdded = 0
  let diversityTarget = min(FULL_EVAL_DIVERSITY_EXTRA, pop_size - fullIds.len)
  # Farthest-point's min distance is monotone: after choosing one new point,
  # update each candidate with distance to ONLY that point. The old code
  # recomputed distance against the entire growing FULL set on every slot.
  # This is mathematically identical but removes an O(K * pool * fullIds) loop.
  var diversityMinD = newSeq[float](pop_size)
  diversityMinD.fill(Inf)
  for jp in diversityCandidates:
    if fullSeen[jp]: continue
    for sid in fullIds:
      diversityMinD[jp] = min(diversityMinD[jp],
        localGenomeDistance(population[jp], population[sid]))

  while diversityAdded < diversityTarget:
    var bestId = -1
    var bestValue = -Inf
    for jp in diversityCandidates:
      if fullSeen[jp]:
        continue
      let value = diversityMinD[jp] + 0.08 * accs[jp]
      if value > bestValue:
        bestValue = value
        bestId = jp

    # 候補poolだけでは足りなかった場合の安全弁。通常は64候補に対して
    # 8枠なのでここへ来ない。fallbackだけは完全計算で正確性を保つ。
    if bestId < 0:
      for jp in allIds:
        if fullSeen[jp]:
          continue
        var minD = Inf
        for sid in fullIds:
          minD = min(minD, localGenomeDistance(population[jp], population[sid]))
        let value = minD + 0.08 * accs[jp]
        if value > bestValue:
          bestValue = value
          bestId = jp

    if bestId < 0:
      break
    fullSeen[bestId] = true
    fullIds.add(bestId)
    inc diversityAdded
    for jp in diversityCandidates:
      if not fullSeen[jp]:
        diversityMinD[jp] = min(diversityMinD[jp],
          localGenomeDistance(population[jp], population[bestId]))

  # Jev requires the configured number of genuinely FULL-evaluated candidates, never FAST scores.
  # If the ordinary GA's FULL set is smaller, promote screened candidates
  # BEFORE executing Stage 2. They share the exact same FULL objective.
  if jevEnabled and (iter + 1) mod JEV_GENERATION_INTERVAL == 0 and fullIds.len < jevModelCount:
    for jp in screeningOrder:
      if fullIds.len >= jevModelCount: break
      if not fullSeen[jp]:
        fullSeen[jp] = true
        fullIds.add(jp)
    echo "  jev: guaranteed FULL evaluation pool=", fullIds.len,
         " requested=", jevModelCount
  echo "  genetic parent pool(full)=", fullIds.len, " / ", pop_size,
       " (elite=", protectedCount, ", meritTop=", fullCount,
       ", diversity=", diversityAdded, ")"

  let tAfterFullIdSelect = epochTime()

  # ------------------------------------------------------------
  # Stage 2: FULL評価。ここも個体単位で並列化する。
  # ------------------------------------------------------------
  var caseScores = newSeq[seq[float]](pop_size)
  var caseTimings = newSeq[seq[float]](pop_size)
  var freshIds: seq[int] = @[]
  var reusedIds: seq[int] = @[]
  for jp in fullIds:
    var canReuse = jp < carriedCaseScores.len and jp < carriedCaseTimings.len
    when defined(disableCarryCache): canReuse = false
    if canReuse and carriedCaseScores[jp].len == aggs_x.len and
       carriedCaseTimings[jp].len == aggs_x.len:
      reusedIds.add(jp)
      caseScores[jp] = carriedCaseScores[jp]
      caseTimings[jp] = carriedCaseTimings[jp]
    else:
      freshIds.add(jp)
  var freshTimings: seq[seq[float]]
  let freshResults = parallelEvaluateCasesBatch(
    population, compiledPopulation, candidatePopulation, freshIds,
    unsafeAddr aggs_x, unsafeAddr aggs_y, unsafeAddr evalRankedY, AAA,
    SCORE_ROUNDS, "FULL", addr freshTimings)
  for i, jp in freshIds:
    caseScores[jp] = freshResults[i]
    caseTimings[jp] = freshTimings[i]

  # Only the refreshed cases of unchanged survivors need recomputation.
  # Paired old-vs-new case scores for an IDENTICAL genome expose dataset drift
  # directly, instead of falsely attributing a case swap to a bad mutation.
  var refreshDeltaSum = newSeq[float](aggs_x.len)
  var refreshDeltaN = newSeq[int](aggs_x.len)
  let refreshed = refreshedCases
  for ci in 0 ..< aggs_x.len:
    if ci >= refreshed.len or refreshed[ci]:
      var slotX = @[aggs_x[ci]]
      var slotY = @[aggs_y[ci]]
      var slotRank = @[evalRankedY[ci]]
      var changedTimings: seq[seq[float]]
      let changedResults = parallelEvaluateCasesBatch(
        population, compiledPopulation, candidatePopulation, reusedIds,
        addr slotX, addr slotY, addr slotRank, AAA, SCORE_ROUNDS,
        "REFRESH", addr changedTimings)
      for i, jp in reusedIds:
        caseTimings[jp][ci] = changedTimings[i][0]
        let oldCaseScore = caseScores[jp][ci]
        let newCaseScore = changedResults[i][0]
        refreshDeltaSum[ci] += newCaseScore - oldCaseScore
        inc refreshDeltaN[ci]
        caseScores[jp][ci] = newCaseScore

  for ci in 0 ..< aggs_x.len:
    if ci < refreshed.len and refreshed[ci]:
      if refreshDeltaN[ci] > 0:
        echo "  case_refresh_delta[", ci, "]=",
          formatFloat(refreshDeltaSum[ci] / float(refreshDeltaN[ci]), ffDecimal, 6),
          " identical_survivors=", refreshDeltaN[ci]
      else:
        echo "  case_refresh_delta[", ci, "]=unavailable (no carried survivors)"

  if "--verify-cache" in commandLineParams() and carriedCaseScores.len > 0:
    let checked = safeEvaluateIndividualCases(population[0], compiledPopulation[0],
      candidatePopulation[0], unsafeAddr aggs_x, unsafeAddr aggs_y,
      unsafeAddr evalRankedY, AAA, SCORE_ROUNDS)
    for ci in 0 ..< checked.len:
      if abs(checked[ci] - caseScores[0][ci]) > 1.0e-10:
        raise newException(ValueError, "Survivor cache mismatch at case " & $ci)

  let previousChampionOnCurrent = if carriedCaseScores.len > 0 and
                                      caseScores[0].len == aggs_x.len:
                                   aggregateCaseScores(caseScores[0])
                                 else: -Inf

  # Compare parent and child on this generation's identical FULL cases,
  # before local weight optimization can change either member of the pair.
  if mutationProbes.len > 0:
    var successes = 0
    var failures = 0
    var meanDelta = 0.0
    var changedProbes = 0
    var regimeChanged: array[MutationRegime, int]
    var regimeWins: array[MutationRegime, int]
    var regimeLosses: array[MutationRegime, int]
    var regimeDelta: array[MutationRegime, float]
    for probe in mutationProbes:
      let regime = probe.regime
      if sameGenomeContent(population[probe.parent], population[probe.child]):
        continue
      inc changedProbes
      inc regimeChanged[regime]
      let delta = aggregateCaseScores(caseScores[probe.child]) -
                  aggregateCaseScores(caseScores[probe.parent])
      meanDelta += delta
      regimeDelta[regime] += delta
      if delta > 2.0e-4:
        inc successes
        inc regimeWins[regime]
      elif delta < -2.0e-4:
        inc failures
        inc regimeLosses[regime]

      # Paired FULL delta is an unbiased operator reward because parent and
      # child see identical cases and probes bypass FAST. Clip outliers so one
      # lucky rolling batch cannot monopolize reproduction for many generations.
      let adjusted = max(-0.10, min(0.10, delta))
      mutationRegimePreference[regime] =
        0.90 * mutationRegimePreference[regime] + 0.10 * adjusted

    # Most sparse rule mutations have *no rank effect* on a small batch.
    # A tie is not evidence that the mutation step is too large. The slow base
    # controller therefore reacts only when at least two probes are decisive;
    # the regime bandit handles local-vs-exploratory allocation separately.
    let decisive = successes + failures
    let successRate = float(successes) / float(max(1, decisive))
    if decisive >= 2:
      mutationSuccessEma = 0.94 * mutationSuccessEma + 0.06 * successRate
      feedbackMutationScale = max(0.72, min(1.28,
        feedbackMutationScale * exp(0.05 * (mutationSuccessEma - 0.20))))
    echo "  mutation_probe_success=", successRate,
         " paired_delta=", meanDelta / float(max(1, changedProbes)),
         " changed_probes=", changedProbes,
         " decisive=", decisive,
         " neutral=", changedProbes - decisive,
         " mutation_base_scale=", feedbackMutationScale
    for regime in MutationRegime:
      echo "    mutation_regime ", $regime,
           " changed=", regimeChanged[regime],
           " wins=", regimeWins[regime],
           " losses=", regimeLosses[regime],
           " paired_delta=", formatFloat(
             regimeDelta[regime] / float(max(1, regimeChanged[regime])), ffDecimal, 6),
           " preference=", formatFloat(mutationRegimePreference[regime], ffDecimal, 6),
           " generated=", mutationRegimeLastCounts[regime]

  # Paired FULL tests isolate crossover from mutation and case refresh.
  # The unchanged parent is in the survivor prefix; both IDs are compulsory
  # FULL, so adaptive scores are not selected by their FAST screening results.
  if crossoverProbes.len > 0:
    var changed: array[CrossoverOp, int]
    var gains: array[CrossoverOp, float]
    var wins: array[CrossoverOp, int]
    var neutrals: array[CrossoverOp, int]
    for probe in crossoverProbes:
      let op = probe.op
      if sameGenomeContent(population[probe.parent], population[probe.child]):
        inc neutrals[op]
        continue
      let delta = aggregateCaseScores(caseScores[probe.child]) -
                  aggregateCaseScores(caseScores[probe.parent])
      inc changed[op]
      gains[op] += delta
      if delta > 2.0e-4: inc wins[op]
      # Use crossover construction time as a small, measured resource penalty.
      # FULL worker evaluation time is shared/parallel and NOT attributed here.
      let adjusted = max(-0.10, min(0.10, delta)) /
        (1.0 + min(1.0, probe.seconds))
      crossoverPreference[op] = 0.90 * crossoverPreference[op] +
                                0.10 * adjusted
    for op in CrossoverOp:
      echo "  crossover_probe op=", $op,
           " changed=", changed[op], " wins=", wins[op],
           " neutral=", neutrals[op],
           " paired_delta=", formatFloat(gains[op] / float(max(1, changed[op])), ffDecimal, 6),
           " ema=", formatFloat(crossoverPreference[op], ffDecimal, 6),
           " generated=", crossoverLastCounts[op],
           " crossover_seconds=", formatFloat(crossoverLastSeconds[op], ffDecimal, 4)

  # Uniform rejection audit: the merit-only FAST/FULL correlation is
  # selection-biased. An independent random sample of FAST-rejected genomes
  # measures how often the screening missed candidates that beat the FULL
  # score of the last FAST-merit admission. Sampling happens AFTER initial
  # selection, and before the adaptive FULL expansion decision.
  var meritFullCutoff = Inf
  for oi in 0 ..< fullCount:
    let id = screeningOrder[oi]
    meritFullCutoff = min(meritFullCutoff, aggregateCaseScores(caseScores[id]))
  var initialFullBest = -Inf
  for id in fullIds:
    initialFullBest = max(initialFullBest, aggregateCaseScores(caseScores[id]))

  var auditCandidates: seq[int] = @[]
  for id in fastIds:
    if not fullSeen[id]:
      auditCandidates.add(id)
  let auditPopulation = auditCandidates.len
  shuffle(auditCandidates)
  var auditIds: seq[int] = @[]
  for i in 0 ..< min(FAST_REJECT_AUDIT_COUNT, auditCandidates.len):
    auditIds.add(auditCandidates[i])

  var auditMissedMerit = 0
  var auditMissedBest = 0
  var auditMaxMargin = -Inf
  if auditIds.len > 0:
    var auditTimings: seq[seq[float]]
    let auditResults = parallelEvaluateCasesBatch(
      population, compiledPopulation, candidatePopulation, auditIds,
      unsafeAddr aggs_x, unsafeAddr aggs_y, unsafeAddr evalRankedY, AAA, SCORE_ROUNDS,
      "FAST_REJECTION_AUDIT", addr auditTimings)
    for i, id in auditIds:
      caseTimings[id] = auditTimings[i]
      let actual = aggregateCaseScores(auditResults[i])
      if meritFullCutoff < Inf:
        let margin = actual - meritFullCutoff
        auditMaxMargin = max(auditMaxMargin, margin)
        if margin > FAST_AUDIT_MISS_MARGIN:
          inc auditMissedMerit
      if actual > initialFullBest + 2.0e-4:
        inc auditMissedBest
      caseScores[id] = auditResults[i]
      fullSeen[id] = true
      fullIds.add(id)  # Rescue the audited individuals into genuine selection.
  echo "  fast_rejection_audit: sampled=", auditIds.len,
       " remaining=", auditPopulation,
       " missed_merit=", auditMissedMerit,
       " missed_best=", auditMissedBest,
       " max_merit_margin=", auditMaxMargin

  # FASTがFULLの順位をどの程度保存しているかを毎世代実測する。
  # 相関が悪い世代だけFULL枠を追加し、次世代のFAST round数も少し増やす。
  # これにより固定ハイパラで「速いが誤選別」か「常に重い」の二択にしない。
  proc screeningCorrelation(ids: openArray[int]): float =
    if ids.len < 4:
      return 1.0
    var fastValues: seq[float] = @[]
    var fullValues: seq[float] = @[]
    for id in ids:
      # Incumbents/probes were deliberately not FAST-evaluated. Compare
      # only paired observations actually measured by both evaluators.
      if id >= 0 and id < fastCaseScores.len and id < caseScores.len and
         fastCaseScores[id].len > 0 and fastCaseScores[id].len == caseScores[id].len:
        fastValues.add(aggregateCaseScores(fastCaseScores[id]))
        fullValues.add(aggregateCaseScores(caseScores[id]))
    if fastValues.len < 4:
      1.0
    else:
      spearman(fastValues, fullValues)

  var screeningCorr = screeningCorrelation(fullIds)
  var extraTarget = 0
  if screeningCorr < 0.58:
    extraTarget = ADAPTIVE_FULL_EXTRA_MAX
  elif screeningCorr < 0.70:
    extraTarget = min(16, ADAPTIVE_FULL_EXTRA_MAX)
  elif screeningCorr < 0.80:
    extraTarget = min(8, ADAPTIVE_FULL_EXTRA_MAX)
  # A candidate that FAST actually rejected but FULL found superior is
  # evidence of missed coverage, even when the merit-only correlation is high.
  # Do not spend the entire adaptive budget on ONE marginal miss: with an
  # 8-sample audit that event is too noisy to justify 8 additional FULL runs.
  if auditMissedMerit > 0:
    extraTarget = max(extraTarget,
      min(ADAPTIVE_FULL_EXTRA_MAX, 2 + 2 * auditMissedMerit))
  if auditMissedBest > 0:
    extraTarget = ADAPTIVE_FULL_EXTRA_MAX

  var adaptiveExtraIds: seq[int] = @[]
  if extraTarget > 0:
    # 半分はFASTの次点、半分は順位帯に偏らないランダムprobe。
    let meritTarget = (extraTarget + 1) div 2
    for id in screeningOrder:
      if adaptiveExtraIds.len >= meritTarget:
        break
      if not fullSeen[id]:
        fullSeen[id] = true
        adaptiveExtraIds.add(id)

    var attempts = 0
    while adaptiveExtraIds.len < extraTarget and attempts < pop_size * 4:
      inc attempts
      let id = rand(pop_size - 1)
      if not fullSeen[id]:
        fullSeen[id] = true
        adaptiveExtraIds.add(id)

    if adaptiveExtraIds.len > 0:
      var extraTimings: seq[seq[float]]
      let extraResults = parallelEvaluateCasesBatch(
        population, compiledPopulation, candidatePopulation, adaptiveExtraIds,
        unsafeAddr aggs_x, unsafeAddr aggs_y, unsafeAddr evalRankedY, AAA,
        SCORE_ROUNDS, "ADAPTIVE_FULL", addr extraTimings)
      for i, id in adaptiveExtraIds:
        caseScores[id] = extraResults[i]
        caseTimings[id] = extraTimings[i]
        fullIds.add(id)
      screeningCorr = screeningCorrelation(fullIds)

  # The independent audit is included in FULL-ranked selection and in the
  # paired FAST/FULL correlation above; its observed misses also expand FULL.
  # FAST has two independent cost knobs: temporal rounds and sample stride.
  # Higher stride means fewer sampled observations. A single small merit miss is
  # deliberately treated as a HOLD signal rather than immediately forcing the
  # old dense stride=3 behavior. Genuine best misses / repeated or large misses
  # still make screening denser immediately.
  let previousFastStride = adaptiveFastEvalStride
  let severeAuditMiss = auditMissedBest > 0 or auditMissedMerit >= 2 or
    (auditMissedMerit > 0 and auditMaxMargin > FAST_AUDIT_SEVERE_MARGIN)
  if severeAuditMiss:
    adaptiveFastEvalRounds = min(FAST_EVAL_MAX_ROUNDS, adaptiveFastEvalRounds + 2)
    adaptiveFastEvalStride = max(FAST_EVAL_STRIDE_MIN, adaptiveFastEvalStride - 1)
  elif screeningCorr < 0.66:
    adaptiveFastEvalRounds = min(FAST_EVAL_MAX_ROUNDS, adaptiveFastEvalRounds + 2)
    adaptiveFastEvalStride = max(FAST_EVAL_STRIDE_MIN, adaptiveFastEvalStride - 1)
  elif screeningCorr < 0.76:
    adaptiveFastEvalRounds = min(FAST_EVAL_MAX_ROUNDS, adaptiveFastEvalRounds + 1)
    if auditMissedMerit > 0:
      adaptiveFastEvalStride = max(FAST_EVAL_STRIDE_MIN, adaptiveFastEvalStride - 1)
  elif auditMissedMerit == 1:
    # One modest rejection-audit miss with otherwise decent rank correlation is
    # not enough evidence to pay for a denser dataset next generation.
    discard
  elif screeningCorr > 0.93:
    adaptiveFastEvalRounds = max(FAST_EVAL_MIN_ROUNDS, adaptiveFastEvalRounds - 2)
    adaptiveFastEvalStride = min(FAST_EVAL_STRIDE_MAX, adaptiveFastEvalStride + 1)
  elif screeningCorr > 0.86:
    adaptiveFastEvalRounds = max(FAST_EVAL_MIN_ROUNDS, adaptiveFastEvalRounds - 1)
    adaptiveFastEvalStride = min(FAST_EVAL_STRIDE_MAX, adaptiveFastEvalStride + 1)

  if adaptiveFastEvalStride != previousFastStride:
    let nextFast = buildFastEvalDataset(evalAggsX, evalAggsY, adaptiveFastEvalStride)
    fastAggsX = nextFast.x
    fastAggsY = nextFast.y
    fastRankedY = buildRankedY(fastAggsY)

  refreshAdaptiveLexicaseEpsilon(caseScores, fullIds)

  var fastPointsSummary = "["
  for ci, xs in fastAggsX:
    if ci > 0: fastPointsSummary.add(",")
    fastPointsSummary.add($xs.len)
  fastPointsSummary.add("]")
  echo "  screening fast/full spearman=", formatFloat(screeningCorr, ffDecimal, 3),
       " adaptive_extra=", adaptiveExtraIds.len,
       " fast_rounds_next=", adaptiveFastEvalRounds,
       " fast_stride_next=", adaptiveFastEvalStride,
       " fast_points_next=", fastPointsSummary,
       " lex_eps=", adaptiveLexicaseEpsilon

  for jp in fullIds:
    accs[jp] = aggregateCaseScores(caseScores[jp])
    bar.increment()

  # FULL 評価対象外も「評価済み」として扱うため、進捗バーを埋める。
  for _ in fullIds.len ..< pop_size:
    bar.increment()
  bar.finish()

  let tAfterFullEval = epochTime()

  # IMPORTANT: FAST-only individuals must never participate in final ranking.
  # Their score is only a screening estimate and can be higher than a true FULL score.
  # Rank only the individuals that actually received FULL evaluation.
  var fullScores = newSeq[float](fullIds.len)
  for oi in 0 ..< fullIds.len:
    fullScores[oi] = accs[fullIds[oi]]
  let fullOrderLocal = sortIndicesByScore(fullScores)

  var order: seq[int] = @[]
  for oi in fullOrderLocal:
    order.add(fullIds[oi])

  if order.len == 0:
    # Defensive fallback; protected elites should normally make this impossible.
    for i in 0 ..< pop_size:
      order.add(i)

  var bestScore = accs[order[0]]

  # 数世代に1回だけ、best個体のweightをcredit-guidedに局所探索する。
  # a/b構造は触らないのでコンパイル済みpatternをそのまま再利用できる。
  if stagnation >= 4 and
     (iter mod WEIGHT_LOCAL_SEARCH_INTERVAL == 0 or
      (stagnation >= WEIGHT_LOCAL_SEARCH_STAGNATION_TRIGGER and
       iter mod WEIGHT_LOCAL_SEARCH_STAGNANT_INTERVAL == 0)):
    let bestIdx = order[0]
    let beforeLocalGenome = cloneGenome(population[bestIdx])
    let beforeLocalCases = caseScores[bestIdx]
    let beforeLocalTimings = caseTimings[bestIdx]
    let beforeLocalScore = bestScore
    var improved = bestScore
    try:
      improved = optimizeWeightsByCredit(
        population[bestIdx],
        compiledPopulation[bestIdx],
        candidatePopulation[bestIdx],
        aggs_x,
        aggs_y,
        evalRankedY,
        bestScore
      )
    except EvaluationBudgetExceeded:
      population[bestIdx] = beforeLocalGenome
      echo "  weight local search skipped: evaluation budget"
    if improved > bestScore + 1.0e-9:
      # optimizeWeightsByCreditはaggregate scoreだけ返すので、caseScoresも必ず
      # 現在のweightで再評価して同期する。そうしないと親選択だけ古いcase scoreを使う。
      var updatedTimings: seq[float] = @[]
      caseScores[bestIdx] = safeEvaluateIndividualCases(
        population[bestIdx],
        compiledPopulation[bestIdx],
        candidatePopulation[bestIdx],
        unsafeAddr aggs_x,
        unsafeAddr aggs_y,
        unsafeAddr evalRankedY,
        AAA,
        SCORE_ROUNDS,
        addr updatedTimings
      )
      # Weight changes can change 2-gram stopping time. Keep the measured
      # FULL runtime aligned with the EXACT genome used by the Pareto filter.
      caseTimings[bestIdx] = updatedTimings
      accs[bestIdx] = aggregateCaseScores(caseScores[bestIdx])
      if accs[bestIdx] < beforeLocalScore or accs[bestIdx] != accs[bestIdx]:
        population[bestIdx] = beforeLocalGenome
        caseScores[bestIdx] = beforeLocalCases
        caseTimings[bestIdx] = beforeLocalTimings
        accs[bestIdx] = beforeLocalScore
        echo "  local search rejected: FULL objective regression"
      bestScore = accs[bestIdx]
      # Re-sort only the FULL-evaluated pool.
      var rescored = newSeq[float](order.len)
      for oi, idx in order:
        rescored[oi] = accs[idx]
      let rescoredOrder = sortIndicesByScore(rescored)
      var newOrder: seq[int] = @[]
      for oi in rescoredOrder:
        newOrder.add(order[oi])
      order = newOrder
      bestScore = accs[order[0]]
      echo "  weight local search: ", bestScore

  # Alternate normal GA evaluation with actual generated-text evaluation.
  # Only CURRENT FULL-scored models participate; no historical FAST scores.
  var jevRankedIds: seq[int] = @[]
  var jevMetricUpdated = false
  var jevWallSeconds = 0.0
  if jevEnabled and (iter + 1) mod JEV_GENERATION_INTERVAL == 0:
    echo "  jev: FULL evaluation completed; choosing generation candidates"
    # Build the observation cohort ONLY from genuinely FULL-scored models, but
    # do not require them all to be near the GA champion. 64 slots let us keep
    # a merit lane AND a broad score-stratified exploration lane. Jev scoring is
    # observational here: no Jev value is ever written into `accs`.
    var isFull = newSeq[bool](pop_size)
    for id in fullIds:
      if id >= 0 and id < pop_size: isFull[id] = true
    var fullOrder: seq[int] = @[]
    for id in order:
      if id >= 0 and id < pop_size and isFull[id]: fullOrder.add(id)
    let fullBeforeDedup = fullOrder.len
    var fullFp: seq[Hash] = @[]
    fullOrder = uniqueGenomeIds(population, fullOrder, fullFp)
    echo "  jev-cohort: duplicate_full_genomes_skipped=", fullBeforeDedup-fullOrder.len
    let target = min(jevModelCount, fullOrder.len)
    var jevCandidates: seq[int] = @[]
    var selected = initHashSet[int]()

    # Exact anchors and pending offspring precede merit selection. No parent
    # score is assigned to an offspring. Hash collisions are resolved by content.
    var hallCandidates: seq[int] = @[]
    for anchor in fusionAnchors:
      if jevCandidates.len >= target: break
      for id in fullOrder:
        if id notin selected and fullFp[id] == anchor.fingerprint and
           sameGenomeContent(population[id], anchor.genome):
          hallCandidates.add(id)
          selected.incl(id)
          jevCandidates.add(id)
          break
    for pending in fusionPending:
      let pendingFp = genomeFingerprint(pending.genome)
      for id in fullOrder:
        if jevCandidates.len >= target: break
        if id notin selected and fullFp[id] == pendingFp and
           sameGenomeContent(population[id], pending.genome):
          selected.incl(id)
          jevCandidates.add(id)
          break
    for member in fusionArchive:
      for id in fullOrder:
        if jevCandidates.len >= target: break
        if id notin selected and fullFp[id] == member.fingerprint and
           sameGenomeContent(population[id], member.genome):
          selected.incl(id)
          jevCandidates.add(id)
          break

    # Merit lane: at most half the cohort. This protects ordinary GA winners
    # without allowing the GA score to monopolize the Jev observation set.
    let meritTarget = min(target - jevCandidates.len, max(1, target div 2))
    var meritAdded = 0
    for id in fullOrder:
      if meritAdded >= meritTarget: break
      if id notin selected:
        jevCandidates.add(id)
        selected.incl(id)
        inc meritAdded

    # Exploration lane: evenly sample the ENTIRE remaining FULL ranking. This
    # is cheap, deterministic, and prevents 64 Jev evaluations from collapsing
    # onto one narrow high-GA lineage.
    let remaining = target - jevCandidates.len
    if remaining > 0:
      var remainder: seq[int] = @[]
      for id in fullOrder:
        if id notin selected: remainder.add(id)
      for j in 0 ..< min(remaining, remainder.len):
        let pos = if remaining <= 1: 0
                  else: (j * (remainder.len - 1)) div max(1, remaining - 1)
        let id = remainder[pos]
        if id notin selected:
          jevCandidates.add(id)
          selected.incl(id)
    # Defensive fill for quantile collisions/short lists.
    if jevCandidates.len < target:
      for id in fullOrder:
        if jevCandidates.len >= target: break
        if id notin selected:
          jevCandidates.add(id)
          selected.incl(id)
    if hallCandidates.len > 0:
      echo "  jev-hall: scheduled_repeat_candidates=", hallCandidates.len
    echo "  jev-cohort: merit=", meritAdded,
         " hall_repeat=", hallCandidates.len,
         " broad_explore=", jevCandidates.len - meritAdded - hallCandidates.len
    echo "  jev: selected ", jevCandidates.len, "/", jevModelCount,
         " FULL candidates for full Jev scoring and KwikSort"
    let jevStarted = epochTime()
    let jevResult = jevRun(iter, jevPrompt, jevModel, jevBridge,
      jevCandidates, population, compiledPopulation, candidatePopulation,
      jevAlphabet, jevCumulative, jevGrams, jevGenerationWorkers)
    jevWallSeconds = epochTime() - jevStarted
    if jevResult.valid:
      # Bridge already validates uniqueness and id range. Index the quality
      # scores once instead of linearly scanning up to 32 items for each rank.
      var qualityById = newSeq[float](pop_size)
      for individual in jevResult.qualities:
        qualityById[individual.id] = individual.quality
      var oldRepeat, newRepeat: seq[float] = @[]
      for anchor in fusionAnchors:
        for id in jevCandidates:
          if fullFp[id] == anchor.fingerprint and
             sameGenomeContent(population[id], anchor.genome):
            oldRepeat.add(anchor.quality)
            newRepeat.add(qualityById[id])
            break
      fusionUpdateTrust(oldRepeat, newRepeat)
      var points: seq[FusionPoint] = @[]
      var uniqueGenomes: seq[int] = @[]
      # Cohort was already deduplicated with exact collision checks.
      for id in jevCandidates:
        uniqueGenomes.add(id)
        points.add(FusionPoint(id: id, g: accs[id], j: qualityById[id]))
      let front = fusionFront(points, FUSION_CAP)
      # All archive points use THIS cohort's J and THIS generation's FULL G.
      fusionArchive.setLen(0)
      for pos in front:
        let point = points[pos]
        fusionArchive.add(fusionSnapshot(population[point.id], point.j, iter))
        jevRankedIds.add(point.id)
      # Rotate a broad score-stratified panel, never just selected winners.
      # Sorting the same unique set by quality balances the observed range.
      uniqueGenomes.sort(proc(a,b: int): int = cmp(qualityById[a], qualityById[b]))
      fusionAnchors.setLen(0)
      let anchorCount = min(FUSION_ANCHORS, uniqueGenomes.len)
      for k in 0..<anchorCount:
        let pos = if anchorCount < 2: 0
                  else: k*(uniqueGenomes.len-1) div (anchorCount-1)
        let id = uniqueGenomes[pos]
        fusionAnchors.add(fusionSnapshot(population[id], qualityById[id], iter))
      fusionPending.setLen(0)
      echo "  jev-fusion: trust=", formatFloat(fusionReliability, ffDecimal, 3),
           " repeat_pairs=", oldRepeat.len, " evidence_rounds=", fusionEvidence,
           " repeat_noise=", fusionNoise, " front=", fusionArchive.len
      var topQualitySum = 0.0
      var topQualityCount = 0
      for rank, id in jevResult.ranked:
        if rank < JEV_HALL_TOP_FINISH:
          topQualitySum += qualityById[id]
          inc topQualityCount
      let topQuality = if topQualityCount > 0: topQualitySum / float(topQualityCount) else: 0.0
      # Give each candidate's externally scored quality a LINEAR rank weight.
      # Rank 1 gets N, rank N gets 1 (for N candidates: N, N-1, ..., 1).
      # The mean is computed within this successful Jev round; MA20 then
      # averages these round-level weighted means across the last 20 observations.
      # Neither the Jev ranking used for selection nor the GA fitness changes.
      var rankWeightedSum = 0.0
      var rankWeightTotal = 0.0
      for rank, id in jevResult.ranked:
        let weight = float(jevResult.ranked.len - rank)
        rankWeightedSum += weight * qualityById[id]
        rankWeightTotal += weight
      let rankWeightedQuality = rankWeightedSum / rankWeightTotal
      let jevMA = jevAverage(jevHistory, rankWeightedQuality)
      jevHistory.add(JevPoint(generation: iter,
        quality: rankWeightedQuality, average: jevMA))
      writeJevProgress(jevProgressPath, jevHistory)
      jevMetricUpdated = true
      echo "  jev: quality_rank_weighted=", rankWeightedQuality,
           " cohort_mean=", jevResult.meanQuality,
           " top8_quality=", topQuality,
           " rank_weighted_ma20=", jevMA, " ranked=", jevRankedIds.len,
           " comparisons=", jevResult.pairComparisons,
           " requests=", jevResult.apiRequests,
           " local_evaluations=", jevResult.localEvaluations,
           " actual_score_calls=", jevResult.actualScoreCalls,
           " cache_hits=", jevResult.cacheHits,
           " budget_cutoffs=", jevResult.budgetCutoffs,
           " score_cpu_s_sum=", formatFloat(jevResult.scoreSeconds, ffDecimal, 2),
           " max_score_ms=", formatFloat(jevResult.maxScoreSeconds * 1000.0, ffDecimal, 1),
           " seconds=", formatFloat(jevWallSeconds, ffDecimal, 3)
    else:
      fusionReliability *= 0.5
      fusionPending.setLen(0) # no indefinite protection after API failure
      echo "  jev: unavailable; trust decayed; ordinary GA remains active"

  # Exact telescoping decomposition, not an accumulated 'corrected' score:
  # best_now - best_previous = dataset_delta + selection_gain.
  # The same previous champion supplies both sides of dataset_delta.
  if previousChampionScore > -Inf and previousChampionOnCurrent > -Inf:
    let datasetDelta = previousChampionOnCurrent - previousChampionScore
    let selectionGain = bestScore - previousChampionOnCurrent
    let observedDelta = bestScore - previousChampionScore
    if selectionGain < -1.0e-9:
      raise newException(ValueError,
        "Elitism invariant violated: previous champion beats selected winner")
    echo "  champion_change: dataset_delta=", datasetDelta,
         " selection_gain=", selectionGain, " observed_delta=", observedDelta
    let driftPath = "progress_diagnostics_v13.csv"
    let needsHeader = not fileExists(driftPath) or getFileSize(driftPath) == 0
    var driftFile = open(driftPath, fmAppend)
    try:
      if needsHeader:
        driftFile.writeLine("generation,previous_champion_old,previous_champion_current,best_current,dataset_delta,selection_gain,observed_delta,refresh_fast,refresh_medium,refresh_slow")
      driftFile.writeLine($iter & "," & $previousChampionScore & "," &
        $previousChampionOnCurrent & "," & $bestScore & "," & $datasetDelta & "," &
        $selectionGain & "," & $observedDelta & "," & $refreshedFast & "," &
        $refreshedMedium & "," & $refreshedSlow)
    finally:
      driftFile.close()

  # The previous teacher mines examples only for refreshed rolling cases.
  # No extra per-individual mining or fixed corpus anchor is introduced.
  if iter mod 32 == 0 or miningGenome.len == 0:
    miningGenome = cloneGenome(population[order[0]])
    miningCompiled = newSeq[SeqPattern](miningGenome.len)
    for i, rule in miningGenome: miningCompiled[i] = compileSeqPattern(rule.a)
    miningCandidates = buildCandidateIndex(miningCompiled)
  if iter mod CHECKPOINT_INTERVAL == 0 or iter == TARGET_GENERATIONS - 1:
    exportEnergyModel(checkpointPath & ".model", population[order[0]])

  echo "gen=", iter, " best=", bestScore,
       " score_rounds=ceil(2*sqrt(input.len))", " rules=", AAA

  # 今世代のベストスコアを記録し、一定間隔でグラフを更新する。
  block:
    let rawTransformed = spearmanToSigma(bestScore)

    # Plot/history is updated now; admission below deliberately examines the
    # PREVIOUS 75 generations before pushing this generation to the MA ring.
    progressHistory.add(ProgressPoint(
      iter: iter,
      best: bestScore,
      transformed: rawTransformed,
      sqrtMovingAvg: 0.0,
      fixedMovingAvg: 0.0
    ))

    # 新しく確定した200/200MAの点だけ更新する。
    updatePlotMovingAverages(progressHistory)

    # Keep one CSV point per generation; throttle only expensive PNG rendering.
    writeProgressCsv("progress.csv", progressHistory)
    # After writing both CSVs, refresh every 5 generations and immediately
    # on a newly successful Jev observation. Avoid unnecessary plot work when
    # the API fails. The first full GA MA window is available at iter 199.
    # Synchronous atomic rendering prevents a previous generation from clobbering this one.
    if ((iter + 1) mod PLOT_INTERVAL == 0 or jevMetricUpdated) and
       progressHistory.len >= PLOT_MA_SHORT_WINDOW:
      plotProgress()

  # The paired comparison is diagnostic only. Admission is STRICTLY based on
  # the champion exceeding mean + 1 stddev of PREVIOUS log-transformed scores.
  # Rolling dataset changes can still affect this historical score baseline.
  var incumbentBest = -Inf
  for id in 0 ..< protectedCount:
    incumbentBest = max(incumbentBest, accs[id])
  # Paired incumbent-vs-champion delta is comparable even if a rolling
  # case was refreshed this generation; unlike the plot, it does not compare
  # scores obtained on DIFFERENT datasets. A negative value cannot normally
  # occur because protected incumbents are included in the FULL ranking.
  let pairedGain = if incumbentBest > -Inf: bestScore - incumbentBest else: 0.0
  echo "  paired_incumbent_gain=", pairedGain,
       " refreshed_fast=", refreshedFast,
       " refreshed_medium=", refreshedMedium,
       " refreshed_slow=", refreshedSlow,
       " fast_screened=", fastIds.len,
       " full_evaluated=", fullIds.len
  let transformedBest = spearmanToSigma(bestScore)
  let archiveGate = archiveLogThreshold()
  let shouldArchive = archiveGate.ready and transformedBest > archiveGate.value
  echo "  archive_gate: ready=", archiveGate.ready,
       " window=", maCount, "/", MA_WINDOW,
       " log_score=", transformedBest,
       " log_mean=", archiveGate.mean,
       " log_stddev=", archiveGate.stddev,
       " log_threshold=", archiveGate.value,
       " qualified=", shouldArchive
  # This observation affects NEXT generation's threshold, never its own.
  maPush(transformedBest)

  # stagnationを「rolling caseの難易度差」で暴れさせない。
  # slow caseは32世代だけ同一なので、そのepoch内でのbest改善だけを比較する。
  # slow case交換直後の最初の世代はbaseline設定のみで、stagnationは増減させない。
  var slowChampionScore = -Inf
  if SLOW_ROLLING_SLOT < MULTI_CASE_COUNT:
    for id in fullIds:
      if id >= 0 and id < caseScores.len and
         caseScores[id].len > SLOW_ROLLING_SLOT:
        var slowSum = 0.0
        var slowCount = 0
        for ci in countup(SLOW_ROLLING_SLOT, caseScores[id].len - 1, 3):
          slowSum += caseScores[id][ci]
          inc slowCount
        if slowCount > 0:
          slowChampionScore = max(slowChampionScore, slowSum / float(slowCount))

  var slowImproved = false
  if slowChampionScore > -Inf:
    if not slowEpochInitialized:
      slowEpochBest = slowChampionScore
      slowEpochInitialized = true
    # 48前後のrank点でのSpearmanは微小な順位入替えでも1e-4級に動く。
    # 1e-6でリセットすると実質ノイズでも「改善」扱いになり、mutation pressureが
    # ほとんど育たないため、意味のある最小改善量を要求する。
    elif slowChampionScore > slowEpochBest + 2.0e-4:
      slowEpochBest = slowChampionScore
      # 改善1回で探索圧をゼロへ落とすと mutation regime が鋸歯状になる。
      # 数世代の連続改善で自然に低圧へ戻す。
      stagnation = max(0, stagnation - 4)
      slowImproved = true
    else:
      inc stagnation

  # bestEverはrolling dataset間の絶対比較には使わない。保存形式互換のため値は保持し、
  # 現在値の上限だけ記録するが、mutation/stagnation制御には一切使わない。
  if bestScore > bestEver:
    bestEver = bestScore

  if shouldArchive:
    archiveRecord(bestScore, population[order[0]], iter)
    echo "  ★ Elite-of-Elites archived: size=", eliteOfElites.len,
         " score=", bestScore, " admission=log-mean-plus-1stddev",
         " threshold_rho=", sigmaToProbability(archiveGate.value),
         " slow_improved=", slowImproved,
         " generation=", iter

  # FULL-measured, identical-case runtime; FAST-only timings are never mixed.
  var fullEvalSeconds = newSeq[float](pop_size)
  for id in fullIds:
    var elapsed = 0.0
    for seconds in caseTimings[id]: elapsed += seconds
    fullEvalSeconds[id] = max(1.0e-6, elapsed)
  let paretoPool = looseRuntimeParetoPool(fullIds, accs, fullEvalSeconds)
  var paretoFastest = order[0]
  for id in paretoPool:
    if fullEvalSeconds[id] < fullEvalSeconds[paretoFastest]: paretoFastest = id
  echo "  runtime_pareto: candidates=", fullIds.len,
       " near_best_front=", paretoPool.len,
       " champion_seconds=", formatFloat(fullEvalSeconds[order[0]], ffDecimal, 5),
       " fastest_front_seconds=", formatFloat(fullEvalSeconds[paretoFastest], ffDecimal, 5),
       " score_slack=", PARETO_FITNESS_SLACK,
       " parent_share=", PARETO_PARENT_PERCENT, "%"

  # best 1体だけをarchiveへ入れると、別系統の優秀building blockが失われる。
  # 今世代の上位候補から、同じ世代内で互いに離れた個体も少数保存する。
  if shouldArchive and order.len > 1:
    var archivedThisGeneration: seq[int] = @[order[0]]
    let extraLimit = min(3, order.len - 1)
    for oi in 1 ..< min(order.len, 37):
      if archivedThisGeneration.len > extraLimit:
        break
      let candidateId = order[oi]
      # Runner-ups must individually pass the same strict admission gate;
      # being close to a qualifying champion alone is insufficient.
      if spearmanToSigma(accs[candidateId]) <= archiveGate.value:
        continue
      if accs[candidateId] < bestScore - 0.01:
        continue
      var minD = Inf
      for sid in archivedThisGeneration:
        minD = min(minD, localGenomeDistance(population[candidateId], population[sid]))
      if minD < 0.45:
        continue
      archiveRecord(accs[candidateId], population[candidateId], iter)
      archivedThisGeneration.add(candidateId)

  # 数世代に1回だけ、今世代bestのrule-level creditを測る。
  # これは「各ruleを全消去して再評価」という高コストな厳密アブレーションではなく、
  # 通常の変換過程から寄与を統計的に抽出する軽量診断。
  if iter mod CREDIT_REFRESH_INTERVAL == 0 or
     (stagnation >= CREDIT_STAGNATION_TRIGGER and
      iter mod CREDIT_STAGNANT_REFRESH_INTERVAL == 0):
    try:
      refreshGuidedModules(
        population[order[0]],
        compiledPopulation[order[0]],
        candidatePopulation[order[0]],
        aggs_x,
        aggs_y,
        evalRankedY
      )
      echo "  guided modules=", guidedModules.len
    except EvaluationBudgetExceeded:
      guidedModules.setLen(0)
      echo "  guided modules skipped: evaluation budget"

  var new_population = newSeqOfCap[Genome](pop_size)

  # --------------------------
  # 1. 現世代Elitism
  # --------------------------
  let keepElites = min(eliteCount, pop_size)
  # 37枠のうち20体はaggregate上位をそのまま保護し、case championを追加。
  # rolling datasetの交換直後に良系統を一気に薄めないため、旧12体から少し厚くする。
  # 残り枠は従来どおり構造多様性に使う。
  let coreEliteCount = min(20, min(keepElites, order.len))
  var survivorIds: seq[int] = @[]
  # Preserve distinct complete genomes, including weight-only improvements.
  var eliteFingerprints = newSeq[Hash](pop_size)
  for id in order: eliteFingerprints[id] = genomeFingerprint(population[id])
  proc duplicatesSurvivor(id: int): bool =
    for sid in survivorIds:
      if eliteFingerprints[id] == eliteFingerprints[sid] and
         sameGenomeContent(population[id], population[sid]): return true
    false
  for id in order:
    if survivorIds.len >= coreEliteCount: break
    if not duplicatesSurvivor(id): survivorIds.add(id)

  var survivorUsed = newSeq[bool](pop_size)
  for id in survivorIds:
    survivorUsed[id] = true
  # Protect each rolling-case champion as well as the aggregate leaders.
  # These occupy existing elite slots and receive FULL evaluation next generation.
  for caseId in 0 ..< min(aggs_x.len, MULTI_CASE_COUNT):
    if survivorIds.len >= keepElites:
      break
    var championId = -1
    var championScore = -Inf
    for id in order:
      if caseId < caseScores[id].len and caseScores[id][caseId] > championScore:
        championScore = caseScores[id][caseId]
        championId = id
    if championId >= 0 and not survivorUsed[championId] and
       not duplicatesSurvivor(championId):
      survivorUsed[championId] = true
      survivorIds.add(championId)

  # Two light Pareto preservation slots, AFTER core elites and case specialists.
  # Never replace champion or a case champion; remaining slots still favor diversity.
  var paretoPreserved = 0
  var fastPareto = @paretoPool
  fastPareto.sort(proc(a, b: int): int =
    cmp(fullEvalSeconds[a], fullEvalSeconds[b]))
  for id in fastPareto:
    if paretoPreserved >= 2 or survivorIds.len >= keepElites: break
    if not survivorUsed[id] and not duplicatesSurvivor(id) and
       accs[id] >= bestScore - PARETO_FITNESS_SLACK:
      survivorUsed[id] = true
      survivorIds.add(id)
      inc paretoPreserved

  let worstFull = accs[order[^1]]
  let fitnessRange = max(1.0e-9, bestScore - worstFull)
  while survivorIds.len < keepElites:
    var bestId = -1
    var bestValue = -Inf
    for id in order:
      if survivorUsed[id] or duplicatesSurvivor(id):
        continue
      var minD = Inf
      for sid in survivorIds:
        minD = min(minD, localGenomeDistance(population[id], population[sid]))
      # diversityを主軸にしつつ、極端に低fitnessな個体がelite枠を
      # 取り続けないよう0..1正規化したfitnessを少量だけ加える。
      let normalizedFitness = max(0.0, min(1.0, (accs[id] - worstFull) / fitnessRange))
      let value = minD + 0.35 * normalizedFitness
      if value > bestValue:
        bestValue = value
        bestId = id
    if bestId < 0:
      break
    survivorUsed[bestId] = true
    survivorIds.add(bestId)

  for idx in survivorIds:
    new_population.add(cloneGenome(population[idx]))

  # --------------------------
  # 2. Elite-of-Elites の永久系統
  # --------------------------
  # archive 全体からランダムに選んで注入する。
  # スコア順固定にすると常に同じ少数の個体へ収束しやすいため、
  # 古いが異なる探索経路を持つeliteも再投入できるようにする。
  let archiveRestartLane = stagnation >= ARCHIVE_STAGNATION_TRIGGER
  if (iter mod ARCHIVE_INJECT_INTERVAL == 0 or archiveRestartLane) and
     eliteOfElites.len > 0 and new_population.len < pop_size:
    let requestedArchiveInject =
      if archiveRestartLane: ARCHIVE_INJECT_STAGNANT else: ARCHIVE_INJECT_BASE
    let injectTarget = min(requestedArchiveInject, pop_size - new_population.len)
    var injected = 0
    var attempts = 0
    let maxAttempts = max(32, injectTarget * 8)

    # ★変更: 以前はここで候補ごとに survivorIds / 既注入分との
    # 全genome(AAA=3100行)比較を sameRule で逐一行っており、
    # 世代あたり最大 maxAttempts×(survivorIds数+injectTarget)×AAA 回の
    # rule比較が発生していた(hidden bottleneck)。
    # survivorIdsのfingerprintは候補を引く前に1回だけ計算しておき、
    # 以降はuint64/Hash同士のO(1)比較で済ませる。
    # 衝突(fingerprint一致だが中身が違う)は理論上あり得るため、
    # 一致した場合だけ sameGenomeContent で厳密確認する。
    var survivorFingerprints = newSeq[Hash](survivorIds.len)
    for i, sid in survivorIds:
      survivorFingerprints[i] = genomeFingerprint(population[sid])
    # 今回のラウンドで実際に注入したarchiveエントリのindexだけ覚えておく。
    # fingerprintが一致した場合のみ、元genome同士をsameGenomeContentで
    # 厳密確認する(fingerprintは衝突し得るため)。
    var seenArchiveIndices: seq[int] = @[]

    while injected < injectTarget and attempts < maxAttempts:
      inc attempts
      let ai = randomArchiveIndex()
      if ai < 0:
        break

      let candFp = eliteOfElites[ai].fingerprint
      let candGenome = eliteOfElites[ai].genome

      # 同一(a,b)構造の完全重複個体を何体も注入しない。
      var duplicate = false
      for i, sid in survivorIds:
        if survivorFingerprints[i] == candFp and
           sameGenomeContent(population[sid], candGenome):
          duplicate = true
          break
      if duplicate:
        continue
      for seenAi in seenArchiveIndices:
        if eliteOfElites[seenAi].fingerprint == candFp and
           sameGenomeContent(eliteOfElites[seenAi].genome, candGenome):
          duplicate = true
          break

      if duplicate:
        continue

      new_population.add(cloneGenome(candGenome))
      seenArchiveIndices.add(ai)
      inc injected

  # Adaptive archive and pending children cannot displace ordinary GA elites.
  # Inject exact calibration anchors only in the generation before Jev.
  if jevEnabled:
    fusionInject(new_population, iter + 1, pop_size)

  # --------------------------
  # 2. 子個体生成
  # --------------------------
  var parentUseCount = newSeq[int](pop_size)

  # FULL親poolの実効多様性を少数pairで測る。絶対距離そのものではなく、
  # elite archiveでも使っている0.45閾値を越えるpair比率にすることで
  # genome長や局所距離スケールへの依存を減らす。
  # Exact duplicate genotypes must not receive multiple lottery tickets merely
  # because they occupy several population slots. One representative is enough
  # for parent selection; this raises effective population size at no fitness cost.
  var parentFingerprints: seq[Hash] = @[]
  let parentPool = uniqueGenomeIds(population, fullIds, parentFingerprints)
  var paretoFingerprints: seq[Hash] = @[]
  let uniqueParetoPool = uniqueGenomeIds(population, paretoPool, paretoFingerprints)
  var jevDonorPool: seq[int] = @[]
  if jevEnabled and fusionReliability > 0.0:
    var currentPoints: seq[FusionPoint] = @[]
    for id in parentPool:
      let fi = fusionMatch(population[id], fusionArchive)
      if fi >= 0 and iter - fusionArchive[fi].generation < FUSION_INTERVAL:
        currentPoints.add(FusionPoint(id: id, g: accs[id], j: fusionArchive[fi].quality))
    for pos in fusionFront(currentPoints, FUSION_CAP):
      jevDonorPool.add(currentPoints[pos].id)
  var jevDonorChildren = 0
  let jevDonorRate = int(round(25.0 * fusionReliability))
  let nurseryCap = min(FUSION_NURSERY, int(round(FUSION_NURSERY.float*fusionReliability)))
  var diversePairs = 0
  var measuredPairs = 0
  if parentPool.len >= 2:
    for _ in 0 ..< min(37, parentPool.len * 2):
      let a = parentPool[rand(parentPool.len - 1)]
      var b = parentPool[rand(parentPool.len - 1)]
      var retry = 0
      while b == a and retry < 4:
        b = parentPool[rand(parentPool.len - 1)]
        inc retry
      if a == b:
        continue
      inc measuredPairs
      if localGenomeDistance(population[a], population[b]) >= 0.45:
        inc diversePairs
  let diversePairRate = if measuredPairs > 0:
                          float(diversePairs) / float(measuredPairs)
                        else:
                          1.0
  let stagnationPressure = min(1.0, sqrt(float(stagnation)) / 6.0)
  let diversityPressure = max(0.0, min(1.0, (0.45 - diversePairRate) / 0.45))
  let parentRandomRate = min(PARENT_RANDOM_RATE_MAX,
    PARENT_RANDOM_RATE_BASE + 0.045 * stagnationPressure + 0.045 * diversityPressure)
  let adaptiveImmigrantRate = min(0.12,
    IMMIGRANT_RATE + STAGNATION_IMMIGRANT_BONUS * stagnationPressure +
    0.02 * diversityPressure)
  # Exploit locally while the population is healthy; increase recombination and
  # distant mating only when diversity collapses or progress stalls. This avoids
  # paying destructive crossover cost at a fixed rate in every regime.
  let adaptiveAsexualPercent = int(round(max(18.0, min(52.0,
    44.0 - 16.0 * stagnationPressure - 14.0 * diversityPressure))))
  let adaptiveDistantPercent = int(round(max(8.0, min(32.0,
    12.0 + 10.0 * stagnationPressure + 14.0 * diversityPressure))))

  echo "  diversity_pair_rate=", formatFloat(diversePairRate, ffDecimal, 3),
       " parent_unique=", parentPool.len, "/", fullIds.len,
       " parent_random=", formatFloat(parentRandomRate, ffDecimal, 3),
       " immigrant=", formatFloat(adaptiveImmigrantRate, ffDecimal, 3),
       " asexual_pct=", adaptiveAsexualPercent,
       " distant_pct=", adaptiveDistantPercent

  # この世代の先頭で次世代へそのまま継ぐ個体数を正確に覚える。
  # archive注入が少なかった場合に、ただの子個体まで次世代FULL保護しない。
  let nextProtectedPrefixCount = new_population.len

  # Shared-input, one-path execution fingerprints are generated once per
  # parent/genome per generation. They guide matching/placement only.
  var crossoverTraces = newSeq[RuleTrace](pop_size)
  var traceReady = newSeq[bool](pop_size)
  var traceBuilds = 0
  var traceSeconds = 0.0
  var traceSamples: seq[seq[int]] = @[]
  for ci in 0 ..< evalAggsX.len:
    if evalAggsX[ci].len > 0:
      traceSamples.add(evalAggsX[ci][0])
    if traceSamples.len >= 2: break
  proc parentTrace(id: int): RuleTrace =
    if not traceReady[id]:
      let started = epochTime()
      crossoverTraces[id] = buildCrossoverTrace(population[id],
        compiledPopulation[id], candidatePopulation[id], traceSamples)
      traceSeconds += epochTime() - started
      inc traceBuilds
      traceReady[id] = true
    crossoverTraces[id]
  for op in CrossoverOp:
    crossoverLastCounts[op] = 0
    crossoverLastSeconds[op] = 0.0
  for regime in MutationRegime:
    mutationRegimeLastCounts[regime] = 0
  crossoverProbes.setLen(0)
  mutationProbes.setLen(0)
  # A small controlled mutation-only lane: no crossover/FFT confounds.
  # Parents remain in the protected prefix and are re-evaluated after refresh.
  for k in 0 ..< min(MUTATION_PROBE_COUNT, pop_size - new_population.len):
    if survivorIds.len == 0: break
    let regime = MutationRegime(k mod MUTATION_REGIME_COUNT)
    let parent = rand(min(coreEliteCount, survivorIds.len) - 1)
    let probeScale = mutationScaleForRegime(feedbackMutationScale, regime)
    var probeChild = cloneGenome(new_population[parent])
    mutateGenome(probeChild, probeScale, 0)
    if sameGenomeRevisionState(probeChild, new_population[parent]):
      mutateGenome(probeChild, probeScale, 0)
    mutationProbes.add((parent: parent, child: new_population.len, regime: regime))
    inc mutationRegimeLastCounts[regime]
    new_population.add(probeChild)

  # One non-mutated matched FULL crossover probe per generation. The operator
  # rotates deterministically, so all four operators receive evidence over four
  # generations without consuming four compulsory FULL slots every generation.
  for probeId in 0 ..< min(CROSSOVER_PROBE_COUNT, pop_size - new_population.len):
    if survivorIds.len < 2 or coreEliteCount <= 0: break
    let op = CrossoverOp((iter * CROSSOVER_PROBE_COUNT + probeId) mod CROSSOVER_OP_COUNT)
    let parent = rand(min(coreEliteCount, survivorIds.len) - 1)
    let pId = survivorIds[parent]
    var donorId = survivorIds[rand(survivorIds.len - 1)]
    if donorId == pId:
      donorId = survivorIds[(parent + 1) mod survivorIds.len]
    let t1 = if op in {coHomologous, coModule}: parentTrace(pId) else: newSeq[RuleTraceRow](0)
    let t2 = if op in {coHomologous, coModule}: parentTrace(donorId) else: newSeq[RuleTraceRow](0)
    let started = epochTime()
    let child = crossoverGenome(population[pId], population[donorId], op, t1, t2)
    let elapsed = epochTime() - started
    crossoverProbes.add((parent: parent, child: new_population.len,
      op: op, seconds: elapsed))
    inc crossoverLastCounts[op]
    crossoverLastSeconds[op] += elapsed
    new_population.add(child)

  # Exact-genotype nursery dedupe. FULL/FAST evaluation is vastly more expensive
  # than hashing one 3100-rule genome, so do not knowingly send clones into the
  # next generation. Diagnostic probes above are intentionally left untouched.
  var offspringBuckets = initTable[Hash, seq[int]]()
  for id in 0 ..< new_population.len:
    let fp = genomeFingerprint(new_population[id])
    offspringBuckets.mgetOrPut(fp, @[]).add(id)
  var duplicateOffspringRetries = 0
  var duplicateOffspringImmigrants = 0

  proc nurseryDuplicate(g: Genome, fp: Hash): bool =
    for id in offspringBuckets.getOrDefault(fp):
      if sameGenomeContent(g, new_population[id]): return true
    false

  proc registerNurseryGenome(id: int, fp: Hash) =
    offspringBuckets.mgetOrPut(fp, @[]).add(id)

  while new_population.len < pop_size:
    let immigrantRate = adaptiveImmigrantRate
    if rand(1000) < int(immigrantRate * 1000.0):
      var immigrant: Genome
      # Pure random 3100-rule genomes are valuable for ergodicity but usually
      # spend an evaluation slot rediscovering that random structure is poor.
      # Most restart slots therefore hypermutate a CURRENT FULL-scored lineage;
      # one quarter remain genuine from-scratch immigrants for global coverage.
      if parentPool.len > 0 and rand(99) < 75:
        let seedId = parentPool[rand(parentPool.len - 1)]
        immigrant = cloneGenome(population[seedId])
        let restartScale = mutationScaleForRegime(max(1.10, feedbackMutationScale), mrExplore)
        mutateGenome(immigrant, restartScale, max(12, stagnation))
        if sameGenomeRevisionState(immigrant, population[seedId]):
          mutateGenome(immigrant, restartScale, max(12, stagnation))
        inc mutationRegimeLastCounts[mrExplore]
      else:
        immigrant = randomImmigrant()
      let immigrantFp = genomeFingerprint(immigrant)
      if not nurseryDuplicate(immigrant, immigrantFp):
        new_population.add(immigrant)
        registerNurseryGenome(new_population.len - 1, immigrantFp)
      continue

    # ------------------------------------------------------------
    # Multi-case parent selection
    # ------------------------------------------------------------
    # 平均scoreだけでなく、FULL評価されたcase別スコアを使う。
    # 30%はepsilon-Lexicase、70%はBatch Tournament。
    # どちらも「毎世代複数の問題/入力で評価した結果」を直接使う。
    # fullIdsは保護elite + FAST上位 + 遺伝的に遠い個体からなる。
    # 400個体でも親候補を約1/5まで広げ、実効集団サイズを確保する。
    let p1Pool = if rand(99) < PARETO_PARENT_PERCENT: uniqueParetoPool else: parentPool
    let p2Pool = if rand(99) < PARETO_PARENT_PERCENT: uniqueParetoPool else: parentPool
    var p1i = selectParentCapped(caseScores, p1Pool, parentUseCount, parentRandomRate)
    var p2i = selectParentCapped(caseScores, p2Pool, parentUseCount, parentRandomRate)
    if p1i < 0:
      p1i = order[0]
      inc parentUseCount[p1i]
    if p2i < 0:
      p2i = order[0]
      inc parentUseCount[p2i]

    var fusionMating = false
    if jevDonorPool.len >= 2 and rand(99) < jevDonorRate:
      releaseParentUse(parentUseCount, p1i)
      releaseParentUse(parentUseCount, p2i)
      var available: seq[int] = @[]
      for id in jevDonorPool:
        if parentUseCount[id] < PARENT_MAX_USES: available.add(id)
      if available.len >= 2:
        let pos = rand(available.len-1)
        p1i = available[pos]
        # Front is ordered along G. Mostly adjacent, occasionally complementary.
        let mate = if rand(99) < 25: (if pos < available.len div 2: available.len-1 else: 0)
                   elif pos+1 < available.len: pos+1 else: pos-1
        p2i = available[mate]
        inc parentUseCount[p1i]
        inc parentUseCount[p2i]
        fusionMating = true
      else:
        inc parentUseCount[p1i]
        inc parentUseCount[p2i]

    # 同一個体同士の交配を抑える。4回だけ引き直すので、選択圧を壊さない。
    if parentPool.len > 1 and p1i == p2i:
      for _ in 0 ..< 4:
        let alt = selectParentCapped(caseScores, parentPool, parentUseCount, parentRandomRate)
        if alt >= 0 and alt != p1i:
          replaceParentUse(parentUseCount, p2i, alt)
          p2i = alt
          break
        releaseParentUse(parentUseCount, alt)

    # crossoverGenomeはp1を骨格としてp2から1種類の交叉操作を適用するので、
    # p1を35%の確率で低fitness側にしていた旧挙動は不必要に退行しやすい。
    # donorの多様性はp2に残るため、骨格だけは常に高fitness側へ寄せる。
    if not fusionMating and accs[p2i] > accs[p1i]:
      swap(p1i, p2i)

    # Diversity/stagnationに応じて遠い系統を意図的に組み合わせる。
    # 早期収束時だけ強め、健全な集団では不要な破壊的交叉を抑える。
    if not fusionMating and parentPool.len >= 4 and rand(99) < adaptiveDistantPercent:
      let distant = selectDistantParentId(p1i, parentPool, parentUseCount)
      if distant >= 0:
        if p2i >= 0 and p2i < parentUseCount.len and parentUseCount[p2i] > 0:
          dec parentUseCount[p2i]
        p2i = distant
        if p2i < parentUseCount.len:
          inc parentUseCount[p2i]

    # distant donorへ差し替えた後も、clone元のp1は必ず高fitness側にする。
    if not fusionMating and accs[p2i] > accs[p1i]:
      swap(p1i, p2i)

    # Archive由来の系統も、populationへ再注入→CURRENT FULL評価→parentPool採用
    # という同じ関門を通す。過去datasetだけで良かった個体を親へ直接抜け道投入しない。
    let p1 = population[p1i]
    let p2 = population[p2i]

    # 全子個体を必ず交叉すると、良い骨格を保存した局所探索の経路がなくなる。
    # 健全な集団ではasexual+mutationの近傍探索を厚くし、停滞・多様性低下時は
    # 自動的に交叉へ予算を戻す。固定40%より探索/活用の切替を状態に合わせる。
    var child: Genome
    var didCrossover = false
    if rand(99) < adaptiveAsexualPercent:
      releaseParentUse(parentUseCount, p2i) # asexual child used only p1
      child = cloneGenome(p1)
    else:
      didCrossover = true
      let op = if fusionMating: (if rand(1) == 0: coHomologous else: coModule)
               else: chooseCrossoverOp(crossoverPreference)
      let t1 = if op in {coHomologous, coModule}: parentTrace(p1i) else: newSeq[RuleTraceRow](0)
      let t2 = if op in {coHomologous, coModule}: parentTrace(p2i) else: newSeq[RuleTraceRow](0)
      let started = epochTime()
      child = crossoverGenome(p1, p2, op, t1, t2)
      crossoverEmbedding(child, p2)
      inc crossoverLastCounts[op]
      crossoverLastSeconds[op] += epochTime() - started

    # Keep an explicit exploration lane, but let paired FULL probes allocate
    # the remaining mutation budget among local/balanced/exploratory regimes.
    # Population state supplies only a mild prior; measured offspring quality
    # remains the dominant signal through mutationRegimePreference.
    let explorationLane = rand(99) < 20
    let mutationPressure = max(stagnationPressure, diversityPressure)
    let mutationRegime = if explorationLane:
                           mrExplore
                         else:
                           chooseMutationRegime(mutationRegimePreference, mutationPressure)
    let mutationScale = mutationScaleForRegime(feedbackMutationScale, mutationRegime)
    inc mutationRegimeLastCounts[mutationRegime]
    mutateGenome(child, mutationScale, if explorationLane: stagnation else: 0)
    # The asexual lane exists for local search, not for cloning the parent into
    # another expensive evaluation slot. Mutation operators are probabilistic,
    # so an exact no-op would otherwise consume a full evaluation next round.
    # Retry once; affected-rule counts still use the same log-uniform sampler.
    if not didCrossover and sameGenomeRevisionState(child, p1):
      mutateGenome(child, max(0.85, mutationScale), if explorationLane: stagnation else: 0)
    # Count only actual parent contributions. A subsequent diversity replacement
    # or an asexual clone may have discarded the initially selected Jev donor.
    if p1i in jevDonorPool or (didCrossover and p2i in jevDonorPool):
      inc jevDonorChildren
    # Reject exact nursery duplicates before they consume a FAST/FULL slot next
    # generation. Re-mutate a few times; if the lineage still cannot leave an
    # existing genotype, replace that slot with an immigrant instead of paying
    # to rediscover an already measured point.
    var childFp = genomeFingerprint(child)
    var dedupeAttempt = 0
    while nurseryDuplicate(child, childFp) and dedupeAttempt < 3:
      inc duplicateOffspringRetries
      inc dedupeAttempt
      mutateGenome(child, max(0.95, mutationScale) * (1.0 + 0.15 * dedupeAttempt.float),
                   if explorationLane: stagnation else: 0)
      childFp = genomeFingerprint(child)
    if nurseryDuplicate(child, childFp):
      inc duplicateOffspringImmigrants
      var immigrantTries = 0
      while true:
        child = randomImmigrant()
        childFp = genomeFingerprint(child)
        inc immigrantTries
        if not nurseryDuplicate(child, childFp) or immigrantTries >= 4: break

    if fusionMating and fusionPending.len < nurseryCap:
      # A distinct unmeasured child gets one bounded chance at the NEXT Jev.
      var duplicate = fusionMatch(child, fusionArchive) >= 0
      for pending in fusionPending:
        if sameGenomeContent(child, pending.genome): duplicate = true
      if not duplicate:
        let due = ((iter + 1) div FUSION_INTERVAL + 1)*FUSION_INTERVAL - 1
        fusionPending.add(FusionPending(genome: cloneGenome(child), due: due))
    new_population.add(child)
    registerNurseryGenome(new_population.len - 1, childFp)

  # Local weight search has already refreshed caseScores before this capture.
  carriedCaseScores = newSeq[seq[float]](survivorIds.len)
  carriedCaseTimings = newSeq[seq[float]](survivorIds.len)
  for i, id in survivorIds:
    carriedCaseScores[i] = caseScores[id]
    carriedCaseTimings[i] = caseTimings[id]
  population = new_population
  protectedPrefixCount = nextProtectedPrefixCount

  echo "  nursery_dedupe: remutations=", duplicateOffspringRetries,
       " immigrant_fallbacks=", duplicateOffspringImmigrants

  if jevDonorPool.len > 0:
    echo "  jev-breeding: donor_children=", jevDonorChildren,
         " (only CURRENT FULL-scored donors; bounded by parent cap)"

  # 世代交代が完了した時点で保存。
  # nextIter=iter+1 なので、ロード時に同じ世代を二重実行しない。
  # Persist newly observed Jev winners immediately at the generation boundary.
  # Do not save mid-generation: the checkpoint must contain the NEXT population.
  if (iter + 1) mod CHECKPOINT_INTERVAL == 0 or jevMetricUpdated:
    saveCheckpoint(
      checkpointPath,
      iter + 1,
      population,
      bestEver,
      stagnation,
      maHistory,
      progressHistory,
      eliteOfElites,
      evalAggsX,
      evalAggsY,
      sourceChunkCursor,
      runSeed,
      progressSigmaOffset
    )

  let tGenEnd = epochTime()
  echo "  work_counts: pattern_revision_updates=", patternRevisionUpdates,
       " candidate_index_rebuilds=", candidateIndexRebuilds,
       " trace_builds=", traceBuilds,
       " trace_seconds=", formatFloat(traceSeconds, ffDecimal, 4)
  # ★追加: tGenStartはこれまで取得されているだけで一切出力されて
  # いなかった(=死んだ計測コード)。ここでステージ別の所要時間を
  # 出力し、実測でボトルネックを特定できるようにする。
  # 目安: compile≈patternキャッシュ/候補index再構築、fast≈全個体スクリーニング、
  # select≈多様性選抜等のCPU計算(並列化されない部分)、full≈FULL評価、
  # repro≈交叉・突然変異・checkpoint保存を含む世代交代。
  echo "  gen=", iter, " timing(s): total=",
       formatFloat(tGenEnd - tGenStart, ffDecimal, 3),
       " compile=", formatFloat(tAfterCompile - tGenStart, ffDecimal, 3),
       " fast_eval=", formatFloat(tAfterFastEval - tAfterCompile, ffDecimal, 3),
       " full_id_select=", formatFloat(tAfterFullIdSelect - tAfterFastEval, ffDecimal, 3),
       " full_eval=", formatFloat(tAfterFullEval - tAfterFullIdSelect, ffDecimal, 3),
       " jev=", formatFloat(jevWallSeconds, ffDecimal, 3),
       " repro_other=", formatFloat(max(0.0, tGenEnd - tAfterFullEval - jevWallSeconds), ffDecimal, 3)

# TARGET_GENERATIONS が CHECKPOINT_INTERVAL の倍数なら、最終世代は
# すでにループ内で保存済み。二重保存を避け、完了直後のcheckpointを
# 不要にローテーションしない。
if TARGET_GENERATIONS mod CHECKPOINT_INTERVAL != 0:
  saveCheckpoint(
    checkpointPath,
    TARGET_GENERATIONS,
    population,
    bestEver,
    stagnation,
    maHistory,
    progressHistory,
    eliteOfElites,
    evalAggsX,
    evalAggsY,
    sourceChunkCursor,
    runSeed,
    progressSigmaOffset
  )

# 最終状態も200/200MAの遡及補正済みCSVへ保存する。
rebuildPlotMovingAverages(progressHistory)
writeProgressCsv("progress.csv", progressHistory)
