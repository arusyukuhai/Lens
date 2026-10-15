## Deterministic frequency-weighted alphabet tokenization for Lens.
## Bytes are mapped bijectively to opaque tokens; original ASCII ordering is lost.
## Each byte receives a distinct token in a frequency-weighted interval.
## The intervals are sorted by descending corpus frequency (hashed tie breaks),
## with interval lengths proportional to observed counts.
## Alphabet tokens represent original corpus bytes (0..255).
## Diffusion noise is sampled exclusively from OBSERVED alphabet tokens,
## in proportion to corpus frequencies, excluding the original byte.
## Rule tokens remain in a separate namespace.
import std/[algorithm, os, strutils]

const
  AlphabetBase* = 1_000_000
  AlphabetSpan = 900_000_000'i64
  AlphabetLimit = AlphabetBase + int(AlphabetSpan) + 256
  RuleBase* = 1_550_000_000
  RuleBucketSpan = 16_000_000
  MaxRuleBucket = 20

type
  TokenCodec* = object
    alphabet*: array[256, int]
    frequencies*: array[256, int64]
    seed*: uint64
    # Derived in-memory index (rebuilt on load, not written to token_map.tsv).
    cumulative: array[256, int64]
    totalFrequency: int64
    sortedTokens: array[256, int]
    sortedBytes: array[256, int]

proc mix64*(v: uint64): uint64 {.inline.} =
  ## SplitMix64 finalizer; pure and stable between executions.
  var x = v + 0x9E3779B97F4A7C15'u64
  x = (x xor (x shr 30)) * 0xBF58476D1CE4E5B9'u64
  x = (x xor (x shr 27)) * 0x94D049BB133111EB'u64
  result = x xor (x shr 31)

proc rebuildNoiseIndex(codec: var TokenCodec) =
  ## Build a compact, reusable weighted sampler and inverse token lookup.
  ## Zero-frequency symbols receive no probability mass.
  var byToken = newSeq[tuple[token: int, sourceByte: int]](256)
  var total = 0'i64
  for b in 0..<256:
    total += codec.frequencies[b]
    codec.cumulative[b] = total
    byToken[b] = (token: codec.alphabet[b], sourceByte: b)
  codec.totalFrequency = total
  byToken.sort(proc(a, b: tuple[token: int, sourceByte: int]): int {.closure.} =
    cmp(a.token, b.token))
  for i in 0..<256:
    codec.sortedTokens[i] = byToken[i].token
    codec.sortedBytes[i] = byToken[i].sourceByte

proc byteFromToken(codec: TokenCodec, token: int): int {.inline.} =
  ## Binary search, rather than 256 comparisons at each corrupted position.
  var left = 0
  var right = 256
  while left < right:
    let mid = (left + right) shr 1
    if codec.sortedTokens[mid] < token:
      left = mid + 1
    else:
      right = mid
  if left >= 256 or codec.sortedTokens[left] != token:
    raise newException(ValueError, "diffusion expects an alphabet token")
  result = codec.sortedBytes[left]

proc makeTokenCodec*(source: seq[seq[int]], seed: uint64): TokenCodec =
  result.seed = seed
  for chunk in source:
    for value in chunk:
      if value < 0 or value > 255:
        raise newException(ValueError, "token codec expects original UTF-8 bytes 0..255")
      inc result.frequencies[value]
  var symbols = newSeq[int](256)
  for i in 0..<256: symbols[i] = i
  let freq = result.frequencies
  symbols.sort(proc(a, b: int): int =
    if freq[a] > freq[b]: return -1
    if freq[a] < freq[b]: return 1
    let ah = mix64(seed xor uint64(a))
    let bh = mix64(seed xor uint64(b))
    if ah < bh: return -1
    if ah > bh: return 1
    return cmp(a, b)
  )
  var totalWeight = 0'i64
  for c in 0..<256:
    totalWeight += max(1'i64, result.frequencies[c])
  var cumulative = 0'i64
  for rank, symbol in symbols:
    let weight = max(1'i64, result.frequencies[symbol])
    let lo = (cumulative * AlphabetSpan) div totalWeight
    let hi = ((cumulative + weight) * AlphabetSpan) div totalWeight
    let width = max(1'i64, hi - lo)
    let offset = int64(mix64(seed xor uint64(symbol) xor 0xA1B2C3D4'u64) mod uint64(width))
    result.alphabet[symbol] = AlphabetBase + rank + int(lo + offset)
    cumulative += weight
  rebuildNoiseIndex(result)

proc encodeAlphabet*(source: seq[seq[int]], codec: TokenCodec): seq[seq[int]] =
  result = newSeq[seq[int]](source.len)
  for i, chunk in source:
    result[i] = newSeq[int](chunk.len)
    for j, value in chunk:
      if value < 0 or value > 255:
        raise newException(ValueError, "original corpus is not byte-valued")
      result[i][j] = codec.alphabet[value]

proc noiseToken*(codec: TokenCodec, chunkId, position,
                 originalToken: int): int {.inline.} =
  ## Sample a DIFFERENT, corpus-observed alphabet token, deterministically.
  ## P(noise byte = b | original byte = a) = freq[b] / (total - freq[a])
  ## for b != a. The skipped interval avoids potentially unbounded rejection
  ## sampling when the original byte dominates the corpus.
  let originalByte = byteFromToken(codec, originalToken)
  let originalWeight = codec.frequencies[originalByte]
  let available = codec.totalFrequency - originalWeight
  if available <= 0:
    raise newException(ValueError,
      "Diffusion requires at least two distinct corpus-observed bytes")
  let key = codec.seed xor
    (uint64(chunkId + 1) * 0x9E3779B97F4A7C15'u64) xor
    (uint64(position + 1) * 0xBF58476D1CE4E5B9'u64)
  let sampled = int64(mix64(key) mod uint64(available))
  var beforeOriginal = 0'i64
  if originalByte > 0:
    beforeOriginal = codec.cumulative[originalByte - 1]
  # Skip over the original byte's interval in the CDF.
  var needle = sampled
  if sampled >= beforeOriginal:
    needle += originalWeight
  var left = 0
  var right = 256
  while left < right:
    let mid = (left + right) shr 1
    if needle >= codec.cumulative[mid]:
      left = mid + 1
    else:
      right = mid
  doAssert left < 256
  doAssert left != originalByte
  doAssert codec.frequencies[left] > 0
  result = codec.alphabet[left]

proc freshRuleToken*(codec: TokenCodec, pattern: openArray[int],
                     frequency: int, salt = 0): int =
  ## New token for a rule, disjoint from the alphabet (including diffusion).
  ## Bucket order encodes the log2 frequency of the candidate pattern;
  ## within a bucket its identity is keyed by a hash of the pattern.
  var hash = mix64(codec.seed xor 0x4352554C45534F4C'u64)
  for v in pattern:
    hash = mix64(hash xor uint64(cast[uint](v)))
  hash = mix64(hash xor uint64(salt))
  var bucket = 0
  var f = max(1, frequency)
  while f > 1 and bucket < MaxRuleBucket:
    f = f shr 1
    inc bucket
  result = RuleBase + bucket * RuleBucketSpan +
    int(hash mod uint64(RuleBucketSpan))

proc saveTokenMap*(path: string, codec: TokenCodec) =
  var content = "# Lens token map v1; seed=" & $codec.seed & "\n"
  content.add("# byte token count\n")
  for c in 0..<256:
    content.add($c & "\t" & $codec.alphabet[c] & "\t" & $codec.frequencies[c] & "\n")
  writeFile(path, content)

proc loadTokenMap*(path: string): TokenCodec =
  if not fileExists(path):
    raise newException(IOError, "Missing token map: " & path)
  var seen: array[256, bool]
  var sawSeed = false
  var n = 0
  for line in lines(path):
    if line.startsWith("# Lens token map v1; seed="):
      result.seed = uint64(parseBiggestUInt(line.split('=')[1].strip()))
      sawSeed = true
      continue
    if line.len == 0 or line[0] == '#': continue
    let fields = line.split('\t')
    if fields.len != 3: raise newException(ValueError, "Malformed token map row")
    let symbol = parseInt(fields[0])
    if symbol < 0 or symbol > 255 or seen[symbol]:
      raise newException(ValueError, "Invalid/duplicate token map byte")
    result.alphabet[symbol] = parseInt(fields[1])
    result.frequencies[symbol] = int64(parseBiggestInt(fields[2]))
    if result.alphabet[symbol] < AlphabetBase or
        result.alphabet[symbol] >= AlphabetLimit or result.frequencies[symbol] < 0:
      raise newException(ValueError, "Invalid token map value")
    seen[symbol] = true
    inc n
  if n != 256 or not sawSeed:
    raise newException(ValueError, "Token map needs a seed and 256 entries")
  for i in 0..<256:
    for j in i+1..<256:
      if result.alphabet[i] == result.alphabet[j]:
        raise newException(ValueError, "Token map must be one-to-one")
  rebuildNoiseIndex(result)
