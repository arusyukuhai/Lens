## Fast Count-Min Sketch for integer keys and whole integer arrays.
## Nim 1.6 / 2.x. No substring, n-gram, rolling hash, or slicing helpers.
##
## - add/estimate accept scalar int/uint64 and openArray[T: SomeInteger].
## - Integer arrays (seq, fixed-size array, openArray) are hashed by CONTENT.
## - Array hash is order- and length-sensitive; no allocation per operation.
## - One contiguous seq[uint32] of counters; power-of-two row width.
## - SplitMix64-style finalization and double hashing for row indices.
## - Saturating counters; simple fast updates (not conservative updates).
## - Cost: O(depth) per scalar, O(array.len + depth) per array.
##
## Not thread-safe for concurrent writes. Not cryptographically collision-resistant.
## Approximate frequencies may be too high due to collisions.
## Frequency estimates saturate at uint32.high.
##
## Self-tests:
##   nim c -r -d:release --opt:speed fast_count_min_sketch.nim

const DefaultCmsSeed* = 0xA0761D6478BD642F'u64

type CountMinSketch* = object
  width, depth: int
  mask, seed: uint64
  counters: seq[uint32]

proc mix64(x: uint64): uint64 {.inline.} =
  var z = x
  z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9'u64
  z = (z xor (z shr 27)) * 0x94D049BB133111EB'u64
  result = z xor (z shr 31)

proc initCountMinSketch*(width = 65536, depth = 4,
                         seed: uint64 = DefaultCmsSeed): CountMinSketch =
  ## Each counter is uint32; memory = width * depth * 4 bytes.
  if width <= 0 or (width and (width - 1)) != 0:
    raise newException(ValueError, "Count-Min Sketch width must be a positive power of 2")
  if depth <= 0 or width > high(int) div depth:
    raise newException(ValueError, "Invalid Count-Min Sketch depth or size")
  result.width = width
  result.depth = depth
  result.mask = uint64(width - 1)
  result.seed = seed
  result.counters = newSeq[uint32](width * depth)

proc hashArray[T: SomeInteger](s: CountMinSketch,
                               key: openArray[T]): uint64 {.inline.} =
  ## Hash the WHOLE array without allocating a temporary buffer.
  ## Signed integers are sign-extended so @[1'i8] and @[1'i64]
  ## have identical hashes, while @[-1'i8] differs from @[255'u8].
  ## The tag makes array hashing separate from scalar hashing.
  var h = s.seed xor 0xD6E8FEB86659FD93'u64 xor
          (uint64(key.len) * 0x9E3779B185EBCA87'u64)
  for x in key:
    var v: uint64
    when T is SomeSignedInt:
      v = cast[uint64](int64(x))
    else:
      v = uint64(x)
    h = (h xor (v + 0xA0761D6478BD642F'u64)) * 0xE7037ED1A0B428DB'u64
    h = (h shl 27) or (h shr 37)
  result = mix64(h xor 0x8EBC6AF09C88C6E3'u64)

proc addHashed(s: var CountMinSketch, h: uint64,
               amount: uint32) {.inline.} =
  if amount == 0'u32: return
  let step = (h shr 32) or 1'u64
  var index = h
  var offset = 0
  if amount == 1'u32:
    for row in 0 ..< s.depth:
      let p = offset + int(index and s.mask)
      if s.counters[p] != high(uint32):
        inc s.counters[p]
      index = index + step
      offset += s.width
  else:
    let limit = high(uint32) - amount
    for row in 0 ..< s.depth:
      let p = offset + int(index and s.mask)
      let old = s.counters[p]
      s.counters[p] = (if old > limit: high(uint32) else: old + amount)
      index = index + step
      offset += s.width

proc estimateHashed(s: CountMinSketch, h: uint64): uint32 {.inline.} =
  let step = (h shr 32) or 1'u64
  var index = h
  var offset = 0
  result = high(uint32)
  for row in 0 ..< s.depth:
    let value = s.counters[offset + int(index and s.mask)]
    if value < result: result = value
    if result == 0'u32: return
    index = index + step
    offset += s.width

proc setHashed(s: var CountMinSketch, h: uint64,
               value: uint32) {.inline.} =
  ## Set the sketch estimate to at least value; never decrease counts.
  let step = (h shr 32) or 1'u64
  var index = h
  var offset = 0
  var oldEstimate = high(uint32)
  for row in 0 ..< s.depth:
    let p = offset + int(index and s.mask)
    let old = s.counters[p]
    if old < oldEstimate: oldEstimate = old
    if old < value: s.counters[p] = value
    index = index + step
    offset += s.width
  if value < oldEstimate:
    raise newException(ValueError,
      "Count-Min Sketch does not support decreasing a count; use clear()")

# Integer keys (original fast path).
proc add*(s: var CountMinSketch, key: uint64,
          amount: uint32 = 1'u32) {.inline.} =
  s.addHashed(mix64(key xor s.seed), amount)

proc estimate*(s: CountMinSketch, key: uint64): uint32 {.inline.} =
  s.estimateHashed(mix64(key xor s.seed))

proc add*(s: var CountMinSketch, key: int,
          amount: uint32 = 1'u32) {.inline.} =
  s.add(uint64(cast[uint](key)), amount)

proc estimate*(s: CountMinSketch, key: int): uint32 {.inline.} =
  s.estimate(uint64(cast[uint](key)))

# Whole-array keys (seq[int], array[N, int], seq[uint64], etc.).
proc add*[T: SomeInteger](s: var CountMinSketch, key: openArray[T],
                          amount: uint32 = 1'u32) {.inline.} =
  if amount != 0'u32:
    s.addHashed(s.hashArray(key), amount)

proc estimate*[T: SomeInteger](s: CountMinSketch,
                               key: openArray[T]): uint32 {.inline.} =
  s.estimateHashed(s.hashArray(key))

proc `[]`*(s: CountMinSketch, key: uint64): uint32 {.inline.} =
  s.estimate(key)

proc `[]`*(s: CountMinSketch, key: int): uint32 {.inline.} =
  s.estimate(key)

proc `[]`*[T: SomeInteger](s: CountMinSketch,
                           key: openArray[T]): uint32 {.inline.} =
  s.estimate(key)

# Indexed assignment only raises an estimate. To increment quickly, use add().
proc `[]=`*(s: var CountMinSketch, key: uint64,
            value: uint32) {.inline.} =
  s.setHashed(mix64(key xor s.seed), value)

proc `[]=`*(s: var CountMinSketch, key: int,
            value: uint32) {.inline.} =
  s[uint64(cast[uint](key))] = value

proc `[]=`*[T: SomeInteger](s: var CountMinSketch, key: openArray[T],
                            value: uint32) {.inline.} =
  s.setHashed(s.hashArray(key), value)

proc clear*(s: var CountMinSketch) {.inline.} =
  if s.counters.len > 0:
    zeroMem(addr s.counters[0], s.counters.len * sizeof(uint32))

proc memoryBytes*(s: CountMinSketch): int {.inline.} =
  s.counters.len * sizeof(uint32)

when isMainModule:
  var cms = initCountMinSketch(width = 4096, depth = 4)
  doAssert cms.estimate(123) == 0'u32
  for i in 0 ..< 100:
    cms.add(123)
  doAssert cms[123] == 100'u32
  cms[123] += 1'u32
  doAssert cms[123] == 101'u32
  cms[777'u64] = 10'u32
  doAssert cms[777'u64] == 10'u32
  var rejectedDecrease = false
  try:
    cms[777'u64] = 3'u32
  except ValueError:
    rejectedDecrease = true
  doAssert rejectedDecrease
  cms.add(-7, 30'u32)
  doAssert cms.estimate(-7) >= 30'u32

  # An array's entire content is one key (not its elements separately).
  let a = @[1, 2, 3]
  let b = @[3, 2, 1]
  let c = @[1, 2]
  cms.add(a)
  cms.add(a, 4'u32)
  doAssert cms.estimate(a) == 5'u32
  doAssert cms.estimate(b) == 0'u32
  doAssert cms.estimate(c) == 0'u32
  doAssert cms.estimate(@[1, 2, 3]) == 5'u32

  let fixed: array[3, int] = [1, 2, 3]
  doAssert cms.estimate(fixed) == 5'u32
  let unsigned = @[1'u64, 2'u64, 3'u64]
  doAssert cms.estimate(unsigned) == 5'u32

  let empty: seq[int] = @[]
  cms.add(empty)
  doAssert cms.estimate(empty) == 1'u32
  cms[b] = 7'u32
  doAssert cms[b] == 7'u32
  cms.add(@[-1, -2])
  doAssert cms.estimate(@[-1, -2]) == 1'u32

  cms.clear()
  doAssert cms.estimate(a) == 0'u32
  doAssert cms.estimate(123) == 0'u32
  cms.add(a, high(uint32))
  cms.add(a)
  doAssert cms.estimate(a) == high(uint32)
  echo "Count-Min Sketch self-tests passed"
