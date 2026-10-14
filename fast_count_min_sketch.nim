## Minimal, fast Count-Min Sketch for integer keys.
## Nim 1.6 / 2.x. No sequence/span/n-gram helpers.
##
## - One contiguous array of uint32 counters
## - Power-of-two width: index masking instead of remainder
## - One 64-bit avalanche mix, then double-hash row indexing
## - Saturating counters (never silently wrap to zero)
## - O(depth) add/estimate; no allocation after initialization
##
## Not thread-safe for simultaneous writes.
## The hash is optimized for speed, not adversarial collision resistance.
## Counts are upper estimates, except when counters hit uint32.high.
##
## Build and run the self-tests:
##   nim c -r -d:release --opt:speed fast_count_min_sketch.nim

const DefaultCmsSeed* = 0xA0761D6478BD642F'u64

type CountMinSketch* = object
  width, depth: int
  mask, seed: uint64
  counters: seq[uint32]

proc mix64(x: uint64): uint64 {.inline.} =
  ## SplitMix64 finalizer. Arithmetic on uint64 wraps modulo 2^64.
  var z = x
  z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9'u64
  z = (z xor (z shr 27)) * 0x94D049BB133111EB'u64
  result = z xor (z shr 31)

proc initCountMinSketch*(width = 65536, depth = 4,
                         seed: uint64 = DefaultCmsSeed): CountMinSketch =
  ## Width must be a power of two; memory = width * depth * 4 bytes.
  if width <= 0 or (width and (width - 1)) != 0:
    raise newException(ValueError, "Count-Min Sketch width must be a positive power of 2")
  if depth <= 0 or width > high(int) div depth:
    raise newException(ValueError, "Invalid Count-Min Sketch depth or size")
  result.width = width
  result.depth = depth
  result.mask = uint64(width - 1)
  result.seed = seed
  result.counters = newSeq[uint32](width * depth)

proc add*(s: var CountMinSketch, key: uint64,
          amount: uint32 = 1'u32) {.inline.} =
  ## Insert `amount` copies of key; saturate at uint32.high.
  if amount == 0'u32: return
  let h = mix64(key xor s.seed)
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

proc estimate*(s: CountMinSketch, key: uint64): uint32 {.inline.} =
  ## Minimum counter across all rows. Collisions may inflate estimates.
  let h = mix64(key xor s.seed)
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

proc add*(s: var CountMinSketch, key: int,
          amount: uint32 = 1'u32) {.inline.} =
  ## Convenience overload for Nim's native signed integers.
  s.add(uint64(cast[uint](key)), amount)

proc estimate*(s: CountMinSketch, key: int): uint32 {.inline.} =
  s.estimate(uint64(cast[uint](key)))

proc clear*(s: var CountMinSketch) {.inline.} =
  ## Reuse the allocated memory.
  if s.counters.len > 0:
    zeroMem(addr s.counters[0], s.counters.len * sizeof(uint32))

proc memoryBytes*(s: CountMinSketch): int {.inline.} =
  s.counters.len * sizeof(uint32)

when isMainModule:
  var cms = initCountMinSketch(width = 4096, depth = 4)
  doAssert cms.estimate(123) == 0'u32
  for i in 0 ..< 100:
    cms.add(123)
  doAssert cms.estimate(123) == 100'u32
  cms.add(-7, 30'u32)
  doAssert cms.estimate(-7) >= 30'u32
  cms.clear()
  doAssert cms.estimate(123) == 0'u32
  cms.add(123'u64, high(uint32))
  cms.add(123'u64)
  doAssert cms.estimate(123'u64) == high(uint32)
  echo "Count-Min Sketch self-tests passed"
