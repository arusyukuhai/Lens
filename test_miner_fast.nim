## Differential tests: original miner vs incremental-hash miner.
## Run: nim c -r -d:release --opt:speed test_miner_fast.nim
import std/[unittest, random]
import cms_miner
import cms_miner_legacy as legacy
import fast_count_min_sketch

proc legacyCfg(c: cms_miner.MiningConfig): legacy.MiningConfig =
  legacy.MiningConfig(
    minGram: c.minGram, maxGram: c.maxGram,
    maxTotalGram: c.maxTotalGram, maxWildcards: c.maxWildcards,
    topPerShape: c.topPerShape, width: c.width, depth: c.depth,
    verbose: false)

suite "Exhaustive mining semantics and hash compatibility":
  test "merged CMS update has identical counters, estimates and signed hashes":
    let keys: seq[seq[int]] = @[
      @[], @[1], @[-1], @[1,-1,3,-1,5], @[256,257],
      @[1,-1,3,-1,5], @[0,0,0], @[1,2,3,4]]
    var oldCms = initCountMinSketch(width = 16, depth = 7)
    var newCms = initCountMinSketch(width = 16, depth = 7)
    for repeat in 0..<10:
      for key in keys:
        oldCms.add(key)
        let oldEstimate = oldCms.estimate(key)
        var prefix = newCms.beginArrayHash(key.len)
        for t in key:
          prefix = extendArrayHash(prefix, t)
        let newEstimate = newCms.addHashedAndEstimate(finishArrayHash(prefix))
        check newEstimate == oldEstimate
        for k in keys:
          check oldCms.estimate(k) == newCms.estimate(k)

  test "exhaustive random chunks yield exactly the same ranked patterns":
    var rng = initRand(20261013)
    for round in 0..<12:
      var chunks: seq[seq[int]]
      for ci in 0..<3:
        var c: seq[int]
        for i in 0..<rand(rng, 8..18):
          c.add(rand(rng, 0..4))
        chunks.add(c)
      var cfg = cms_miner.defaultMiningConfig()
      cfg.startStride = 1
      cfg.gapSamples = 0
      cfg.width = 16          # Force collisions to test CMS ordering.
      cfg.depth = 3
      cfg.maxGram = 3
      cfg.maxTotalGram = 5
      cfg.maxWildcards = 3
      cfg.topPerShape = 9
      let a = legacy.minePatterns(chunks, legacyCfg(cfg))
      let b = cms_miner.minePatterns(chunks, cfg)
      check a.len == b.len
      if a.len == b.len:
        for i in 0..<a.len:
          check a[i].value == b[i].value
          check a[i].rank == b[i].rank

  test "single-gram shape and four-wildcard suffix enumeration remain unchanged":
    var cfg = cms_miner.defaultMiningConfig()
    cfg.startStride = 1
    cfg.gapSamples = 0
    cfg.width = 64
    cfg.depth = 4
    cfg.maxWildcards = 4
    cfg.maxGram = 2
    cfg.maxTotalGram = 5
    cfg.topPerShape = 8
    let chunks = @[@[1,2,3,1,2,3,1,2,3,1,2,3,1,2]]
    let oldResult = legacy.minePatterns(chunks, legacyCfg(cfg))
    let newResult = cms_miner.minePatterns(chunks, cfg)
    check oldResult.len == newResult.len
    if oldResult.len == newResult.len:
      for i in 0..<oldResult.len:
        check oldResult[i].value == newResult[i].value
        check oldResult[i].rank == newResult[i].rank
