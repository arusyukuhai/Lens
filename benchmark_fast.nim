## Synthetic repeatable component benchmark, not an end-to-end corpus claim.
import std/[times, random, sequtils, strformat]
import lens_core, batch_presence, nsga2_pareto

var rng = initRand(20261013)
var checksum = 0.0
block:
  let n = 1024
  let p = 64
  let c = 256
  var data = DataSet(texts: newSeq[seq[int]](n))
  for i in 0..<n:
    data.target.add(float64(i mod 256)/255.0)
    data.cases.add(i div 256)
  var cols: seq[seq[float64]]
  var candidates: seq[seq[float64]]
  for j in 0..<p+c:
    var col: seq[float64]
    for i in 0..<n: col.add(float64(rand(rng,1)))
    if j < p: cols.add(col)
    else: candidates.add(col)
  let before = cpuTime()
  for col in candidates:
    let ev = scoreColumns(cols & @[col],cols & @[col],data,data,0.01,false)
    checksum += ev.score
  let reference = cpuTime()-before
  let now = cpuTime()
  let cache = prepareRankAppend(cols,data,0.01)
  let ranks = prepareSpearman(data)
  for col in candidates:
    let w = fitAppend(cache,col)
    checksum += scoreSpearman(ranks,predictAppend(cols,col,w))
    checksum += scoreSpearman(ranks,predictAppend(cols,col,w))
  let fast = cpuTime()-now
  echo &"score N={n} R={p} C={c}: reference={reference:.6f}s fast={fast:.6f}s speedup={reference/fast:.2f}x"
block:
  let n = 512
  let length = 512
  let c = 1000
  var texts,patterns: seq[seq[int]]
  for i in 0..<n:
    var text: seq[int]
    for j in 0..<length: text.add(rand(rng,15))
    texts.add(text)
  for i in 0..<c:
    var pat: seq[int]
    for j in 0..rand(rng,2):
      if j > 0: pat.add(-1)
      for k in 0..rand(rng,3): pat.add(rand(rng,15))
    patterns.add(pat)
  let before = cpuTime()
  for pat in patterns:
    let column = presence(texts,pat)
    for v in column: checksum += v
  let reference = cpuTime()-before
  let now = cpuTime()
  let matrix = batchPresence(compileBatch(patterns),texts)
  for col in matrix.columns: checksum += float64(bitCount(col))
  let fast = cpuTime()-now
  echo &"presence N={n} L={length} C={c}: reference={reference:.6f}s fast={fast:.6f}s speedup={reference/fast:.2f}x"
block:
  let n = 5000
  var scores: seq[seq[float64]]
  for i in 0..<n: scores.add(@[rand(rng,1.0),rand(rng,1.0)])
  let before = cpuTime()
  let general = nsga2SortGeneral(scores,@[odMaximize,odMinimize])
  let reference = cpuTime()-before
  let now = cpuTime()
  let fastResult = nsga2Sort(scores,@[odMaximize,odMinimize])
  let fast = cpuTime()-now
  doAssert general.order == fastResult.order
  echo &"pareto C={n}: reference={reference:.6f}s fast={fast:.6f}s speedup={reference/fast:.2f}x"
echo "checksum=",checksum
