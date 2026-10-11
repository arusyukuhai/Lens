import std/os
import std/streams

import fast_count_min_sketch

var sourceChunks: seq[seq[int]] = @[]

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
    j += 1
    if(j > n):
      break

  if sourceChunks.len == 0:
    quit("No source chunks found.")

loadTrainingCorpus(1)

var cms = initCountMinSketch(
  width = 262144,
  depth = 7
)

var a = sourceChunks[0]

var ranks : seq[tuple[cms: int, value:seq[int]]]

var aaa : seq[int]
var bbb : uint32
for j in 0..len(a)-16:
  var t: seq[int] = @[]
  for g in 0..2:
    t.add(a[j+g])
  t.add(-1)
  for i in j+4..len(a)-12:
    var t2 = t
    for g in 0..2:
      t2.add(a[i+g])
    cms.add(t2)
    bbb = cms.estimate(t2)
    aaa = t2

echo cms.estimate(aaa)
