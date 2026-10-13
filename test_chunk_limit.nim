## Standalone regression test: nim c -r test_chunk_limit.nim
import std/[os, strutils, times]
import cms

let path = getTempDir() / ("lens_chunk_limit_test_" & $int(epochTime()) & ".txt")
try:
  # readChunk always appends one newline to each stored source line.
  # The first and third chunks have exactly 4096 Unicode tokens.
  let corpus = "a".repeat(4095) & "\n===SPLIT===\n" &
    "b".repeat(4096) & "\n===SPLIT===\n" &
    "界".repeat(4095) & "\n===SPLIT===\n" &
    "tail\n"
  writeFile(path, corpus)

  var skipped = 0
  let unicodeChunks = indexCorpus(path, false, 4096, skipped)
  doAssert skipped == 1, "4097-token chunk should be skipped"
  doAssert unicodeChunks.len == 3, "4096-token chunks should be included"

  skipped = 0
  let byteChunks = indexCorpus(path, true, 4096, skipped)
  doAssert skipped == 2, "Unicode chunk is >4096 bytes"
  doAssert byteChunks.len == 2

  skipped = 0
  let noLimit = indexCorpus(path, false, 0, skipped)
  doAssert skipped == 0
  doAssert noLimit.len == 4

  skipped = 0
  let verySmall = indexCorpus(path, false, 3, skipped)
  doAssert skipped == 4
  doAssert verySmall.len == 0

  echo "test_chunk_limit: OK"
finally:
  if fileExists(path): removeFile(path)
