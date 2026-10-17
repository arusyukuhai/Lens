import std/json
import replace_fixed
proc ints(j: JsonNode): seq[int] =
  for v in j: result.add(v.getInt)
for line in stdin.lines:
  let j = parseJson(line)
  let x = ints(j["x"])
  let a = ints(j["a"])
  let b = ints(j["b"])
  let pat = compileSeqPattern(a, j["minimum"].getInt)
  let r = replaceSeqCompiled(x, pat, b)
  echo $(%r.data)
