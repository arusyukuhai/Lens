#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v nim >/dev/null 2>&1; then
  echo 'Nim compiler not found; install Nim 2.2.x and retry' >&2
  exit 1
fi
nim --version
nim check cms.nim
nim c -d:release --opt:speed --mm:orc -o:lens_pve_check cms.nim
./lens_pve_check --self-test
python3 verify_multifidelity.py
python3 verify_rotation.py
python3 verify_variation.py
python3 verify_prefix_cache.py
python3 verify_block_cache.py
if python3 -c 'import numpy' >/dev/null 2>&1; then
  python3 verify_ga_math.py
fi
# Small smoke run through two generations, without reliance on user corpus.
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT
python3 - "$TMPDIR_TEST/test.txt" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); chunks=[]
for i in range(8):
    chunks.append(''.join(chr(65+((j*j+i*j+2*i) % 18)) for j in range(64)))
p.write_text('\n===SPLIT===\n'.join(chunks),encoding='utf-8')
PY
./lens_pve_check --corpus="$TMPDIR_TEST/test.txt" --out="$TMPDIR_TEST/run" \
  --chunks=8 --chunk-len=64 --train=2 --valid=2 --rules=16 \
  --population=4 --elites=1 --archive=2 --generations=2 \
  --noise-steps=4 --stage1-noise=2 --stage2-noise=3 --stage3-noise=4 \
  --stage2-chunks=3 --stage3-chunks=4 --candidate-factor=1 \
  --finalists=1 --local-trials=1 --residual-candidates=4 \
  --linkage-size=4 --virus=0 --plot-every=0 --ma-window=3
[[ -s "$TMPDIR_TEST/run/multifidelity_stats.dat" ]]
echo '[check] Nim compile, Nim self-test, Python checks, and two-generation smoke run completed'
