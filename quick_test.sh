#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nim c -d:release --opt:speed lens_ga.nim
./lens_ga --self-test
./lens_ga --corpus=demo_corpus.txt --nodes=32 --generations=2 \
  --population=4 --elites=1 --offspring=4 --immigrants=1 \
  --train-chunks=4 --anchor-chunks=1 --rotate-chunks=2 \
  --valid-chunks=1 --test-chunks=1 --noise-levels=6 \
  --selected-features=4 --plot-every=1 --out=quick_plot \
  --save=quick_model.json
