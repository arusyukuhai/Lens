#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v nim >/dev/null 2>&1; then
  echo 'Nim compiler not found (requires Nim 1.6+)' >&2
  exit 1
fi
nim c -d:release --opt:speed -o:lens_ga cms.nim
exec ./lens_ga "$@"
