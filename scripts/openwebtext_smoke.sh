#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CUDA_VISIBLE_DEVICES [openwebtext_analysis.py args...]" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$1"
shift

uv run python examples/openwebtext_analysis.py \
  --run-name smoke \
  --num-reference 2 \
  --num-sampler-source 8 \
  --num-samples 2 \
  --batch-size 1 \
  --skip-plots \
  "$@"
