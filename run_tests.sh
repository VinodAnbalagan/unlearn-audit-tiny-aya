#!/usr/bin/env bash
# All local CPU checks (no downloads). Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
python -c "import torch, transformers, jlens, unidecode" 2>/dev/null || {
  echo "Missing packages in this Python ($(which python)). Run once: bash setup_local.sh"; exit 1; }
for t in tests/test_data.py tests/test_fictional_data.py tests/test_judge.py tests/test_core_cpu.py tests/test_jaccess_cpu.py tests/test_name_rank_cpu.py; do
  echo "== $t"; python "$t"
done
if python -c "import safetune.unlearn" 2>/dev/null; then
  echo "== tests/test_st_npo_cpu.py"; python tests/test_st_npo_cpu.py
else
  echo "== SKIPPED tests/test_st_npo_cpu.py: safetune not installed (needs Python >= 3.12; see setup_local.sh)"
fi
echo "== ALL LOCAL TESTS PASSED"
