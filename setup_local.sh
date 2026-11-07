#!/usr/bin/env bash
# One-time local setup for the CPU tests (Mac or Linux). Run inside your venv.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt pytest
python -m pip install -q "jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e"
# SafeTune (Python >= 3.12), pinned; only needed for tests/test_st_npo_cpu.py and real unlearning runs
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" \
  && python -m pip install -q datasets "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git@26f59e6d1bdcacafe5f2be3b8d982fb214369659" \
  || echo "NOTE: Python < 3.12 or SafeTune install failed; the SafeTune equivalence test will be skipped"
python -c "import torch, transformers, jlens, unidecode; print('torch', torch.__version__, '| transformers', transformers.__version__, '| jlens ok')"
