#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
work="${VLLM_AUDIT_DIR:-$project_dir/.local/bug-audits/forward-verify}"
mkdir -p "$work"
uv run --no-project --python 3.12 python "$script_dir/prepare_source.py"
uv run --no-project --python 3.12 --with 'modal[api-proxy-support]' \
  modal run "$script_dir/modal_forward.py" 2>&1 \
  | tee "$work/modal-run-$(date -u +%Y%m%dT%H%M%SZ).log"
