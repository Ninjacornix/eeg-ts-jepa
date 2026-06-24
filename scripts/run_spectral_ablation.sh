#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "scripts/run_spectral_ablation.sh is kept as a compatibility alias." >&2
echo "Prefer: bash scripts/run_dreyer_mi.sh <preset>" >&2

exec bash "$script_dir/run_dreyer_mi.sh" "${1:-all}" "${@:2}"
