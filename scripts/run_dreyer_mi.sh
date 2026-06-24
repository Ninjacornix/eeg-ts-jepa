#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/run_dreyer_mi.sh PRESET [extra experiment.py args...]

presets:
  primary             alias for raw_mu_beta_aug
  raw_broadband       broadband raw-only input
  raw_mi_band         5-35 Hz raw-only input
  raw_mu_beta_aux     broadband raw + mu/beta auxiliary loss
  raw_mu_beta_aug     raw_mu_beta_aux + mild train-only SSL augmentation
  filterbank          fixed MI filter-bank spectral input
  learned_spectral    learned 5-35 Hz spectral input
  raw_plus_learned    broadband raw + learned spectral frontend
  all                 six spectral-prior conditions, excluding augmentation
  all_plus_aug        all plus raw_mu_beta_aug
  eval_old_raw        diagnostic eval of an existing old raw checkpoint
  reeval              retrain the shared decoder from a saved primary encoder
  calibration         few-label calibration sweep from a saved primary encoder

common overrides:
  DEVICE=cuda|mps|cpu        default: cuda
  WORKERS=N                 default: 8
  BATCH_SIZE=N              default: 32
  EPOCHS=N                  default: 25
  SAVE_DIR=PATH             default: .
  LOG_DIR=PATH              default: .
  PRETRAIN_EXTRA="..."      default: PhysionetMI:109 BNCI2014_001:9 Schirrmeister2017
  LOAD=checkpoint.pt        used by load presets
  DRY_RUN=1                 print the resolved command without running it

examples:
  bash scripts/run_dreyer_mi.sh primary
  BATCH_SIZE=16 bash scripts/run_dreyer_mi.sh raw_mu_beta_aug
  bash scripts/run_dreyer_mi.sh raw_mu_beta_aux --probe-epochs 300
  DRY_RUN=1 bash scripts/run_dreyer_mi.sh primary
  bash scripts/run_dreyer_mi.sh all
EOF
}

preset="${1:-primary}"
if [[ "$preset" == "-h" || "$preset" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-.venv/bin/python}"
device="${DEVICE:-cuda}"
workers="${WORKERS:-8}"
batch_size="${BATCH_SIZE:-32}"
epochs="${EPOCHS:-25}"
save_dir="${SAVE_DIR:-.}"
log_dir="${LOG_DIR:-.}"
pretrain_extra_string="${PRETRAIN_EXTRA:-PhysionetMI:109 BNCI2014_001:9 Schirrmeister2017}"

mkdir -p "$save_dir" "$log_dir"

common=(
  "$python_bin" scripts/experiment.py
  --dataset Dreyer2023
  --n-subjects 87
  --split 0.7 0.15 0.15
  --split-stratify subdataset
  --crop-align end
  --dim 384
  --heads 8
  --depth 8
  --batch-size "$batch_size"
  --patch-ms 250
  --lr 4e-4
  --warmup-epochs 8
  --var-reg 2
  --cov-reg 2
  --dropout 0.1
  --epochs "$epochs"
  --patience 5
  --eval-protocol cross-subject
  --probe-epochs 200
  --probe-lr 3e-3
  --probe-patience 25
  --seed 0
  --probe-seed 0
  --workers "$workers"
  --device "$device"
  --pool spatial
)

if [[ "${AMP:-1}" == "1" ]]; then
  common+=(--amp)
fi

pretrain_extra=()
if [[ -n "$pretrain_extra_string" ]]; then
  read -r -a pretrain_extra <<< "$pretrain_extra_string"
  common+=(--pretrain-extra "${pretrain_extra[@]}")
fi

run_one() {
  local name="$1"
  shift || true
  local save_base="$name"
  local -a mode
  local -a eval_flags
  eval_flags=(--raw-baseline --mdfb-analysis)

  case "$name" in
    primary)
      run_one raw_mu_beta_aug "$@"
      return
      ;;
    raw_broadband)
      save_base="spectral_raw_broadband"
      mode=(--bandpass 0.5 45 --input-mode raw --spectral-frontend none --spectral-aux 0)
      ;;
    raw_mi_band)
      save_base="spectral_raw_mi_band"
      mode=(--bandpass 5 35 --input-mode raw --spectral-frontend none --spectral-aux 0)
      ;;
    raw_mu_beta_aux)
      save_base="spectral_raw_mu_beta_aux"
      mode=(
        --bandpass 0.5 45
        --input-mode raw
        --spectral-frontend none
        --spectral-window-ms 2000
        --spectral-aux 1.0
        --spectral-aux-bands 8 13 13 30
      )
      ;;
    raw_mu_beta_aug)
      save_base="enc_cross_subject_aug_v5"
      mode=(
        --bandpass 0.5 45
        --input-mode raw
        --spectral-frontend none
        --spectral-window-ms 2000
        --spectral-aux 1.0
        --spectral-aux-bands 8 13 13 30
        --augment
        --aug-crop-jitter-ms 250
        --aug-time-jitter-ms 125
        --aug-amplitude-jitter 0.10
        --aug-gaussian-noise 0.02
        --aug-channel-dropout 0.05
        --aug-freq-mask-prob 0.25
        --aug-freq-mask-width-hz 2.0
        --aug-freq-mask-range 5 35
      )
      ;;
    filterbank)
      save_base="spectral_filterbank"
      mode=(
        --bandpass 0.5 45
        --input-mode spectral
        --spectral-frontend filterbank
        --spectral-window-ms 2000
        --spectral-aux 0
      )
      ;;
    learned_spectral)
      save_base="spectral_learned_spectral"
      mode=(
        --bandpass 0.5 45
        --input-mode spectral
        --spectral-frontend learned
        --spectral-window-ms 2000
        --spectral-range 5 35
        --spectral-aux 0
      )
      ;;
    raw_plus_learned)
      save_base="spectral_raw_plus_learned"
      mode=(
        --bandpass 0.5 45
        --input-mode raw
        --spectral-frontend learned
        --spectral-window-ms 2000
        --spectral-range 5 35
        --spectral-aux 0
      )
      ;;
    eval_old_raw)
      save_base="enc_raw_cross_subject"
      mode=(
        --bandpass 8 32
        --load "${LOAD:-enc_raw_v2_all.pt}"
      )
      ;;
    reeval|reevaluate)
      save_base="reevaluate_cross_subject_aug_v5"
      mode=(
        --bandpass 0.5 45
        --load "${LOAD:-enc_cross_subject_aug_v5_all.pt}"
      )
      ;;
    calibration)
      save_base="calibration_cross_subject_aug_v5"
      mode=(
        --bandpass 0.5 45
        --load "${LOAD:-enc_cross_subject_aug_v5_all.pt}"
        --eval-protocol calibration
        --calib-trials 5 10 20 40 120
      )
      eval_flags=()
      ;;
    *)
      echo "unknown Dreyer MI preset: $name" >&2
      usage
      exit 2
      ;;
  esac

  local save_path="${SAVE:-$save_dir/$save_base.pt}"
  local log_path="${LOG:-$log_dir/$save_base.log}"
  local -a command
  command=("${common[@]}" "${mode[@]}")
  if ((${#eval_flags[@]})); then
    command+=("${eval_flags[@]}")
  fi
  command+=(--save "$save_path" "$@")

  echo "=== Dreyer MI preset: $name ==="
  echo "save: $save_path"
  echo "log:  $log_path"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'PYTORCH_CUDA_ALLOC_CONF=%q ' \
      "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${command[@]}" \
    2>&1 | tee "$log_path"
}

case "$preset" in
  all)
    for name in \
      raw_broadband \
      raw_mi_band \
      raw_mu_beta_aux \
      filterbank \
      learned_spectral \
      raw_plus_learned
    do
      run_one "$name" "$@"
    done
    ;;
  all_plus_aug)
    for name in \
      raw_broadband \
      raw_mi_band \
      raw_mu_beta_aux \
      raw_mu_beta_aug \
      filterbank \
      learned_spectral \
      raw_plus_learned
    do
      run_one "$name" "$@"
    done
    ;;
  *)
    run_one "$preset" "$@"
    ;;
esac
