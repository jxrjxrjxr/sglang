#!/usr/bin/env bash
# PUT -> EXIST loss matrix. Each case changes one factor against the repro, so
# the first case that stops losing keys names the trigger. Every run uses a
# fresh run token (fresh keys). Logs and summary.txt go to ${LOG_DIR}.
#
#   bash run_put_exist_matrix.sh                # all cases, REPEAT=3 each
#   REPEAT=5 bash run_put_exist_matrix.sh repro two_gpu
#
# Cases:
#   repro      Zhang Yuan's run: 8 GPUs initialized, GPU 0 PUTs 1000 pages x
#              78 layers in one call, EXIST on GPU 0 and 1
#   two_gpu    same, only GPUs 0,1 initialized (no idle connections from 2-7)
#   layers1    1000 pages x 1 layer   (does loss need many layers per key?)
#   pages256   256 pages x 78 layers  (does loss need a large PUT?)
#   prod_shape 1000 pages as 256-page PUTs, one call per layer, exactly like
#              submit_put_layer() in production
set -u

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPEAT="${REPEAT:-3}"
LOG_DIR="${LOG_DIR:-/home/j00977581/gids_probe_logs/put_exist_$(date +%m%d%H%M)}"
mkdir -p "${LOG_DIR}"
SELECTED="${*:-repro two_gpu layers1 pages256 prod_shape}"

case_env() {  # print the env overrides of one case
  case "$1" in
    repro)      echo "GPUS=0,1,2,3,4,5,6,7" ;;
    two_gpu)    echo "GPUS=0,1" ;;
    layers1)    echo "GPUS=0,1 LAYERS=1" ;;
    pages256)   echo "GPUS=0,1 PAGES=256" ;;
    prod_shape) echo "GPUS=0,1 PUT_MODE=per-layer PUT_CHUNK_PAGES=256" ;;
    *) echo "unknown case $1" >&2; return 1 ;;
  esac
}

for name in ${SELECTED}; do
  overrides="$(case_env "${name}")" || exit 2
  for i in $(seq 1 "${REPEAT}"); do
    log="${LOG_DIR}/${name}_${i}.log"
    echo "===== ${name} #${i}: ${overrides}" | tee -a "${LOG_DIR}/summary.txt"
    # shellcheck disable=SC2086
    env ${overrides} SYNC_DIR="${LOG_DIR}/${name}_${i}.sync" \
      bash "${here}/run_oceandisk_dispatch_roundtrip.sh" > "${log}" 2>&1
    code=$?
    grep -h -E "^SUMMARY|^PUT: mode=" "${log}" | tee -a "${LOG_DIR}/summary.txt"
    echo "  exit=${code}" | tee -a "${LOG_DIR}/summary.txt"
    sleep 5  # let the box drop the previous run's connections
  done
done
echo "summary: ${LOG_DIR}/summary.txt"
