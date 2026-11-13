#!/usr/bin/env bash
# Run the GIDS probes without sglang. Each test is its own process (GIDS can
# only be initialized once per process). Logs go to ${LOG_DIR}.
#
#   bash run_probe.sh            # quick suite (~10 min)
#   bash run_probe.sh t1         # only T1 (full / launch / none)
#   EXTRA="--pages 64" bash run_probe.sh t4
set -u

export OPEXTENSION="${OPEXTENSION:-/home/j00977581/gids/lib/libop_extension.so}"
export LIBGIDS="${LIBGIDS:-/home/j00977581/gids/lib/libgids_kv_host.so}"
export GIDS_DIR="${GIDS_DIR:-/home/j00977581/gids/cfg_fbs-h20-01/}"
export USE_SINGLE_PORT="${USE_SINGLE_PORT:-0}"   # same as run_prefill.sh
export PYTHONPATH="/home/j00977581/gids/lib:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

DEVICE="${DEVICE:-0}"          # single-GPU tests
DEVICES="${DEVICES:-0,1}"      # T5 writer,reader
EXTRA="${EXTRA:-}"
LOG_DIR="${LOG_DIR:-/home/j00977581/gids_probe_logs/$(date +%m%d%H%M)}"
PY="${PYTHON_BIN:-python3}"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${LOG_DIR}"
sha256sum "${LIBGIDS}" "${OPEXTENSION}" > "${LOG_DIR}/gids_lib.sha256" 2>/dev/null

run() {  # run <name> <script> [args...]
  local name=$1; shift
  echo "===== ${name}: $*" | tee -a "${LOG_DIR}/summary.txt"
  (cd "${here}" && "${PY}" -u "$@" ${EXTRA}) > "${LOG_DIR}/${name}.log" 2>&1
  local code=$?
  grep -h "^RESULT" "${LOG_DIR}/${name}.log" | tee -a "${LOG_DIR}/summary.txt"
  [ ${code} -eq 3 ] && echo "  (exit 3 = HANG, see ${name}.log)" | tee -a "${LOG_DIR}/summary.txt"
  # Native "ERR! recv cid" lines are printed by the GIDS kernel on box errors.
  local errs; errs=$(grep -c "ERR! recv" "${LOG_DIR}/${name}.log")
  echo "  exit=${code} gids_ERR_lines=${errs}" | tee -a "${LOG_DIR}/summary.txt"
  sleep 5  # let the box drop the previous process's connections
}

want() { [ $# -eq 0 ] || [[ " ${SELECTED} " == *" $1 "* ]]; }
SELECTED="${*:-t0 t1 t2 t4 t5}"

want t0 && run t0_serial t0_serial.py --device "${DEVICE}" --rounds 20
if want t1; then
  for mode in full launch none; do
    run "t1_${mode}" t1_get_put_overlap.py --device "${DEVICE}" --mode "${mode}" --rounds 30
  done
fi
if want t2; then
  for mode in full none; do
    run "t2_${mode}" t2_exist_get_overlap.py --device "${DEVICE}" --mode "${mode}" --seconds 60
  done
  run t2_none_with_put t2_exist_get_overlap.py --device "${DEVICE}" --mode none --seconds 60 --with-put
fi
if want t4; then
  for mode in full none; do
    run "t4_${mode}" t4_multiround.py --device "${DEVICE}" --mode "${mode}" --rounds 10
  done
fi
want t5 && run t5_two_process t5_two_process.py --devices "${DEVICES}" --rounds 30 --reader-get-load

echo "summary: ${LOG_DIR}/summary.txt"
