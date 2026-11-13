#!/usr/bin/env bash
# One PUT -> EXIST run of oceandisk_dispatch_roundtrip.py (Zhang Yuan's test).
# Defaults reproduce his case: GPU 0 PUTs 1000 pages x 78 layers, GPU 0 and
# GPU 1 then run EXIST, no GET. Override any variable on the command line:
#   PAGES=256 GPUS=0,1 bash run_oceandisk_dispatch_roundtrip.sh
#   PUT_MODE=per-layer PUT_CHUNK_PAGES=256 bash run_oceandisk_dispatch_roundtrip.sh
set -euo pipefail

# Same GIDS install and config as dual_node_pd/run_prefill.sh.
export OPEXTENSION="${OPEXTENSION:-/home/j00977581/gids/lib/libop_extension.so}"
export LIBGIDS="${LIBGIDS:-/home/j00977581/gids/lib/libgids_kv_host.so}"
export GIDS_DIR="${GIDS_DIR:-/home/j00977581/gids/cfg_fbs-h20-01/}"
# Per-GPU configs gids_config_<N>.txt need 0; gids_config.txt needs 1.
export USE_SINGLE_PORT="${USE_SINGLE_PORT:-0}"
export PYTHONPATH="${GIDS_PYTHONPATH:-$(dirname "${OPEXTENSION}")}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

GPUS="${GPUS-0,1,2,3,4,5,6,7}"   # empty string -> single-GPU mode
DEVICE="${DEVICE:-0}"            # used only without --gpus
EXISTS_DEVICE="${EXISTS_DEVICE-1}" # empty -> skip EXISTS; 1 -> query on GPU 1
EXISTS_ON_WRITER="${EXISTS_ON_WRITER:-1}" # 1 -> also query on the PUT GPU
SKIP_GET="${SKIP_GET:-1}"        # 1 -> PUT and EXISTS only
CONN="${CONN-}"                  # empty -> route each key to its DPU
CONNS="${CONNS-}"                # empty -> all connections; e.g. 0,1,2
LAYERS="${LAYERS:-78}"
PAGES="${PAGES:-1000}"
KV_BYTES="${KV_BYTES:-73728}"
INDEX_BYTES="${INDEX_BYTES:-8448}"
WARMUP="${WARMUP:-0}"
GET_STAGGER_MS="${GET_STAGGER_MS:-0}"   # per-worker delay: i * N ms
PUT_MODE="${PUT_MODE:-all}"             # all | per-layer (production shape)
PUT_CHUNK_PAGES="${PUT_CHUNK_PAGES:-0}" # 0 = one PUT; production chunk = 256
SETTLE_S="${SETTLE_S:-15}"
RECHECK_AFTER="${RECHECK_AFTER:-30}"    # seconds, comma separated
SYNC_DIR="${SYNC_DIR:-$(mktemp -d /tmp/oceandisk-roundtrip-XXXXXX)}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

args=(--device "${DEVICE}")
if [[ -n "${CONN}" ]]; then
  args+=(--conn "${CONN}")
fi
if [[ -n "${CONNS}" ]]; then
  args+=(--conns "${CONNS}")
fi
if [[ -n "${GPUS}" ]]; then
  args+=(--gpus "${GPUS}" --sync-dir "${SYNC_DIR}")
fi
if [[ -n "${EXISTS_DEVICE}" ]]; then
  args+=(--exists-device "${EXISTS_DEVICE}")
fi
if [[ "${EXISTS_ON_WRITER}" == "1" ]]; then
  args+=(--exists-on-writer)
fi
if [[ "${SKIP_GET}" == "1" ]]; then
  args+=(--skip-get)
fi

# Record which GIDS build ran; compare with the build Zhang Yuan used.
sha256sum "${LIBGIDS}" "${OPEXTENSION}" 2>/dev/null || true
exec "${PYTHON_BIN:-python3}" -u "$script_dir/oceandisk_dispatch_roundtrip.py" \
  "${args[@]}" \
  --layers "${LAYERS}" \
  --pages "${PAGES}" \
  --kv-bytes "${KV_BYTES}" \
  --index-bytes "${INDEX_BYTES}" \
  --warmup "${WARMUP}" \
  --get-stagger-ms "${GET_STAGGER_MS}" \
  --put-mode "${PUT_MODE}" \
  --put-chunk-pages "${PUT_CHUNK_PAGES}" \
  --settle-s "${SETTLE_S}" \
  --recheck-after "${RECHECK_AFTER}" \
  "$@"
