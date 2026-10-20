#!/usr/bin/env bash
set -euo pipefail
# 使用方法
# base：	首次预热 → 第二次相同请求	HBM prefix cache 命中，不启用 Linker
# oceandisk：	首次预热写出 → 清空本地缓存 → 第二次相同请求	外部 Linker 命中，要求 HBM 命中为 0、外部命中大于 0
# bash examples/runtime/engine/run_oceandisk_offline_tp8.sh base
# bash examples/runtime/engine/run_oceandisk_offline_tp8.sh oceandisk

# Edit these paths for your machine.
MODEL_PATH="${MODEL_PATH:-/models/your-model}"
DATASET_PATH="${DATASET_PATH:-/home/z00591349/dataset/multifieldqa_zh.jsonl}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
LIBGIDS="${LIBGIDS:-/opt/oceandisk/libgids_kv_host.so}"
GIDS_DIR="${GIDS_DIR:-/opt/oceandisk/config/}"
GIDS_LOCAL_IP="${GIDS_LOCAL_IP:-}"       # 本机 RDMA/RoCE 网卡 IP
GIDS_SERVER_IP="${GIDS_SERVER_IP:-}"     # 存储服务端 IP
GIDS_SERVER_PORT="${GIDS_SERVER_PORT:-4420}"
GIDS_GID_INDEX="${GIDS_GID_INDEX:-}"
GIDS_ROCE_PORT="${GIDS_ROCE_PORT:-1}"
GIDS_QP_COUNT="${GIDS_QP_COUNT:-}" # 每个 IP 的 QP 数量

mode=base
if [[ "${1:-}" == base || "${1:-}" == oceandisk ]]; then
    mode="$1"
    shift
fi
linker_args=()
if [[ "$mode" == oceandisk ]]; then
    : "${GIDS_LOCAL_IP:?请在脚本顶部填写本机 RDMA/RoCE IP}"
    : "${GIDS_SERVER_IP:?请在脚本顶部填写存储服务端 IP}"
    : "${GIDS_GID_INDEX:?请填写本机网卡 GID index}"
    : "${GIDS_QP_COUNT:?请填写每个 IP 的 QP 数量}"
    mkdir -p -- "$GIDS_DIR"
    printf 'ip_config = %s, %s, %s, %s, %s, %s\n' \
        "$GIDS_LOCAL_IP" "$GIDS_SERVER_IP" "$GIDS_SERVER_PORT" \
        "$GIDS_GID_INDEX" "$GIDS_ROCE_PORT" "$GIDS_QP_COUNT" \
        > "${GIDS_DIR%/}/gids_config.txt"
    export USE_SINGLE_PORT=1
    export OPEXTENSION LIBGIDS GIDS_DIR
    linker_args=(
        --enable-unified-cache-external-linker
        --unified-cache-external-linker-backend oceandisk
    )
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND="${SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND:-rust}"

exec "${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_offline_tp8.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --context-length 8000 \
    --tp-size 8 \
    --disaggregation-mode null \
    --trust-remote-code \
    --page-size 64 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 12000 \
    --max-running-requests 1 \
    "${linker_args[@]}" \
    "$@"
