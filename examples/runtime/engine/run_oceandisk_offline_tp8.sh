#!/usr/bin/env bash
set -euo pipefail
# 使用方法
# base：	首次预热 → 第二次相同请求	HBM prefix cache 命中，不启用 Linker
# oceandisk：	首次预热写出 → 清空本地缓存 → 第二次相同请求	外部 Linker 命中，要求 HBM 命中为 0、外部命中大于 0
# bash examples/runtime/engine/run_oceandisk_offline_tp8.sh base
# bash examples/runtime/engine/run_oceandisk_offline_tp8.sh oceandisk

# Edit these paths for your machine.
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
DATASET_PATH="${DATASET_PATH:-/home/z00591349/dataset/multifieldqa_zh.jsonl}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
GIDS_CONFIG="${GIDS_CONFIG:-/opt/oceandisk/config/gids_config.txt}"

mode=base
if [[ "${1:-}" == base || "${1:-}" == oceandisk ]]; then
    mode="$1"
    shift
fi
linker_args=()
if [[ "$mode" == oceandisk ]]; then
    if [[ "$(basename -- "$GIDS_CONFIG")" != gids_config.txt ]]; then
        printf 'GIDS_CONFIG must point to a file named gids_config.txt\n' >&2
        exit 1
    fi
    GIDS_DIR="$(cd -- "$(dirname -- "$GIDS_CONFIG")" && pwd)/"
    export USE_SINGLE_PORT=1
    export OPEXTENSION GIDS_DIR
    export PYTHONPATH="$(dirname -- "$OPEXTENSION")${PYTHONPATH:+:${PYTHONPATH}}"
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
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"
export SGLANG_DG_USE_NVRTC="${SGLANG_DG_USE_NVRTC:-1}"

exec "${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_offline_tp8.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --context-length 8000 \
    --tp-size 8 \
    --trust-remote-code \
    --page-size 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --moe-dense-tp-size 1 \
    --data-parallel-size 1 \
    --moe-a2a-backend none \
    --disable-prefill-cuda-graph \
    --mem-fraction-static 0.85 \
    --max-total-tokens 12000 \
    "${linker_args[@]}" \
    "$@"
