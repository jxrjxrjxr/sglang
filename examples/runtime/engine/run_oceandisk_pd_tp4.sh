#!/bin/sh
set -eu
# 使用方法：单机 8 卡 PD，P 用 GPU0-3、D 用 GPU4-7（各 TP4），router 对外
# check：同一请求依次要求 D 从 OceanDisk 读回 → HBM 命中 → 清空 D 后再读回；另测并发与 D 部分命中
# bench：单请求冷启动与清空 D 后读回的 TTFT，以及不同并发下的吞吐
# sh examples/runtime/engine/run_oceandisk_pd_tp4.sh check 2>&1 | tee pd_check.log
# sh examples/runtime/engine/run_oceandisk_pd_tp4.sh bench 2>&1 | tee pd_bench_linker.log
# 原生 PD 对照（mooncake，不经过 OceanDisk）：TRANSFER_BACKEND=mooncake sh ... bench
# GIDS_DIR 内放 gids_config_0..7.txt（D 用 --base-gpu-id 4，读 4..7）：
# OPEXTENSION=/path/to/libop_extension.so GIDS_DIR=/path/to/config sh ... check

# Edit these paths for your machine.
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
DATASET_PATH="${DATASET_PATH:-/home/z00591349/dataset/multifieldqa_zh.jsonl}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
LIBGIDS="${LIBGIDS:-$(dirname -- "$OPEXTENSION")/libgids_kv_host.so}"
GIDS_DIR="${GIDS_DIR:-/opt/oceandisk/config}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-linker}"
MOONCAKE_IB_DEVICES="${MOONCAKE_IB_DEVICES:-mlx5_3,mlx5_5,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15}"
run_id="$(date +%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/oceandisk_pd/${run_id}}"

mode=check
case "${1:-}" in
    check|bench)
        mode="$1"
        shift
        ;;
esac
# OceanDisk trace：check 默认开启，bench 默认关闭以免影响计时。
if [ "$mode" = check ]; then trace=1; else trace=0; fi
export SGLANG_OCEANDISK_TRACE="${SGLANG_OCEANDISK_TRACE:-$trace}"

case "$TRANSFER_BACKEND" in
    linker)
        GIDS_DIR="$(cd -- "$GIDS_DIR" && pwd)/"
        export USE_SINGLE_PORT=0 OPEXTENSION LIBGIDS GIDS_DIR
        export PYTHONPATH="$(dirname -- "$OPEXTENSION")${PYTHONPATH:+:${PYTHONPATH}}"
        # P、D 用同一个 namespace（存储 key 的前缀），每次运行换新的。
        set -- \
            --enable-unified-cache-external-linker \
            --unified-cache-external-linker-backend oceandisk \
            --hicache-storage-backend-extra-config \
            "{\"namespace\":\"pd-${run_id}\",\"connection_indices\":[0],\"layerwise_save\":true}" \
            --disaggregation-transfer-backend linker \
            "$@"
        ;;
    mooncake)
        set -- --disaggregation-transfer-backend mooncake \
            --disaggregation-ib-device "$MOONCAKE_IB_DEVICES" "$@"
        ;;
    *)
        printf 'TRANSFER_BACKEND must be linker or mooncake\n' >&2
        exit 1
        ;;
esac

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP="${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}"
export SGLANG_DG_USE_NVRTC="${SGLANG_DG_USE_NVRTC:-1}"
# 黄区代理不放行本机地址。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

set -- \
    --model-path "${MODEL_PATH}" \
    --tp-size 4 \
    --host 127.0.0.1 \
    --trust-remote-code \
    --page-size 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --enable-cache-report \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend none \
    --disable-prefill-cuda-graph \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    "$@"

mkdir -p "$LOG_DIR"
printf 'mode=%s backend=%s logs=%s\n' "$mode" "$TRANSFER_BACKEND" "$LOG_DIR"
pids=
trap 'kill $pids 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM

python3 -m sglang.launch_server "$@" --base-gpu-id 0 --port 31002 \
    --dist-init-addr 127.0.0.1:31500 --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 31145 > "$LOG_DIR/prefill.log" 2>&1 &
pids="$pids $!"
python3 -m sglang.launch_server "$@" --base-gpu-id 4 --port 31001 \
    --dist-init-addr 127.0.0.1:31501 --disaggregation-mode decode \
    --disaggregation-decode-enable-radix-cache > "$LOG_DIR/decode.log" 2>&1 &
pids="$pids $!"
# 等 P、D 就绪后再起 router；任一服务退出时 kill -0 失败，脚本退出并停掉其余服务。
until curl -sf http://127.0.0.1:31002/health >/dev/null &&
    curl -sf http://127.0.0.1:31001/health >/dev/null; do
    kill -0 $pids
    sleep 10
done
python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:31002 31145 --decode http://127.0.0.1:31001 \
    --host 127.0.0.1 --port 31345 > "$LOG_DIR/router.log" 2>&1 &
pids="$pids $!"
until curl -sf http://127.0.0.1:31345/health >/dev/null; do
    kill -0 $pids
    sleep 5
done

"${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_pd_tp4.py" "$mode" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}"
