#!/bin/sh
set -eu
# 使用方法：单机 8 卡 PD，P 用 GPU0-3、D 用 GPU4-7（各 TP4），router 对外
# check：同一请求依次要求 D 从 OceanDisk 读回 → HBM 命中 → 清空 D 后再读回；另测并发与 D 部分命中
# bench：单请求冷启动与清空 D 后读回的 TTFT，以及不同并发下的吞吐
# sh examples/runtime/engine/run_oceandisk_pd_tp4.sh check 2>&1 | tee pd_check.log
# sh examples/runtime/engine/run_oceandisk_pd_tp4.sh bench 2>&1 | tee pd_bench_chunkload.log
# 默认 chunkload：OceanDisk 存储前缀 + Mooncake RDMA 尾部，D 分窗口装载。
# rounds：把整个 Round 目录（含 doupocangqiong.txt）拷到 134，再指定路径：
# ROUNDS_DIR=/path/to/Round sh ... rounds
# ROUNDS_ARGS 覆盖整组多轮参数；router 不支持 request-usage 时，从覆盖值中去掉它。
# 10 x 128K 约需 1.3M token 的 D 容量，不足时排队；保留日志里的容量与排队数据。
# chunkload、mooncake 各跑一次，日志、对话和 JSON 留在各自 LOG_DIR。
# 原生 PD 对照（mooncake，不经过 OceanDisk）：TRANSFER_BACKEND=mooncake sh ... bench
# GIDS_DIR 内放 gids_config_0..7.txt（D 用 --base-gpu-id 4，读 4..7）：
# OPEXTENSION=/path/to/libop_extension.so GIDS_DIR=/path/to/config sh ... check

# Optional P-side diagnostics (all disabled by default):
# SGLANG_MOONCAKE_TRANSFER_TRACE=1
# SGLANG_MOONCAKE_TRANSFER_WARN_SECONDS=30
# SGLANG_MOONCAKE_TRANSFER_TIMEOUT_SECONDS=120
# SGLANG_MOONCAKE_TRANSFER_FAIL_FAST=1  # stop P on deadline/worker exception

# Edit these paths for your machine.
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
DATASET_PATH="${DATASET_PATH:-/home/z00591349/dataset/multifieldqa_zh.jsonl}"
OPEXTENSION="${OPEXTENSION:-/opt/oceandisk/libop_extension.so}"
LIBGIDS="${LIBGIDS:-$(dirname -- "$OPEXTENSION")/libgids_kv_host.so}"
GIDS_DIR="${GIDS_DIR:-/opt/oceandisk/config}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-chunkload}"
MOONCAKE_IB_DEVICES="${MOONCAKE_IB_DEVICES:-mlx5_3,mlx5_5,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15}"
run_id="$(date +%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/oceandisk_pd/${run_id}}"

mode=check
case "${1:-}" in
    check|bench|rounds)
        mode="$1"
        shift
        ;;
esac
if [ "$mode" = check ]; then
    export SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS="${SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS:-0}"
fi
if [ "$mode" = rounds ]; then
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-163840}"
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1800}"
    # P 等待 D 握手的超时，D 容量不足时也可能长时间排队。
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1800}"
    ROUNDS_DIR="$(cd -- "${ROUNDS_DIR:?Set ROUNDS_DIR to the external Round directory}" && pwd)"
    test -f "$ROUNDS_DIR/run_rounds.py"
    test -f "$ROUNDS_DIR/doupocangqiong.txt"
    ROUNDS_ARGS="${ROUNDS_ARGS:---rounds 3 --batch-size 10 --initial-tokens 128000 --increment-tokens 256 --output-len 16 --record-response --request-usage}"
    # 检查本地模型配置，不自动放宽模型的位置长度限制。
    "${PYTHON_BIN:-python3}" - "$MODEL_PATH/config.json" "$CONTEXT_LENGTH" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
limit = config.get("text_config", config).get("max_position_embeddings")
if not isinstance(limit, int) or limit < int(sys.argv[2]):
    raise SystemExit(f"Model max_position_embeddings={limit}; need >= {sys.argv[2]}")
PY
fi
# OceanDisk trace：check 默认开启，bench 默认关闭以免影响计时。
if [ "$mode" = check ]; then trace=1; else trace=0; fi
export SGLANG_OCEANDISK_TRACE="${SGLANG_OCEANDISK_TRACE:-$trace}"

decode_args=
case "$TRANSFER_BACKEND" in
    chunkload)
        decode_args=--disaggregation-decode-linker-load
        GIDS_DIR="$(cd -- "$GIDS_DIR" && pwd)/"
        export USE_SINGLE_PORT=0 OPEXTENSION LIBGIDS GIDS_DIR
        export PYTHONPATH="$(dirname -- "$OPEXTENSION")${PYTHONPATH:+:${PYTHONPATH}}"
        # P、D 用同一个 namespace（存储 key 的前缀），每次运行换新的。
        set -- \
            --enable-unified-cache-external-linker \
            --unified-cache-external-linker-backend oceandisk \
            --hicache-storage-backend-extra-config \
            "{\"namespace\":\"pd-${run_id}\",\"connection_indices\":[0],\"layerwise_save\":true}" \
            --disaggregation-transfer-backend mooncake \
            --disaggregation-ib-device "$MOONCAKE_IB_DEVICES" \
            "$@"
        ;;
    mooncake)
        set -- --disaggregation-transfer-backend mooncake \
            --disaggregation-ib-device "$MOONCAKE_IB_DEVICES" "$@"
        ;;
    *)
        printf 'TRANSFER_BACKEND must be chunkload or mooncake\n' >&2
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
    --enable-prefill-cp \
    --cp-strategy interleave \
    --attention-backend dsa \
    --host 127.0.0.1 \
    --trust-remote-code \
    --page-size 64 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --enable-cache-report \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend none \
    --disable-prefill-cuda-graph \
    --mem-fraction-static 0.85 \
    --context-length "${CONTEXT_LENGTH:-65536}" \
    "$@"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd -- "$LOG_DIR" && pwd)"
printf 'mode=%s backend=%s logs=%s\n' "$mode" "$TRANSFER_BACKEND" "$LOG_DIR"
# Separate process groups let cleanup also stop scheduler/client children.
command -v setsid >/dev/null 2>&1 || { echo 'setsid is required' >&2; exit 1; }
pids=
client_pid=
cleanup() {
    trap '' INT TERM
    for pid in $pids $client_pid; do kill -TERM "-$pid" 2>/dev/null || true; done
    alive=
    for attempt in 1 2 3 4 5; do
        alive=
        for pid in $pids $client_pid; do
            if kill -0 "-$pid" 2>/dev/null; then alive="$alive $pid"; fi
        done
        [ -n "$alive" ] || break
        sleep 1
    done
    for pid in $alive; do kill -KILL "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_client() {
    while kill -0 "$client_pid" 2>/dev/null; do
        for pid in $pids; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Backend pid=$pid exited; stopping experiment (logs: $LOG_DIR)" >&2
                return 1
            fi
        done
        sleep 1
    done
    wait "$client_pid"
}

# P、D、router 依次启动，前一个就绪后再起后一个，避免 P、D 同时建链互相影响；
# 等待中任一服务退出时 kill -0 失败，脚本退出并停掉其余服务。
setsid python3 -m sglang.launch_server "$@" --base-gpu-id 0 --port 31002 \
    --dist-init-addr 127.0.0.1:31500 --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 31145 > "$LOG_DIR/prefill.log" 2>&1 &
pids="$pids $!"
until curl -sf http://127.0.0.1:31002/health >/dev/null; do
    kill -0 $pids
    sleep 10
done
setsid python3 -m sglang.launch_server "$@" --base-gpu-id 4 --port 31001 \
    --dist-init-addr 127.0.0.1:31501 --disaggregation-mode decode \
    --disaggregation-decode-enable-radix-cache $decode_args > "$LOG_DIR/decode.log" 2>&1 &
pids="$pids $!"
until curl -sf http://127.0.0.1:31001/health >/dev/null; do
    kill -0 $pids
    sleep 10
done
setsid python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://127.0.0.1:31002 31145 --decode http://127.0.0.1:31001 \
    --host 127.0.0.1 --port 31345 > "$LOG_DIR/router.log" 2>&1 &
pids="$pids $!"
# router 的 /health 只表示进程存活，/readiness 要等 P、D 都被判为健康。
until curl -sf http://127.0.0.1:31345/readiness >/dev/null; do
    kill -0 $pids
    sleep 5
done

if [ "$mode" = rounds ]; then
    cd -- "$ROUNDS_DIR"
    printf 'backend=%s context_length=%s rounds_args=%s\n' \
        "$TRANSFER_BACKEND" "$CONTEXT_LENGTH" "$ROUNDS_ARGS" > "$LOG_DIR/rounds-config.txt"
    status=0
    # ROUNDS_ARGS 是以空格分隔的参数列表，不作为 shell 代码执行。
    set -f
    setsid "${PYTHON_BIN:-python3}" -u run_rounds.py $ROUNDS_ARGS \
        --model "$MODEL_PATH" --base-url http://127.0.0.1:31345 \
        --conversation-dir "$LOG_DIR/conversations" \
        --output-json "$LOG_DIR/rounds.json" > "$LOG_DIR/rounds.log" 2>&1 &
    client_pid=$!
    wait_for_client || status=$?
    cat "$LOG_DIR/rounds.log"
    exit "$status"
fi
setsid "${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_pd_tp4.py" "$mode" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" --decode-log "$LOG_DIR/decode.log" &
client_pid=$!
wait_for_client
