#!/bin/sh
# Run from the prepared SGLang environment. See oceandisk_pd_tp4.md.
REPO="${REPO:-/home/j00977581/sglang-pd}"
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8/}"
LOCAL_IP="${LOCAL_IP:?Set LOCAL_IP to the local reachable IP}"
OD_NAMESPACE="${OD_NAMESPACE:?Set the same OD_NAMESPACE on P and D}"
GIDS_DIR="${GIDS_DIR:?Set GIDS_DIR to the local gids_config_0..7 directory}"
MOONCAKE_IB_DEVICES="${MOONCAKE_IB_DEVICES:?Set the verified local Mooncake devices}"
OPEXTENSION="${OPEXTENSION:-/home/j00977581/gids/lib/libop_extension.so}"
LIBGIDS="${LIBGIDS:-${OPEXTENSION%/*}/libgids_kv_host.so}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-163840}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
STORAGE_CONFIG="{\"namespace\":\"${OD_NAMESPACE}\",\"layerwise_save\":true}"
STORAGE_CONFIG="${OD_STORAGE_CONFIG:-$STORAGE_CONFIG}"

# Keep physical GPU numbering for the native per-GPU GIDS config files.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SGLANG_HOST_IP="$LOCAL_IP"
export USE_SINGLE_PORT=0 OPEXTENSION LIBGIDS
export GIDS_DIR="${GIDS_DIR%/}/"
export PYTHONPATH="${REPO}/python:${OPEXTENSION%/*}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/home/cache/deep_gemm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/cache/inductor}"
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGL_DG_USE_NVRTC="${SGL_DG_USE_NVRTC:-1}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=6
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-600}"
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-600}"
export SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS="${SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS:-50}"
export SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS="${SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS:-8192}"
export SGLANG_OCEANDISK_TRACE="${SGLANG_OCEANDISK_TRACE:-0}"
export SGLANG_OCEANDISK_CACHE_DEBUG="${SGLANG_OCEANDISK_CACHE_DEBUG:-0}"
# UCL trace (see oceandisk_pd_tp4.md section 11): off unless both are set.
export SGLANG_UCL_TRACE="${SGLANG_UCL_TRACE:-0}"
export SGLANG_UCL_TRACE_DIR="${SGLANG_UCL_TRACE_DIR:-}"

export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export SGLANG_DISAGGREGATION_QUEUE_SIZE="${SGLANG_DISAGGREGATION_QUEUE_SIZE:-1}"
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE="${SGLANG_DISAGGREGATION_THREAD_POOL_SIZE:-1}"

python -m sglang.launch_server \
  --disaggregation-mode prefill \
  --port "${PORT:-31002}" \
  --base-gpu-id "${BASE_GPU_ID:-0}" \
  --dist-init-addr "${DIST_INIT_ADDR:-${LOCAL_IP}:31500}" \
  --data-parallel-size 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --disable-prefill-cuda-graph \
  --model-path "$MODEL_PATH" \
  --host "$LOCAL_IP" \
  --tensor-parallel-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --attention-backend dsa \
  --kv-cache-dtype bf16 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --moe-dense-tp-size 1 \
  --moe-a2a-backend none \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --watchdog-timeout 3600 \
  --log-level info \
  --trust-remote-code \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$MOONCAKE_IB_DEVICES" \
  --disaggregation-linker-notify \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend oceandisk \
  --hicache-storage-backend-extra-config "$STORAGE_CONFIG" \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-31145}"
