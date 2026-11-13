#!/bin/sh
# 132 混部（不分 P/D），参数照 run_prefill_xinlang.sh 的 P 侧：TP8 CP8 DP1，
# 去掉 PD 专用的 mooncake / bootstrap / notify。在 nsys 下启动，只采集
# /start_profile 与 /stop_profile 之间（由 run_profile_client.sh 触发），8 个 rank 都录。
# 采集结束后 Ctrl+C 停服务，nsys 写出 ${log_dir}/p_mixed_4k.nsys-rep，再跑 run_summary.sh。
set -x

local_ip=141.111.65.132
port=31010
model_path=/home/models/GLM-5-FP8
log_dir=/home/j00977581/p_profile
gids_lib=/home/j00977581/gids/lib

mkdir -p ${log_dir}
git -C /home/j00977581/sglang-pd rev-parse HEAD > ${log_dir}/commit.txt

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export SGLANG_HOST_IP=${local_ip}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_TIMEOUT_KEEP_ALIVE=600

# Kernel cache
mkdir -p /home/cache/deep_gemm /home/cache/triton /home/cache/inductor
export SGLANG_DG_CACHE_DIR=/home/cache/deep_gemm
export TRITON_CACHE_DIR=/home/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/cache/inductor
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGL_DG_USE_NVRTC=1

# xinlang config
export SGLANG_ENABLE_JIT_DEEPGEMM=true
export SGLANG_USE_DEEPGEMM_BMM=true

# for GIDS kernel
export GIDS_DIR=/home/j00977581/gids/cfg_fbs-h20-01/
export PYTHONPATH=/home/j00977581/sglang-pd/python:${gids_lib}
export OPEXTENSION=${gids_lib}/libop_extension.so
export LIBGIDS=${gids_lib}/libgids_kv_host.so
export USE_SINGLE_PORT=0

# nsys 录全部 8 个 rank（默认只录 GPU0）
export SGLANG_CUDA_PROFILER_ALL_RANKS=1

# UCL trace 默认关闭；需要和 nsys 对照时取消注释
# export SGLANG_UCL_TRACE=1
# export SGLANG_UCL_TRACE_DIR=${log_dir}

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o ${log_dir}/p_mixed_4k \
python -m sglang.launch_server \
  --log-level info \
  --model-path ${model_path} \
  --watchdog-timeout 3600 \
  --host ${local_ip} \
  --port ${port} \
  --tensor-parallel-size 8 \
  --moe-dense-tp-size 1 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --mem-fraction-static 0.85 \
  --deepep-mode normal \
  --dist-init-addr ${local_ip}:31500 \
  --page-size 64 \
  --enable-metrics-for-all-schedulers \
  --enable-metrics \
  --enable-cache-report \
  --trust-remote-code \
  --chunked-prefill-size 16384 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend oceandisk \
  --hicache-storage-backend-extra-config '{"layerwise_save":true}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8_e4m3 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --attn-cp-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --attention-backend dsa \
  --enable-layerwise-nvtx-marker \
  2>&1 | tee -i ${log_dir}/server_$(date +%m%d%H%M).log

# torch profiler（ACTIVITIES=CPU,GPU）时用这段：不带 nsys，trace 写到客户端的 PROFILE_DIR
# python -m sglang.launch_server \
#   --log-level info \
#   --model-path ${model_path} \
#   --watchdog-timeout 3600 \
#   --host ${local_ip} \
#   --port ${port} \
#   --tensor-parallel-size 8 \
#   --moe-dense-tp-size 1 \
#   --enable-prefill-cp \
#   --cp-strategy interleave \
#   --mem-fraction-static 0.85 \
#   --deepep-mode normal \
#   --dist-init-addr ${local_ip}:31500 \
#   --page-size 64 \
#   --enable-metrics-for-all-schedulers \
#   --enable-metrics \
#   --enable-cache-report \
#   --trust-remote-code \
#   --chunked-prefill-size 16384 \
#   --enable-unified-cache-external-linker \
#   --unified-cache-external-linker-backend oceandisk \
#   --hicache-storage-backend-extra-config '{"layerwise_save":true}' \
#   --tool-call-parser glm47 \
#   --reasoning-parser glm45 \
#   --kv-cache-dtype fp8_e4m3 \
#   --pipeline-parallel-size 1 \
#   --data-parallel-size 1 \
#   --attn-cp-size 8 \
#   --enable-dp-attention \
#   --enable-dp-lm-head \
#   --moe-a2a-backend deepep \
#   --attention-backend dsa \
#   2>&1 | tee -i ${log_dir}/server_$(date +%m%d%H%M).log
