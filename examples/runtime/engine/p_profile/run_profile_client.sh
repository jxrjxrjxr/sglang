#!/bin/sh
# 单请求 4K、90% 命中（前缀从 OceanDisk 读回）的 P 侧 profile 客户端。
# 先起 run_mixed_server.sh，看到 "The server is fired up" 后再跑本脚本。
set -x

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export BASE_URL=http://141.111.65.132:31010
export INPUT_LEN=4096
# 58 页 × 64 = 3712，命中率 90.6%
export PREFIX_LEN=3712
export OUTPUT_LEN=1
export WARMUP=2
# 种子请求写盘后等多久再 flush GPU 缓存
export SETTLE_S=5
# 请求返回后再等多久停止采集（包住尾部 PUT）
export STOP_DELAY_S=2

# nsys：CUDA_PROFILER（服务端必须在 nsys 下启动）
export ACTIVITIES=CUDA_PROFILER
# torch profiler：改成 CPU,GPU，服务端用不带 nsys 的那段命令启动
# export ACTIVITIES=CPU,GPU
export PROFILE_DIR=/home/j00977581/p_profile
export PROFILE_PREFIX=p_mixed_4k

mkdir -p ${PROFILE_DIR}

python3 /home/j00977581/sglang-pd/examples/runtime/engine/p_profile/p_profile_client.py 2>&1 | tee -i /home/j00977581/p_profile/client_$(date +%m%d%H%M).log
