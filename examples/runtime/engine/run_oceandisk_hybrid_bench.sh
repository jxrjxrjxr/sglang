#!/bin/sh
set -eu
# 使用方法：对已经起好的混合部署（单机、不分 PD）服务跑 bench，输出两张表，格式与
# run_oceandisk_pd_tp4.sh bench 相同。本脚本不起服务，先自行启动，例如
# run_oceandisk_online_tp8.sh，再把它的地址填到下面的 --host、--port。
# sh examples/runtime/engine/run_oceandisk_hybrid_bench.sh 2>&1 | tee hybrid_bench.log
# 单请求表：每个长度冷启动发一次，/flush_cache 清空服务的本地缓存后再发一次，
# 第二次的前缀从 OceanDisk 读回；清缓存要求服务空闲，测试期间不要有其他请求。
# 数据集每个请求用一行新的，行数要不少于：长度个数 + 2 x 并发档位之和（默认 62）。
# 参数直接改下面各行；--model-path 是本机 tokenizer 路径，删掉这一行就用服务端的路径。

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 黄区代理不放行内网地址。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

"${PYTHON_BIN:-python3}" -u "${script_dir}/oceandisk_hybrid_bench.py" \
    --host 127.0.0.1 \
    --port 9101 \
    --model-path /home/models/GLM-5.2-W4AFP8/ \
    --dataset-path /home/z00591349/dataset/multifieldqa_zh.jsonl \
    --latency-prompt-tokens 4096 16384 32768 60000 \
    --latency-output-tokens 128 \
    --throughput-prompt-tokens 8192 \
    --throughput-concurrency 1 4 8 16 \
    --throughput-output-tokens 256 \
    "$@"
