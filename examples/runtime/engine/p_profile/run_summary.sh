#!/bin/sh
# 把 nsys-rep 压成几百行文本，用来粘贴回蓝区给 AI。服务端 Ctrl+C、nsys 写完报告后再跑。
# 第一次会在报告旁边生成同名 .sqlite（耗时几分钟），之后直接读 sqlite。
set -x

export TOP=15
export GAP_MS=1.0
export GAPS_PER_GPU=10
export STREAMS_PER_GPU=6
export SEGMENTS=40
export KEYWORDS=gids,dispatch,kvcache,oceandisk,nccl,deep_ep

python3 /home/j00977581/sglang-pd/examples/runtime/engine/p_profile/nsys_summary.py \
  /home/j00977581/p_profile/p_mixed_4k.nsys-rep \
  > /home/j00977581/p_profile/summary.txt

wc -l /home/j00977581/p_profile/summary.txt
