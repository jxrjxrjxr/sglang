# OceanDisk notify 运行指南（单机 4P4D／双机 8P8D）

更新：2026-11-06（双机改用 `dual_node_pd/` 固定脚本，直接运行，不用 export 和 `OD_NAMESPACE`，见第 4 节；第 2、3 节及其余单机内容不变）。分支 `dev_oceandisk_linker_pd_notify`，参数对齐实现基于 `9dc1fcf964`，本版包含独立 P／D／router 入口及 D DP 支持。两机按第 2 节拉取 notify 分支最新提交，并记录实际提交和工作区 diff。**GPU、RDMA、GIDS 及双机实测仍待验证。**两台机器继续使用 `/home/j00977581/sglang-pd`，复用已准备的环境、模型和 GIDS 库。

## 1. 五个入口与固定行为

| 入口（相对 `examples/runtime/engine/`） | 用法 |
|---|---|
| `run_oceandisk_prefill.sh` | 必要赋值＋一条 `python -m sglang.launch_server`；前台运行 P |
| `run_oceandisk_decode.sh` | 同上；独立配置 D 的 TP、DP、物理 GPU 起点和 IP |
| `run_oceandisk_router.sh` | 一条 router 命令；P/D 地址独立 |
| `Round/run_rounds.py` | 基于 `D:\CodeSpace\LLMTest\Round`，保留原 TTFT／缓存统计口径；增加 TPOT；轮间不暂停、不清缓存，直接跑完 |
| `Round/run_rounds.sh` | 只有 `python3 run_rounds.py` 与参数；16000 起步、每轮 +16000、输出上限 16、10 会话 × 10 轮（约 16K→160K） |

以上五个入口用于单机和单项对照。132 P＋134 D 双机用 `dual_node_pd/` 下的固定脚本，见第 4 节。

服务脚本不启动其他服务、不等待健康检查、不清理进程、不运行压测。先在三个终端启动 P、D、router，再用第四个终端跑 Round。前台服务用 Ctrl+C 停止；需要后台日志时使用第 5 节的外部命令。

| 场景 | P | D | Router |
|---|---|---|---|
| 132 或 134 单机 | GPU0–3；TP4、CP4、DP1 | GPU4–7；TP4、DP2、attention TP2 | 同机 |
| 双机 | 132 GPU0–7；TP8、CP8、DP1 | 134 GPU0–7；TP8、DP4、attention TP2 | 134 |

默认模型 `GLM-5.2-W4AFP8`；BF16 KV、DSA、page 64、mem-fraction 0.85、dense TP1、MoE A2A `none`。P 用 interleave CP，禁用 prefill CUDA graph；D 开启 DP attention、DP LM head、round-robin 和 radix cache，max-running／CUDA graph max batch 均为 128。P chunk 和 D 加载窗口均为 16384。

notify 保持 **50 层一组，RDMA tail 显式为 8192**：P PUT 完成后发 READY，D 按通知 GET；P CP0 写出，D 每个 attention TP rank 读回；D 无存储 EXIST／PUT。加载完成、失败和 DMA 排空在请求所属 attention TP 组同步。复制式 KV 的存储 namespace 使用实例总 TP：P `attention TP × CP`、D `attention TP × DP`，DP rank 不进入 key。NULL 混部和非复制缓存规则不变。

D radix cache 用于本方案的前缀匹配、锁定、回收及装载槽位复用，必须保留。原生 PD 可用 chunk cache 管理请求生命周期，因而可以关闭跨请求 radix 复用；这不意味着 D 无需管理 KV。DP round-robin 不保证同一会话回到同一 DP 组，不清 D 时的命中率仍取决于分组路由。

## 2. 每个终端的环境与代码检查

在已有 pd 容器、已准备好的 SGLang Python 环境中操作。132／134 的 `/home` **不共享**；代码、模型、动态库、本机 GIDS 配置各自准备，客户端机器另备 Round 语料和 bench 数据集。不要把 132 的本地网卡/IP 配置直接复制给 134。

每个服务终端先执行本节公共块，再选一个机器块，连续实验时保留。namespace 及依赖它的变量单独放在第 2.1 节，一次实验所有终端、双机两端使用**同一个手工填写的值**。

```bash
export W=/home/j00977581
export REPO=$W/sglang-pd
export ENGINE=$REPO/examples/runtime/engine
export MODEL_PATH=/home/models/GLM-5.2-W4AFP8/
export DATASET_PATH=/home/z00591349/dataset/multifieldqa_zh.jsonl
export OPEXTENSION=$W/gids/lib/libop_extension.so
export LIBGIDS=$W/gids/lib/libgids_kv_host.so
export GIDS_DIR=$W/gids/cfg_$(hostname)/
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p /home/cache/deep_gemm /home/cache/triton /home/cache/inductor
cd "$REPO"
```

132 的机器块：

```bash
export LOCAL_IP=141.111.65.132
export MOONCAKE_IB_DEVICES=mlx5_1,mlx5_3,mlx5_5,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15
```

134 的机器块：

```bash
export LOCAL_IP=141.111.65.134
export MOONCAKE_IB_DEVICES=mlx5_2,mlx5_4,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14
```

上述设备来自已有环境记录：132 有全部 16 个 mlx5 设备；134 未发现 mlx5_0／1，偶数列表去掉 mlx5_0 后为 7 个。Mooncake 与 GIDS 分开网口：132 的 GIDS 用 port0（偶数），Mooncake 用奇数；134 的 GIDS 用 port1（奇数），Mooncake 用偶数。2026-11-04 前手册给 134 的是奇数列表，与 GIDS 共用 port1，按旧手册跑出的 134 结果可能受此影响。132 奇数到 134 偶数的跨机连通性尚未实测，双机前先用小流量验证。运行前用 `ibv_devices` 和 `nvidia-smi topo -m` 复核本机当前设备。GIDS 使用自己配置中的地址；Mooncake IB 列表不能替代 GIDS 配置。132 的 `gids_config_4.txt` 应已修正为 GID index 3，134 本地 `.200` 应已修正为 `.201`，以当前机器文件为准。

脚本保留全部 8 卡可见，通过 `--base-gpu-id` 选卡；不要在外层改成 `CUDA_VISIBLE_DEVICES=4,5,6,7`，否则 native 选择 `gids_config_<GPU编号>.txt` 的编号会改变。

两机各在原目录拉取远端 notify 分支；远端 `mine` 沿用已配置的个人 fork 和网络设置。切换前检查、保存已有工作，不做 reset 或强制覆盖：

```bash
git -C "$REPO" status --short --branch
git -C "$REPO" remote get-url mine
git -C "$REPO" fetch mine dev_oceandisk_linker_pd_notify
git -C "$REPO" switch dev_oceandisk_linker_pd_notify
git -C "$REPO" pull --ff-only mine dev_oceandisk_linker_pd_notify
git -C "$REPO" rev-parse HEAD
git -C "$REPO" diff --stat
```

没有本地分支时，用 `git switch --track -c dev_oceandisk_linker_pd_notify mine/dev_oceandisk_linker_pd_notify` 替换 switch 行。若分支被旧 worktree 占用，先检查 `git worktree list`，保存旧目录的工作，再在旧目录 detach；目录仍保留。

### 2.1 每次实验只改这一块

连续跑任务时保留 P、D、Router、Client 终端和上面的环境，只处理这里。先停 P、D，再把下面这块粘贴到 **P、D、客户端终端**，只改第一行；双机两端填相同的值。字符使用字母、数字、`-`、`_`。

```bash
export OD_NAMESPACE=notify-20261104-run01
export LOGS=/home/j00977581/pd_logs/$(hostname)/$OD_NAMESPACE
export SGLANG_UCL_TRACE_DIR=/home/j00977581/ucl_trace/$OD_NAMESPACE
mkdir -p "$LOGS"
```

- `run_oceandisk_prefill.sh`、`run_oceandisk_decode.sh` 启动时读取这些变量，改完必须重启 P、D。
- trace 目录跟随 namespace，不用时间戳，避免 P、D 的目录不一致。同一 namespace 重跑时新旧 trace 会混在一个目录，下一组实验要换值。
- 用过第 7 节的 `OD_STORAGE_CONFIG` 时，改 namespace 后要重新生成它。
- Router 不读取 namespace，启动命令不变；只有后台运行时的日志路径用到 `$LOGS`。

两机应为同一提交，若部署未提交改动则额外对比 patch／文件哈希。下面的命令写入 `$LOGS`，每次实验在每台服务机器上执行一次：

```bash
git -C "$REPO" rev-parse HEAD > "$LOGS/commit.txt"
git -C "$REPO" diff > "$LOGS/worktree.patch"
sha256sum "$OPEXTENSION" "$LIBGIDS" "$MODEL_PATH/config.json" > "$LOGS/artifacts.sha256"
sha256sum "$ENGINE"/run_oceandisk_{prefill,decode,router}.sh "$ENGINE/Round/run_rounds.py" > "$LOGS/scripts.sha256"
sha256sum "$GIDS_DIR"/gids_config_*.txt > "$LOGS/gids-config.sha256"
printf '%s\n' "$OD_NAMESPACE" > "$LOGS/namespace.txt"
```

模型需同一权重版本、tokenizer 和模型配置；`config.json` 相同不能单独证明权重相同，另对照已有模型 manifest。两端 BF16 KV、page64、总 TP、层数、层组50必须一致。本机配置文件哈希允许不同，记录它们用于追溯。

## 3. 单机：132 或 134 的 4P4D

选择目标机器，在三个终端都完成第 2 节公共块和该机器块，P、D 终端再完成第 2.1 节。P31002、D31001、router31345、P bootstrap31145 保持固定。

P 终端：

```bash
TP_SIZE=4 BASE_GPU_ID=0 sh "$ENGINE/run_oceandisk_prefill.sh"
```

D 终端：

```bash
TP_SIZE=4 DP_SIZE=2 BASE_GPU_ID=4 sh "$ENGINE/run_oceandisk_decode.sh"
```

Router 终端：

```bash
PREFILL_IP="$LOCAL_IP" DECODE_IP="$LOCAL_IP" sh "$ENGINE/run_oceandisk_router.sh"
```

P、D 的 dist-init 分别为本机 `:31500`、`:31501`。等待日志完成加载／预热，用第 5 节检查服务。三个脚本均保持前台运行，不会自行退出并提示总的 Ready。

## 4. 双机：132 P＋134 D/router

双机用 `dual_node_pd/` 下的四个 sh，所有取值都写在脚本里，**不用 export、不用第 2、2.1 节的变量，也不再用 `OD_NAMESPACE`**。需要改参数时直接改脚本里的那一行。

容器不限于 `j00977581_pd`，任何人起的 0.5.20 容器都可以，只要满足：`--gpus all --network host --ipc host --privileged --ulimit memlock=-1`，并挂载 `/home:/home`。代码固定在 `/home/j00977581/sglang-pd`，它在 `PYTHONPATH` 里优先于镜像自带的 sglang。

132，一个终端：

```bash
cd /home/j00977581/sglang-pd
git pull --ff-only mine dev_oceandisk_linker_pd_notify
sh examples/runtime/engine/dual_node_pd/run_prefill.sh
```

134，D 终端：

```bash
cd /home/j00977581/sglang-pd
git pull --ff-only mine dev_oceandisk_linker_pd_notify
sh examples/runtime/engine/dual_node_pd/run_decode.sh
```

134，router 终端（D 启动之后）：

```bash
cd /home/j00977581/sglang-pd
sh examples/runtime/engine/dual_node_pd/run_router.sh
```

134，client 终端（等下面的 readiness 返回成功）：

```bash
cd /home/j00977581/sglang-pd
curl --noproxy '*' -fsS http://127.0.0.1:31345/readiness
sh examples/runtime/engine/dual_node_pd/run_rounds.sh
```

| 脚本 | 机器 | 固定内容 |
|---|---|---|
| `dual_node_pd/run_prefill.sh` | 132 | P：GPU0–7、TP8、CP8、DP1；Mooncake 用奇数网卡；GIDS 配置 `cfg_fbs-h20-01` |
| `dual_node_pd/run_decode.sh` | 134 | D：GPU0–7、TP8、DP4、attention TP2；Mooncake 用偶数网卡；GIDS 配置 `cfg_fbs-h20-02`；`LOAD_PAGES=0` |
| `dual_node_pd/run_router.sh` | 134 | router：P `132:31002`，bootstrap 31145；D `134:31001`；监听 31345 |
| `dual_node_pd/run_rounds.sh` | 134 | Round：16000 起步、每轮 +16000、输出上限 16、10 会话 × 10 轮 |

P、D 都开 `SGLANG_UCL_TRACE=1` 和 `SGLANG_UCL_TRACE_GPU=1`，这是 run03 的打点条件。连接数保持默认 8。

**日志。** P、D 每次启动在本机建 `/home/j00977581/pd_logs/<月日时分>/`（如 `11061644`），并让 `/home/j00977581/pd_logs/latest` 指向它。两台机器各用各的目录，目录名按各自的启动时间，不要求相同。

| 机器 | 目录内容 | 生成时机 |
|---|---|---|
| 132 | `prefill.log`、`commit.txt`、`gids_lib.sha256`、P 的 trace `*.jsonl`；`report.md`、`summary.json`、`trace.json` | report 在 P 上按 Ctrl+C 退出后生成，只含 P 的事件 |
| 134 | `decode.log`、`router.log`、`client.log`、`rounds.json`、`conversations/`、`commit.txt`、`gids_lib.sha256`、D 和客户端的 trace `*.jsonl`；`report.md`、`summary.json`、`trace.json` | report 在 Round 结束后自动生成，含 D 和客户端的事件 |

- 134 的 router、Round 写入 `latest`，所以**先起 D，再起 router 和 Round**。
- 重跑 Round 前重启 D：否则新旧 trace 进同一个目录、`rounds.json` 被覆盖。重启 P 同理。
- 132 的 P 不是用 Ctrl+C 退出时，手动生成：`python3 /home/j00977581/sglang-pd/examples/runtime/engine/Round/ucl_trace_tool.py report /home/j00977581/pd_logs/latest`。
- 想看 P、D 合在一张报告里：把 132 目录的 `*.jsonl` 拷到 134 的一个新目录（连同 134 的 `*.jsonl`），再对新目录跑上面的 report；跨机差值不计算。

**namespace。** 脚本的存储配置不带 `namespace`：原生层用 GIDS 库编译时固定的 ns id（`g_kvnsid`、`g_kvnsid_index`），每次都一样，不刷新。SGLang 的 key 哈希退回模型路径，两端都是 `/home/models/GLM-5.2-W4AFP8/`，因此 P、D 的 key 一致；改模型路径时两个脚本要一起改。

- 两机必须是同一份 GIDS 库：对比两边目录下的 `gids_lib.sha256`，不同就可能各写各的 ns。
- 存储上的旧数据不再被 namespace 隔开。Round 默认每次随机取语料，新旧前缀不会撞上；**加 `--use-fix-prompt` 重跑相同文本时，P 可能命中上一次写入的数据**，TTFT 不再是冷启动口径。

这是两个独立 TP8 实例。P dist-init 是 `141.111.65.132:31500`，D 是 `141.111.65.134:31501`，不是跨两机的 TP16。两端要能访问服务/bootstrap 和 Mooncake 建立的数据／控制连接；仅 HTTP 端口通不能证明 RDMA 正常。

多连接沿用张圆现有 `ConnectionRouter`：省略 `connection_indices` 选择配置中的全部连接（最多 8 条）；同一 DPU 的不同连接可轮转。同 key 的 P 写和 D 读必须选择同一个逻辑 DPU。

- Python router 用配置第 6 字段作为 DPU 分组 ID；当前提供的 native 源码把它解释为 `qp_cnt`。实施未改 ABI、库或配置格式。先向库维护者核对部署版本中该字段的实际契约，不能直接把任意 QP 数当唯一 DPU ID。
- 132 的 `.10/.12/.14/.16` 与 134 的 `.11/.13/.15/.17` 是不同存储端口。P/D 必须有一致的逻辑 DPU 集合／hash ring，并验证跨端口共享 key 空间及 PUT 完成后 GET 可见。
- 同 key 可能从同一 DPU 的不同连接读取；即使单机，也需确认该 DPU 的端口间可见性。单连接成功不能替代多连接验证。

**双机路径尚未实测。**先用小请求、tail=0 验证 P PUT→READY→D GET，再清 D 重复读回；逐个 DP 组核对日志、输出和 native 完成状态。双机先跑同拓扑原生 Mooncake 小请求排除网络问题，再启用 notify。不满足跨端口可见性时先停在诊断阶段，不运行十轮长负载。

## 5. 健康检查、外部后台运行与停止

客户端／检查终端设置地址：单机全部填实际目标机器 IP；双机按下例。P/D 绑定实际 IP，不使用 `127.0.0.1` 检查它们；router 默认监听 `0.0.0.0`，在 router 所在机器可用 localhost。

```bash
export P_IP=141.111.65.132
export D_IP=141.111.65.134
export ROUTER_IP=141.111.65.134
curl --noproxy '*' -fsS "http://$P_IP:31002/health"
curl --noproxy '*' -fsS "http://$D_IP:31001/health"
curl --noproxy '*' -fsS "http://$ROUTER_IP:31345/readiness"
curl --noproxy '*' -fsS "http://$D_IP:31001/model_info"
curl --noproxy '*' -sS "http://$ROUTER_IP:31345/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/models/GLM-5.2-W4AFP8/","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"max_tokens":16,"temperature":0,"stream":false}'
```

若想保持终端可用，在**第 3／4 节对应启动位置**把前台命令换成下面的外部后台命令。不要重复启动；以下是单机示例，双机 P 改 TP8，D 改 TP8/DP4/base0，router 改远端 IP。PID 文件、日志均按角色分开。

```bash
TP_SIZE=4 BASE_GPU_ID=0 nohup setsid sh "$ENGINE/run_oceandisk_prefill.sh" > "$LOGS/prefill.log" 2>&1 < /dev/null &
echo $! > "$LOGS/prefill.pid"
TP_SIZE=4 DP_SIZE=2 BASE_GPU_ID=4 nohup setsid sh "$ENGINE/run_oceandisk_decode.sh" > "$LOGS/decode.log" 2>&1 < /dev/null &
echo $! > "$LOGS/decode.pid"
PREFILL_IP="$LOCAL_IP" DECODE_IP="$LOCAL_IP" nohup setsid sh "$ENGINE/run_oceandisk_router.sh" > "$LOGS/router.log" 2>&1 < /dev/null &
echo $! > "$LOGS/router.pid"
```

`setsid` 可能 fork，因此 `$!` 是启动 PID 提示，停止前必须检查实际进程组；不能把旧 PID 文件直接当 kill 目标。用本次日志目录定位、核对当前命令与端口，然后向确认的 PGID 发 TERM（同一 pd 容器内）：

```bash
cat "$LOGS/decode.pid"
ps -eo pid,ppid,pgid,sid,args --forest | grep -E 'run_oceandisk_|sglang.launch_server|sglang_router'
ss -ltnp | grep -E ':31001|:31002|:31345'
# 将下面数字替换为刚核对的本次 decode 进程组；P/router 分别处理。
kill -TERM -- -12345
```

前台启动用各终端 Ctrl+C。停止后复核监听和 GPU worker 已退出，再启动下一组。客户端 Ctrl+C 只结束 Round，不会停止服务。日志查看：`tail -f "$LOGS/decode.log"`。短请求诊断在 P/D 启动前设置 `SGLANG_OCEANDISK_TRACE=1`，保存 `PUT_DONE/READY_SEND` 和 `READY_RECV/LOAD_STARTED/LOAD_DONE`；正式计时回到默认 0。load tag 为 `rid@start_page:layer_start`。结构化的流程与耗时记录见第11节 UCL trace。

## 6. Round 与缓存条件

在 router 所在机器准备语料。仓库不存语料副本；已有文件或链接时跳过 ln：

```bash
cd /home/j00977581/sglang-pd/examples/runtime/engine/Round
ln -s /home/j00977581/Round/doupocangqiong.txt ./doupocangqiong.txt
```

默认十轮（客户端在 router 机器上，模型位于默认路径）：

```bash
cd /home/j00977581/sglang-pd/examples/runtime/engine/Round
sh run_rounds.sh
```

Round 的 `--model`、`--base-url` 是显式参数，**不会读取服务脚本的 MODEL_PATH**；改模型或从远端跑客户端时相应修改 sh 参数。默认输出为 `rounds.json` 和 `conversations/`，重跑前移走旧结果或改输出路径。

`run_rounds.py` 轮间不再暂停，也不清缓存，启动后直接跑完全部轮次并保存最终 JSON。默认协议是 **P、D 均不 flush**，模拟真实场景；D 命中率仍取决于 DP round-robin 的分组路由。

需要专项诊断时，可在请求全部结束后手动 flush。Round 不再提供轮间等待，因此这些命令不用于多轮连续负载：

```bash
# 使用第 5 节的实际 P_IP/D_IP。只清 D：
curl --noproxy '*' -i --max-time 60 -X POST "http://$D_IP:31001/flush_cache?timeout=30"
# 清 P+D 时，再清 P：
curl --noproxy '*' -i --max-time 60 -X POST "http://$P_IP:31002/flush_cache?timeout=30"
```

确认每次均为 HTTP200／`Cache flushed.`；400 是清理失败。D 的服务端 flush 面向该实例全部 DP 组，仍需从各 scheduler 日志确认无在途请求。flush 释放缓存状态，模型和预分配 KV 池仍在，`nvidia-smi` 不会归零。独立实验重启服务并换 namespace；仅清 P+D 不清 OceanDisk。

先用相同负载的 batch=1 分离单请求成本，再跑默认 batch=10；完整 batch=1 参数：

```bash
python3 run_rounds.py \
  --base-url http://127.0.0.1:31345 \
  --model /home/models/GLM-5.2-W4AFP8/ \
  --rounds 10 --batch-size 1 --initial-tokens 16000 --increment-tokens 16000 \
  --output-len 16 --record-response --request-usage \
  --conversation-dir ./conversations_16k_increment_b1 --output-json ./rounds_16k_increment_b1.json
```

默认负载对齐张圆旧命令：initial=16000、increment=16000、10 轮、并发 10、输出上限 16（目标 16K→160K，包含实际历史／模板开销，以 usage 为准）。`bash run_rounds.sh` 使用同样的负载参数；以下命令另指定结果保存路径：

```bash
python3 run_rounds.py \
  --base-url http://127.0.0.1:31345 \
  --model /home/models/GLM-5.2-W4AFP8/ \
  --rounds 10 --batch-size 10 --initial-tokens 16000 --increment-tokens 16000 \
  --output-len 16 --record-response --request-usage \
  --conversation-dir ./conversations_16k_increment --output-json ./rounds_16k_increment.json
```

Round 的输出 16 是请求上限，保留原件的 EOS／思考处理；可能提前结束，实际 output tokens 必须记录。它的 TTFT 是**首条非空 SSE 行**，不是首个内容 token；不能与 bench TTFT 混用。原始 128K 小增量对照仍可使用 `--initial-tokens 128000 --increment-tokens 256`。

新增 `Avg TPOT`／`Max TPOT`，计算、聚合及输出直接复用 `run_test_rounds_v6.py`：每请求 `TPOT = (total_latency_ms - ttft_ms) / (output_tokens - 1)`，其中 TTFT 仍为首条非空 SSE，`output_tokens` 沿用回复正文的本地计数。输出不足两个 token 时该请求 TPOT 为 `None`；每轮只聚合成功且 TPOT 非 `None` 的请求，无有效值时 avg/max/min 为 0；跨轮平均只纳入 `avg_tpot_ms > 0` 的成功轮。JSON 与 v6 一致增加 `avg_tpot_ms`、`max_tpot_ms`、`min_tpot_ms`，终端显示 Avg／Max 两列、两位小数。

Round Python 已在 LLMTest 原件上增加上述统计，当前文件不再与原件哈希相同；历史原件 SHA256 为 `d94abd42e4da3b8447ca4992658e9d99dcd83f6e4f66147ae0fa47acc02a04f9`。

## 7. 逐项性能对照与诊断客户端

旧 `9dc1fcf964` 的实测用于历史参照。为了只改一个因素，新的 A/B 共用模型、GPU、cache 条件、max-running128、graph128、timeouts、queue/thread、层组50和 tail8192；同一组 P/D 的 namespace 相同，不同组换值：

| 档位 | D DP | P chunk／D window | 连接 |
|---|---:|---:|---|
| A | 1 | 8192 | `[0]` |
| B | 2（双机4） | 8192 | `[0]` |
| C | 2（双机4） | 16384 | `[0]` |
| D（新默认） | 2（双机4） | 16384 | 全部配置 |

在 P/D 启动终端都设置 `CHUNKED_PREFILL_SIZE=8192` 或 16384，D 用 `DP_SIZE=1/2/4`。单连接对照两端显式设置：

```bash
export OD_STORAGE_CONFIG="{\"namespace\":\"${OD_NAMESPACE}\",\"layerwise_save\":true,\"connection_indices\":[0]}"
```

恢复全部连接用 `unset OD_STORAGE_CONFIG`。跨机 `[0]` 必须对应同一个逻辑 DPU。override JSON 会整体替代默认配置，改 namespace 后要重新生成它。tail 对照在两端都设 `SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=0`；其他对照固定 8192，不随 chunk 改动。

D 单次 GET 默认最多 256 页（`SGLANG_DISAGGREGATION_LINKER_LOAD_PAGES`），大范围分片读回，片间让 forward 推进；设为 0 恢复一次读完，用于 A/B。

验证 D 的 GET 是否与 forward 共用硬件队列：P、D 启动前 `export CUDA_DEVICE_MAX_CONNECTIONS=32`（默认 8），其他条件不变，对比持续 decode 间隔 p99；报告第 8.3 节列出各条 GPU 流。

默认旧超时均为 600，watchdog3600。长队列实验在两端启动前覆盖：

```bash
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_HEALTH_CHECK_TIMEOUT=1800
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
```

每档保存各 DP 组 `max_total_num_tokens`、实际 running／prealloc／queue、请求数、输入输出 tokens、TTFT 分布及错误。247616 token 是**旧 D TP4/DP1** 的容量，不能沿用为新 DP2 的容量。一个 attention TP 组内的复制 KV 不能按 rank 相加；跨 DP 组才是独立容量。先确认每组能容纳一条 128K，再评估总并发是否提升。

现有诊断 Python 可直接连接已启动服务，不再用聚合 launcher 重启它们：

```bash
python "$ENGINE/oceandisk_pd_tp4.py" bench \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --router-url "http://$ROUTER_IP:31345" --decode-url "http://$D_IP:31001"
```

bench 有自己的输入长度、输出数和采样口径，报告 TBT mean/max；不是默认十轮的 TBT。它会在单请求 cold/reload 对照中清 D，运行时不要同时跑 Round；cold 标签只表示该次新 prompt，是否真正冷缓存仍需控制初始状态。

旧 `check` 第二次请求要求同一 D HBM 命中，不能直接作为 DP round-robin 的验收。做这项专项诊断时，重启 D 为 DP1，P/D 打开 trace、tail=0，然后直接运行：

```bash
python "$ENGINE/oceandisk_pd_tp4.py" check \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --router-url "http://$ROUTER_IP:31345" --decode-url "http://$D_IP:31001" \
  --decode-log "$LOGS/decode.log"
```

DP 小请求示例：P/D 两端先设置 tail=0 和 trace=1 后启动，随后在 Round 目录执行；两轮之间不清 D，覆盖的是不 flush 时的路径。D 恢复路径用上面的 bench reload 对照验证。batch4 可覆盖单机两组或双机四组，但是否实际分发到每组仍以日志为准。

```bash
python3 run_rounds.py \
  --base-url http://127.0.0.1:31345 \
  --model /home/models/GLM-5.2-W4AFP8/ \
  --rounds 2 --batch-size 4 --initial-tokens 16000 --increment-tokens 256 \
  --output-len 16 --record-response --request-usage \
  --conversation-dir ./conversations_dp_smoke --output-json ./rounds_dp_smoke.json
```

DP2/4 验收另外进行：先发超过 DP 组数的小请求，确认每组收到请求且能独立完成；清 D 后恢复；再混合不同前缀长度和完成时刻，观察一个组等 READY／GET 时其他组仍能推进。查各组 rid 与 READY/load 范围，确认 D 无 EXIST／PUT、失败时 DMA 排空。小请求用 tail0 才能覆盖 OceanDisk 读回，默认 tail8192 可能让短 prompt 全走 RDMA。

## 8. 同配置原生 Mooncake 对照

停掉本组服务、更换实验 namespace／日志目录，保持相同模型、TP/DP、chunk、cache 和负载。用下面命令从新脚本生成临时对照脚本，删除四条 notify/linker 参数；原始日常入口不变，router 与 Round 照常使用：

```bash
sed '/^  --disaggregation-linker-notify /d; /^  --enable-unified-cache-external-linker /d; /^  --unified-cache-external-linker-backend /d; /^  --hicache-storage-backend-extra-config /d' \
  "$ENGINE/run_oceandisk_prefill.sh" > "$LOGS/prefill_mooncake.sh"
sed '/^  --disaggregation-linker-notify /d; /^  --enable-unified-cache-external-linker /d; /^  --unified-cache-external-linker-backend /d; /^  --hicache-storage-backend-extra-config /d' \
  "$ENGINE/run_oceandisk_decode.sh" > "$LOGS/decode_mooncake.sh"
```

在 P/D 各自机器和终端、用第 3／4 节相同的 TP/DP/base 设置启动对应临时脚本。剩余 notify 环境变量在关闭该功能时不参与传输，原生 Mooncake 负责全部缺失 KV；native 基线也保留 D radix cache，避免比较时改变命中条件。

## 9. 0.5.13 → 当前 0.5.20 分支的参数迁移

以下按提供的 132 prefill、134 decode 原件和本分支实际消费点核对；旧脚本同名不代表同参数。

| 旧项 | 新入口处理／原因 |
|---|---|
| `--enable-nsa-prefill-context-parallel`、`--nsa-prefill-cp-mode round-robin-split` | P 用 `--enable-prefill-cp --cp-strategy interleave`；D 不开 CP |
| D `TP8/DP4 + --enable-dp-attention --enable-dp-lm-head` | 双机恢复；单机等比例为 TP4/DP2，attention TP 都是2 |
| `--prefill-round-robin-balance` | 移除；旧版本已是 deprecated no-op；D 用仍有效的 `--load-balance-method round_robin` |
| `--cuda-graph-max-bs 128` | 当前版本已拆分；D 必须用 `--cuda-graph-max-bs-decode 128`，旧拼写会因 decode／prefill 两个候选而报 ambiguous option。脚本环境变量仍为 `CUDA_GRAPH_MAX_BS`；P 保持禁用 prefill graph |
| P chunk16384 | 恢复；D notify load window 同为16384；tail 独立固定8192 |
| P/D queue、thread | 恢复实际最终值 P=1/1、D=2/8；旧 D 后面的 export 覆盖了前面的1/1 |
| dense-attention threshold | 恢复 P=0、D=20；对纯 D decode 的主路径收益不能据此推算 |
| `SGLANG_DG_CACHE_DIR`、Triton/Inductor cache、JIT/fast warmup | 恢复持久目录和有效开关；主要减少重复启动编译，不能直接算进稳态 TTFT/TBT 收益 |
| `SGLANG_DG_USE_NVRTC=1` | 使用实际被 `srt/layers/deep_gemm_wrapper/compile_utils.py` 消费的 `SGL_DG_USE_NVRTC=1`（映射 `DG_JIT_USE_NVRTC`）；旧 SGLANG 拼写只声明而未被这条路径消费 |
| `SGLANG_USE_DEEPGEMM_BMM=true` | 移除；GLM 路径无该拼写消费点，不自动换成会改变行为的 `SGL_USE_DEEPGEMM_BMM` |
| `--deepep-mode normal/low_latency`、`moe_bk=deepep` | `--moe-a2a-backend none` 下不选择 DeepEP；移除无效搭配和未引用赋值 |
| `ENABLE_SWAPAB`、`USE_MULTI_LINK`、`--enable-transfer-metadata` | 不复制到当前路径；前两者未找到有效消费，后者为旧私有扩展；现有 Mooncake metadata 机制继续使用 |
| 单连接 `connection_indices:[0]` | 默认省略，使用 native 配置中全部连接；单连接留作显式 A/B |
| watchdog／旧超时 | watchdog3600、bootstrap/wait/health/keepalive600、heartbeat6；长队列另按第7节覆盖1800 |

源码定位：`arg_groups/pd_disaggregation_hook.py`（支持范围）、`mem_cache/storage/oceandisk/oceandisk_direct_linker.py`（key）、`connection_router.py`／`oceandisk_storage.py`（连接及 native ABI）、`disaggregation/decode_linker_mixin.py`／`unified_radix_cache.py`（attention 组同步）、`srt/layers/deep_gemm_wrapper/compile_utils.py`（NVRTC）。

仍有意保留的差异：0.5.13 使用 GLM-5-FP8，现用 GLM-5.2-W4AFP8；单机4P4D资源少于旧双机8P8D；notify 50层通知、存储前缀＋RDMA tail8192、P、D 全程不 flush，均不等同于张圆 NULL 混部。Q8KV8、FP8 KV、layer split、NVLS 和 speculative decoding 不进入本轮对齐基线，单列后续实验，不承诺耗时下降比例。notify 当前拒绝非复制 DSA、PP>1、DCP>1、P DP>1、无 DP attention 的 D DP，以及 speculative／D offload／staging／HiCache 组合。

## 10. 验证状态与旧入口

本地用现有 uv Python、torch 执行实际源码的 CPU/mock 隔离测试：notify/加载/脚本74项中68通过、6项POSIX进程测试因Windows跳过；key专项5项、路由及descriptor专项6项通过。shell语法与真实参数展开（含必填变量、单双机和tail覆盖）通过；首次对齐时 Round Python与用户原件哈希一致；本轮 TPOT 扩展后以当前文件为准。模拟 hash 验证的是路由逻辑，跨语言 xxh3 和 native 可见性仍须实机确认。

扩大执行旧存储测试时发现既有失配：旧 fixture 缺 `_put_io_lock`、仍传 `max_ranges`，及旧逐层失败断言不适配当前批量接口。用 `9dc1fcf964` 原件复跑仍是1 failure／14 error记录（含subtest）；本轮不改这些旧接口和测试预期。完整Linux导入、GPU/RDMA/GIDS、DP实际容量、双机正确性和性能尚未验证。

黄区顺序：小请求各DP组→清D恢复→A/B单因素→128K batch1→batch10十轮→16K递增十轮→同配置Mooncake及独立TBT；双机先验证版本、网络和跨端口可见性再重复。100/100客户端完成不替代原生状态和答案核查；1842／Batch_Retrive 等 native 错误必须单独记录。

旧 `run_oceandisk_pd_tp4.sh`、`oceandisk_pd_tp4.py` 保留；原有 launcher 停止修复与测试随本版交付。旧 launcher 的 `start/check/bench/rounds/stop` 继续按旧参数运行，不会继承新脚本的 DP／chunk／连接默认值。日常使用本指南五个入口。

旧 launcher 的 `server.pids` 只是当次容器进程组快照，与宿主机或GPU worker PID不一定一致。它的 `stop` 按当前监听的 `127.0.0.1:31002/:31001/:31345` 查找 SGLang，先收集主进程及子进程，TERM后等5秒再KILL残留，不依赖旧PID文件，且不结束端口上的非SGLang进程。**该 stop 仅用于旧 launcher 的 localhost 服务**；新入口绑定实际IP，按第5节停止，避免留下P/D却只停掉router。

## 11. 性能打点（UCL trace）

默认关闭；关闭时每个埋点只多一次布尔判断，调度阶段装饰器和 `exists_many` 不加任何包装层。打开后各进程把事件写入同一目录（`<host>_<pid>.jsonl`，由后台线程建文件并每秒刷盘；目录不可写时只在 stderr 提示一次并自动关闭打点，不影响服务），跑完生成报告和时间线。与上面的 `SGLANG_OCEANDISK_*` 日志相互独立，做开销对照时两组保持相同设置，建议都关闭。

1. P、D、客户端终端各设置一次开关，连续实验时保留；目录由第 2.1 节的 `SGLANG_UCL_TRACE_DIR` 按 namespace 设置。两者都在启动时读取，不能热切换，改后要**重启 P/D**：

   ```bash
   export SGLANG_UCL_TRACE=1
   ```

   `dual_node_pd/` 的脚本已自带开关和目录，不用设置，目录与报告位置见第 4 节。合并两机报告时，工具标注“跨机未校准”，不计算跨机差值。

2. 照常 `sh run_rounds.sh`。结束后自动等待3秒并生成 `report.md`、`trace.json`、`summary.json`。客户端会给每个请求传 `rid=r<轮>-<序号>-<随机>`；若 router 拒绝该字段，设 `SGLANG_UCL_TRACE_CLIENT_RID=0`（仍记录客户端时间，但无法与服务端按 rid 关联）。自动报告在 P/D 仍运行时执行，会占用少量 CPU 和磁盘；连续正式测数时不要让它与下一次运行重叠。

3. 手动生成或对比：

   ```bash
   cd /home/j00977581/sglang-pd/examples/runtime/engine/Round
   python3 ucl_trace_tool.py report  "$SGLANG_UCL_TRACE_DIR"
   python3 ucl_trace_tool.py compare 目录A 目录B
   ```

   短时抓 GPU 时间线时，在 P、D 启动前另设 `SGLANG_UCL_TRACE_GPU=1`，报告第 8.1 节会多出 GPU forward 和 GPU GET 两列。这些事件进入 GPU 流，做开销对照时保持关闭。

   `trace.json` 用 Chrome 的 `chrome://tracing` 或 Perfetto 打开：每个进程一组原始线程轨道；另有“请求并行轨道”，每个请求一组泳道（客户端、P 调度、P PUT、P READY、D READY、D 状态、D 提示、D load、D 传输/放行、D 调度），按 rid、room、load tag、GET batch 和 PUT session 关联，可以直接对比并发请求的进度；“D 调度”含该请求参与的每个 decode batch，可与客户端帧间隔对照；卡住的 load／GET batch 画到最后一次刷盘并标“进行中”。

4. 服务卡住时不必先停服务：直接对目录执行 report。第1节列出没有结束标记的进程和未闭合的操作，第7节列出每个未放行请求最后到达的事件，以及它在最后状态已经停留的时间（计到该进程最后一次刷盘）。

报告口径：

| 报告项 | 含义与限制 |
|---|---|
| 客户端首帧／首内容 | 首帧是 rounds.json 旧 TTFT 口径；首内容是第一个带 content 的帧；帧间隔不等于逐 token TBT |
| D 请求状态 | 等待READY → READY已到·未提交 → 本rank就绪·等其他rank → load进行中 → prefix完成·等传输/放行，互斥且加总等于 D 侧时间（到放行或 abort；未结束的计到最后一次刷盘）；“READY已满足·被inflight限制”是可重叠的提示，属于候选优化点，需再验证 |
| GET／PUT 事件链 | 入队 → worker 取出 → 前置依赖就绪 → native 调用（另列进入前的锁/上下文等待）→ worker 观察完成 → scheduler 消费；“worker 观察完成”不等于设备完成；入队时间在放入队列之前记录 |
| notify 传递 | 发送时间在发送调用之前记录，延迟包含发送调用本身；按同一接收方 session 顺序配对，仅同机计算；多个 P 进程向同一接收方发送相同范围、或找不到更早发送记录的接收，分别计数、不计入延迟 |
| 调度 | 实际执行的 batch 逐个记录；其他调度阶段和 rank 一致性检查按次数／合计／最大值累计，超过 `SGLANG_UCL_TRACE_SLOW_MS`（默认2）单独记录；run_batch 是 CPU 侧耗时 |

L1 不证明 GPU 计算与 I/O 的真实重叠；要下“已实现掩盖”的结论，先用短时 nsys 核验。开销初筛：同负载 L0/L1 交替各跑3次，比较 TTFT、平均 TPOT、帧间隔尾部和吞吐；六次本身波动超过2%时判为暂时无法判断。
