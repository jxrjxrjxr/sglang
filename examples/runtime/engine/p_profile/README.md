# P 侧单请求 profile（混部，4K，90% OceanDisk 命中，8 个 rank）

目的：找 PD 分离中 P 侧的卡点。用混部近似 P：参数照 `run_prefill_xinlang.sh`（TP8 CP8 DP1、DSA、DeepEP normal、fp8 KV、layerwise save），去掉 mooncake / bootstrap / notify。

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `run_mixed_server.sh` | 132 终端 1 | nsys 下启动混部服务端，8 个 rank 都录 |
| `run_profile_client.sh` → `p_profile_client.py` | 132 终端 2 | 预热 2 轮，然后只对 1 个 4K 请求开关 profile |
| `run_summary.sh` → `nsys_summary.py` | 132 | 把 `.nsys-rep` 压成约 300 行文本，用于贴回蓝区 |

## 1. 黄区切到正确的分支

代码在 `jxr/dev`（基于吴穹 `od_linker_dev`）。先在蓝区把改动提交并 `git push mine jxr/dev`，然后在 132：

```bash
cd /home/j00977581/sglang-pd
git status --short                       # 有本地改动先 git stash，别直接覆盖
git config --replace-all remote.mine.fetch '+refs/heads/*:refs/remotes/mine/*'
git fetch mine
git switch -c jxr/dev --track mine/jxr/dev   # 第一次；以后用 git switch jxr/dev && git pull --ff-only
git log -1 --oneline                     # 和蓝区推送的提交对一下
```

`remote.mine.fetch` 那一行是给单分支克隆用的：不加的话，`git fetch mine` 看不到新分支。服务端脚本会把 HEAD 写到 `/home/j00977581/p_profile/commit.txt`。

容器里先确认有 nsys：`nsys --version`。

## 2. 跑

```bash
sh run_mixed_server.sh       # 终端 1，等到 "The server is fired up"
sh run_profile_client.sh     # 终端 2，约 1 分钟
# 终端 1 Ctrl+C，等 nsys 写完 /home/j00977581/p_profile/p_mixed_4k.nsys-rep
sh run_summary.sh            # 生成 /home/j00977581/p_profile/summary.txt
```

**先看客户端最后一行 `[profile] ... hit=90.6%`**。不是 90.6% 说明前缀没从盘框读回来，这次的 trace 不能用。

### 90% 命中是怎么构造的

每一轮：随机生成 3712 token 的前缀（58 页 × 64），发 1 个种子请求，P 按层把它写进 OceanDisk → 等 `SETTLE_S` 秒 → `/flush_cache` 清掉 GPU 上的 radix 树（不清盘框）→ 发 前缀 + 384 个新 token = 4096 的请求。GPU 上没有命中，`UnifiedCacheLinkerWrapper.match` 去 OceanDisk 查到 58 页。每轮换新的随机前缀，盘框上的旧 key 不会让命中率虚高。

所以 profile 的这一个请求包含：EXISTS → 58 页 GET → 4096 token 的 prefill → 新增 6 页的逐层 PUT。`STOP_DELAY_S` 用来把请求返回后仍在跑的尾部 PUT 也录进去。

## 3. 看结果

### 人看（你和同事）：在黄区看

`.nsys-rep` 一般有几百 MB，出不了黄区，所以人工分析在黄区内完成：

- **有能装 Nsight Systems 的黄区桌面机**：把 `p_mixed_4k.nsys-rep` scp 过去，用 Nsight Systems GUI 打开（GUI 版本要不低于 132 上的 `nsys --version`）。8 个进程各有一组 GPU 行，重点看：
  - GET/PUT 所在的 stream 和计算 stream 是不是重叠，还是串行；
  - GPU 空洞期间 CPU 主线程在哪个调用上（`cudaStreamSynchronize`、`pthread_cond_wait` 等）；
  - 逐层 NVTX（`--enable-layerwise-nvtx-marker`）对应的每层时长；8 个 rank 之间是不是有人慢、其他人在 NCCL 上等。
- **只有 132 的命令行**：`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum p_mixed_4k.nsys-rep | less`，或者直接看下面的 `summary.txt`。

torch profiler 模式的 `*.trace.json.gz` 用 Chrome 的 `chrome://tracing`（离线可用）或 Perfetto 打开。

### 给 AI：只回传文本

蓝区只能粘贴几百行，所以回传以下内容，**不回传 nsys-rep / sqlite / trace**：

1. `cat /home/j00977581/p_profile/commit.txt`（1 行）；
2. 客户端日志全文 `client_*.log`（约 15 行，含每轮 `hit` 和 `meta_info`）；
3. `summary.txt`（约 300 行，`##` 开头分 8 节：总览、kernel、stream、GIDS/NCCL 关键 kernel、GPU0 时间线、空洞与 CPU 主线程、NVTX、CUDA API）。一次贴不下就按 `##` 分节分几次贴；
4. 只有出错时：`grep -n -i -E "error|traceback|ERR!" server_*.log | head -50`。

summary 还是太长时，在 `run_summary.sh` 里调小 `GAPS_PER_GPU`、`STREAMS_PER_GPU`、`SEGMENTS`。AI 看完如果需要某一段的细节，会告诉你改哪个参数重跑 `run_summary.sh`；sqlite 已经生成，重跑只需几秒。

## 4. 采集方式

| `ACTIVITIES`（客户端） | 服务端 | 产物 |
|---|---|---|
| `CUDA_PROFILER`（默认） | `run_mixed_server.sh` 里带 nsys 的那段 | `p_mixed_4k.nsys-rep`，8 个 rank 都在里面 |
| `CPU,GPU` | 同一脚本末尾注释掉的不带 nsys 那段 | `PROFILE_DIR` 下每个 rank 一个 `*.trace.json.gz` |

两者不要同时开：nsys 和 torch profiler 都要用 CUPTI。

## 5. 已知限制

- sglang 默认只在 GPU0 上调用 `cudaProfilerStart`；本分支加了 `SGLANG_CUDA_PROFILER_ALL_RANKS=1`（服务端脚本已设），每个 rank 各自开关，所以 8 个进程都会被录到。
- 混部 + CP8 + OceanDisk linker 这个组合此前没在 132 上跑过；启动报错请回传第 3 节第 4 条的 grep 结果。
- DeepEP normal 会自动关闭 CUDA graph，混部的 decode 很慢；这里只要 1 个输出 token，不影响 P 侧结论。
- 混部只能近似 P：没有 PD 的 bootstrap、READY 通知和 aux 发送。
- `nsys_summary.py` 只在构造的 sqlite 上测过，真实报告的表结构随 nsys 版本可能有差异；报错时把 traceback 贴回来。
