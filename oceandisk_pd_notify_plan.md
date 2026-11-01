# OceanDisk PD：PUT 完成通知驱动 GET

更新：2026-11-01。分支：`dev_oceandisk_linker_pd_notify`。

本方案复用原生 Mooncake PD 的握手、ZMQ socket、尾部 RDMA、metadata 和失败处理。P 对每个 token chunk 分组完成 layerwise PUT，等待本地 save event 完成后发送 READY；D 收到通知后直接 GET 对应页和层，不调用 EXIST。D 保留 HBM radix cache，只有 P 做存储卸载。

## 基线与交接

- 直接基于张圆 `origin/dev_oceandisk_linker` 的 `32be4a3e00925b23884040f3c181a151d4d8882f`。2026-11-01 再次用 `git ls-remote` 核对，远端仍是这个提交。
- 与 `dev_oceandisk_linker_pd_chunkload` 并列，公共前缀为 `dev_oceandisk_linker_pd_`，没有从 chunkload 分支派生。
- 从 chunkload 的 `6dc02220a1e2830ca65bb172e2f89eca26bbe612` 选择复用 D 预分配、token 窗口、DMA 回收、P CP 存储 namespace、启动及实验脚本。D 的 EXIST 轮询替换为通知驱动。
- 工作树：`D:/CodeSpace/sglang-od-notify`。原工作树 `D:/CodeSpace/sglang-od-linker` 保留在 chunkload 分支。
- 目标个人 Fork：`mine = https://github.com/jxrjxrjxr/sglang.git`，分支 pushRemote 指向 `mine`。实现与运行指南在此分支提交发布。132／134 均在 `/home/j00977581/sglang-pd` 切换分支，运行步骤见 [notify 指南](examples/runtime/engine/oceandisk_pd_tp4.md)。

## 数据边界和运行时序

设 `L` 为 D 已有的 HBM 前缀 token 数，`C` 为请求可缓存前缀向下按 page 对齐后的 token 数，`T` 为保留的 RDMA 尾部 token 数，page 大小为 `B`：

`S = max(L, floor((C - T) / B) * B)`。

- `[0, L)` 复用 D HBM；`[L, S)` 由 OceanDisk GET；`[S, prompt_end)` 和原生 metadata 继续由 Mooncake 传输。
- D 从 `L` 分配缺少的 KV 槽位，保持原生 prefix lock 和内存预算；只把发送给 P 的 `decode_prefix_len` 及 RDMA 目的页切到 `S`。
- D 的 token 窗口最大为 `chunked_prefill_size`，从实际 `L` 开始；P 的 chunk 边界不必与 D 的窗口边界重合。
- 层组大小由 `SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS` 控制，默认 50，最后一组可以不足 50 层。层号和页号均使用左闭右开区间。

一次正常传输依次执行：

1. D 沿用 Mooncake metadata 握手，追加 `LINKER_NOTIFY`、`L/page`、层组大小和层数。P 校验两端模式和层配置。
2. P 仍由已有 layerwise save hook 和 offload worker 提交 PUT。每组最后一层提交后，在该 worker 上记录并等待 save event。
3. P 通过已有 ZMQ socket 发送 `LINKER_READY(room, decode_session, page_start, page_end, layer_start, layer_end)`。
4. D 按层组保存合并后的已完成页区间。重复、相邻、重叠通知可以合并；存在缺口时不会宣布整段可读。旧 session 通知不被采用。
5. D 对当前 token 窗口、当前层组做 TP 就绪共识后，直接排队 GET。GET 只选对应层的 descriptor，保持原始存储 offset；每个请求最多有一个层组 GET 在途。
6. 所有层组完成后推进到下一个 token 窗口。所有窗口 GET 完成，且原生 RDMA/metadata 完成，才能进入正常 decode。

P 的原生 Success 会等待所有 D 目的端需要的存储区间已发布 READY，避免 RDMA 先完成时销毁后续通知所需的目的地址。D 的失败和超时仍要等已经发起的 GET DMA 排空，才能释放槽位。P 的源页沿用张圆代码的 I/O lease，直到整个 save session 完成并经 scheduler 回收后释放。

## P 已有缓存、共享 PUT 和 CP

P 自身缓存恢复沿用张圆的混部逻辑，没有新增 GET/EXIST 互斥锁，也没有改变 `oceandisk_storage.py` 的锁和 native 调用。

对于 P 已有 HBM、但 D 尚需从存储获取的前缀：

- 有 PUT 回执：绑定原 Future，成功后发布 READY；在途回执不依赖 EXIST，失败回执直接报失败。
- 没有回执：P 存储 owner 在原有 match 阶段调用 EXIST。命中的完整页可发布 READY；从首个缺失页开始，用现有 HBM 槽位补写连续后缀。
- 多个请求共享一个 save session 或同一在途页时，绑定同一回执，不重复创建在途 PUT。当前 session 可按层组发布，复用的旧 session 以整个 Future 完成为准。
- P CP 采用 DSA interleave，只有 CP0/TP owner 做存储检查和 PUT。各 rank 同步补写边界，维持相同的 forward/save 和源页保留流程。
- P CP 的复制缓存 namespace 与 D 全宽 TP 对齐，避免相同逻辑页在两端生成不同存储 key。
- 乐观 prefill 早于 D 握手完成时，P 暂存已完成区间，握手完成后裁剪到该 D 的 `[L, S)` 并发送。

## 文件级改动与必要性

以下是相对张圆基线的代码规模，包含选择复用的 chunkload 代码；数字为约数。

| 位置 | 规模 | 必要性 |
| --- | ---: | --- |
| `arg_groups/field_order.py`、`fields/disagg.py`、`pd_disaggregation_hook.py`、`environ.py` | 约 60 行 | 开关、默认层组/尾部配置及当前支持范围校验。 |
| `disaggregation/mooncake/conn.py` | 约 255 行 | 在既有 sender/receiver/manager 中扩展握手、READY 区间、晚握手补发和完成门控。复用 socket 与发送锁，不增加通信线程或协议类。 |
| `mem_cache/storage/oceandisk/oceandisk_direct_linker.py` | 约 90 行变动 | 每组 PUT 后 sync/通知、按层范围 GET、按需选择 Future 数量；保留 chunkload CP namespace 修正。 |
| `mem_cache/unified_cache/unified_cache_linker.py` | 约 230 行变动 | P 回执绑定、HBM 缺失页补写；复用 chunkload 的哈希链、已分配槽位 GET 和完成回收。 |
| `disaggregation/decode_linker_mixin.py` | 约 150 行 | 在 chunkload 的独立装载逻辑中增加层游标、READY 共识、失败与 DMA 排空门控。 |
| `disaggregation/decode.py`、`mem_cache/registry.py` | 约 60 行变动 | 衔接 L/S 预分配、每轮进度处理；D 跳过存储 match，禁用 write-through 和 layerwise saver。 |
| `examples/runtime/engine/run_oceandisk_pd_tp4.sh`、`oceandisk_pd_tp4.py` | 约 465 行复用/调整 | 维持 chunkload 的 check/bench/rounds、P CP、服务监督和日志；改默认模式，并让 abort/无 token 响应使实验失败。 |
| `test/registered/unit/disaggregation/test_linker_notify.py` | 约 490 行 | 新增完成时序、区间、共享回执、P 补写和按层 GET 回归。 |
| `test_linker_decode_load.py`、`test_oceandisk_pd_launcher.py`、`test_oceandisk_direct_linker.py` | 约 930 行复用/调整 | 保留 chunkload 的预分配/回收/CP/脚本回归，将 D 轮询断言改为通知和分层行为。 |

生产代码没有新增 transport class、worker thread 或独立调度器。新增状态为活动 room 上的 sender/receiver 引用、按层组合并的页区间、请求的少量标量游标，以及原 chunkload 的在途 GET 映射。与此方案无关的 chunkload Mooncake 诊断开关未搬入。

## 配置与启动

保持单机 8 卡布局：P GPU 0–3，D GPU 4–7，各 TP4，P 显式启用 `--enable-prefill-cp --cp-strategy interleave`，D 不开 CP。P、D 都带 `--disaggregation-linker-notify`；脚本会自动添加。

两端需要同样的模型、page 大小、总 TP 宽度、namespace、层数及层组大小。当前支持 DSA 复制缓存、PP=1、DP=1、DCP=1，无 speculative decoding、staging、hierarchical cache、host_pool 回撤备份或 D offload；P 必须开启 interleave CP，保持 attention TP=1，并使用 `layerwise_save=true`。DSA 层切分和所有 CP rank 同时传输会在启动阶段被拒绝。

```sh
# 先设置实际模型、数据集、GIDS 配置和动态库路径。
export MODEL_PATH=/path/to/model
export DATASET_PATH=/path/to/multifieldqa_zh.jsonl
export OPEXTENSION=/path/to/libop_extension.so
export GIDS_DIR=/path/to/gids/config
export SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS=50

sh examples/runtime/engine/run_oceandisk_pd_tp4.sh check
sh examples/runtime/engine/run_oceandisk_pd_tp4.sh bench
ROUNDS_DIR=/path/to/Round sh examples/runtime/engine/run_oceandisk_pd_tp4.sh rounds

# 同一脚本的原生 Mooncake 对照。
TRANSFER_BACKEND=mooncake sh examples/runtime/engine/run_oceandisk_pd_tp4.sh bench
```

- 默认 `TRANSFER_BACKEND=notify`。为兼容旧命令，本分支的 `chunkload` 值是 notify 的别名；要比较 EXIST 轮询方案，请在黄区同一 `sglang-pd` 目录切换到 chunkload 分支。
- `SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS=-1` 默认保留一个 `chunked_prefill_size` 的尾部；设为非负数可指定保留量。`check` 未显式设置时使用 0，让可缓存缺失前缀尽量覆盖 GET；不可缓存尾部和 metadata 仍由原生 PD 传输。
- `--chunked-prefill-size` 可作为脚本尾随参数传入，P、D 取同值。需要至少一个 page。
- `rounds` 保留原参数、容量检查和 1800 秒握手/等待设置；`Round` 输入文件仍使用外部目录。
- `check` 默认开 `SGLANG_OCEANDISK_TRACE=1`，`bench` 默认关。日志在 `LOG_DIR` 下；P、D、router 顺序启动及进程组清理逻辑保持一致。

## 验证与实机验收

本地 CPU 检查通过 49 项，另有 4 项 POSIX 启动监督测试因 Windows 环境跳过。检查使用现有 uv Python、torch/numpy，通过临时 AST 隔离加载实际方法并执行仓库测试，绕过 Linux 专用导入。它覆盖：

- READY 早于/晚于 RDMA、晚握手、重复/有缺口通知、L/S 裁剪、旧 session、发送失败及失败状态不被后续成功覆盖。
- P 在途/成功/失败回执、缺失回执 EXIST、owner 共识、HBM 补写页范围和源页 lease、共享请求回执。
- D 仅 GET、不调用 lookup/offload；部分通知不发 GET，层组逐一推进；多请求、TP 失败、超时、DMA 排空、预分配与原生 metadata 门控。
- 部分层 GET 保持原始存储 offset；PUT group event 完成后才发 READY；PUT 失败时不发布失败层组。

本机未安装格式化或测试依赖。完整 SGLang 常规导入在 Windows 的 `resource` 模块处失败；CPU 隔离检查不代表 Linux 模块集成、CUDA、RDMA、GIDS 或性能验证。Python 语法、shell 语法和暂存 diff 空白检查另行核对。

在已有依赖的 Linux 环境运行正式单测：

```sh
PYTHONPATH=python python3 -m unittest discover -s test/registered/unit/disaggregation -p 'test_linker_notify.py'
PYTHONPATH=python python3 -m unittest discover -s test/registered/unit/disaggregation -p 'test_linker_decode_load.py'
PYTHONPATH=python python3 -m unittest discover -s test/registered/unit/disaggregation -p 'test_oceandisk_pd_launcher.py'
```

实机先跑 `check`，对齐 P 的 `PUT_DONE`、`READY_SEND`，D 的 `READY_RECV`、`LOAD_STARTED/LOAD_DONE`；D 的 load tag 是 `rid@start_page:layer_start`。确认无 D EXIST/PUT，覆盖冷请求、P HBM 命中/D 无缓存、D 部分或全量 HBM 命中、多 chunk、末组不足 N 层、并发和 abort。确认所有 GET 完成与 metadata 就绪后才进入 decode，再进行同输入的 chunkload/notify/mooncake 对照。

本方案的 native 运行契约是：**P save event 完成代表该层组 PUT 完成，且相同 namespace/key 的 D GET 已可见**。Python 层捕获的失败通过原生失败路径传递；GIDS native 内部错误码传播和跨进程可见性保障不在本次 SGLang 改动内，需在实机确认此契约。

提交审阅入口：

```sh
git show --stat
git show -- python/sglang/srt/disaggregation/mooncake/conn.py
git show -- python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py
git show -- python/sglang/srt/disaggregation/decode_linker_mixin.py
```
