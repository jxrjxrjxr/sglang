# GIDS 独立探测脚本（不启动 sglang）

这个目录有两组互相独立的脚本，都只依赖 `torch` 和 GIDS 的 `od_utils` / `libop_extension.so`，不 import sglang，拷到任何装了 GIDS 的容器里都能跑。

| 问题 | 脚本 | 状态 |
|---|---|---|
| A. 串行 PUT 完、无任何并发，EXIST 仍查不到（盘框上 key 存在但 `complete=0`） | `oceandisk_dispatch_roundtrip.py`、`run_oceandisk_dispatch_roundtrip.sh`、`run_put_exist_matrix.sh`、`oceandisk_exists_key.py` | **当前优先** |
| B. 同一进程内 GET∥PUT 共用 QP 抢 CQE | `gids_probe_lib.py`、`t0`–`t5`、`run_probe.sh` | 保留，用于验证 QP 拆分修复 |

## A. PUT→EXIST 丢 key（张圆的脚本）

`oceandisk_dispatch_roundtrip.py`、`oceandisk_exists_key.py` 取自 `origin/dev_oceandisk_linker` 的 `2c17a9ccc8`。在不改变原默认行为的前提下加了：

| 新增 | 作用 |
|---|---|
| 逐 key EXIST（`--full-exists`，默认开） | 原生 EXIST 只返回前缀，而且是每条连接各自的前缀；第一个未命中之后的 key 全看不到。逐 key 查出完整未命中列表和 `miss_by_dpu` / `miss_by_conn` |
| 延时复查（`--recheck-after 30,90`） | 未命中的 key 过一段时间再查：变成命中 = 晚到（`late`），仍未命中 = 真丢（`lost`） |
| PUT 逐层耗时 | 打印每层完成间隔的中位数、最大值，以及超过 `--slow-layer-ms`（默认 50，GIDS `data_io_timeout_ms=100`）的层：有慢层说明走了超时/重传路径 |
| `--put-mode per-layer`、`--put-chunk-pages 256` | 按生产 `submit_put_layer()` 的方式下发：每层一次调用、每次 256 页。原脚本是 1000 页 ×78 层一次调用，超出生产的单次规模 |
| `--settle-s` | 原来写死的 PUT 后 sleep 15 秒 |
| 末尾 `SUMMARY` / `RESULT` 行 | 每个查询卡一行 `RESULT`（`miss`、`late`、`lost`、PUT 耗时、慢层数、`run_token`），以及每张卡日志里 GIDS `ERR!` 打印的行数 |

### 怎么跑

```bash
cd examples/runtime/engine/gids_probe
bash run_oceandisk_dispatch_roundtrip.sh 2>&1 | tee repro.log   # 复现张圆的一次运行
bash run_put_exist_matrix.sh                                     # 五组对照，每组 3 次
```

- 路径默认值与 `dual_node_pd/run_prefill.sh` 相同，可用 `OPEXTENSION`、`LIBGIDS`、`GIDS_DIR`、`GIDS_PYTHONPATH`、`PYTHON_BIN` 覆盖。张圆用的是 `/home/wq/code/0520/GIDS/` 下的库和 `/home/wq/code/0513/sglang_deploy_prefill/multi` 配置，两边脚本开头都会打印库的 sha256，结果不一致时先比对这个。
- 单卡模式：`GPUS= EXISTS_DEVICE=0 bash run_oceandisk_dispatch_roundtrip.sh`。
- 只看某个 key：`python oceandisk_exists_key.py --device 0 --dpu <dpu> --key <key>`（需要先 export 同样的 `OPEXTENSION`、`GIDS_DIR`）。
- 不要和 sglang 同时占用同一组卡。

### 怎么判读（看 `summary.txt` 的 `SUMMARY ... RESULT` 行）

| 观察 | 结论 |
|---|---|
| `gids_ERR_lines>0` 或 `put_slow_layers>0` | 客户端收到过失败/超时但被吞掉，查 GIDS 的超时、重传和错误处理 |
| 两者都为 0 但 `lost>0` | 客户端认为全部成功；带着 `EXISTS FULL MISS key` 里的 key/DPU 找盘框侧 |
| `late>0`、`lost=0` | 写入晚到；盘框侧的 complete 置位有延迟 |
| `miss_by_dpu` 集中在一个 DPU | 偏向该 DPU 或链路问题 |
| `repro` 丢、`two_gpu` 不丢 | 和空闲卡上的 GIDS 连接有关 |
| `layers1` 不丢 | 需要每个 key 多层才触发 |
| `pages256` 或 `prod_shape` 不丢 | 和单次 PUT 规模有关；生产形态可能不受影响 |
| 所有对照都丢且频率接近 | 和这些因素无关，回到盘框侧 |

## B. GET∥PUT 并发（原有内容）

目的：在不启动 sglang 的前提下，用和线上完全相同的 GIDS 调用方式，验证“同一进程内 GIDS 操作并发会丢写或查错”这一根因。

> 更正：下文“TP1 是 EXIST∥GET”不成立。P 上 EXIST 总是在 GET 开始之前完成；真正的并发是写入 rank 上的 GET∥PUT（P TP0 同一次 forward 内，以及 D TP0 的 decode PUT 与 handoff GET）。`t5_two_process.py` 和 A 组脚本覆盖的场景重叠，优先用 A 组。

## 背景（为什么怀疑并发）

GIDS 判断一次请求完成的方式是：kernel 从连接的 QP 接收队列里**按顺序数够 N 个完成事件（CQE）**就退出。

- `find_qp` 返回 `rcs[lane % 8]`，GET、PUT、EXIST 用的是**同一组 8 条 QP**。
- `gids_io_rdma_cpl_sync_poll_cq_of_rq` 的做法是先读 `cons_idx`，再自旋等待这个位置出现 CQE，最后才对 `cons_idx` 做 `atomicAdd`。这三步合起来不是原子的，取到的 CQE 也不核对 `cid`/`wr_id` 是不是自己的。
- 所以只要两个 kernel 同时在同一条 QP 上取 CQE，就可能取到同一个 CQE，或者把对方的 CQE 当成自己的。结果是有的 kernel 提前“完成”，`cons_idx` 和 `qp_cur_inflight` 的计数逐渐错乱，有的 kernel 永远等不到 CQE。
- 错误被三层吞掉：IB 层的 `error`、盘框返回的 `status`、逐 IO 的 `d_result`，都不会传到上层，`kvcache_put` 恒返回成功。
- 原生用法是 exist→get→forward→put 严格串行，同一时刻只有一个 kernel 在取 CQE，所以不会出问题。线上 P 的 TP0 是 GET（get stream）∥PUT（put stream），TP1 是 EXIST（GIDS 自己的 stream）∥GET，都在并发。`_put_io_lock` 只在 CPU 侧锁“提交”，管不住 GPU 上已经在跑的 kernel，也从来没有锁过 GET。

## 文件

| 文件 | 内容 |
|---|---|
| `gids_probe_lib.py` | 极简封装：初始化、写入校验戳、PUT/GET/EXIST、看门狗、工作线程、三种排队模式 |
| `t0_serial.py` | 严格串行基线，必须 100% 干净；失败说明是环境问题，后面的测试不用看 |
| `t1_get_put_overlap.py` | **决定性测试**：GET 线程不停读取的同时，PUT 线程写入新数据 |
| `t2_exist_get_overlap.py` | 在 GET（可选再加 PUT）进行时反复 EXIST，统计假未命中 |
| `t4_multiround.py` | 多会话多轮回放，节奏和 UnifiedCacheLinker 一致 |
| `t5_two_process.py` | 双卡、双进程：一个进程只写，另一个进程查询并读取（对照） |
| `run_probe.sh` | 一键跑完整套，日志和 `summary.txt` 写到 `LOG_DIR` |

## 怎么跑

```bash
cd examples/runtime/engine/gids_probe
bash run_probe.sh                 # 整套，大约 10 分钟
bash run_probe.sh t1              # 只跑 T1 的三种模式
DEVICE=3 EXTRA="--pages 64" bash run_probe.sh t1 t4
```

- 环境变量默认值取自 `dual_node_pd/run_prefill.sh`（`OPEXTENSION`、`LIBGIDS`、`GIDS_DIR`、`USE_SINGLE_PORT=0`），可以直接覆盖。
- 默认使用真实页布局：78 层、KV 页 73728 B、Indexer 页 8448 B（对齐到 8704 B），每次操作 32 页。
- 每次运行都用随机生成的新 key，不会读到旧数据，也不会覆盖线上数据。
- 不要和 sglang 同时占用同一张卡。

## 三种模式（`--mode`）

| 模式 | 含义 | 对应线上 |
|---|---|---|
| `full` | 锁从提交一直持有到 GPU 操作完成，任意时刻只有一个 GIDS kernel 在跑 | GIDS 原生约定（对照组） |
| `launch` | 只在提交时加锁；EXIST 在锁内先同步 PUT stream | 当前 `_put_io_lock` 的语义 |
| `none` | 不做任何排序 | 当前 GET 与 PUT 之间的关系 |

## 怎么看结果

每个测试最后都会输出一行 `RESULT ... verdict=PASS|FAIL|HANG`，`summary.txt` 会汇总这些行，并统计 GIDS 在 kernel 里打印的 `ERR! recv` 行数。

| 字段 | 含义 |
|---|---|
| `get_bad_pages` | 并发期间 GET 读回的页不对（`untouched` = 目标区域根本没被写过，说明 GET 提前“完成”了） |
| `late` | PUT 报告完成后马上 EXIST 未命中，`--settle` 秒后又命中了：完成事件被别人取走，写入其实晚到 |
| `lost` | 等过 settle 时间仍未命中：真正丢写，对应线上“盘框里查不到这个 key” |
| `data_bad` | key 存在，但内容不是自己写的 |
| `false_miss_calls` | T2 中对确定存在的 key 报了未命中 |
| `missing_after_put_done` | T4 中 PUT 已报告完成，下一轮却查不到 |
| `health_after` | 并发结束后再做一次串行检查；如果 FAIL，说明 QP 状态已经被破坏、计数漂移了，之后的串行操作也不可靠 |
| 退出码 3 / `verdict=HANG` | 有操作超过 `--timeout` 秒没有完成，日志里会列出当时在跑的操作 |

## 判读

| 观察到的情况 | 结论 |
|---|---|
| T0 PASS，T1 `full` PASS，T1 `none` 出现 late/lost/get_bad/HANG | 坐实“同一进程内 GET∥PUT 共用 QP” |
| T1 `launch` 也出问题 | 坐实“只锁提交不够”，修复必须保证 GPU 上不重叠 |
| T1 只有 late、没有 lost | CQE 被偷导致提前完成；线上的源页释放和复用会在这个窗口内出问题 |
| `health_after=FAIL` | 计数漂移是累积的，对应线上“前几轮好，第 5 轮左右开始变差” |
| T2 `none` 有 false miss，`full` 没有 | EXIST 假阴性成立；线上一部分“未命中”其实是查错，不是没写 |
| T4 `none` 的异常从某一轮开始出现 | 和线上多轮现象对应 |
| T5 干净 | 跨进程没有这个问题，和 QP 按进程独立的推断一致 |
| T5 不干净 | 问题更可能在盘框侧（可见性/落盘），需要盘框同事介入 |
| 所有 `none` 都 PASS | 并发假设不成立，回到盘框侧容量/淘汰/覆盖写语义排查 |
