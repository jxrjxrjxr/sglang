# Unified Cache Linker 实现与对接导读

> 版本提示：本文记录的是前次梳理时的代码形态。后续核对发现工作区 PC 加载已改为先插树、锁定节点，并通过 `num_completed_loads/pop_completed_load` 回收加载锁；下文的请求私有加载、`finish_layer_wise_loading()`、counter 位置等描述不适用于本次核对的版本。当前接入方式与 Mooncake 实现请优先阅读 [新缓存后端接入指南](new-cache-backend-integration-guide.md)，尤其是第 8 节。

本文基于当前工作区代码，介绍 Unified Cache Linker 的架构、读写流程，以及以 Mooncake 为例的后端对接方式。第 1–9 节介绍 prefix cache 的外部存储直连，第 10 节补充 PD 分离及 unified memory 的传输适配；两者使用不同的接口和传输链路。

## 1. 定位与架构

Unified Cache Linker 为 `UnifiedRadixCache` 提供 GPU KV cache 与外部存储直接传输的能力，即 direct L3 路径。数据传输跳过 CPU host cache。

```text
Scheduler / Model Forward
    │ 调度、逐层等待、加载失败处理
    ▼
UnifiedRadixCache / TreeCore
    │ prefix match、节点插入、BackupKV action
    ▼
UnifiedCacheLinkerWrapper
    │ 命中协调、请求私有槽位、节点锁、完成确认
    ▼
UnifiedCacheLinker
    │ lookup / load / offload / 生命周期
    ▼
MooncakeDirectLinker
    │ 逻辑 pool → 物理 GPU buffer、异步传输
    ▼
MooncakeStore / Mooncake 客户端
```

| 层 | 职责 |
| --- | --- |
| TreeCore / Component | 管理树和组件状态，描述 Full、SWA 要传输哪些页 |
| UnifiedCacheLinkerWrapper | 将存储接入请求与树生命周期，协调多 rank 状态 |
| UnifiedCacheLinker 后端 | 查询外部缓存，执行异步加载和写出，报告完成状态 |
| DevicePoolGroup / DevicePoolEntry | 将逻辑 transfer 转换为物理 pool、GPU 地址、大小和对象内偏移 |
| MooncakeStore | 存储 key 编码、存在性查询和底层读写 |

核心实现：[unified_cache_linker.py](python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py)。

## 2. 初始化与挂接

[registry.py](python/sglang/srt/mem_cache/registry.py) 根据 `unified_cache_external_linker_backend` 选择后端。当前 `mooncake` 对应 `MooncakeDirectLinker`，`mori` 对应 `UMBPDirectLinker`。

Mooncake 的挂接逻辑可概括为：

```python
linker = MooncakeDirectLinker(
    server_args, params, components=set(cache.components)
)
cache.init_cache_linker(linker)

counter = cache.linker.layer_done_counter
kvcache.register_layer_transfer_counter(counter)
tp_worker.register_hicache_layer_transfer_counter(counter)
```

`init_cache_linker()` 创建 Wrapper。Wrapper 初始化时：

- 校验树组件，当前只接受 `FULL` 和 `SWA`。
- 设置 `tree_core.enable_external_cache_linker = True`。
- 将 `write_through_threshold` 设为 `1`。
- 初始化命中标记、请求私有加载槽位、在途加载和待确认写出列表。

需要区分树组件与物理 pool：树组件描述缓存语义，物理 pool 描述内存布局。一个 FULL 逻辑 transfer 可以展开为 KV、Indexer 等多个物理 pool，因此物理 pool 的范围可以大于树组件集合。

## 3. 后端接口与传输描述

### 3.1 UnifiedCacheLinker 接口

| 方法 | 约定 |
| --- | --- |
| `lookup(rid, transfers)` | 返回当前 rank 所有可恢复前缀长度，单位为 page |
| `load(rid, transfers)` | 将加载任务排队，实际执行由下一次 start 触发 |
| `start_layer_wise_loading()` | 启动已排队加载，返回批次 counter index |
| `finish_layer_wise_loading(index)` | 等待该批次停止写入 device memory，报告是否全部成功 |
| `cancel_queued_load(rid)` | 取消尚未启动的请求加载 |
| `offload(transfers)` | 接受一组需要整体持久化的 transfers |
| `num_completed_offloads()` | 查询可消费的已完成写出数量 |
| `pop_completed_offload()` | 按顺序消费最早完成任务的成功状态 |
| `reset()` | 等待传输静止并重置后端状态 |
| `close()` | 等待传输静止并释放资源 |

### 3.2 PoolTransfer

[hicache_storage.py](python/sglang/srt/mem_cache/hicache_storage.py) 中的 `PoolTransfer` 是通用传输描述：

```python
PoolTransfer(
    name=PoolName.KV,
    device_indices=slots,
    keys=page_hashes,
    hit_policy=PoolHitPolicy.ALL_PAGES,
)
```

关键字段：

- `name`：逻辑或物理 pool 名称。
- `keys`：每页对应的存储 key。
- `device_indices`：逻辑 device 槽位索引。
- `host_indices`：复用存储接口的索引字段；在 direct linker 展开后可承载物理 device pool 索引。
- `hit_policy`：连续前缀或尾部窗口命中策略。

这里 `host_indices` 的命名不意味着数据经过 CPU host cache。

## 4. 读路径：查询、分配、逐层读回

核心原则：外部数据先加载到请求私有 GPU 槽位，确认成功后，再通过正常插入流程进入共享树。

### 4.1 Prefix match 与外部查询

`UnifiedRadixCache.match_prefix()` 完成本地匹配后，调用 Wrapper 的 `match()`：

1. 根据本地 `device_prefix_len` 找到未命中的尾部。
2. 将尾部截断到完整 page 边界，计算 storage hash。
3. 如果已有本地命中，以本地最后一个 hash 为锚点继续 hash 链；缺少锚点时跳过外部匹配。
4. 各 Component 构造 `LOOKUP` transfer。
5. 调用后端 `lookup()`，获取可恢复前缀长度集合。
6. 对各 rank 的集合求交集，选取最长共同边界。
7. 将结果保存在 `hit_markers[rid]`，通过 `MatchResult.host_hit_length` 通知调度器。

`host_hit_length` 是复用的调度字段，此路径的数据实际直接读入 GPU。

### 4.2 为什么返回集合

Full 与 SWA 的可恢复条件不同：

| 组件 | 命中策略 | 可恢复条件 |
| --- | --- | --- |
| Full | `ALL_PAGES` | 目标边界之前的所有页存在 |
| SWA | `TRAILING_PAGES` | 目标边界末尾所需窗口存在 |

SWA 等窗口状态使可恢复边界可能稀疏，不能只返回最大命中长度。

例如 rank 0 可恢复 `{4, 8}` 页，rank 1 可恢复 `{4, 6}` 页。共同可恢复边界为 `{4}`，最终应选择 4 页；对各 rank 最大值取最小值会错误地选择 6 页。

Wrapper 使用 0/1 mask 和 `MIN` all-reduce 实现集合求交。

### 4.3 分配请求私有槽位

`init_load_back()` 检测到命中标记后，转到 Wrapper 的 `load_back()`：

1. 消费 `hit_markers[rid]`。
2. Full 分配完整命中尾部所需的 GPU slots。
3. SWA 分配尾部窗口所需的 slots。
4. 调用后端 `load()` 排队。
5. 成功排队后执行 Component 的 `PREPARE`，建立 Full → SWA 索引映射等状态。
6. 保存 `private_load_indices[rid]` 和在途请求列表。

此时 `req.last_node` 保持原值，加载尾部尚未进入共享树。

如果槽位不足或后端拒绝排队，则执行 `ABORT`，释放已分配资源并回退到重新计算。SWA request-ring 模式有专门分支：跳过 SWA transfer，并记录对应的不可恢复范围，避免使用未初始化的映射。

Component 实现：

- [full.py](python/sglang/srt/mem_cache/unified_cache/components/full.py)：Full 的 transfer 构造、分配与回滚。
- [swa.py](python/sglang/srt/mem_cache/unified_cache/components/swa.py)：窗口选择、SWA 分配与 Full → SWA 映射。

### 4.4 逐层加载与 forward 重叠

`start_layer_wise_loading()` 将排队加载合并为一个批次，并返回 counter index。模型访问某层 KV 时通过已注册的 counter 等待该层 ready。

```text
后台加载：读 layer 0 → complete(0) → 读 layer 1 → complete(1) → …
模型执行：等 layer 0 → 算 layer 0 → 等 layer 1 → 算 layer 1 → …
```

`LayerWiseLoadCounter` 为每个加载批次维护逐层 Future，从而允许加载和计算重叠。

### 4.5 完成确认与失败隔离

Scheduler 在 forward 后调用 `finish_external_linker_loads()`：

- 等待加载结束，并跨 rank 归约成功状态。
- 成功后，请求通过正常流程将 KV 插入树。
- 失败时，关联请求设置 `skip_radix_cache_insert`，丢弃输出，并标记为 HTTP 503 abort。

逐层等待遇到失败时不会直接向 forward 抛异常；forward 可能继续执行，但失败批次的加载请求输出不会被使用。私有槽位保证失败数据不会污染共享树。

相关挂钩见 [unified_radix_cache.py](python/sglang/srt/mem_cache/unified_radix_cache.py) 和 [scheduler.py](python/sglang/srt/managers/scheduler.py)。

## 5. 写路径：异步写出与状态发布

写出由树的 write-through 流程产生 `BackupKV` action，转到 Wrapper 的 `offload_nodes()`。

```text
构造 OFFLOAD transfers
    → 对源节点加锁
    → 后端 offload() 排队
    → 标记 pending
    → 后台写入
    → 跨 rank 完成确认
    → 更新节点外部可用状态
    → 释放节点锁
```

关键机制：

- **源数据保护**：写出期间持有节点锁，防止 GPU 源 slots 被回收。
- **延迟发布**：排队成功不等于持久化成功，完成确认后才更新节点状态。
- **顺序消费**：Wrapper 的 pending 列表与后端完成结果按顺序对应。
- **节点 split**：写出期间节点发生 split 时，更新该任务对应的待发布节点列表。
- **多 rank 一致性**：先对已完成数量取最小值，再对任务成功状态做归约，所有 rank 一致推进。

TreeCore 相关方法包括 `build_external_linker_offload_transfers()`、`mark_external_linker_offload_pending()` 和 `finish_external_linker_offload()`。

## 6. Mooncake 对接实例

实现入口：[mooncake_direct_linker.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py)。

### 6.1 初始化 GPU pool 与客户端

MooncakeDirectLinker 初始化过程：

1. 从 allocator 获取模型 KV cache。
2. 调用 `resolve_hybrid_device_pool_group()`，构造物理 device pool group。
3. 根据并行拓扑确定 rank 信息、存储 suffix 和写出 owner。
4. 解析额外配置，创建 `MooncakeStore(storage_config, mem_pool=None)`。
5. 将 pool group 作为存储接口的 pool 视图，设置 registered pools。
6. 注册 GPU buffers。
7. 创建逐层 counter、加载线程和写出线程。

`register_buffers()` 按底层 allocation 的地址与大小去重，再调用 Mooncake `register_buffer()`，避免共享 storage 的多个 tensor 被重复注册。

### 6.2 逻辑 transfer 到物理 buffer

[linker_pool_assembler.py](python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py) 提供布局适配：

- `DevicePoolGroup.resolve_transfers()`：将逻辑 transfers 展开到物理 pools，转换索引并保留命中语义。
- `DevicePoolEntry.get_page_buffer_meta()`：生成整页读写需要的 GPU 指针和字节数。
- `DevicePoolEntry.prepare_locations()`：预处理页位置。
- `DevicePoolEntry.get_prepared_layer_range_meta()`：生成某层的数据指针、大小、存储对象内偏移。

底层会检查索引是否构成对齐、连续的 page，以及是否越界。模型布局差异主要封装在这一层，使传输逻辑能够复用。

### 6.3 查询

```python
expanded = pool_group.resolve_transfers(transfers)
result = storage.batch_exists_v2(page_keys, expanded)
return result.restorable_prefix_pages
```

[MooncakeStore.batch_exists_v2()](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) 根据各物理 pool 的 `ALL_PAGES` 或 `TRAILING_PAGES` 策略筛选合法恢复边界。Wrapper 再负责跨 rank 求交。

### 6.4 逐层读回

`load()` 只将请求放入 `pending_loads`。`start_layer_wise_loading()` 切出当前批次，记录设备 ready event，并交给后台线程。

后台读取流程：

```text
合并请求的各 pool keys 和目标位置
    → batch_get_session_start(keys)
    → 对每一层、每个 pool：
        get_prepared_layer_range_meta(...)
        batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
        检查实际传输字节数
    → 完成该层 counter
    → batch_get_session_end(keys)
```

这里读取页对象中属于当前层的 ranges，从而实现逐层恢复。异常会标记 counter 失败，最终交给 Wrapper 和 Scheduler 处理。

### 6.5 异步写出

`offload()` 展开物理 transfers，记录设备 ready event 并排队。后台线程等待源数据 ready 后执行：

```python
results = storage.batch_set_v2(expanded)
success = all(all(pool_results) for pool_results in results.values())
```

只有所有 pool 返回成功，才回报该任务成功。接口要求将整组 transfers 作为一个持久化单元；底层对象分组语义由 MooncakeStore 配置和实现承担，不能仅凭这个布尔汇总推断存储具有事务原子性。

### 6.6 多 rank 与 MLA 去重

| 场景 | 写出与 key 处理 | 读取 |
| --- | --- | --- |
| rank 间分片的 KV | 各 rank 写各自分片，suffix 包含 TP rank | 各 rank 读取自己的分片 |
| rank 间复制的 KV | TP rank 0 负责写出，suffix 省略 TP rank | 默认各 rank 可自行读取 |
| 复制 KV 且开启 MLA dedup | 同上 | 源 rank 读取，再逐层广播 |

CP、PP rank 信息也进入 suffix。

开启 `--enable-linker-mla-dedup` 后，启动批次前先比较各 rank 的逻辑加载计划摘要。若不一致，回退到各 rank 自行读取；一致时，数据在 forward stream 上按层广播。

当前代码还注明：此处生成的 storage config 未设置 `dp_rank`，后续若调整，需要考虑已持久化 key 的兼容性。

## 7. 启用参数

核心参数：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend mooncake
```

连接信息可使用：

- `--hicache-storage-backend-extra-config`，例如提供 `master_server_address` 或 `client_server_address`。
- `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` 指向的配置文件。
- Mooncake 环境变量配置。

额外配置包含连接地址时优先使用额外配置，否则按实现选择配置文件或环境变量。可选启用：

```text
--enable-linker-mla-dedup
```

配置约束：

- 与 `--enable-hierarchical-cache` 互斥。
- direct linker 模式不设置 `--hicache-storage-backend`，使用独立的 linker backend 参数。
- MLA dedup 参数要求启用 Mooncake linker。
- 当前 Wrapper 仅支持 FULL、SWA 树组件；其他功能兼容性仍需结合参数校验与模型布局检查。

参数定义与校验见 [memory.py](python/sglang/srt/arg_groups/fields/memory.py) 和 [hicache_hook.py](python/sglang/srt/arg_groups/hicache_hook.py)。

## 8. 新存储后端如何对接

新增后端的主要工作：

1. 实现 `UnifiedCacheLinker` 全部抽象方法。
2. 提供兼容的 `layer_done_counter`，确保模型访问某层前该层数据已 ready。
3. 尽量复用 `DevicePoolGroup`，将物理布局适配与传输实现解耦。
4. 在 registry 中增加后端选择，并更新参数 choices。
5. 验证查询边界、异步传输、取消、失败处理、节点锁和多 rank 一致性。

必须保持的接口约定：

- `lookup()` 返回所有合法恢复边界，不能只返回最大值。
- `load()` 先排队，由 `start_layer_wise_loading()` 启动。
- 完成某层 counter 时，必须保证该层数据可以安全读取。
- `finish_layer_wise_loading()` 返回前，该批次必须停止写入 GPU 内存。
- offload 完成结果必须与提交顺序对应。
- `reset()`、`close()` 返回前，传输必须静止。
- Wrapper 负责树锁、请求私有槽位和跨 rank 状态；后端负责物理 I/O 与本地完成报告。

## 9. 推荐阅读顺序与测试参考

| 顺序 | 文件 | 阅读重点 |
| --- | --- | --- |
| 1 | [unified_cache_linker.py](python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py) | 接口、Wrapper、逐层 counter |
| 2 | [registry.py](python/sglang/srt/mem_cache/registry.py) | 创建后端、注册等待机制 |
| 3 | [unified_radix_cache.py](python/sglang/srt/mem_cache/unified_radix_cache.py) | match/load/action/完成确认挂钩 |
| 4 | [full.py](python/sglang/srt/mem_cache/unified_cache/components/full.py)、[swa.py](python/sglang/srt/mem_cache/unified_cache/components/swa.py) | 组件 transfer 与资源管理 |
| 5 | [mooncake_direct_linker.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py) | Mooncake 线程、队列、读写与去重 |
| 6 | [linker_pool_assembler.py](python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py) | 物理 pool 和 buffer range 适配 |
| 7 | [mooncake_store.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) | key 编码、命中策略与底层存储操作 |

测试参考：

- [test_unified_cache_linker.py](test/registered/unit/mem_cache/test_unified_cache_linker.py)：接口与树侧行为，包含内存后端示例和 Python/Rust TreeCore 覆盖。
- [test_linker_pool_assembler.py](test/registered/unit/mem_cache/test_linker_pool_assembler.py)：物理 pool 布局适配。
- [test_unified_cache_linker_kl_glm52.py](test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_glm52.py)：Mooncake 加载回读后的 KL 一致性，并确认发生远端命中。
- [test_unified_cache_linker_kl_dsv4.py](test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_dsv4.py)：DeepSeek V4 linker 测试入口。

本文为静态代码梳理，未启动 GPU 或 Mooncake 服务进行运行验证。

## 10. PD 分离：另一条传输适配链路

### 10.1 与 UnifiedCacheLinker 的边界

严格按照代码命名，`UnifiedCacheLinker` 是 prefix cache 的外部存储接口。PD 分离不是它的另一个实现，而是独立的 disaggregation 子系统。若讨论统一缓存/统一内存的完整对接，需要同时介绍两条路径。

| 对比项 | Prefix cache 外部存储 | PD 分离 |
| --- | --- | --- |
| 目标 | 将 KV 持久化并按前缀复用 | 将 Prefill 结果交给 Decode 继续生成 |
| 传输端点 | GPU ↔ 外部 KV store | Prefill 内存 → Decode 接收内存，常规为 GPU → GPU |
| Mooncake 实现 | `MooncakeDirectLinker` + `MooncakeStore` | `MooncakeKVManager` + Sender/Receiver + Transfer Engine |
| 数据定位 | storage hash / page key | 对端 session、buffer 地址和目标 page/slot 索引 |
| 内存布局接口 | `DevicePoolGroup` / `DevicePoolEntry` | `get_contiguous_buf_infos()` / `KVArgs` / state buffer 描述 |
| 完成机制 | layer counter、offload 结果、跨 rank 归约 | bootstrap room、传输状态、`KVPoll` |
| 对端消费 | 加载到请求私有槽位，确认后插入树 | Decode 确认接收完成后进入后续调度 |

PD 中 Mooncake 的主入口是 [disaggregation/mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py)，并不调用 `MooncakeDirectLinker.lookup/load/offload`，也不依赖 Mooncake Store 中已有 page keys。不能用开启 external linker 的参数代替 PD 配置。

### 10.2 端到端流程

```text
Prefill                                       Decode
   │                                            │
   │ 注册源 KV / state buffers                  │ 注册目标 buffers
   │                                            │ 分配请求接收槽位
   │ ◄──── bootstrap / receiver metadata ─────── │
   │       room、session、目标索引、prefix 长度   │
   │                                            │
   │ 计算一个 prefill chunk                     │
   │ 源虚拟索引 → 源物理 page 索引              │
   │                                            │
   │ ─── Mooncake Transfer Engine 写目标页 ────► │
   │ ─── 最后一个 segment 附带 state / aux ────► │
   │                                            │
   │ 等待发送完成、安全释放源资源               │ 轮询接收状态
   │                                            │ 成功后继续 decode
```

主要入口：

- [prefill.py](python/sglang/srt/disaggregation/prefill.py)：计算 chunk、构造源 page/state 索引、调用 sender。
- [decode.py](python/sglang/srt/disaggregation/decode.py)：预分配目标槽位、发布 metadata、处理传输状态。
- [utils.py](python/sglang/srt/disaggregation/utils.py)：组织 KV/state buffer 描述、统一内存搬迁门控等公共适配。
- [mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py)：注册内存、bootstrap 交互、实际传输及完成状态。

### 10.3 Unified memory：页 envelope 暴露

这里的 `--enable-unified-memory` 与 `UnifiedRadixCache`、external linker 是不同概念。统一内存中，多个子 pool 可以共享同一个底层 allocation；PD 需要暴露可直接传输的物理布局。

[unified_memory_pool.py](python/sglang/srt/mem_cache/unified_memory_pool.py) 的 `get_contiguous_buf_infos()` 返回：

```python
([raw_ptr], [active_allocation_bytes], [page_envelope_bytes])
```

在 whole-envelope 路径中，一个传输 item 包含该子 pool 一页的数据，地址按以下方式计算：

```text
page_address = raw_ptr + physical_page_id * page_envelope_bytes
```

这与传统按层 K/V buffer 列表的方式不同。Full、SWA 等逻辑条目可能共享底层 allocation，同时使用不同的 envelope 大小。

`MooncakeKVManager._registerable_regions()` 对 `(ptr, length)` 去重：底层内存只注册一次，但 `KVArgs` 中用于传输的各逻辑条目仍保留。否则同一 allocation 作为 KV 和 state 多次注册可能失败。

### 10.4 两端都必须转换虚拟索引

统一 allocator 下，`req_to_token` 保存虚拟 token IDs，RDMA 需要物理地址。因此发送端与接收端都调用：

```python
physical_indices = allocator.translate_kv_indices_for_transfer(raw_indices)
page_indices = kv_to_page_indices(physical_indices, page_size)
```

- Prefill 对每个发送 segment 转换源索引。
- Decode 转换预分配的目标索引，并将对应目标页信息发布给 Prefill。
- SWA 使用 `translate_swa_indices_for_transfer()`，因为 Full 和 SWA 可能有独立的物理映射。
- Mamba state 使用 `translate_mamba_indices()` 将虚拟 state slot 转换为物理 slot。

源页号与目标页号不要求相等；传输按请求中的逻辑位置配对。

相关实现包括 [unified_hybrid_swa.py](python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py)、[unified_mamba.py](python/sglang/srt/mem_cache/allocator/unified_mamba.py) 和两端的 prefill/decode 代码。

### 10.5 Decode 发布接收位置，Prefill 分块发送

Decode 预分配后，`MooncakeKVReceiver.send_metadata()` 发布请求级信息，包括 bootstrap room、本端 session、目标 KV 索引、aux/state 索引以及 `decode_prefix_len`。buffer 指针等注册信息通过注册和 bootstrap 协议维护。

已有 Decode prefix 可用于减少需要传输的 KV。Prefill 按 chunk/segment 选择源范围，通过 `MooncakeKVSender.send()` 排队，最终进入 `MooncakeKVManager.send_kvcache()` 和 `_send_kvcache_generic()`。

常规数据面根据源/目标 buffer、item 大小和页索引生成传输范围，并由 Transfer Engine 执行。unified-memory 路径设置 `force_flat=True`，按整页 envelope 处理，而不是将 envelope 当作按层 buffer 再切片。

### 10.6 除 Full KV 外，还要传请求状态

PD 不仅需要搬运 Full KV。两端根据 `KVArgs.state_types` 构造位置对应的 state 索引，例如：

| 状态 | 适配要点 |
| --- | --- |
| SWA | 选择所需尾部窗口，使用 SWA 物理索引 |
| Mamba | 转换请求对应的 state slot |
| DSA / indexer 等 | 根据其 buffer 和 state 描述提供对应索引 |
| SWA ring | 按窗口的逻辑位置顺序构造 ring rows |
| 请求级压缩状态 | 使用模型 pool 提供的 request-state 索引接口 |

例如 SWA ring 的 row 计算为：

```text
row = req_pool_idx * ring_stride + position % ring_stride
```

Prefill 和 Decode 的 `req_pool_idx` 可以不同，只要两端按同一组逻辑 positions 排序，就能正确配对。prefill 调用处只给最终 segment 附带 state 索引，Sender 也在最后一个 chunk 路径附加 state/aux。

这些状态类型来自 PD 独立的 state 协议，不代表 external linker 的 Wrapper 已支持同名树组件。例如 PD 能处理 Mamba state，与前述 Wrapper 仅接受 FULL/SWA 并不矛盾。

### 10.7 传输期间禁止 compaction 搬迁

虚拟到物理映射允许统一 allocator 搬迁物理页，但地址一旦发布给对端，就不能在传输期间改变。

`unified_memory_disagg_move_gate()` 为 PD 节点提供搬迁门控：

- Prefill 检查 inflight queue 和尚有 chunk 在途的请求集合。
- Decode 检查 transfer queue、已发布的目标地址，以及 decode offload 的在途 device transfer。

不能只检查队列为空：目标地址可能已经发布，但请求尚未加入 transfer queue；Prefill 的前面 chunk 也可能仍在传输，而最后 chunk 已离开当前调度状态。

因此 PD 适配需要同时保证地址转换正确，以及该物理地址在整个传输生命周期中稳定。

### 10.8 Mooncake whole-envelope 的布局约束

`MooncakeKVManager._validate_envelope_kv_layout()` 在首次写入前检查 unified-memory 路径：

- 本端 KV item 描述和对端 KV region 数量满足该路径预期。
- 两端 envelope 的 item 字节数一致；代码错误提示要求两端启用 unified memory，并使用相同 page size 和模型 spec。
- 两端 attention TP size 一致；当前 whole-envelope 状态不能按不同 TP size 重新切片，因此会拒绝此组合。

这项限制针对当前 unified-memory whole-envelope 路径，不能推广为所有 Mooncake PD 路径都不支持异构并行配置。

### 10.9 完成与资源释放

PD 使用 `KVPoll` 等传输状态协调后续调度，而不是 external linker 的 layer counter。Decode 侧成功后才推进请求；发送侧也需要确认传输生命周期结束，才能安全回收源资源。

Mooncake 代码还针对 deferred decode KV release 处理失败后的在途写入：取消 Future 不会停止已运行的传输，因此需要 drain 已启动任务。不同开关对应的释放协议需要结合具体路径判断，不能将超时或请求取消直接视为 RDMA 已停止。

### 10.10 新后端或新布局的 PD 对接检查点

若新增传输后端，需要实现 PD 的 Manager/Sender/Receiver 协议；仅实现 `UnifiedCacheLinker` 不足以完成 PD 对接。

若新增统一内存布局，需要检查：

1. 正确暴露 KV/state 的 buffer 地址、注册长度和 item 大小。
2. 在两端将虚拟索引转换为对应物理索引。
3. 维护 state 类型、窗口范围和两端逻辑顺序的一致性。
4. 正确处理 chunk、Decode prefix 命中和最后一块的 state/aux。
5. 地址发布到传输结束期间，禁止移动或释放相关物理页。
6. 对不能支持的 page/envelope/并行布局组合提前拒绝。
7. 对成功、超时、取消和失败路径分别确认传输停止及资源释放条件。

Prefix cache 与 PD 可以共享底层 pool、allocator 及部分布局能力，但当前实现通过两套独立协议接入；阅读或扩展时应分别跟踪两条调用链。
