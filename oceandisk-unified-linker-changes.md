# OceanDisk UnifiedCacheLinker 工作区修改说明

**当前布局：** 本地地址和有效长度复用 assembler；远端长度及偏移参考 `bak_1020_modify` 的 cache controller，按 512 字节补齐。同一 page 的 KV/Indexer 使用相同 key，由原生 DSA 路径区分。key 包含模型/并行身份、page size 和 page hash，不包含布局指纹或额外版本号。

**当前命名：** 后端入口为 `oceandisk`，包目录为 `storage/oceandisk/`，核心类为 `OceanDiskDirectLinker`、`OceanDiskStorage`，范围类型为 `OceanDiskRange`。

本文说明当前工作区已经写入的代码及其行为。设计背景及底层 ABI 要求见 [设计文档](oceandisk-unified-linker-design.md)，运行配置见本文第 10 节。

**文档分工：** 本文用于查看“实际新增哪些文件、代码怎样运行、改动如何验证”；设计文档用于理解方案与后续扩展。本文只描述当前实现，历史过渡方案已移除。

**2026-10-19 对照工作区核对：** 类名为 `OceanDiskStorage(HiCacheStorage)`，当前实际实现 UC 所需范围接口；当前 Linker 调用 `exists_many/get_ranges/put_ranges`。HiCache host-cache 路径仍未实现。布局为每 page/物理 pool 一个对象、对象内分层 offset，同层合批并等待逐层 event；尚无跨 pool bundle。本次核对仅检查代码与文档一致性，没有新增 GPU 或多卡测试结果。

当前是 Prefix Cache 后端的初始实现，尚未通过真实 GIDS `.so`、CUDA 或 GLM-5 多卡端到端验证。本文中的“实现”指代码已写入，不表示已经完成部署验收。本次没有实现 PD 分离传输。

## 1. 修改范围

| 文件 | 改动及用途 |
| --- | --- |
| `python/sglang/srt/mem_cache/storage/oceandisk/__init__.py` | 新增 OceanDisk 后端包。 |
| `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_direct_linker.py` | 新增 `OceanDiskDirectLinker`、逐层完成计数器、加载和写回工作线程。 |
| `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_storage.py` | 新增 `OceanDiskStorage`，负责 GPU 内存注册、地址转换、参数打包及 GIDS 算子调用。 |
| `python/sglang/srt/mem_cache/registry.py` | 增加 `oceandisk` 分支，实例化 `OceanDiskDirectLinker`。 |
| `python/sglang/srt/arg_groups/fields/memory.py` | 将 `oceandisk` 加入 Unified Cache 外部 Linker 的后端参数选项。 |
| `test/registered/unit/mem_cache/test_oceandisk_direct_linker.py` | 新增参数打包、key、命中判断、部分 pool 加载及失败传播等测试。 |
| `oceandisk-unified-linker-design.md` | 记录适配设计和范围。 |

本次复用了现有 UnifiedRadixCache、Wrapper、pool assembler 和模型侧等待入口，没有修改它们，也没有修改 `od/` 中的早期 HiCache 适配代码。

## 2. 接入结构与职责

```text
UnifiedRadixCache / UnifiedCacheLinkerWrapper  现有
    │ PoolTransfer
    ▼
OceanDiskDirectLinker                             新增
    │ pool 展开、key、命中策略、任务队列、完成通知
    ▼
OceanDiskStorage                                 新增
    │ GPU 地址 → MR/offset、元数据打包、调用和同步
    ▼
torch.ops.my_ops
    ▼
libop_extension.so → GIDS 核心库
```

`OceanDiskDirectLinker` 继承 `UnifiedCacheLinker`，实现 Wrapper 已经约定的接口。`OceanDiskStorage` 继承 `HiCacheStorage`，为后续 HiCache 接入保留基类契约。当前实际实现 Direct Linker 范围 I/O，未实现的 host-cache 入口明确报错；Storage 不接触 radix tree、请求调度或节点锁。

`PoolTransfer`、`PoolName`、`PoolHitPolicy` 虽定义在 `hicache_storage.py`，仍可直接导入使用；使用这些类型不要求继承该文件里的 `HiCacheStorage` 类。

原设计中的 `OceanDiskRangePlanner` 没有单独建类：物理 pool 到地址范围的转换放在 Linker 的 `_page_fragments()`，地址到 MR/offset 及 ABI 打包放在 Storage 和辅助函数中。

## 3. 初始化：识别布局并注册 GPU 内存

初始化入口在 `registry.py`：选择 `linker_cls = OceanDiskDirectLinker` 后，调用 `linker_cls(server_args, params, components=...)` 创建对象。Python 会自动执行 `OceanDiskDirectLinker.__init__()`；不需要 Wrapper 或 registry 再手动调用一次 `__init__()`，也不要求抽象基类声明相同的初始化方法。

按当前代码顺序，Linker 初始化分七步：

1. **获取已有 GPU KV cache。** 从 `params.token_to_kv_pool_allocator.get_kvcache()` 取得 SGLang 已分配的 pool，不重新分配 KV cache。
2. **解析物理布局。** 调用 `resolve_hybrid_device_pool_group()`，保存 `entry_map` 和 `num_layers`，收集 `get_hybrid_pool_buffer()` 返回的 GPU tensors，并取得当前 device。DSA 示例包括 KV、Indexer 两个物理 pools。
3. **确定多卡写入规则。** 获取当前 attention TP cache group（未提供时使用 TP cache group）的 rank/size。`rank_replicated=True` 时组内 rank 0 是写回 owner；否则各 rank 写自己的分片。各 rank 保留独立读取能力。
4. **读取配置并校验远端布局。** 读取 namespace、`max_ranges_per_call` 等，生成模型、page size 和并行拓扑身份。调用 `_validate_dsa_layout()` 校验 KV/Indexer 配对，建立 assembler 原始偏移到补齐后远端偏移的映射，并计算有界补齐缓冲大小；此时不传输具体 page 数据。
5. **创建存储封装并注册 GPU 内存。** 执行 `OceanDiskStorage(config, self.device, buffers)`，自动进入 Storage 的 `__init__()`。可选 `storage=` 参数用于测试注入，注入时不在这里构造默认 Storage。
6. **建立完成通知和队列。** 创建 `LayerWiseLoadCounter`、`pending_loads`、`load_queue`、`completed_loads`、`offload_queue` 和 `offload_results`。
7. **启动工作线程。** 启动 `_load_worker` 与 `_offload_worker` 后台线程，等待后续任务；此时初始化不主动加载或写回任何请求的 KV 数据。

第 5 步中，`OceanDiskStorage.__init__()` 进一步完成：

```text
检查连接列表
    ↓
检查 buffers 都位于目标 CUDA device
    ↓
按底层 allocation 去重，保存 GPU 地址和长度（regions）
    ↓
设置 LIBGIDS / GIDS_DIR，加载扩展库
    torch.ops.load_library(op_extension)
    ↓
保存 torch.ops.my_ops 算子命名空间，在初始化、查询和读写时调用对应算子
    ↓
创建专用 CUDA stream 和 output tensor
    ↓
构造 CUDA int64 tensor：[addresses..., lengths...]
    ↓
调用 kvcache_init()，同步 stream 并检查返回状态（0 为成功）
```

Storage 保留注册 buffers 的引用，并拒绝同进程重复初始化；当前 ABI 没有注销/重新注册接口。这里的 `kvcache_init()` 是底层初始化和内存注册，区别于 Python 对象的 `__init__()`。

Linker 构造返回后，现有 registry 再将其挂到 cache Wrapper，并把 `layer_done_counter` 注册给 KV pool 和 TP worker。后续调度才通过 `lookup/load/offload` 等接口发起具体工作。

```text
registry 实例化 Linker
    → 获取现有 GPU pool、解析布局
    → 确定 rank 策略、建立远端布局
    → 构造 Storage、加载库并注册内存
    → 建 counter/队列、启动线程
    → registry 挂接 Wrapper 和模型侧 counter
```

Storage 通过每个 tensor 的 `untyped_storage()` 取得底层 allocation 的地址和字节数，去重后形成：

```text
regions = [(allocation_base_0, bytes_0), (allocation_base_1, bytes_1), ...]
```

随后以 CUDA int64 tensor `[addresses..., lengths...]` 调用 `kvcache_init`。注册的是当前进程参与这些物理 pool 的底层 GPU allocations，不是某一个请求的 slots，也不是其他 rank 的 GPU 内存。

实际读写时再用 `resolve_mr()` 将具体范围 `(ptr, size)` 转成 `(MR_ID, ptr - allocation_base)`；越界或非正长度会报错。MR 编号等于注册列表下标这一点，是当前适配要求底层满足的 ABI 契约。

### 3.1 MR/offset 转换示例与分层理由

假设初始化时的注册表是：

```text
regions[0] = (base=10000, size=4000)   → MR 0，范围 [10000, 14000)
regions[1] = (base=20000, size=8000)   → MR 1，范围 [20000, 28000)
```

Linker 根据某个 page、层及物理 pool 得到：

```python
OceanDiskRange(key=K, ptr=21500, size=256, storage_offset=0)
```

Storage 的 `resolve_mr()` 找到完整包含这 256 字节的第二个注册区域，`pack_io()` 将其打包为：

```text
key            = K
MR_ID          = 1
local_offset   = 21500 - 20000 = 1500
length         = 256
storage_offset = 0
```

| 字段 | 含义 | 当前实现 |
| --- | --- | --- |
| `local_offset` | 相对本地 GPU 注册区域起点的偏移 | `ptr - allocation_base` |
| `storage_offset` | 相对远端 key 对象起点的偏移 | 将 assembler 的原始 offset 映射为按 512 字节补齐后的对象内偏移 |

这个转换只生成 GIDS 寻址参数，不搬动 KV 数据，也不重新注册内存。当前实现对未注册、跨区域或非正长度范围直接报错，不会自动拆分。

职责边界是：

```text
OceanDiskDirectLinker / assembler
    page、层、pool、slots → key、GPU ptr、size
                         ↓ OceanDiskRange
OceanDiskStorage
    GPU ptr → MR_ID + local_offset
    打包 length、storage_offset 等参数，调用 .so
```

因此 MR/offset 转换放在 Storage 合适：它依赖 GIDS 的内存注册规则和 ABI，而 page/层到 GPU 地址的转换依赖缓存布局，留在 Linker 和 assembler。Linker 无需知道底层 MR 如何编号。

当前 `MR_ID=注册列表下标` 与所提供 .cu 的注册数组索引方式对应，代码没有其他编号模式。若底层以后返回独立 MR handle，应在 Storage 记录并转换真实 handle。

## 4. 物理布局与存储对象

本次沿用当前 Direct Linker assembler 已有的布局策略，包括 DSA 的 KV + Indexer，以及现有 DeepSeek V4 对应布局。没有新增普通 MLA 或 GQA 的 Direct Linker 布局策略，因此不能据此声称 DeepSeek V2 Lite、Qwen2.5 已接通。

逻辑 `PoolTransfer` 经 `pool_group.resolve_transfers()` 展开为物理 transfers。随后：

```text
transfer.host_indices
    → entry.prepare_locations()
    → entry.get_prepared_layer_range_meta(locations, layer)
    → OceanDiskRange(key, ptr, size, storage_offset)
```

这里 `host_indices` 是现有传输结构复用的字段，在这条路径中承载物理 device pool 索引，不表示 KV 数据需要经过 CPU host cache。

`_validate_dsa_layout()` 在初始化时检查 packed KV/Indexer 的配对范围，并按物理 buffers 顺序建立补齐后的远端偏移。`_page_fragments()` 保留真实 GPU ptr 和有效 size，仅将 storage_offset 替换为补齐布局中的偏移。KV/Indexer 共享同一 page key，传输时按 KV 在前、Indexer 在后的顺序提交；模型权重或布局变化时应清理旧对象或更换 namespace。

key 的身份信息包括模型 namespace、并行拓扑、page size、pool 名和 page hash。GPU 指针、MR_ID、请求 ID 和本地缓存容量不进入 key，因此同一对象可读入不同 slots。

`encode_key()` 将身份信息做 SHA-256，并截取为正 63 位整数。截断存在碰撞概率，当前没有远端元数据校验机制。模型权重或不兼容配置变化时，应使用新的 namespace。

### 4.1 两层布局与 SGLang 代码的对应关系

**SGLang assembler 描述 GPU 地址及 packed 对象内偏移；OceanDisk Linker 将原始偏移映射到补齐后的远端偏移并生成 key。没有改变 SGLang 原来的 GPU KV 布局。** 远端多个片段相邻，也不要求它们在 GPU 上相邻；Storage 使用各自的本地地址和远端 offset 传输，只有未满足 512 字节长度的片段通过独立缓冲补齐。

#### 第一步：Wrapper 提供逻辑传输

例如某个前缀 page 需要卸载：

```python
PoolTransfer(
    name=PoolName.KV,
    keys=[page_hash],
    device_indices=token_slots,
)
```

`page_hash` 表示前缀内容身份，`token_slots` 表示当前本地位置。这一层还没有展开 KV 和 Indexer 的具体 GPU 地址。一个 page 包含 `page_size` 个 token slots；恢复到另一组 slots 不会改变它的内容身份。

#### 第二步：assembler 描述实际 GPU pools

以无 MTP 的 GLM-5 DSA 路径为例，代码位于 `python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py` 的 `_build_dsa_device_pool_group()`：

```text
DevicePoolGroup
  ├─ KV 的 DevicePoolEntry
  │    components = [kvcache.kv_buffer]
  │    rows_are_pages = False：buffer 的一行对应一个 token
  │
  └─ INDEXER 的 DevicePoolEntry
       components = [kvcache.index_k_with_scale_buffer]
       rows_are_pages = True：buffer 的一行对应一个 page
```

两者都设置 `indices_from_pool=PoolName.KV`。`DevicePoolGroup.resolve_transfers()` 因此将逻辑 KV transfer 展开为 KV、Indexer 两份物理 transfers，复用相同的 page hashes 和来源 slots，再按各自 pool 的规则寻址。两者行粒度不同，但表示的是同一批 token pages。

| `DevicePoolEntry` 字段或方法 | 作用 |
| --- | --- |
| `components` | 按 component 分组的实际 GPU tensors。 |
| `layer_mapping` | 逻辑传输层对应哪些物理 buffer；无 MTP 的普通 DSA 示例中为 `layer → layer`。 |
| `buffer_meta` | 每个 buffer 的基地址、行字节跨度和一页有效数据字节数。 |
| `prepare_locations()` | 将传入的 slots 转成该物理 pool 的 page 起始行位置。 |
| `get_prepared_layer_range_meta()` | 根据位置及层映射，返回具体 GPU ptrs、sizes 和 assembler 的布局 offsets。 |

#### 第三步：计算补齐后的远端偏移

`_storage_offsets[pool][assembler_offset]` 将物理 buffer 的原始偏移映射为补齐偏移；按物理排列累计 `(size + 511) & ~511`，保留 MTP 非连续层映射的物理顺序。

例如 page size 为 64、KV 为 BF16、Indexer 含 FP32 scale 时：

| 数据 | 本地有效长度 | 远端 I/O 长度 | 第 n 层远端偏移（每层一个 fragment） |
| --- | ---: | ---: | ---: |
| KV | 73728 | 73728 | n × 73728 |
| Indexer | 8448 | 8704 | n × 8704 |

#### 第四步：构造范围与补齐读写

`_page_fragments()` 返回原始 ptr、有效 size 和补齐后的 storage_offset。Storage 先验证真实内存范围，再为不对齐片段改用独立注册的 staging MR 和补齐长度。

写出：将有效字节拷入 staging，补齐部分置零，再调用原生 PUT。读回：GET 到 staging，等待层 event，然后仅将有效字节拷回目标 GPU slots；拷贝完成后才通知该层 Future。缓冲按最大批次预分配，分块间等待完成后复用；原始 buffers 的注册长度不扩大。

## 5. 查询路径：`lookup()`

`lookup(rid, transfers)` 返回可恢复的 page 前缀边界集合，不返回 KV 数据。

1. 展开逻辑 transfers。
2. 对候选 page 生成各物理 pool 的对象 keys。
3. 调用 `storage.exists_many()` 查询对象是否存在。
4. 要求底层对象命中代表全部层和 fragment 完整提交；所有必需 pool 对象命中才视为可恢复。
5. 根据 `ALL_PAGES` 或 `TRAILING_PAGES` 策略计算可用边界，对各 pool 的结果取交集。
6. 返回排序后的边界，由现有上层逻辑协调各 rank 的共同命中。

当前逐个 key 调用 `kvcache_exist`，只接受返回 0 或 1，避免把批量命中数量误当作连续前缀。每 page/pool 一个对象减少了被查询的 keys 数量；尚未实现逐 key bitmap 的批量查询。

## 6. 加载路径：排队、逐层传输与等待

### 6.1 一次加载批次是什么

`load(rid, transfers)` 将 Wrapper 的逻辑 transfers 经 assembler 展开后放入 `pending_loads`，支持部分 pool 加载，包括只加载独立来源的 SWA 等组件、没有 KV transfer 的情况；DSA Indexer 仍由逻辑 KV transfer 展开，不接受仅传物理 Indexer 的逻辑输入。

`start_layer_wise_loading()` 把当前所有 pending 请求一起封装成一个加载批次，为它创建一个 counter index，并将任务放入加载队列。空队列时返回 `-1`。

因此，批次不是单个 block，也不一定是单个请求。例如，本次一起恢复 20 个 blocks，模型有 3 个传输层：

```text
futures[batch_index] = [Future_0, Future_1, Future_2]

Future_0：本批次第 0 层所需的全部 blocks / 物理 pools 已可读
Future_1：本批次第 1 层所需的全部 blocks / 物理 pools 已可读
Future_2：本批次第 2 层所需的全部 blocks / 物理 pools 已可读
```

这里共有 3 个 Future。20 个 blocks 不会自动创建 20 个批次。

### 6.2 加载线程何时通知完成

加载线程先等待提交时记录的 CUDA event，再调用 `_run_transfers()`：

```text
for 每一层:
    汇总本批次所有请求、所有物理 transfer 的本层地址范围
    storage.get_ranges(ranges)：分块提交，记录并等待本层 event
    counter.complete(batch_index, layer)
```

只有该层所有请求、所有必需物理 pool 的 get 都成功返回，才将该层 Future 置为成功。对于 DSA，这包括该层需要恢复的 KV 和 Indexer。

### 6.3 模型侧怎样等待

复用现有模型执行和 memory pool 的 counter 接入机制：模型侧先设置当前 consumer batch index，在获取对应层缓存 buffer 时调用 `wait_until(layer)`。

```python
# 模型侧：等待当前批次的这一层
futures[consumer_index][layer].result()

# 加载线程：这一层全部完成
futures[batch_index][layer].set_result(None)
```

模型逐层使用缓存时检查对应 Future，不在开始计算前一次等待所有层。若加载线程已经完成该层，调用立即返回；尚未完成则阻塞。等待不会发起 GIDS load，真正的 load 由后台线程执行。

`threshold` 是传输层编号，`consumer_index` 是加载批次编号。`wait_until(threshold)` 只等待该层 Future，不循环等待 `0..threshold`。加载线程当前按层顺序处理。

当前 Mooncake 的 counter 也采用“每加载批次、每传输层一个 Future”的机制；区别主要在实际传输调用。GIDS 使用 `storage.get_ranges()`，Mooncake 使用其 store 的范围读取接口。

### 6.4 加载失败与完成队列

加载异常时，`fail()` 将该批次尚未完成的 Future 置为异常。模型等待对应层时，异常被包装为 `RuntimeError("OceanDisk layer-wise KV load failed.")`。

线程最终将该批次的请求 ID 列表放入 `completed_loads`，供 Wrapper 排空完成任务和释放节点锁。这条队列通知不等于“加载成功”；加载错误通过 Future 传播。

`cancel_queued_load()` 当前返回 `False`。原因是现有 Wrapper 已经发布树上的目标 slots，取消需要配套回滚，本次没有实现回滚流程。

## 7. 写回路径：`offload()`

`offload()` 展开 transfers 后，由写回 owner 将任务和 CUDA event 放入队列。线程等待 event 后逐层生成 ranges，分块调用 `storage.put_ranges()`。

每层最后一个分块后记录并等待 event，全部成功后产生 `True` ACK；发生异常则记录日志并产生 `False` ACK。Wrapper 通过 `num_completed_offloads()` 和 `pop_completed_offload()` 获取结果，继续使用现有上层完成协调。

## 8. 多卡策略

复制关系取自 `pool_group.rank_replicated`，没有用模型名称硬编码判断。

| 布局 | 远端 key | 写回 | 加载 |
| --- | --- | --- | --- |
| 当前 group 内复制的 pool group | group 内各 TP rank 共享 key | group rank 0 实际写入，其他 rank 生成本地成功占位 ACK | 每个 rank 独立读取到自己的 GPU slots |
| 分片的 pool group | key 包含 TP rank | 每个 rank 写自己的分片 | 每个 rank 读自己的分片 |

共享 key 只减少重复存储和写入；本次没有实现 rank 间读取广播。各 rank 的连接配置必须按相同顺序指向相同远端逻辑分片，否则相同 key 不能保证访问相同对象。

key 保守地包含 TP size，并区分相关 CP/PP 拓扑；本次没有验证跨 TP size 的对象复用。代码包含多 rank 策略，但 GLM-5 多卡正确性仍需真实运行验证。

## 9. `OceanDiskStorage` 怎样调用底层

本次参考 `od/` 的 tensor ABI，直接调用 `torch.ops.my_ops`，没有导入旧 `cache_controller.py`，也没有通过会丢弃算子返回码的 `od_utils` 批量封装。

```text
kvcache_init       注册 GPU allocations
kvcache_exist      查询一个 pool/page 对象是否完整存在
kvcache_load_v2    读取到 GPU 范围
kvcache_store_v2   从 GPU 范围写入
```

读写元数据采用列式排列：

```text
[task_id,
 MR_IDs...,
 keys...,
 local_offsets...,
 storage_offsets...,
 lengths...,
 connection_indices...]
```

同层各请求、各 pool 的范围先合并，每个算子最多提交 `max_ranges_per_call` 个范围，默认 256。所有分块提交后记录一个 CUDA event，等待该 event 后才 complete 该层 Future；正常路径不再逐分块同步 stream。metadata 保留到 event 完成；异常路径同步 stream 后才释放。连接按 key 确定性路由。Storage 仍用一把锁和一个 CUDA stream 串行访问底层算子，不保证读写并发，也没有实现提前提交全部层的流水线。

初始化和读写会按返回码 0 判断成功，初始化及查询仍同步 stream，读写改为等待逐层 event。event 是否等价于底层 RDMA 真正结束，必须由实际 GIDS 二进制保证；Python 层无法凭空补出该保证。

## 10. 配置与生命周期

直接通过脚本中的环境变量配置两个库及 GIDS 原生配置目录，无需外部 JSON 文件：

```bash
export OPEXTENSION=/opt/oceandisk/libop_extension.so
export LIBGIDS=/opt/oceandisk/libgids_kv_host.so
export GIDS_DIR=/opt/oceandisk/config/
```

启用参数为 `--enable-unified-cache-external-linker --unified-cache-external-linker-backend oceandisk`。`GIDS_DIR` 中仍需提供原生库读取的 `gids_config.txt`（或按设备编号的文件），其中记录存储服务连接信息。

`namespace` 默认模型路径，`connection_indices` 默认 `[0]`，`max_ranges_per_call` 默认 256。路径也可由 `OPEXTENSION/LIBGIDS/GIDS_DIR` 提供。模型权重或缓存布局变化时应清理旧缓存或更换 namespace。

初始化及读写统一按原生返回值 0 表示成功、非零报错，不需要成功码配置。CUDA event 等待和注册顺序编号是当前固定实现，不要求填写声明式配置。对象完整性没有通过配置开关实现。

`reset()` 会启动尚未提交给线程的 pending loads，等待读写队列排空，再清理本地完成通知和 counter；不会清除远端对象。`close()` 排空任务、停止工作线程并关闭 Storage。

当前提供的 ABI 没有注销内存或销毁全局状态的接口。因此注册内存保留到进程退出，同一进程内再次初始化 Storage 会被拒绝；不能将 `reset()` 当成注销后重新注册。

## 11. 验证情况与后续工作

当前已完成语法编译、OceanDisk 模块及测试的静态和格式检查、修改文件空白检查。使用模拟 tensor 和依赖隔离运行现有测试中的 18 项，通过；涉及真实 tensor 索引运算的一项未运行。这不等价于完整 PyTorch 单元测试或 CUDA/GIDS 验证。

当前环境没有 PyTorch/CUDA，也没有可运行的 GIDS `.so`。完整新增测试文件尚未运行，以下项目也尚未验证：

- 实际二进制的 MR 编号、返回码、key 范围、长度及对齐约束。
- stream 同步后数据是否真正可读，错误路径是否仍存在未结束的 DMA。
- `exist` 是否只报告完整且已提交的对象。
- GLM-5 多卡 KV + Indexer 写回和恢复，与重新计算结果的一致性。
- owner rank 写入失败、读写错误、进程重启后复用等路径。

后续先在具备 PyTorch 的环境运行新增单元测试，再核对真实 GIDS ABI，最后完成 GPU 和多卡端到端测试。进一步优化、普通 MLA/GQA 布局扩展和 PD 分离属于后续范围。


## 12. 当前边界和验证

- 布局要求 `packed=True`，各层 KV/Indexer 范围数量配对且相同；本地地址使用 assembler，远端偏移映射为补齐布局。
- DSA 的 KV 与 Indexer 为不同物理 pool，同一 page 使用相同 key；MTP 按物理 buffer 顺序计算补齐偏移，不重新按传输层排列。
- `OceanDiskStorage(HiCacheStorage)` 当前实际服务 UC Linker，尚未注册 HiCache factory。五个抽象方法保留基类签名并抛出 NotImplementedError，保证 Storage 可以实例化；基类 v2 报错方法直接继承，v1/host 注册/clear 的静默入口显式拒绝。后续还需实现 host 内存注册、host 读写与 key 映射，以及构造和 factory 接线。
- `MAX_GIDS_*`、`gids_config_path()`、`torch.ops.my_ops`、`kvcache_*`、原生库文件名及环境变量与现有二进制保持一致；这不是旧适配代码。`kvcache_load_v2/store_v2` 名称中的 v2 是真实算子名，不能作为对象格式标签删除。
- 参考旧 controller 的 512 字节补齐规则，用独立 staging 保存补齐内容；不扩大原始注册长度，不跨 page 读写原始 GPU buffers。旧的 io_alignment 配置及直接拒绝不对齐载荷的校验已移除。
- 配置目录按原生 USE_SINGLE_PORT 规则选择文件；连接编号限制为 0..7，且必须小于配置文件实际连接数，每次提交范围数限制为 409600。
- 原生桥接层会吞掉部分 init/get/put 错误，exist 也可能把错误变成 miss。Python 返回码检查无法恢复这些状态；CUDA event 是否覆盖真实 DMA、对象部分写入后的可见性仍需真实库验证。未增加声明底层保证的布尔配置。
- 进程级 `_initialized` 防止重复注册，`_registered_tensors` 和元数据引用保证异步操作期间内存存活，不能当作无用状态删除。当前原生 API 没有暴露注销注册和完整提交操作。

当前测试覆盖 DSA 结构、跨 slots 稳定 key、多 component offsets、MTP 非连续映射、unpacked 拒绝、偏移变化、重叠范围、查询边界、MR/offset 打包、逐层 event/Future 和失败传播。完整新增单元测试需在具备 PyTorch 的环境运行；当前没有真实 PyTorch/CUDA/GIDS 服务验证。代码不修改 od/，不新增普通 GQA、HiCache host-cache 或 PD 支持。


## 13. 恢复后续 HiCache 接入契约

此前精简错误删除了已经明确要求保留的 HiCacheStorage 继承，本次恢复。未实现方法不使用 pass 或成功返回值：父类抽象方法提供兼容签名和明确异常，避免类保持抽象而阻止 UC 创建实例。父类已有明确 NotImplementedError 的方法不重复实现。exists_many/get_ranges/put_ranges 与父类 exists/get/set 保持不同名称，防止整数 key/GPU 范围参数覆盖 HiCache 字符串 key/tensor 接口。

核对保留了 UC registry/参数入口、原生算子加载、GPU allocation 注册与生命周期引用、MR 转换、查询、assembler offsets、逐层读写、event/Future、完成队列及 offload owner 策略。未恢复已删除的 README，也未修改 od/ 文件。此次恢复不表示 HiCache host-cache 路径已可运行。

本轮 20 项隔离测试通过，包含恢复的继承关系、可实例化、父类参数签名一致性及未实现调用明确失败测试。使用实际 HiCacheStorage 类和 Python 方法、合成 buffers 及模拟 CUDA，替换 tensor slots 归一化；静态检查、格式检查和语法编译通过，真实 GPU/GIDS 验证仍未进行。


## 14. 精简注释与重复处理

删除重复职责解释、历史背景和对代码的逐行复述，保留原生 ABI、CUDA 元数据生命周期、异常清理和 HiCache 未实现边界等必要说明。元数据维度检查统一使用 zip(strict=True)，不重复编写多组长度分支；配置目录只规范化一次，初始化配置文件路径改为局部变量，移除原生算子逐个 getattr 预探测。

范围读写在提交前一次生成整层 CPU 描述符，MR 边界查找与地址转换只在 pack_io 中执行一次，移除 _io 中重复的 MR 查找和负偏移检查。每个批次对应的 CUDA 元数据仍保持到层 event 完成，已提交操作异常时仍同步清理。测试检查每个范围只查一次 MR，以及后续批次地址或偏移无效时不会提交前面的读写。

HiCacheStorage 继承及未实现接口保留；原生返回码、内存边界、重复初始化保护和 Future 失败传播保留。没有更改对象 key、存储排列或 od/ 代码。

本轮 20 项隔离测试及静态、格式、语法检查通过，新增断言验证读写打包只查一次 MR、后续范围无效时零提交。测试使用合成 buffers 和模拟 CUDA，不代表真实 GIDS 端到端验证。


## 15. JSONL 离线 Prefix Cache 测试

[run_oceandisk_offline_tp8.sh](examples/runtime/engine/run_oceandisk_offline_tp8.sh) 调用 [oceandisk_offline_tp8.py](examples/runtime/engine/oceandisk_offline_tp8.py)。模型、数据集、原生库路径及网络连接参数写在 Bash 顶部，也可用同名环境变量覆盖。无需外部 JSON 配置文件；JSONL 是用户提供的测试数据集。

Python 读取 multifieldqa_zh.jsonl 第一行，使用 context 和 input 字段，按用户提供的 system/user 消息构造中文问答 prompt，再调用本地 tokenizer.apply_chat_template。模型路径沿用 MODEL_PATH，未固定为参考脚本中的 Qwen2.5；当前 Direct Linker 的模型支持范围仍由 assembler 决定。

```bash
# 编辑 Bash 顶部的模型、数据集及库路径后运行。

# 两次请求：首次预热，再用相同 prompt 检查 HBM prefix cache。
bash examples/runtime/engine/run_oceandisk_offline_tp8.sh base

# 两次请求：预热写出 → 清空本地缓存 → 验证 OceanDisk 读回。
bash examples/runtime/engine/run_oceandisk_offline_tp8.sh oceandisk

# 覆盖数据集或上下文长度。
bash examples/runtime/engine/run_oceandisk_offline_tp8.sh oceandisk --dataset-path /data/multifieldqa_zh.jsonl --context-length 16000 --max-total-tokens 24000
```

两种模式均为单实例 prefill/decode 混部，默认 TP=8、page_size=64、context_length=8000，使用 temperature=0、top_p=0.95、max_new_tokens=16。每次打印耗时、prompt_tokens、cached_tokens、cached_tokens_details 和输出文本。默认启用 cache report；请求间等待 2 秒。

base 模式不加载 GIDS 库。第二次请求以 device 命中数判断 HBM 复用，统计缺失或为零时打印 MISS / not confirmed。oceandisk 模式在首次预热后直接调用 Engine.flush_cache()，排空 Linker 工作并清空本地缓存，保留远端对象，然后发送第二次相同请求，不进行 HBM 复用测试。由于没有启用 HiCache host tier，第二次的 host（当前 Direct Linker 统计路径）和 storage 命中数用于确认外部恢复；远端未命中、flush 失败或仍有 device 命中都会报错。不能通过 --disable-radix-cache 关闭整棵树，否则会破坏当前 UC Linker 依赖的树路径；这里只清空本地驻留缓存来排除 HBM 命中。

Engine 在 finally 中关闭，不生成额外报告文件。模型和数据集使用本地路径，不自动截断数据；输入加生成长度需在设置的上下文上限内。开启 Linker 时原生库、连接配置及服务需预先就绪；仅凭这项 smoke 不能证明 KV/Indexer 全量数据、数值精度和多卡失败路径均已正确。

不传模式等价于 base；每次运行创建新进程并重新加载模型，不是在已有 Engine 内热切换。

本次通过四种模拟 Engine/tokenizer 场景：HBM 复用、远端读回、远端未命中失败和 flush 失败，并检查异常时关闭 Engine；同时验证只读取 JSONL 第一条样本和生成参数。静态、格式及语法检查通过，未运行真实模型/GPU/GIDS 测试。


## 16. 按更新的 GIDS README 配置 IP 与等长 Get

启动脚本顶部新增填写项：

| 变量 | 填写内容 |
| --- | --- |
| GIDS_LOCAL_IP | 推理机器 RDMA/RoCE 网卡的本机 IP，不是 GPU 编号。 |
| GIDS_SERVER_IP | GIDS 存储服务端 IP。 |
| GIDS_SERVER_PORT | 服务监听端口，脚本以 README 示例的 4420 为默认值，需与实际服务匹配。 |
| GIDS_GID_INDEX | 本机网卡的 GID index，必须按部署环境填写。 |
| GIDS_ROCE_PORT | RoCE 端口，默认 1，可修改。 |
| GIDS_QP_COUNT | 每个 IP 的 QP 数量，对应 .cu 的 qp_num_per_ip → qp_cnt。 |

以当前 gids_kernel.cu 实现为准：第六个字段为每个 IP 的 QP 数量，process_config() 将其保存到 qp_num_per_ip，随后传入 config[i].qp_cnt。脚本使用 GIDS_QP_COUNT，填写实际部署需要的 QP 数量；不保留旧字段名别名。od/readme.md 保留用户原文，代码按 .cu 实现适配。

运行 oceandisk 模式时，会检查必要填写项，然后创建 GIDS_DIR 并写入（覆盖）其中的 gids_config.txt：

```text
ip_config = 本机IP, 服务端IP, 服务端口, GID_index, RoCE端口, QP数量
```

脚本显式设置 USE_SINGLE_PORT=1，让八个 rank 使用同一连接配置文件。该值只决定原生库选择哪份配置，不表示只用一张 GPU。当前脚本生成一条连接配置，对应 Storage 默认的 connection_indices=[0]。base 模式不要求填写 IP，也不生成此文件。不会自动启动存储服务；这里生成的是 GIDS 原生文本配置，不需要外部 JSON。

更新 README 还要求 Get 的 IO 长度一致。Storage.get_ranges() 因此在每层内按 size 分组，每组再按 max_ranges 分批提交；Put 保留不同长度混合提交。所有读分组完成后，仍只等待一次层 event，再由 Linker 完成该层 Future，未改变 key、offset、GPU 注册和外部命中测试逻辑。

本轮 20 项隔离测试通过，新增检查混合长度 Get 的分组、批次上限、Put 顺序和一次层 event。使用 Git Bash 检查了脚本语法，并用 Python 替身实际执行 base、配置缺失失败和配置生成场景；未连接 RDMA 服务或运行真实 GPU 模型。未修改用户更新的 od/ 文件。
