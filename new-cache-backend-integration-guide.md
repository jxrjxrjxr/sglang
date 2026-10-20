# 新缓存后端接入指南：Prefix Cache 与 PD 分离

本文说明如何在当前 SGLang 工作区接入一个类似 Mooncake 的新系统。以 `NewStore` 作为示例名称，分别介绍 PC（Prefix Cache）外部存储和 PD（Prefill/Decode）分离。

本文是接入设计与代码导读，示例代码为说明性片段，不是已实现的新后端。第 8 节对照当前 Mooncake 实现。未启动 GPU 或外部服务验证。

版本说明：本次补充时重新核对了工作区源码。当前 PC 使用“插树、锁定加载节点、按完成队列解锁”的实现；此前导读中的“私有槽位加载成功后插树”和 `finish_layer_wise_loading()` 不适用于本次核对的版本。

## 1. 先明确两条独立接入路径

Mooncake 同时提供存储服务和传输引擎，因此在 SGLang 中有两套适配。

| 对比项 | PC 外部存储 | PD 分离 |
| --- | --- | --- |
| 目标 | 保存 KV，供后续请求按前缀复用 | 将 Prefill 的结果交给 Decode 继续生成 |
| 端点 | GPU ↔ 外部 KV store | Prefill → Decode，常规为 GPU → GPU |
| 数据定位 | page hash / storage key | 对端 session、buffer 地址、目标索引 |
| 接入接口 | `UnifiedCacheLinker` | PD Manager / Sender / Receiver / BootstrapServer |
| Mooncake 实现 | `MooncakeDirectLinker`、`MooncakeStore` | `MooncakeKVManager` 等、Transfer Engine |
| 完成机制 | layer counter、顺序 offload 完成队列 | `KVPoll`、传输状态与资源释放协议 |

实现 `UnifiedCacheLinker` 只能获得 PC 外部存储能力，不能自动获得 PD。两者可以共享底层 SDK、内存注册封装和地址范围工具，但生命周期协议需要分别适配。

## 2. PC：运行时调用关系

```text
UnifiedRadixCache
    → UnifiedCacheLinkerWrapper
        → NewStoreDirectLinker 实例
            → NewStore SDK
```

`UnifiedCacheLinker` 是抽象基类，规定 `NewStoreDirectLinker` 必须实现的方法，不是额外的运行时调用层。Wrapper 直接调用具体实例：

```python
self.cache_linker.lookup(rid, transfers)
self.cache_linker.load(rid, transfers)
self.cache_linker.offload(transfers)
```

### 2.1 已有通用逻辑与后端职责

**加粗的“必须实现”是新 Linker 后端的交付范围。** 可以委托给 SDK 或独立 storage adapter，但不能缺少对应能力；“直接复用”表示上层已有，不需要在新 Linker 中重写。

| 新 Linker 是否要做 | 工作 | 你需要交付的内容 / 已有实现 |
| --- | --- | --- |
| **必须实现** | 初始化存储客户端、配置与 key 命名空间 | 连接新 SDK；隔离模型、pool、分片及数据格式 |
| **必须接入，可复用布局工具** | 将逻辑 transfers 展开到物理 pools，获得 GPU 地址范围 | 调用 `DevicePoolGroup` / `DevicePoolEntry`，按新 SDK 要求组织传输描述 |
| **必须实现** | 内存注册与释放（SDK 需要时） | 注册实际 GPU allocations，处理共享 allocation 去重；SDK 无需注册时可省略注册操作 |
| **必须实现** | 查询真实存储，计算本 rank 所有可恢复边界 | `lookup()`；同时满足各必要 pool 的完整性与命中策略 |
| **必须实现** | 加载排队与批次启动 | `load()`、`start_layer_wise_loading()`；读入上层传入的实际目标槽位 |
| **必须提供，可参考现有 counter** | 逐层 ready、等待与异常传播 | 提供兼容的 `layer_done_counter`，仅在数据可安全读取后标记 ready |
| **必须实现** | 加载完成通知 | `num_completed_loads()`、`pop_completed_load()`；返回完成批次的请求 IDs，通知后不得再写目标内存 |
| **必须实现** | 写出排队与执行 | `offload()`；等待源数据 ready，再调用 SDK 持久化 |
| **必须实现** | 按提交顺序报告写出结果 | `num_completed_offloads()`、`pop_completed_offload()`；汇总整组 transfers 的成功状态 |
| **必须实现** | 取消策略、重置与关闭 | `cancel_queued_load()`、`reset()`、`close()`；取消不安全时返回 False，清理前等待在途传输静止 |
| 直接复用，不重写 | 本地 prefix match、page 对齐与尾部 hash 链 | Cache / Wrapper 已实现 |
| 直接复用，不重写 | 构造 Full、SWA 的逻辑 transfers | Component 已实现，由 Wrapper 调用 |
| 直接复用，不重写 | 跨 rank 求合法恢复边界交集 | Wrapper 已实现；新 `lookup()` 只需返回本 rank 的集合 |
| 直接复用，不重写 | 保存命中标记、更新 `MatchResult` | Wrapper 已实现 |
| 直接复用，不重写 | 分配 GPU slots、插树、确定采用范围和组件映射 | Component / Wrapper 已实现；后端使用最终传入的 transfers |
| 直接复用，不重写 | 加载/写出节点的锁管理、完成后的树状态更新 | Wrapper / TreeCore 已实现；后端必须准确报告完成 |

**此外还有两处接线工作**：在 cache registry 增加新后端的创建分支，在参数 backend choices 中增加名称，详见 3.8。上表针对已有 FULL/SWA 组件和布局接新存储；如果还新增模型布局或树组件，则需要额外扩展对应 Component / assembler。

## 3. PC：实现步骤

### 3.1 新增后端类

建议文件：

```text
python/sglang/srt/mem_cache/storage/new_store/new_store_direct_linker.py
```

继承 [UnifiedCacheLinker](python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py)，参考 [MooncakeDirectLinker](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py)。

不要求继承 `MooncakeStore`，也不要求实现整个 HiCache storage 层；可以直接调用自己的 SDK。

### 3.2 适配物理布局并注册内存

优先复用 [linker_pool_assembler.py](python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py)：

```python
pool_group = resolve_hybrid_device_pool_group(
    kvcache=kvcache,
    page_size=params.page_size,
    params=params,
    components=components,
)
```

- `resolve_transfers()`：将逻辑 transfers 展开为物理 pools，例如 KV 及其 Indexer。
- `get_page_buffer_meta()`：得到整页的 GPU 指针和字节数。
- `prepare_locations()` / `get_prepared_layer_range_meta()`：生成逐层读取的地址、长度及对象内偏移。

按底层 allocation 去重注册内存。展开后的 `host_indices` 是复用存储接口的字段，可以承载物理 device pool 索引，不表示必须经过 CPU host cache。

#### GPU KV cache 具体如何注册给 Mooncake

这里的“内存池”指 **GPU 显存中的 KV cache 内存池**，不是 CPU 内存。`kvcache` 是 Python 管理对象，持有实际位于 GPU 上的 tensors/buffers。

当前 Mooncake 的注册过程如下：

1. **SGLang 在启动时分配 GPU KV cache buffers。** 根据显存和配置预留 KV 存储空间；此时部分或全部 slots 可能尚未存放请求的有效 KV 数据。
2. **初始化 Linker 时获取已有的 kvcache 对象。** 不重新分配或复制 KV cache：

   ```python
   kvcache = params.token_to_kv_pool_allocator.get_kvcache()
   ```

3. **构造物理 pool 视图。** `resolve_hybrid_device_pool_group()` 将模型的 KV 布局整理为可遍历的 pools，引用或建立原 GPU buffers 的视图：

   ```python
   self.pool_group = resolve_hybrid_device_pool_group(
       kvcache=kvcache,
       page_size=self.page_size,
       params=params,
       components=components,
   )
   self.pools = self.pool_group.entry_map
   ```

4. **遍历 buffers，取底层 GPU allocation 的起始地址和完整大小，去重后注册。** Mooncake 构造函数会直接调用 `self.register_buffers()`，其核心逻辑为：

   ```python
   seen = set()
   for pool in self.pools.values():
       for buffer in pool.get_hybrid_pool_buffer():
           storage = buffer.untyped_storage()
           allocation = (
               int(storage.data_ptr()),  # GPU allocation 起始地址
               int(storage.nbytes()),    # 完整 allocation 的字节数
           )
           if allocation in seen:
               continue
           seen.add(allocation)
           result = self.storage.store.register_buffer(*allocation)
           # 实际实现还检查返回码，注册失败时抛出异常。
   ```

`storage.nbytes()` 体现了注册范围是底层完整 allocation，而非某个请求使用的几页。这里没有传入请求 ID 或请求 slots，也没有按请求范围截取。多个 tensor view 可能共享同一个 allocation，因此需要按 `(ptr, size)` 去重。

准确的注册范围是：**当前进程中，参与 Linker 传输的各物理 pool 所引用的完整 GPU allocations**。不能理解为一次注册所有 rank、所有 GPU 上的 KV cache。

```text
SGLang 启动时分配 GPU KV buffers
    → 初始化 MooncakeDirectLinker
    → 获取 kvcache，构造物理 pool 视图
    → 遍历 buffers，提取 GPU 地址和完整大小
    → 去重后 register_buffer(ptr, size)

之后每次 load/offload
    → 根据本次 slots 生成具体页地址和长度
    → 在已注册区域内执行实际读写
```

注册是把访问 GPU 内存所需的地址和长度交给 Mooncake 客户端/传输引擎做准备，**注册本身不传输或持久化 KV 数据**。数据仍留在原 GPU buffer 中；后续 load/offload 才按具体页读写。

UMBP 使用相同的 allocation 遍历与去重方式，只是调用 `self.storage.client.register_memory(ptr, size)`。新后端需要根据 SDK 的内存访问要求实现对应注册流程。

### 3.3 定义存储对象与 key 格式

Wrapper 已提供每页 hash，但最终物理对象 key 与字节布局由后端定义。

需要明确：模型标识、数据格式版本、pool、并行分片以及影响布局的配置如何隔离；一页中的多层、多 tensor 如何排列；逐层读取时的 offsets 如何对应。

不要直接把裸 page hash 当作跨所有模型和布局的全局唯一 key。复制型 KV 可考虑去重，分片型 KV 必须区分 rank；第一版可以先不做复制去重。

### 3.4 实现 lookup

输入为各 pool 的 `PoolTransfer`，查询阶段主要使用 `name`、`keys`、`hit_policy`。

```python
def lookup(self, rid, transfers) -> list[int]:
    # 展开物理 pools，查询存储完整性，返回当前 rank 的合法恢复边界
    ...
```

- `ALL_PAGES`：候选边界之前的页全部存在。
- `TRAILING_PAGES`：候选边界末尾需要的窗口存在。
- 所有必要物理 pools 都满足策略，才算该边界可恢复。
- 返回全部合法边界，单位为 page，且相对于传入的未命中尾部。

例如返回 `[4, 6]` 表示尾部恢复 4 页或 6 页合法。Wrapper 会跨 rank 求交集，无需后端重复归约。

写入部分完成时，lookup 不能误判为完整数据。可以依赖底层原子对象写入、完整性查询、分组提交或完成标记，具体由存储能力决定。

### 3.5 实现异步 offload

```text
展开 transfers → 等 GPU 数据 ready → 后台写入 → 顺序记录成功/失败
```

接口要求：

| 方法 | 实现要求 |
| --- | --- |
| `offload()` | 返回是否接受排队，不能把接受排队视为持久化完成 |
| `num_completed_offloads()` | 返回可按提交顺序消费的已完成任务数量 |
| `pop_completed_offload()` | 消费最早任务的成功状态 |

Wrapper 已持有源节点锁。SDK 若乱序完成，后端需要按提交序号重排完成结果。整组 transfers 成功后才报告成功；部分对象写入后的可见性仍需通过上一节的存储协议保证。

### 3.6 实现加载与 layer counter

Wrapper 已分配 GPU slots 并执行插树，通过 COMMIT 阶段确定需要实际加载的范围。`load()` 只排队，`start_layer_wise_loading()` 才启动当前批次。此时节点已经发布，因此必须保持节点锁直到传输结束。

```text
load(rid, transfers)
    → pending loads
start_layer_wise_loading()
    → 创建 counter index
    → 读取 layer 0 → complete(index, 0)
    → 读取 layer 1 → complete(index, 1)
    → …
num_completed_loads() / pop_completed_load()
    → 消费完成批次的请求 ID，Wrapper 释放对应加载锁
```

可以参考 Mooncake 文件中的 `LayerWiseLoadCounter` 实现兼容 counter。`complete()` 必须表示该层数据已可安全读取，而不只是 SDK 接受了任务。按实际 SDK 的完成语义处理设备 event 或 stream 顺序。

若暂不支持范围读，可以先加载整个批次，确认成功后再标记所有层 ready；这是正确性优先的起点，暂时没有逐层重叠收益。

加载失败时标记 counter 失败。当前 Mooncake 在逐层等待中抛出异常；完成队列返回请求 ID，用于释放锁，不承载成功状态。不能照搬此前“失败后只中止私有请求”的描述。新后端必须保证完成通知发出后，该批次不再写目标 GPU 内存，并明确已发布数据发生加载失败时的处理策略。

### 3.7 取消与生命周期

- `cancel_queued_load(rid)`：只有能够保证树与组件状态安全时才可返回取消成功。当前 Mooncake 始终返回 False，源码注明取消前仍需原子回滚已发布状态。
- `reset()`：清理排队任务，等待已启动任务静止，再重置结果与 counter。
- `close()`：完成上述清理后退出线程、解除注册并释放客户端资源。

请求槽位释放后仍有后台写入，会破坏其他请求的数据，因此超时和失败路径也必须满足静止要求。

### 3.8 注册与参数

在 [mem_cache/registry.py](python/sglang/srt/mem_cache/registry.py) 增加 `newstore` 的类选择，在 [fields/memory.py](python/sglang/srt/arg_groups/fields/memory.py) 的 backend choices 中增加名称。

接入后的预期参数片段：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend newstore
```

这只是新增实现后的示意，当前仓库尚无 `newstore` 后端。连接参数需定义并校验；不必机械沿用 Mooncake 的配置字段。

当前 direct linker 与 hierarchical cache 互斥，Wrapper 仅接受 FULL/SWA 树组件。新增后端不应绕过这些限制，除非同时实现和验证相应能力。

## 4. PD：实现步骤

### 4.1 对接独立的 PD 协议

基础协议在 [base/conn.py](python/sglang/srt/disaggregation/base/conn.py)，公共实现位于 [common/conn.py](python/sglang/srt/disaggregation/common/conn.py)。可以参考公共类复用控制流程，避免完整复制 Mooncake 文件。

建议新增：

```text
python/sglang/srt/disaggregation/new_store/
    __init__.py
    conn.py
```

| 类 | 职责 |
| --- | --- |
| `NewStoreKVManager` | 引擎初始化、内存注册、连接、传输任务与状态管理 |
| `NewStoreKVSender` | Prefill 侧初始化请求、提交 chunk、查询完成及错误 |
| `NewStoreKVReceiver` | Decode 侧发现 Prefill、发送目标 metadata、查询完成及错误 |
| `NewStoreKVBootstrapServer` | 发布/发现 Prefill 端点及拓扑；可复用公共实现 |

当前后端共享 `KVArgs` 容器，通常不必新建一套 KVArgs。

### 4.2 注册后端选择

在 [disaggregation/utils.py](python/sglang/srt/disaggregation/utils.py) 中：

1. 给 `TransferBackend` 增加枚举值。
2. 在 `get_kv_class()` 中增加 Manager、Sender、Receiver、BootstrapServer 映射。
3. 在新包中导出对应类。

更新 [fields/disagg.py](python/sglang/srt/arg_groups/fields/disagg.py) 使用的 `DISAGG_TRANSFER_BACKEND_CHOICES` 定义，并检查 [pd_disaggregation_hook.py](python/sglang/srt/arg_groups/pd_disaggregation_hook.py) 等校验中的后端白名单。

不能仅把新名称加入所有白名单。例如 DCP、特殊模型或统一内存路径是否允许，需要根据新后端实际能力判断。

接入后的预期参数片段：

```text
# Prefill
--disaggregation-mode prefill
--disaggregation-transfer-backend newstore

# Decode
--disaggregation-mode decode
--disaggregation-transfer-backend newstore
```

完整启动还需要模型、端口和路由等配置；这里只表示后端选择。

### 4.3 使用 KVArgs 注册内存

`KVArgs` 提供 KV、aux、state 的指针、总长度、item 大小、layer IDs 及拓扑信息。后端将它们转换为自身引擎的注册描述。

对共享 allocation 去重注册，但保留用于传输配对的逻辑条目。需要逐项处理的零长度 buffer 也应检查，不能提交给不接受零长度注册的引擎。

### 4.4 打通控制面和数据面

```text
Decode 分配目标槽位
    → Receiver 发布 session、目标索引、aux/state 索引和 prefix 长度
Prefill 计算 chunk
    → Sender 提交源索引
Manager 计算源/目标范围
    → 引擎传输
    → 维护状态并通知对端
Decode 确认完成后进入后续调度
```

沿用现有 bootstrap 和请求 room 语义时，需要完整维护其并行拓扑、请求隔离与状态通知协议。若底层是拉取式引擎，也可以调整数据面方向，但不能破坏上层“目标已分配、完成前不消费”的约定。

### 4.5 索引转换与 unified memory

Prefill 和 Decode 的调度代码通过 allocator 的 transfer translation 将虚拟索引转换为物理索引。后端消费传入索引时，应确认对应分支的索引语义，不能重复转换。

统一内存 whole-envelope 路径按以下方式定位：

```text
address = buffer_base + physical_page_id * page_envelope_bytes
```

这不是普通的按层 K/V buffer 列表。需要匹配两端 item 大小、page size 和布局；不支持的并行重切片必须在第一次写入前拒绝。

Mooncake 当前 unified-memory 路径要求两端 attention TP size 一致。新后端若想放宽，需要实现额外布局转换，不能直接删除校验。

### 4.6 chunk、prefix 和附加状态

- 支持 chunked prefill 的多个发送片段，正确判断最终块。
- 支持 Decode 已有 prefix 时的增量传输，避免源/目标范围错位。
- 除主 KV 外，处理模型需要的 SWA、Mamba、Indexer、scale 等状态。
- 按既有调用协议处理最终块的 aux/state。
- 源与目标 slot 编号不要求相等，但必须按相同逻辑位置配对。

这些 PD state 能力与 external linker 的树组件支持范围是独立的。

### 4.7 完成、取消和内存稳定性

将底层状态正确映射为 `KVPoll`：`Bootstrapping`、`WaitingForInput`、`Transferring`、`Success`、`Failed`。

`Success` 需要表示本请求必要数据和状态已完成，不能仅表示任务提交成功。按公共协议完成所需 rank/stage 的协调。

特别注意：

- 目标地址发布后，到传输静止前，不得释放或搬迁目标页。
- 源页在引擎仍可能读取时不得释放。
- 超时、Future 取消不一定能停止正在执行的远端写入。
- 接入统一内存时，应让新后端的在途任务完整进入现有队列和门控生命周期；若存在额外在途窗口，应补充保护。

参考 `unified_memory_disagg_move_gate()` 及 Mooncake 的失败 drain/延迟释放处理，结合新引擎能力实现对应保证。

## 5. 建议实施顺序

PC 和 PD 可以独立推进。每条路径先完成固定布局的闭环，再扩大支持范围。

| 阶段 | PC | PD |
| --- | --- | --- |
| 第一阶段 | 单 rank、Full KV、固定模型/page size，整批读写 | 两端相同模型/page size/attention TP，直接传输 |
| 第二阶段 | 真正远端命中后的输出一致性、失败与清理 | 多 chunk、完成通知、取消与安全回收 |
| 第三阶段 | 多 rank、SWA、附属物理 pools | prefix 增量、SWA/Mamba 等模型状态 |
| 第四阶段 | 逐层加载、批处理、复制去重 | 更多 TP/PP/DCP 组合、staging、性能优化 |

第一版支持范围有限时，应明确拒绝未实现组合。

## 6. 验证与验收

### 6.1 PC

```text
请求 A 生成 KV → 等待写出完成 → 清除本地缓存
    → 请求 B 使用相同前缀 → 确认实际远端加载
    → 与重算的输出/logprobs 对比
```

同时验证：部分写入不可误命中；稀疏恢复边界正确；加载异常能传播且不会误用未完成数据；取消策略与已发布树状态一致；reset、close 后没有在途写入；乱序 SDK 完成不会打乱 offload ACK。

参考：

- [test_unified_cache_linker.py](test/registered/unit/mem_cache/test_unified_cache_linker.py)：内存后端、Wrapper 与树侧行为。
- [test_linker_pool_assembler.py](test/registered/unit/mem_cache/test_linker_pool_assembler.py)：物理布局。
- [test_unified_cache_linker_kl_glm52.py](test/registered/radix_cache/unified_radix_tree/linker/test_unified_cache_linker_kl_glm52.py)：真实存储加载与 KL 验证。

### 6.2 PD

先验证单请求端到端输出与非 PD 基线一致，再覆盖多 chunk、已有 prefix、不同源/目标索引、附加状态、连续请求的槽位复用、连接失败、传输超时和请求取消。

统一内存还要覆盖地址发布到队列入队之间的窗口，以及 compaction/回收与传输的互斥。声明支持多 rank 后，再验证对应拓扑和完成共识。

## 7. 代码阅读索引

| 文件 | 关注点 |
| --- | --- |
| [unified_cache_linker.py](python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py) | PC 抽象接口、Wrapper、counter |
| [mooncake_direct_linker.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py) | PC 后端实现范例 |
| [linker_pool_assembler.py](python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py) | PC 物理地址与布局 |
| [registry.py](python/sglang/srt/mem_cache/registry.py) | PC 创建与 counter 注册 |
| [base/conn.py](python/sglang/srt/disaggregation/base/conn.py) | PD 基础接口、KVArgs、KVPoll |
| [common/conn.py](python/sglang/srt/disaggregation/common/conn.py) | PD 公共实现 |
| [mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py) | PD 具体传输、注册与完成协议 |
| [disaggregation/utils.py](python/sglang/srt/disaggregation/utils.py) | PD 后端选择与内存搬迁门控 |
| [prefill.py](python/sglang/srt/disaggregation/prefill.py)、[decode.py](python/sglang/srt/disaggregation/decode.py) | PD 两端调度、索引与请求生命周期 |

已有架构导读见 [Unified Cache Linker 实现与对接导读](unified-cache-linker-guide.md)。

## 8. 对照现有 Mooncake 实现

### 8.1 PC：初始化与运行对象

入口是 [mooncake_direct_linker.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py) 中的 `MooncakeDirectLinker`，由 cache registry 创建并交给 Wrapper。

初始化主要完成：

1. 通过 `resolve_hybrid_device_pool_group()` 获取物理 pools 和传输层数。
2. 获取并行拓扑，生成 storage suffix。分片 KV 区分 TP rank，复制型 KV 的写出由 TP rank 0 负责。
3. 解析配置，创建 `MooncakeStore(storage_config, mem_pool=None)`。
4. 将 device pool group 接到 storage 的 pool 视图，注册 GPU allocations。
5. 创建 `LayerWiseLoadCounter`、`pending_loads`、加载队列、加载完成队列、写出队列和写出结果队列。
6. 启动独立的 load/offload 后台线程。

`LayerWiseLoadCounter` 在当前版本定义于 Mooncake 后端文件，而不是通用 `unified_cache_linker.py`。

### 8.2 PC：lookup 的实际调用

```text
Wrapper.match()
    → Component 生成 LOOKUP transfers
    → MooncakeDirectLinker.lookup()
        → pool_group.resolve_transfers()
        → MooncakeStore.batch_exists_v2(page_keys, expanded)
        → 返回 restorable_prefix_pages
    → Wrapper 跨 rank 求交集并保存命中标记
```

`MooncakeDirectLinker.lookup()` 是薄适配。实际 key 编码、多 pool 完整性检查及 `ALL_PAGES` / `TRAILING_PAGES` 策略由 [MooncakeStore](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py) 处理。

新后端可在自己的 linker 内完成这些检查，也可以像 Mooncake 一样委托给独立 storage adapter。

### 8.3 PC：先插树，再排队加载

当前 Wrapper 的 `load_back()` 流程：

```text
消费 hit marker
    → Component 构造 LOAD transfers 并分配槽位
    → PREPARE 组件状态
    → cache.insert(..., track_adopted_ranges=True)
    → 收集 canonical tail 索引
    → Component COMMIT，筛选/调整实际加载范围
    → 锁定节点，调用 MooncakeDirectLinker.load()
```

插树时可能复用已有槽位，因此实际加载可以是部分 pool，甚至只需要 SWA 等 side pool。Mooncake 的 `load()` 使用 `allow_partial=True, allow_missing_kv=True` 展开 transfers，再保存到 `pending_loads[rid]`。

这一点对新后端很重要：不能强制每次 load 都有非空主 KV，也不能忽略 COMMIT 后传入的实际索引和范围。

### 8.4 PC：加载线程如何读每一层

`start_layer_wise_loading()` 切出当前 pending dict，创建 counter index，记录 device ready event，并将批次交给 load thread。

```text
load_thread_func()
    → 等 ready event
    → load_layer_wise()
        → 按物理 pool 合并 keys 和 locations
        → 生成 component keys 并追加 storage tags
        → batch_get_session_start(keys)
        → 对每层、每 pool：
            get_prepared_layer_range_meta(locations, layer)
            batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
            检查实际传输字节数
        → 每层 complete(counter_index, layer)
        → batch_get_session_end(keys)
    → 将该批次请求 IDs 放入 completed_loads
```

counter 的 `wait_until()` 调用 Future 的 `result()`；加载失败时会抛出 RuntimeError。线程在 finally 中发布加载完成批次，因此完成通知表示传输工作已结束，不等同于成功。

Wrapper 的 `drain_loads()` 消费完成批次并释放节点锁。若新 SDK 在失败返回后仍有异步 DMA，必须先 drain，不能直接复用相同的完成通知逻辑。

### 8.5 PC：写出线程与生命周期

```text
Wrapper 给源节点加锁
    → MooncakeDirectLinker.offload()
        → 展开物理 transfers
        → 非 owner rank 直接加入成功结果
        → owner 记录 ready event、写出任务入队
    → offload_thread_func()
        → 等 event
        → storage.batch_set_v2(expanded)
        → 汇总所有 pool 结果到 offload_results
    → Wrapper/Cache 跨 rank 确认结果，更新节点状态并解锁
```

当前 `cancel_queued_load()` 不取消已发布加载，直接返回 False。`reset()` 等待线程任务队列清空，再重置结果与 counter；`close()` 在 reset 后发送退出哨兵、join 线程并关闭 storage。

### 8.6 PD：对象关系和线程结构

PD 入口是 [disaggregation/mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py)。当前实现通过 `disaggregation/utils.py` 的 `get_kv_class()` 创建相应对象。

```text
Prefill 调度                          Decode 调度
    ↓                                    ↓
MooncakeKVSender                    MooncakeKVReceiver
    ↓                                    ↓
MooncakeKVManager                  MooncakeKVManager
    ↓                                    ↓
Transfer Engine ───── 数据传输 ───→ 已注册目标内存
    └──────── ZMQ 控制消息 / 状态 ─────────┘
```

Manager 继承 `StagingManagerMixin` 和 `CommonKVManager`。初始化时获取 Transfer Engine，并注册 KV/aux/state buffers。

- Prefill 侧启动接收 bootstrap metadata 的线程，创建多个 transfer queues、executor 和 transfer workers。
- Decode 侧启动接收状态/控制消息的线程；启用 staging 时另有 staging allocator 和到达处理。
- 可选启用 failed-session probe，跟踪失效连接。

### 8.7 PD：metadata 和实际传输

`MooncakeKVReceiver._register_kv_args()` 处理对端所需的注册信息；请求级 `send_metadata()` 通过 ZMQ 发布 room、本端 endpoint/session、目标 KV 索引、aux/state 索引、所需目标信息数量和 Decode prefix 长度等。

Prefill 侧 `MooncakeKVSender.send()` 调用 `add_transfer_request()` 排队。最后一个 chunk 附带 state 和 aux 信息。

常规数据路径：

```text
transfer_worker()
    → 选择当前请求和目标 rank 信息
    → send_kvcache()
        → 校验 whole-envelope 布局（对应模式下）
        → _send_kvcache_generic()
        → 生成并合并源/目标连续地址范围
        → _transfer_data()
        → engine.batch_transfer_sync(session, src_ptrs, dst_ptrs, lengths)
```

不同模型、不同并行布局和 staging 路径会进入额外分支。新后端第一版可以只实现明确声明的直接传输组合，无需复制全部优化，但必须正确校验不支持的组合。

### 8.8 PD：最后一块、完成与失败

主 KV 成功后，最终 chunk 继续通过 `maybe_send_extra()` 传必要 state，再发送 aux。代码在收齐需要的目标 rank 结果后发送最终状态。

Decode 控制线程解析状态消息并调用公共的 `apply_prefill_status()`；Receiver 的 `poll()` 向调度返回聚合后的状态。启用 staging 时还处理 `CHUNK_READY` 等消息。

取消/失败使用控制协议协调。启用 deferred decode KV release 时，`ABORT_ACK` 携带发送 rank 信息，表示相应发送端已 drain；接收端聚合 ACK 后才能按对应协议安全释放。不能将收到任意一条失败通知直接等同于所有写入都停止。

### 8.9 从 Mooncake 学什么、替换什么

| 内容 | 新后端处理方式 |
| --- | --- |
| Wrapper 的 match、插树、锁与完成回收 | 复用 |
| PC pool 展开与地址 range 描述 | 尽量复用 |
| Mooncake 存储 key 编码、session get、batch set | 按新 SDK 实现 |
| PD 公共 bootstrap/状态聚合能力 | 尽量复用并遵守协议 |
| Mooncake Transfer Engine 注册和批量写调用 | 替换为新引擎 |
| GPU ready、逐层可读、DMA 完成及取消 drain 语义 | 必须根据新 SDK 重新确认 |
| MLA 去重、异构并行、staging 等优化 | 按能力逐项实现，不能因参考后端存在就默认支持 |
