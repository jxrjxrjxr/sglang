# OceanDisk 接入 UnifiedCacheLinker：Prefix Cache 设计方案

**命名约定：** Python 后端统一命名为 OceanDisk，入口为 `oceandisk`，核心类为 `OceanDiskDirectLinker` 和 `OceanDiskStorage`，范围描述为 `OceanDiskRange`。底层继续使用现有 GIDS 库；原生算子、环境变量和配置文件名保持不变；对象使用固定标识 `sglang-oceandisk-pool-pages`，复用 assembler offsets；key 由模型/并行身份、page size、pool 名和 page hash 决定。不保留旧后端入口别名。

本文回答“为什么这样设计、职责怎么划分、后续怎样扩展”；已经落地的代码与修改记录见 [工作区修改说明](oceandisk-unified-linker-changes.md)。本文包含后续目标，不能将所有设计目标视为已实现功能。

**2026-10-19 对照工作区核对：** 当前为 `OceanDiskDirectLinker(UnifiedCacheLinker)` → `OceanDiskStorage(HiCacheStorage)` → GIDS 算子；Linker 使用 `exists_many/get_ranges/put_ranges`。远端按 page/物理 pool 建对象，内部按层/fragment 偏移存储，同层合批并等待逐层 event。HiCache host-cache 路径、普通 MLA/GQA 布局扩展和跨 pool bundle 均未实现；真实 GPU/GIDS 与 GLM-5 多卡验证尚未完成。

> 初步实现已落到 `python/sglang/srt/mem_cache/storage/oceandisk/`，沿用当前已有布局，不包含本文讨论的普通 MLA/GQA 布局扩展。实现分为 OceanDiskDirectLinker 和 OceanDiskStorage；为检查返回码，首版直接调用 torch.ops 而不使用丢弃返回码的 batch 门面。运行配置见 [代码修改说明](oceandisk-unified-linker-changes.md) 第 10 节，待验证的二进制约定见本文。未进行 GPU/GIDS 服务验证，不能将代码落地等同于模型支持已验收。

## 1. 目标与范围

使用现有 GIDS `.so` 为 UnifiedRadixCache 提供 prefix cache 外部存储能力，跳过 CPU host cache。复用通用 Wrapper、Component 与物理 pool assembler；不迁移旧 HiCache controller 的调度框架，不涉及 PD。

长期目标是通用 GIDS prefix cache 后端，覆盖 **GLM-5 多卡和 GQA（如 Qwen2.5）模型**；当前实现范围已收敛为复用现有 Direct Linker 布局策略，不新增普通 MLA/GQA 策略。不能按 GLM-5 的类名、单一 KV 布局或复制属性写死。标准 GLM-5（GlmMoeDsaForCausalLM 路径）是具体目标场景，需要主 KV 和 DSA Indexer 的完整保存、查询与恢复，但尚未完成真实多卡验收。Qwen2.5 的部署配置、布局接入和端到端支持属于后续工作，不能假定其缓存布局与 DSA 相同。单 rank 可用于底层接口调试，多卡验证规模按实际部署配置确定。

FULL-only 只描述 GLM-5 目标路径，不作为整个 OceanDisk 后端的固定限制，也不代表只传一个物理 KV pool。DSA assembler 会将其展开为 `PoolName.KV` 和 `PoolName.INDEXER`；Indexer 对应 `index_k_with_scale_buffer`，不能遗漏。GQA（Qwen2.5） 应根据实际模型确定所需树组件、物理 pools、附属状态和命中策略，纳入对应验证；不能遗漏模型必需状态后仍声明支持。MTP、PP/CP 等组合分别确认，不能由 TP 支持自动推断。

### 通用适配原则

- `OceanDiskDirectLinker` 按 Component 产生的 `PoolTransfer` 和 assembler 返回的实际 pools 工作，不硬编码“只有 KV”或“固定 KV + Indexer”。
- 使用 `DevicePoolEntry` 的 layer mapping、component buffers 和 ranges，不假设每层只有一个连续 tensor。
- 使用布局描述的复制/分片属性决定 keyspace 和写出 owner，不用模型名称判断。若一个布局包含复制属性不同的 pools，需要提供准确的逐 pool 描述并分别处理，不能直接用一个组级标记覆盖全部。
- lookup 必须满足所有必需 pools 的命中策略，每层 ready 必须等待该层所有必需 ranges 完成。
- 现有 assembler 能描述的布局直接复用；缺少布局描述时扩展 assembler 的对应策略。若 GQA（Qwen2.5） 需要当前 Wrapper 不支持的树组件，则另行设计组件接入，不能宣称只新增存储 backend 就足够。
- 模型覆盖分别验收 GLM-5 和指定 GQA（Qwen2.5）；GLM-5 测试通过不能代替 GQA（Qwen2.5） 验证。

现有 od 目录已提供 `gids_kernel.cu`，可以核对参数解析、当前 stream 传递和桥接层错误处理；仍没有可运行的 `.so`、核心库实现及其依赖头文件。已发现桥接层对部分 init/get/put 错误仅打印后返回成功，exist 也可能将错误变成 miss。按当前要求不修改 od 原文件，Python 无法恢复被吞掉的状态；真实传输完成及完整对象命中仍需测试。

## 2. 调用结构与代码划分

### 已核对的 GQA 布局接入缺口

当前 `_PlainKvStrategy` 能为普通 KV 布局构建 HiCache stack，但没有覆写 `build_direct_linker_pool_group()`；基类默认会报不支持。因此 Qwen2.5 GQA 的 direct-linker 布局还需补充，不能假定现有 assembler 已能直接接入。

计划在通用 assembler 的 plain-KV 策略中增加 direct pool 构建：按实际 K/V buffers、层映射、page 大小及附属 dtype 数据构造 DevicePoolEntry，不在 GIDS 中硬编码 Qwen 模型。常规 TP 分片路径使用区分 TP rank 的对象 key；GLM-5 DSA 沿用已存在的复制型 KV + Indexer 描述。特殊 KV-head 复制或布局须据实际拓扑单独验证。两者使用同一个 OceanDiskDirectLinker/OceanDiskStorage。

```text
UnifiedRadixCache
    → UnifiedCacheLinkerWrapper             已有，直接复用
        → OceanDiskDirectLinker                  新增：接收 PoolTransfer，复用 assembler
            │                              管理队列、counter、完成通知
            → OceanDiskStorage                   新增：地址 → MR/offset、参数打包
                │                          封装 .so、检查完成与错误
                → torch.ops.my_ops.kvcache_*
                    → libop_extension.so
                        → libgids_kv_host.so
```

当前两个主要实现类已放在 `python/sglang/srt/mem_cache/storage/oceandisk/` 下的 `oceandisk_direct_linker.py` 和 `oceandisk_storage.py`。地址转换和参数打包合并到 `OceanDiskStorage`，不单独设 RangePlanner 类；后续转换规则明显变复杂时再考虑抽取。

| 部分 | 职责 |
| --- | --- |
| Wrapper / Component | 本地 match、hash 链、跨 rank 边界交集、分配、插树、组件映射、节点锁和完成回收 |
| OceanDiskDirectLinker | 继承 UnifiedCacheLinker；接收 PoolTransfer，复用 assembler 生成物理 ranges；组织对象标识和命中策略；管理异步队列、counter、完成队列、reset/close |
| OceanDiskStorage(HiCacheStorage) | 只实现 Direct Linker 所需的范围 I/O；负责延迟加载库、初始化与注册；接收已编码 key 和 GPU 地址范围；MR 查找、偏移转换、conn 路由、tensor ABI 打包、dispatch、底层完成与错误检查 |

`PoolTransfer`、`PoolName` 是 `hicache_storage.py` 的顶层数据类型，直接导入使用，不需要继承 `HiCacheStorage`。已有 Wrapper、Component 与这些通用类型保持不变。只有 `OceanDiskDirectLinker` 需要理解它们；当前 `OceanDiskStorage` 的 get_ranges/put_ranges 接收 `OceanDiskRange(key, ptr, size, storage_offset)`，内部管理 stream，不依赖 radix tree、请求调度或节点锁。

命名按职责区分：`OceanDiskDirectLinker` 实现上层 Linker 契约；`OceanDiskStorage` 是具体存储访问封装，由 `self.storage` 持有，不是抽象接口；按后续 HiCache 接入要求继承 `HiCacheStorage`；当前未实现的 host-cache 入口明确抛出 `NotImplementedError`。`OceanDiskRange` 保留为范围数据类，`LayerWiseLoadCounter` 保留为逐层完成通知类。底层 `torch.ops.my_ops` 算子名不变。

两个类的协作可概括为：

```text
OceanDiskDirectLinker：PoolTransfer → assembler → OceanDiskRange(key, ptr, size, storage_offset)
OceanDiskStorage：OceanDiskRange → key/MR/local_offset/length 等描述符 → .so
```

当前转换逻辑由 `oceandisk_storage.py` 中的 `resolve_mr()` 和 `pack_io()` 辅助函数实现，由 Storage 调用。Linker 的 reset/close 负责协调任务队列，Storage 负责其持有的底层资源，两者共同确保清理前传输已静止。

不直接 import `od/nds_utils.py`：其模块加载有设备判断及库加载副作用，部分 CUDA 方法是 pass/NotImplemented，部分旧接口返回常数。抽取其有效的 ABI 和路由思路，用一个独立适配器封装即可。

### 当前范围接口

`OceanDiskStorage(HiCacheStorage)` 当前实际实现 UC Linker 使用的 `exists_many()`、`get_ranges()`、`put_ranges()` 和 `close()`，使用整数对象 key 和 GPU ranges。父类五个抽象方法保留兼容签名并抛出 `NotImplementedError`，使 UC 路径仍能实例化 Storage；基类已有明确报错的 v2 方法直接继承，基类会静默返回的 v1、host 注册和 clear 入口显式拒绝。继承是后续接入基础，host-cache 读写、host 内存注册、构造适配与 HiCache factory 接线仍待实现。详见 [代码修改说明](oceandisk-unified-linker-changes.md)。

## 3. 使用哪些现有二进制接口

| 接口 | 用途 |
| --- | --- |
| `torch.ops.load_library(OPEXTENSION)` | 加载 libop_extension，注册 my_ops 算子 |
| `import od_utils` | 旧批量门面；当前不导入，参考其 stream/event 实现 |
| `kvcache_init(input, output)` | 初始化后端及注册 GPU allocations |
| `kvcache_exist(input, output)` | 查询 block key 是否存在 |
| `kvcache_load_v2(...)` | 当前在指定 stream 分块读，保留返回码 |
| `kvcache_store_v2(...)` | 当前在指定 stream 分块写，保留返回码 |

`LIBGIDS` 用于核心库路径覆盖，`GIDS_DIR` 指向配置目录。根据 guide 的约定设置路径并检查实际构建 ABI；当前通过 torch.ops 调用对应算子，不要求加载 od_utils 批量门面。

库仅在启用 oceandisk 后端时加载，在实际 CUDA device 上完成初始化，不以全局 cuda:0 为默认假设。核心资源可能为进程全局状态，需要约束同进程重复初始化及不同 buffers 的冲突。

## 4. 初始化与 GPU 内存注册

### 初始化的触发与职责顺序

registry 选择 `OceanDiskDirectLinker` 后执行类实例化表达式，Python 自动调用其 `__init__()`。该方法按顺序获取现有 GPU pool、解析物理布局、确定多 rank 写回 owner、读取配置、建立远端对象布局，然后构造 `OceanDiskStorage`，最后建立 counter/任务队列并启动读写线程。

创建 `OceanDiskStorage` 又会自动执行它的 `__init__()`：检查配置和 device、收集并去重 allocations、设置库路径、通过 `torch.ops.load_library()` 注册算子、创建 stream/output、调用 `kvcache_init()` 注册内存，并同步检查初始化状态。可选 `storage=` 注入仅用于替换默认存储对象，例如单元测试。

```text
registry：实例化 OceanDiskDirectLinker
    → Linker.__init__：pool 布局、rank 策略、对象布局
        → OceanDiskStorage.__init__：库加载、底层初始化、GPU 注册
    → Linker：counter、队列、工作线程
    → registry：挂接 Wrapper 和模型侧 counter
```

两层 Python `__init__()` 都是对象初始化方法；`kvcache_init()` 则是被 Storage 调用的底层算子。初始化使用已经分配的 GPU KV buffers，不复制一份 KV cache，也不执行具体请求的加载。初始化与调用关系见 [代码修改说明](oceandisk-unified-linker-changes.md)。

### GPU 注册的具体步骤

1. 从 allocator 获取已分配的 GPU kvcache。
2. 调用 `resolve_hybrid_device_pool_group()` 获取物理 pools。
3. 遍历 `get_hybrid_pool_buffer()`，按底层 storage 的 `(ptr, nbytes)` 去重，固定顺序。
4. 构造初始化 int64 tensor：`[addr_0, ..., addr_n, size_0, ..., size_n]`，放到目标 CUDA device，调用 `kvcache_init()`。
5. 建立本地 registration table，保存每个已注册区域的 base、size 以及用于 I/O 的 MR 标识。

GIDS 与 Mooncake 的重要差别：注册不是逐块调用 register_buffer，而是通过 init 一次传地址/长度列表；I/O 使用 MR ID 与区域内偏移。

旧 controller 按注册列表顺序构造 MR ID。新实现需先验证“列表序号就是 MR ID”的约定，不能仅凭旧 Python 代码假定底层一定如此。

从 assembler 取得实际范围 `(ptr, length)` 后，查找覆盖它的注册区域：

```text
MR.base <= ptr
ptr + length <= MR.base + MR.size
local_offset = ptr - MR.base
```

注册后保持相关 GPU tensors 的引用。禁止越界范围；如果范围跨注册区域，必须拆分。注册顺序只决定本地 MR，不能影响远端对象 key。

### 4.1 MR/offset 具体做什么，为什么放在 Storage

这里有两次不同的转换：

| 层级 | 输入 | 输出 | 原因 |
| --- | --- | --- | --- |
| `OceanDiskDirectLinker`，复用 assembler | page、层、pool 和 slots | `OceanDiskRange(key, ptr, size)` | 需要理解缓存物理布局、对象身份与传输任务。 |
| `OceanDiskStorage` | `key`、GPU `ptr`、字节数 `size` | GIDS 的 MR_ID、local_offset、length 等 ABI 字段 | 需要理解本地注册表和 GIDS 的寻址约定。 |

以假设的 GPU 地址说明，初始化时注册：

```text
regions[0] = (base=10000, size=4000)   → MR 0，范围 [10000, 14000)
regions[1] = (base=20000, size=8000)   → MR 1，范围 [20000, 28000)
```

Linker 为实际 page/layer 算出 `OceanDiskRange(key=K, ptr=21500, size=256)` 后，Storage 找到完整覆盖该范围的注册区域，并转换为：

```text
key            = K
MR_ID          = 1
local_offset   = 21500 - 20000 = 1500
length         = 256
storage_offset = 0  # 示例取对象首片段；实际值来自 assembler
```

`local_offset` 相对本地 GPU 注册区域起点；`storage_offset` 相对远端 key 对象起点。当前实现将各层片段放入同一个 pool/page 对象，后者由 assembler 的布局 offsets 决定；示例若是对象首片段则为 0，其他片段可以非零，二者不能混淆。

这个步骤只转换寻址参数，不复制或搬动 KV 数据，也不在每次读写时重新注册内存。当前 `resolve_mr()` 对未注册、跨区域或非正长度范围直接报错，不自动拆分；若以后出现跨区域范围，应在调用前拆成合法片段。

放在 Storage 是因为 MR 编号和偏移格式属于 GIDS 二进制接口。Linker 只需决定“哪个 key 对应哪段 GPU 数据”；将 MR 规则留在 Storage，可以避免把后端注册细节带入缓存调度和布局逻辑。

当前代码直接以注册列表下标作为 MR_ID，与所提供 .cu 按注册数组索引寻址的方式对应，不提供其他编号模式。若底层以后返回独立 MR handle，应在 Storage 记录并转换真实 handle。

## 5. 数据布局与 GIDS I/O 描述

先复用 `resolve_transfers()` 展开物理 pools；load 允许 partial 和 missing KV，以兼容插树去重后的 side-pool-only 情况。

复用 `prepare_locations()` / `get_prepared_layer_range_meta()` 获取逐层范围，再转换为 GIDS 的列式描述符。旧 controller 展示的格式是：

```text
[task_id,
 MR_IDs...,
 block_keys...,
 local_offsets...,
 storage_offsets...,
 lengths...,
 conn_indices...]
```

当前一行对应一次算子 dispatch。同层所有请求和物理 pools 的 ranges 先汇总，再按 `max_ranges_per_call` 分块。每个分块独立构造有效长度的 tensor，不填充无效 I/O。当前直接调用 torch.ops，保留每次返回码；旧 pybind 的 length_list、buffer_index 仅作参考，不是当前接口参数。

task ID 用单调递增的整数作为任务标识；不要将请求 rid 当成远端数据 key，也不要依赖旧代码的 crc32 实现存储命名空间。

### 当前实现：pool/page 对象与 assembler offset

**布局分为两层：SGLang assembler 描述 GPU 地址及 packed 对象内偏移；OceanDisk Linker 复用这些偏移、校验布局并生成远端对象 key。这个适配没有改变 SGLang 已分配的 GPU KV 布局，也不会为了远端排列而先把 GPU 数据拼成连续 tensor。**

| 阶段 | 当前代码 | 描述的内容 |
| --- | --- | --- |
| 逻辑传输 | `PoolTransfer` | page hashes、来源 pool、当前请求或节点的 GPU slots。 |
| GPU 物理布局 | `DevicePoolGroup` / `DevicePoolEntry` | 实际 buffers、逻辑层到物理 buffer 的映射，以及 page 的 GPU 地址和大小。 |
| 远端对象布局 | `OceanDiskDirectLinker._object_layouts` | 每个物理 pool 内，各层及 fragment 的对象内 offset 和 size。 |
| 实际传输范围 | `OceanDiskRange` | 将对象 key、实际 GPU ptr、size 和远端 storage_offset 合在一起。 |

以无 MTP 的 DSA 路径为例，`_build_dsa_device_pool_group()` 为 `kvcache.kv_buffer` 和 `kvcache.index_k_with_scale_buffer` 分别生成 KV、Indexer entries。前者按 token 行寻址，后者按 page 行寻址；两者的 `indices_from_pool` 都是 KV，因此一份逻辑 KV transfer 经 `resolve_transfers()` 展开为两份物理 transfers，复用同一组 page hashes 和来源 slots。

`_object_layouts[pool][layer]` 当前是 `[(storage_offset, size), ...]` 列表。它描述一个 page 在远端的排列，与本次分配到哪些 GPU slots 无关。一个物理 fragment 是 assembler 给出的一段连续地址范围，不必等同于一个完整 tensor。详细代码对应见 [代码修改说明](oceandisk-unified-linker-changes.md)。

每个 `(page_hash, physical_pool, model/topology, page_size)` 对应一个远端 key。初始化时遍历该 pool 的全部传输层，直接保存 assembler 返回的 offsets/sizes，校验对齐、非负偏移和范围不重叠，不再按层重新累计。当前要求 packed=True，避免多个独立对象的偏移被误用到同一个 key。层号和 fragment 不单独生成 key，offsets/sizes 只用于读写和校验。

```text
一个 page hash H（DSA，无 MTP；大小仅为示意）
  KV pool → key K1
    第 0 层 KV：offset 0，size 100
    第 1 层 KV：offset 100，size 100
  Indexer pool → key K2
    第 0 层 Indexer：offset 0，size 20
    第 1 层 Indexer：offset 20，size 20
```

`OceanDiskRange(key, ptr, size, storage_offset)` 同时表达本地和远端位置。每层大小可以不同，缺少某层映射时不生成该层传输，其他片段仍保留 assembler 的原始偏移；不用模型专用固定公式。KV 和 Indexer 保持不同对象，不新增跨 pool bundle，以保留现有部分 pool 加载语义。

直接复用 assembler 的物理 buffer 偏移，包含 MTP 等非连续层映射。key 不包含布局指纹；模型权重或缓存布局变化时，应清理旧缓存或更换 namespace。不使用额外版本号或旧格式读取分支，也不宣称与 od 旧格式互通。

**完整性前提：** 同 key 的多次范围写入不等于完整提交。只有全部层和片段可读时，exist 才能报告命中。现有 Python 接口没有 commit 操作，不能自行建立该保证。若实际库首个片段写入就返回 exist=true，必须先在底层补提交/覆盖范围跟踪协议，禁止直接启用此格式。当前没有用于声明“后端已保证完整性”的配置开关；需要真实实现与测试保证。

整数 key 的有效位宽、符号及保留值需核对真实 `.so` ABI。完整 page hash 压缩到有限整数空间存在碰撞，不能宣称无碰撞；根据实际位宽选稳定摘要方案，若需要可验证的严格完整性，需额外元数据或底层更宽 key 支持。不使用 Python 内置 hash，也不将 crc32 当作内容 key。

对象命名由物理布局的复制/分片属性决定，采用与当前 Mooncake Direct Linker 相同的基础策略：

- `rank_replicated=True`：同一 attention TP 组共享对象 key，不加入 TP rank；组内 rank 0 为 offload owner，只写一份，各 rank 从相同 keys 独立读回自己的 GPU slots。
- `rank_replicated=False`：各 rank 保存不同分片，key 区分 TP rank。

当前 GLM-5 DSA pool group 将 KV + Indexer 标记为 `rank_replicated=True`，所以使用共享 key 和单 owner 写出，不采用每 rank 重复存储一份的方案。依据 assembler 提供的布局属性分支，不仅凭模型名称或 MLA 标签猜测。

读写必须使用同一稳定 conn 路由：同一个共享 key 在不同 rank 上必须路由到同一远端存储位置。不得因本地 MR、GPU 地址、请求 ID 或 rank 改变共享对象身份。实际连接若指向隔离的后端命名空间，必须先统一映射。CP/PP 及模型差异通过 key 隔离；布局变化需清理旧缓存或更换 namespace；是否支持跨 TP size 复用要单独验证，不能仅凭省略 TP rank 推断。

### GLM-5 多卡的必要约定

| 项目 | 初版要求 |
| --- | --- |
| 实例与设备 | 每个 TP rank 创建自己的 Linker/Storage，在该 rank 当前 CUDA device 上初始化 |
| GPU 注册 | 每个 rank 注册本进程的 KV 与 Indexer allocations；不把其他 rank 的 GPU 地址当成本地地址 |
| 本地 MR 表 | MR ID 及 local offset 只用于本 rank 寻址，不进入持久化对象身份 |
| namespace | GLM-5 复制型 pools 在 TP 组内共享 key；包含模型、page size、pool 和并行拓扑，不含 TP rank 或布局指纹 |
| 完整命中 | 每个候选边界都要求本 rank 的 KV 和 Indexer 必要片段完整存在 |
| 跨 rank 命中 | 复用 Wrapper 的恢复边界交集；某 rank 缺失时不能让其他 rank 单独采用更长前缀 |
| 每层 ready | 同层涉及的 KV 与 Indexer I/O 全部完成后才 complete counter |
| offload ACK | 组内 rank 0 写全部必要 pools；其他 rank 提交对应的本地成功占位结果；复用上层归约，使 owner 失败时所有 rank 都不能发布成功 |
| 后台线程 | 不新增与调度线程顺序不一致的 distributed collectives，避免多 rank 死锁 |
| GIDS 资源 | 验证多进程各自 init 的连接、MR/QP 资源和容量；不能只验证一个进程初始化成功 |

存储去重与读取去重是两回事：共享 key + 单 owner 写出可以不引入加载广播。基础方案由各 rank 分别从 GIDS 读相同数据，简单沿用现有 layer counter。以后再考虑“rank 0 读取后逐层广播”，它需要额外的加载计划一致性、设备流同步及失败协调，不是基础多卡支持的前提。

## 6. lookup：由完整 pool/page 对象存在性推导前缀

旧 `query_nds_cache()` 构造 `[task_id, keys..., conn_indices...]`，CUDA `kv_exist()` 返回 int。但“命中 block 数”不明确等于连续前缀长度。

第一版使用单 key 查询，将返回结果解释为该 key 的存在性（先验证 ABI）。这样即使批量接口只返回计数，也能构造逐对象 bitmap，不把总命中数误当连续前缀。

步骤：

1. 展开物理 pools，生成每个候选页、每个物理 pool 的对象 key。
2. 查询完整对象存在性，汇总每个 pool 的 page bitmap，不再逐层逐 fragment 查询。
3. 对 `ALL_PAGES` 计算连续前缀合法边界。
4. 后续支持 SWA 时，对 `TRAILING_PAGES` 逐候选边界检查尾部窗口。
5. 对所有必要 pools 的合法边界求交，返回当前 rank 的全部合法边界集合。

Wrapper 已负责跨 rank 求交集。

必须验证 exist 为真是否意味着整个 pool/page 对象的全部层和片段已写完，而不是仅分配元数据。如果不是，需要 GIDS 已有的提交/完成协议，或扩展二进制接口。仅有本进程内的“已写 key 集合”不能支持跨进程复用，也不能解决远端半写可见性。

## 7. load：排队、逐层读与完成回收

当前 Wrapper 先插树、COMMIT 实际采用范围，再锁节点并调用后端 load。因此 GIDS 必须使用最终传入的 slots，且不能取消已发布加载而不回滚树。

```text
load(rid, transfers)
    → 展开 partial transfers，保存 pending
start_layer_wise_loading()
    → 创建 counter index，切出批次，记录 source/metadata ready event
    → load worker
        → 按层建立 GIDS 描述符与 CPU/GPU metadata tensors
        → 同层所有请求/pools 汇总，按 IO size 分组后在专用 CUDA stream 上分块调用 load_v2
        → 最后一个分块后记录 layer event，并等待 event
        → 确认该层全部 pools 的真实 I/O 完成
        → complete(counter_index, layer)
        → 所有任务静止后发布 completed_loads[rids]
Wrapper drain_loads()
    → 消费完成批次，释放加载节点锁
```

Counter 沿用每批次每层一个 Future。worker 提交本层全部分块后记录并等待一个 event，再 complete；下一层随后提交。当前已经做同层组批，尚未实现全部层提前提交或 metadata 缓冲复用。

GPU metadata tensor、CPU task-ID tensor、output tensor 和 event 必须存活到对应任务真正结束，不能提前覆盖。不同并发任务不共用可能被写入的 output tensor。

### 完成语义是接入前提

pybind 在 op.call 后记录 CUDA event，但没有证明底层 RDMA 一定受该 stream 顺序约束。需要真实库保证：同步返回或 event 完成时，get 的 GPU 数据已可读，put 的源数据不再被读取，远端完整性可查询。

若算子只是异步提交到独立线程/设备，必须使用底层 wait/poll/drain；当前提供的 op schema 没有 wait 接口。此时仅靠 Python 包装无法保证正确性，需补最小完成接口或更换可用构建。

## 8. offload：异步写与按序 ACK

```text
offload(transfers)
    → 展开物理 pools，记录源数据 ready event，任务入队
offload worker
    → 等源数据 ready
    → 生成 keys / MR / local offset / storage offset / length / conn
    → 同层所有范围合批，分块调用 store_v2，逐层记录并等待 event
    → 等全部 I/O 结束，检查错误与可见性
    → 按提交顺序记录 bool 结果
Wrapper / Cache
    → 多 rank 完成确认，更新树状态并解锁
```

第一版使用单写线程保持完成顺序。部分片段写入失败则该 offload 返回失败，lookup 必须检查全部必要 pool 对象的完整提交状态。节点不需要在外部一次原子写完所有页，但不能把未完整的恢复边界报告为可用。

复制型 pools 按第 5 节选定组内 offload owner：只有 owner 执行 put，非 owner 为对应任务记录成功占位 ACK，仍由通用上层等待并归约 owner 的真实结果。每个 rank 都保留本地注册和 get 能力。分片型 pools 则由各 rank 执行自己的 put。

当前 pybind 丢弃 `op.call()` 的 int 返回值，未检查 cudaEventRecord 的返回码。直接单算子调用可读取同步返回值，但返回码成功含义仍需验证；即便直接调用算子，也无法检测被 `gids_kernel.cu` 吞掉的错误。当前保留 Python 返回码检查，但不能把调用无异常或返回成功作为完整持久化成功的充分证据。

## 9. 生命周期、接线与配置

- `cancel_queued_load()` 第一版返回 False，与当前已发布树状态保持一致。
- `num_completed_loads/pop_completed_load` 按批次返回请求 IDs，通知前确保不再写入 GPU。
- `reset()` 不清空远端存储：启动已有 pending loads，等待本地队列排空，清理完成通知和 counter。当前没有并发拒收新任务的门控，调用方需串行协调 reset 与任务提交。
- `close()` 在静止后退出线程、释放资源。当前 schema 无显式 deregister/close，不能虚构可用算子；确认进程退出清理约定及是否支持同进程重新初始化。
- registry 增加 `oceandisk` 分支，memory 参数 backend choices 增加 `oceandisk`。
- 当前配置支持 so/config 路径、namespace、连接列表、每次最大 ranges、对齐；初始化与读写统一按返回码 0 判断成功；尚未实现适配器级超时或容量参数。保留 OPEXTENSION/LIBGIDS/GIDS_DIR 环境变量兼容，不要求旧 nds 专用 server args。

启动参数：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend oceandisk
```

## 10. 旧代码取舍

| 参考内容 | 处理 |
| --- | --- |
| guide 的 so 加载链 | 保留，启动时验证实际构建 |
| nds_utils 的 init、CUDA exist 调用 | 抽取有效 ABI，不直接引入整个模块 |
| controller 的 MR/offset/length/conn 打包方式 | 保留字段语义，合并到 OceanDiskStorage 的内部转换与打包方法 |
| controller 的模型专用页/层地址公式 | 用 DevicePoolEntry 元数据替代 |
| controller 的事件与 stream 用法 | 参考，但需验证 RDMA 完成语义 |
| 旧 HiCache 调度、host pool、prefetch 协议 | 不迁移，复用 Unified Wrapper |
| 旧 is_kv_exist 常数返回、CUDA pass 方法 | 不使用 |

## 11. 实施与验证顺序

1. **二进制契约验证**：单 GPU buffers 注册；用非零 MR、本地偏移、不同 key 和不同 conn 做 put/get；核对 key 位宽、MR 编号和 metadata 格式。
2. **完成/失败验证**：故障时返回码与异常，event 是否覆盖真实 I/O，exist 对半写是否误命中，取消/close 后是否仍有 DMA。
3. **地址转换与参数打包验证**：独立验证 OceanDiskStorage 的转换方法，用已知 tensor 内容验证每个 range 指向正确 page/layer/fragment，无越界或漏传；不要求为验证单独拆出 Planner 类。
4. **小规模底层闭环**：使用可运行的小模型或合成 GPU buffers，验证保存、驱逐和恢复；这仅是调试，不要求完整 GLM-5 能单卡装载。
5. **GLM-5 多卡 TP 必测**：在目标 TP 配置启动真实 GLM-5，各 rank 注册 KV + Indexer；写入→等待上层完成确认→驱逐本地前缀→相同前缀触发真实远端恢复→对比重算输出/logprobs。必须确认各 rank 及两个必要 pools 均实际参与恢复。
6. **多卡跨进程重启复用**：producer TP 集群写入后，重启相同 TP 配置的 consumer，各 rank 在不同本地 slots 恢复；验证 key 不依赖请求 ID、MR 顺序、GPU 地址或本地已写集合。
7. **跨 rank 缺失与故障**：删除共享 Indexer/KV 片段验证命中收缩；注入某 rank 的查询故障（区分真实错误与正常 miss）及 owner 写入失败，验证不会误报全局成功；注入加载故障，确认不误报 ready 且无未结束 DMA 导致槽位污染。
8. **GQA（Qwen2.5） 模型覆盖**：确认具体模型后，核对其树组件、全部物理 pools、状态、复制/分片属性和层映射；必要时补 assembler 策略，并完成该模型的多卡远端恢复、结果一致性和故障验证。该项属于方案要求，不以 GLM-5 通过替代。
9. **进一步优化和组合**：在目标模型正确性覆盖后，再考虑 MTP、rank-0 读取后广播、其他并行组合及查询/组批优化。模型本身必需的 SWA 或其他状态不属于可省略的优化项。

可以先完成 Python 设计与纯元数据部分，但在缺少真实二进制和完成/可见性契约验证时，不能宣称该后端已经可正确运行。若契约满足，核心接入可保持为新增 Python backend 和参数/registry 接线；否则需先补 `.so` 的最小状态接口。


## 12. od 更新后的适配边界

- 新 controller 将 KV、DSA 分开管理 MR、地址和每层 item 长度。新 Linker 已通过 DevicePoolGroup/Entry 描述这些物理 pools；不迁移旧 controller 的模型专用索引公式、PD 通知或额外缓存类型枚举。
- 新 controller 对部分长度做 512 字节向上取整，并把 DSA 注册长度人为加 512。新适配保留真实 allocation 边界，禁止照搬超额注册或跨 page 读取。`io_alignment` 默认 512，初始化校验实际 fragment 大小，I/O 校验 size 和远端 offset；不满足时明确报错，当前没有 padded staging。只有真实库验证支持字节粒度时才可配置为 1。
- `gids_kernel.cu` 的 USE_SINGLE_PORT 未设置或等于 1 时选择 gids_config.txt，其他值选择 gids_config_<当前可见CUDA设备号>.txt。Storage 按同一规则预检文件，可用 use_single_port 布尔配置显式设置环境变量，并规范 GIDS_DIR 末尾分隔符。
- Storage 预检六字段 ip_config；connection_indices 必须在文件实际连接数内且为 0..7，单次 ranges 不超过 409600。第六字段按 .cu 传给 qp_cnt，不照搬旧 Python 将它用作 dpu_id 的分组逻辑。当前不引入一致性 hash ring，仍使用显式连接列表的稳定 key 路由。
- .cu 中的 g_devid、g_kvnsid、g_kvnsid_index 是固定值；Python namespace 只隔离对象 key，不能修改这些原生 namespace。部署时需匹配实际后端配置；GIDS_MR_NUM、MAX_IP_NUM 的定义还在未提供的头文件中。
- 不照搬 `time.sleep(device_id * 2)`：它是旧启动路径的错峰措施，不是同步或连接成功保证。若实际服务有初始化并发限制，应根据联调结果处理。
- 接口名称、六列元数据和逐层 stream/event 方案保持不变。HiCache/PD/NPU/GQA 不因 od 更新自动接通；不宣称与旧 controller 的整数 page hash、远端布局或存量对象互通。


## 离线单实例验证入口

[Bash 启动脚本](examples/runtime/engine/run_oceandisk_offline_tp8.sh) 调用 [Python 推理脚本](examples/runtime/engine/oceandisk_offline_tp8.py)，通过 MODEL_PATH 指定本地模型，同一实例同时运行 prefill/decode，默认 TP=8。

本地 prefix cache 默认开启；Bash 的 base 模式执行普通推理，oceandisk 模式自动追加外部 Linker 参数，并导出 Bash 中配置的 OPEXTENSION、LIBGIDS、GIDS_DIR，不使用外部 JSON 文件。两种模式共用模型和八卡参数，每次运行创建新的 Engine。Python 读取用户 JSONL 第一条样本并构造相同 prompt：base 测试首次预热及 HBM 复用，oceandisk 在首次预热后直接清空本地缓存，第二次只验证外部读回，并要求 device 命中为零。完整启动命令见 [代码修改说明](oceandisk-unified-linker-changes.md)。


## GIDS 网络配置与读取长度约束

按更新的 od/readme.md，连接配置需要本机 RoCE IP、存储服务端 IP、服务端口、GID index、RoCE 端口及每个 IP 的 QP 数量。离线 Bash 脚本顶部留有填写位置，oceandisk 模式生成 GIDS_DIR/gids_config.txt，并设置 USE_SINGLE_PORT=1；base 模式不使用这些配置。第六个字段以当前 .cu 实现为准，通过 GIDS_QP_COUNT 配置，对应 qp_num_per_ip → qp_cnt，详见修改说明第 16 节。

Get 批次按实际 range.size 分组，满足 README 的同批读取等长要求；各组完成后统一确认一层就绪。Put 可混合长度，沿用逐层分批写入。此项属于 Storage 的底层传输适配，Python 问答流程不需要修改。
