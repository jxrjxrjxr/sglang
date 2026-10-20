# SPDX-License-Identifier: Apache-2.0
"""Direct GPU prefix-cache storage using the GIDS CUDA operators."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from queue import Empty, Queue

import torch
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
    MAX_GIDS_RANGES,
    OceanDiskRange,
    OceanDiskStorage,
    encode_key,
    read_io_alignment,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_memory, get_model

logger = logging.getLogger(__name__)


class LayerWiseLoadCounter:
    """Resolve layers only after every physical pool in that layer is readable."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("OceanDisk layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.futures.clear()


class OceanDiskDirectLinker(UnifiedCacheLinker):
    def __init__(self, server_args, params, *, components, storage=None):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if self.num_layers <= 0:
            raise ValueError("OceanDisk requires at least one transfer layer.")
        buffers = [
            buffer
            for entry in self.pools.values()
            for buffer in entry.get_hybrid_pool_buffer()
        ]
        if not buffers:
            raise ValueError("OceanDisk has no device buffers.")
        self.device = buffers[0].device
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        tp_rank, tp_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        replicated = self.pool_group.rank_replicated
        self.offload_owner = not replicated or tp_rank == 0
        config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        self.max_ranges = config.get("max_ranges_per_call", 256)
        if (
            type(self.max_ranges) is not int
            or not 0 < self.max_ranges <= MAX_GIDS_RANGES
        ):
            raise ValueError(
                f"OceanDisk max_ranges_per_call must be in [1, {MAX_GIDS_RANGES}]."
            )
        self.io_alignment = read_io_alignment(config)
        namespace = config.get("namespace", get_model().model_path)
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(
                "OceanDisk namespace must be a nonempty model/version identifier."
            )
        self._namespace = (
            "sglang-oceandisk-pool-pages",
            namespace,
            params.page_size,
            tp_size,
            None if replicated else tp_rank,
            params.attn_cp_size,
            params.attn_cp_rank,
            params.pp_size,
            params.pp_rank,
        )
        self._build_object_layouts()
        self.storage = (
            OceanDiskStorage(config, self.device, buffers)
            if storage is None
            else storage
        )
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.load_queue = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        self.offload_queue = Queue()
        self.offload_results: Queue[bool] = Queue()
        self._closed = False
        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name=f"oceandisk-load-tp{tp_rank}"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker,
            daemon=True,
            name=f"oceandisk-offload-tp{tp_rank}",
        )
        self._load_thread.start()
        self._offload_thread.start()
        logger.info(
            "OceanDisk direct linker: tp=%d/%d replicated=%s owner=%s pools=%s",
            tp_rank,
            tp_size,
            replicated,
            self.offload_owner,
            list(self.pools),
        )

    def _build_object_layouts(self) -> None:
        """Save and validate assembler offsets for one page/pool object."""
        self._object_layouts = {}
        for name, entry in self.pools.items():
            if not entry.packed:
                raise ValueError(
                    f"OceanDisk pool {name} requires packed=True for one page/pool object."
                )
            layout = {}
            intervals = []
            for layer in range(self.num_layers):
                meta = entry.get_prepared_layer_range_meta([0], layer)
                if meta is None:
                    continue
                _, sizes, offsets = meta
                fragments = []
                for lengths, positions in zip(sizes, offsets, strict=True):
                    for size, offset in zip(lengths, positions, strict=True):
                        if size <= 0:
                            raise ValueError(
                                "OceanDisk object fragments must be nonempty."
                            )
                        if offset < 0:
                            raise ValueError(
                                "OceanDisk storage offsets must be nonnegative."
                            )
                        if size % self.io_alignment or offset % self.io_alignment:
                            raise ValueError(
                                f"OceanDisk pool {name}, layer {layer} has {size}-byte fragment "
                                f"at offset {offset}, not aligned to io_alignment={self.io_alignment}. "
                                "A padded staging/packing path is required."
                            )
                        fragments.append((int(offset), int(size)))
                        intervals.append((int(offset), int(offset + size)))
                layout[layer] = fragments
            if not intervals:
                raise ValueError(f"OceanDisk pool {name} has no object payload.")
            intervals.sort()
            if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
                raise ValueError(
                    f"OceanDisk pool {name} has overlapping object ranges."
                )
            self._object_layouts[name] = layout

    def _object_key(self, name, page_key) -> int:
        return encode_key((self._namespace, str(name), page_key))

    def _page_fragments(
        self, name, page_keys, locations, layer
    ) -> list[OceanDiskRange]:
        """Map a pool/page object's layer to local ranges and remote offsets."""
        if len(locations) != len(page_keys):
            raise ValueError(
                "OceanDisk page keys and locations must have equal length."
            )
        entry = self.pools[name]
        meta = entry.get_prepared_layer_range_meta(locations, layer)
        if meta is None:
            if self._object_layouts[name].get(layer):
                raise ValueError("OceanDisk required layer has no physical ranges.")
            return []
        if not page_keys:
            return []
        ptrs, sizes, offsets = meta
        if len(ptrs) != len(page_keys):
            raise ValueError("OceanDisk pool ranges do not align with page keys.")
        ranges = []
        schema = self._object_layouts[name].get(layer, ())
        for page_key, pointers, lengths, positions in zip(
            page_keys, ptrs, sizes, offsets, strict=True
        ):
            if list(zip(positions, lengths, strict=True)) != schema:
                raise ValueError(
                    "OceanDisk range differs from registered object layout."
                )
            key = self._object_key(name, page_key)
            ranges.extend(
                OceanDiskRange(key, int(ptr), int(size), int(storage_offset))
                for ptr, size, storage_offset in zip(
                    pointers, lengths, positions, strict=True
                )
            )
        return ranges

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(transfers)
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        if not expanded or kv is None or not kv.keys:
            return []
        page_keys = list(kv.keys)
        valid = set(range(1, len(page_keys) + 1))
        for transfer in expanded:
            bitmap = self.storage.exists_many(
                [self._object_key(transfer.name, page_key) for page_key in page_keys]
            )
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                end = next((i for i, hit in enumerate(bitmap) if not hit), len(bitmap))
                valid.intersection_update(range(1, end + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                window = max(1, len(transfer.keys or ()))
                valid = {
                    end for end in valid if all(bitmap[max(0, end - window) : end])
                }
            else:
                raise ValueError(
                    f"Unsupported OceanDisk hit policy: {transfer.hit_policy}"
                )
            if not valid:
                break
        return sorted(valid)

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"OceanDisk load already queued for {rid!r}.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Wrapper has already published the tree; cancellation requires rollback.
        return False

    def _ready_event(self):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def start_layer_wise_loading(self) -> int:
        self._ensure_open()
        if not self.pending_loads:
            return -1
        # Preserve published requests if event creation/recording fails.
        event = self._ready_event()
        index = self.layer_done_counter.update_producer()
        pending, self.pending_loads = self.pending_loads, {}
        self.load_queue.put((index, pending, event))
        return index

    def _run_transfers(self, transfers, *, write: bool, counter_index=None) -> None:
        prepared = [
            (
                transfer,
                self.pools[transfer.name].prepare_locations(transfer.host_indices),
            )
            for transfer in transfers
        ]
        for layer in range(self.num_layers):
            ranges = []
            for transfer, locations in prepared:
                ranges.extend(
                    self._page_fragments(
                        transfer.name, list(transfer.keys), locations, layer
                    )
                )
            if write:
                self.storage.put_ranges(ranges, max_ranges=self.max_ranges)
            else:
                self.storage.get_ranges(ranges, max_ranges=self.max_ranges)
            if counter_index is not None:
                self.layer_done_counter.complete(counter_index, layer)

    def _load_worker(self) -> None:
        with torch.cuda.device(self.device):
            while True:
                task = self.load_queue.get()
                try:
                    if task is None:
                        return
                    index, pending, event = task
                    try:
                        event.synchronize()
                        self._run_transfers(
                            [t for transfers in pending.values() for t in transfers],
                            write=False,
                            counter_index=index,
                        )
                    except BaseException as error:
                        self.layer_done_counter.fail(index, error)
                        logger.exception("OceanDisk load batch failed")
                    finally:
                        self.completed_loads.put(list(pending))
                finally:
                    self.load_queue.task_done()

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self.completed_loads.get_nowait()

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        if not self.offload_owner:
            self.offload_results.put(True)
        else:
            self.offload_queue.put((expanded, self._ready_event()))
        return True

    def _offload_worker(self) -> None:
        with torch.cuda.device(self.device):
            while True:
                task = self.offload_queue.get()
                try:
                    if task is None:
                        return
                    transfers, event = task
                    success = False
                    try:
                        event.synchronize()
                        self._run_transfers(transfers, write=True)
                        success = True
                    except BaseException:
                        logger.exception("OceanDisk offload failed")
                    self.offload_results.put(success)
                finally:
                    self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OceanDisk linker is closed.")

    def reset(self) -> None:
        # Even queued loads target published nodes; drain them before tree reset.
        if self.pending_loads:
            self.start_layer_wise_loading()
        self.load_queue.join()
        self.offload_queue.join()
        for results in (self.completed_loads, self.offload_results):
            while True:
                try:
                    results.get_nowait()
                except Empty:
                    break
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        self.storage.close()
