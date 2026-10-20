"""CPU tests for OceanDisk layout conversion and backend completion contracts."""

import inspect
import itertools
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker import (
    LayerWiseLoadCounter,
    OceanDiskDirectLinker,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
    OceanDiskRange,
    OceanDiskStorage,
    encode_key,
    gids_config_path,
    pack_io,
    read_connection_count,
    read_io_alignment,
    resolve_mr,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestOceanDiskStorage(CustomTestCase):
    def test_hicache_contract_is_reserved_without_changing_uc_range_io(self):
        self.assertTrue(issubclass(OceanDiskStorage, HiCacheStorage))
        self.assertFalse(inspect.isabstract(OceanDiskStorage))
        storage = OceanDiskStorage.__new__(OceanDiskStorage)
        for name in (
            "get",
            "batch_get",
            "set",
            "batch_set",
            "exists",
            "batch_get_v1",
            "batch_set_v1",
            "register_mem_pool_host",
            "register_mem_host_pool_v2",
            "clear",
        ):
            self.assertEqual(
                inspect.signature(getattr(OceanDiskStorage, name)),
                inspect.signature(getattr(HiCacheStorage, name)).replace(
                    return_annotation=inspect.Signature.empty,
                    parameters=[
                        parameter.replace(annotation=inspect.Parameter.empty)
                        for parameter in inspect.signature(
                            getattr(HiCacheStorage, name)
                        ).parameters.values()
                    ],
                ),
            )
        operations = [
            lambda: storage.get("key"),
            lambda: storage.batch_get(["key"]),
            lambda: storage.set("key"),
            lambda: storage.batch_set(["key"]),
            lambda: storage.exists("key"),
            lambda: storage.batch_exists(["key"]),
            lambda: storage.batch_get_v1(["key"], None),
            lambda: storage.batch_set_v1(["key"], None),
            lambda: storage.register_mem_pool_host(None),
            lambda: storage.register_mem_host_pool_v2(None, PoolName.KV),
            lambda: storage.clear(),
            lambda: storage.batch_get_v2([]),
            lambda: storage.batch_set_v2([]),
            lambda: storage.batch_exists_v2([]),
        ]
        for operation in operations:
            with self.assertRaises(NotImplementedError):
                operation()

    def test_native_config_selection_and_connection_limits(self):
        directory = Path("config")
        self.assertEqual(gids_config_path(directory, 3), directory / "gids_config.txt")
        self.assertEqual(
            gids_config_path(directory, 3, "1"), directory / "gids_config.txt"
        )
        self.assertEqual(
            gids_config_path(directory, 3, "0"), directory / "gids_config_3.txt"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gids_config.txt"
            line = "ip_config = 10.0.0.1, 10.0.0.2, 1234, 3, 1, 8\n"
            path.write_text(line * 2)
            self.assertEqual(read_connection_count(path), 2)
            for text in (
                "",
                line * 9,
                line.replace(", 3,", ", 256,"),
                "ip_config = bad",
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_connection_count(path)
        self.assertEqual(read_io_alignment({}), 512)
        self.assertEqual(read_io_alignment({"io_alignment": 1}), 1)
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                read_io_alignment({"io_alignment": value})
        for connections in ([8], [-1], [True], [0, 0]):
            with self.assertRaisesRegex(ValueError, "connection_indices"):
                OceanDiskStorage(
                    {"connection_indices": connections},
                    "cuda:0",
                    [],
                )

    def test_layer_event_waits_after_all_chunks_and_preserves_status(self):
        storage = OceanDiskStorage.__new__(OceanDiskStorage)
        storage._lock = threading.Lock()
        storage._closed = False
        storage.device = "cuda:0"
        storage.regions = [(1000, 256)]
        storage.connection_indices = (0,)
        storage._task_ids = itertools.count(1)
        storage.io_alignment = 1  # Synthetic tiny ranges, not the deployed ABI.
        trace = []
        fake_torch = Mock()
        fake_torch.cuda.device.side_effect = lambda *_: nullcontext()
        fake_torch.cuda.stream.side_effect = lambda *_: nullcontext()
        fake_torch.tensor.side_effect = lambda values, **_: values
        event = Mock()
        event.record.side_effect = lambda _: trace.append("record")
        event.synchronize.side_effect = lambda: trace.append("wait")
        fake_torch.cuda.Event.return_value = event
        storage._torch = fake_torch
        storage._stream = Mock()
        storage._output = Mock()
        storage._ops = Mock()

        def load(*_):
            trace.append("submit")
            return 0

        storage._ops.kvcache_load_v2.side_effect = load
        ranges = [OceanDiskRange(7, 1000, 8, offset) for offset in (0, 8, 16)]
        mr_lookup = Mock(wraps=resolve_mr)
        with patch.dict(pack_io.__globals__, resolve_mr=mr_lookup):
            storage.get_ranges(ranges, max_ranges=2)
        self.assertEqual(mr_lookup.call_count, len(ranges))
        self.assertEqual(trace, ["submit", "submit", "record", "wait"])
        storage._stream.synchronize.assert_not_called()
        trace.clear()
        batches = []

        def capture_sizes(metadata, *_):
            count = (len(metadata) - 1) // 6
            batches.append(metadata[1 + 4 * count : 1 + 5 * count])
            trace.append("submit")
            return 0

        mixed = [OceanDiskRange(7, 1000, size, 0) for size in (16, 8, 16, 8, 8)]
        storage._ops.kvcache_load_v2.side_effect = capture_sizes
        storage.get_ranges(mixed, max_ranges=2)
        self.assertEqual(batches, [[8, 8], [8], [16, 16]])
        self.assertEqual(trace, ["submit"] * 3 + ["record", "wait"])
        batches.clear()
        trace.clear()
        storage._ops.kvcache_store_v2.side_effect = capture_sizes
        storage.put_ranges(mixed, max_ranges=2)
        self.assertEqual(batches, [[16, 8], [16, 8], [8]])
        self.assertEqual(trace, ["submit"] * 3 + ["record", "wait"])
        storage._ops.kvcache_load_v2.side_effect = load
        trace.clear()
        for operation in (storage.get_ranges, storage.put_ranges):
            for invalid in (
                OceanDiskRange(7, 2000, 8, 16),
                OceanDiskRange(7, 1000, 8, -1),
            ):
                with self.assertRaises(ValueError):
                    operation([*ranges[:2], invalid], max_ranges=2)
                self.assertEqual(trace, [])

        linker = OceanDiskDirectLinker.__new__(OceanDiskDirectLinker)
        linker.pools = {PoolName.KV: Mock()}
        linker.pools[PoolName.KV].prepare_locations.return_value = [0]
        linker.num_layers = 2
        linker.max_ranges = 2
        linker.storage = storage
        linker._page_fragments = lambda *args: ranges
        linker.layer_done_counter = Mock()
        linker.layer_done_counter.complete.side_effect = lambda *args: trace.append(
            "ready"
        )
        trace.clear()
        linker._run_transfers(
            [SimpleNamespace(name=PoolName.KV, host_indices=[0], keys=["p0"])],
            write=False,
            counter_index=0,
        )
        self.assertEqual(trace, ["submit", "submit", "record", "wait", "ready"] * 2)
        storage.io_alignment = 512
        trace.clear()
        with self.assertRaisesRegex(ValueError, "align"):
            storage.get_ranges(ranges, max_ranges=2)
        self.assertEqual(trace, [])
        storage.io_alignment = 1
        storage._ops.kvcache_load_v2.side_effect = lambda *_: -1
        with self.assertRaisesRegex(RuntimeError, "status"):
            storage.get_ranges(ranges, max_ranges=2)
        storage._stream.synchronize.assert_called_once()

    def test_descriptor_uses_registration_relative_offsets_and_stable_routing(self):
        regions = [(1000, 128), (5000, 256)]
        ranges = [OceanDiskRange(7, 1032, 16, 64), OceanDiskRange(8, 5040, 24, 128)]
        self.assertEqual(
            pack_io(11, ranges, regions, (2, 5)),
            [11, 0, 1, 7, 8, 32, 40, 64, 128, 16, 24, 5, 2],
        )
        for ptr, size in ((999, 1), (1120, 16), (1000, 0)):
            with self.assertRaises(ValueError):
                resolve_mr(regions, ptr, size)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            pack_io(1, [OceanDiskRange(7, 1032, 16, -1)], regions, (0,))

    def test_key_and_native_status_codes(self):
        self.assertEqual(encode_key(("model", "page")), encode_key(("model", "page")))
        self.assertNotEqual(encode_key(("kv", "page")), encode_key(("indexer", "page")))
        self.assertTrue(0 < encode_key(("page",)) < 2**63)
        for operation in ("init", "get", "put"):
            OceanDiskStorage._check_status(0, operation)
            with self.assertRaisesRegex(RuntimeError, operation):
                OceanDiskStorage._check_status(-1, operation)


class TestOceanDiskDirectLinker(CustomTestCase):
    def make_linker(
        self, *, packed=True, independent_side_pool=False, multi_component=False
    ):
        linker = OceanDiskDirectLinker.__new__(OceanDiskDirectLinker)
        entries = []
        names = (
            PoolName.KV,
            PoolName.SWA if independent_side_pool else PoolName.INDEXER,
        )
        for name in names:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=name if independent_side_pool else PoolName.KV,
                    device_pool=None,
                    # DSA exposes KV and Indexer as separate single-component pools.
                    components=(
                        [
                            [
                                torch.zeros((16, width), dtype=torch.uint8)
                                for _ in range(2)
                            ]
                            for width in (3, 5)
                        ]
                        if multi_component
                        else [
                            [
                                torch.zeros(
                                    (16, 3 if name == PoolName.KV else 5),
                                    dtype=torch.uint8,
                                )
                                for _ in range(2)
                            ]
                        ]
                    ),
                    layer_mapping={0: 0, 1: 1},
                    page_size=2,
                    rows_are_pages=False,
                    packed=packed,
                )
            )
        linker.pool_group = DevicePoolGroup(entries, 2, 2, rank_replicated=True)
        linker.pools = linker.pool_group.entry_map
        linker.num_layers = 2
        linker.io_alignment = 1  # Synthetic layout with 6/10-byte fragments.
        linker._namespace = ("sglang-oceandisk-pool-pages", "test-model", 2, 1)
        linker._build_object_layouts()
        linker._closed = False
        linker.storage = Mock()
        linker.max_ranges = 3
        linker.layer_done_counter = LayerWiseLoadCounter(2)
        linker.pending_loads = {}
        linker.offload_results = Queue()
        return linker

    def test_unaligned_layout_fails_before_gpu_registration(self):
        linker = self.make_linker()
        linker.io_alignment = 512
        with self.assertRaisesRegex(ValueError, "staging/packing"):
            linker._build_object_layouts()

    def test_fragment_keys_do_not_depend_on_gpu_slots(self):
        linker = self.make_linker()
        first = linker._page_fragments(PoolName.KV, ["p0", "p1"], [0, 2], 0)
        second = linker._page_fragments(PoolName.KV, ["p0", "p1"], [4, 6], 0)
        self.assertEqual([r.key for r in first], [r.key for r in second])
        self.assertNotEqual([r.ptr for r in first], [r.ptr for r in second])
        self.assertEqual([r.size for r in first], [6, 6])
        other_pool = linker._page_fragments(PoolName.INDEXER, ["p0"], [0], 0)
        other_layer = linker._page_fragments(PoolName.KV, ["p0"], [0], 1)
        self.assertNotEqual(first[0].key, other_pool[0].key)
        self.assertEqual(first[0].key, other_layer[0].key)
        self.assertEqual([r.storage_offset for r in first], [0, 0])
        self.assertEqual([r.storage_offset for r in other_layer], [6])

    def test_offsets_come_from_assembler_for_multiple_components(self):
        linker = self.make_linker(multi_component=True)
        first = linker._page_fragments(PoolName.KV, ["p0"], [0], 0)
        second = linker._page_fragments(PoolName.KV, ["p0"], [0], 1)
        # Component-major offsets differ from the old layer-major packing.
        self.assertEqual([r.storage_offset for r in first], [0, 12])
        self.assertEqual([r.storage_offset for r in second], [6, 22])

    def test_keys_use_namespace_pool_and_page_without_layout_fingerprinting(self):
        linker = self.make_linker()
        other_layout = self.make_linker(multi_component=True)
        key = linker._object_key(PoolName.KV, "p0")
        self.assertEqual(key, other_layout._object_key(PoolName.KV, "p0"))
        linker._build_object_layouts()
        self.assertEqual(key, linker._object_key(PoolName.KV, "p0"))
        self.assertNotEqual(key, linker._object_key(PoolName.INDEXER, "p0"))
        self.assertNotEqual(key, linker._object_key(PoolName.KV, "p1"))
        linker._namespace = (*linker._namespace, "other-model")
        self.assertNotEqual(key, linker._object_key(PoolName.KV, "p0"))

    def test_unpacked_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "packed=True"):
            self.make_linker(packed=False)

    def test_draft_mapping_uses_physical_buffer_offsets(self):
        linker = self.make_linker()
        entry = linker.pools[PoolName.KV]
        linker.pools[PoolName.KV] = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[
                entry.components[0] + [torch.zeros((16, 3), dtype=torch.uint8)]
            ],
            layer_mapping={0: (0, 2), 1: 1},
            page_size=2,
            rows_are_pages=False,
        )
        linker._build_object_layouts()
        first = linker._page_fragments(PoolName.KV, ["p0"], [0], 0)
        second = linker._page_fragments(PoolName.KV, ["p0"], [0], 1)
        self.assertEqual([r.storage_offset for r in first], [0, 12])
        self.assertEqual([r.storage_offset for r in second], [6])

    def test_changed_or_overlapping_offsets_are_rejected(self):
        linker = self.make_linker()
        entry = linker.pools[PoolName.KV]
        original = entry.get_prepared_layer_range_meta

        def changed(locations, layer):
            ptrs, sizes, offsets = original(locations, layer)
            return ptrs, sizes, [[offset + 1 for offset in group] for group in offsets]

        entry.get_prepared_layer_range_meta = changed
        with self.assertRaisesRegex(ValueError, "registered object layout"):
            linker._page_fragments(PoolName.KV, ["p0"], [0], 0)
        entry.get_prepared_layer_range_meta = original
        entry.layer_mapping = {0: 0, 1: 0}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            linker._build_object_layouts()

    def test_layer_batches_all_pools_before_counter_completion(self):
        linker = self.make_linker()
        transfers = linker.pool_group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV, keys=["p0"], device_indices=torch.tensor([2, 3])
                )
            ]
        )
        calls = []
        linker.storage.get_ranges.side_effect = lambda ranges, **kw: calls.append(
            ("get", len(ranges), [r.storage_offset for r in ranges], kw)
        )
        linker.layer_done_counter.complete = lambda index, layer: calls.append(
            ("complete", index, layer)
        )
        linker._run_transfers(transfers, write=False, counter_index=7)
        self.assertEqual([c[0] for c in calls], ["get", "complete", "get", "complete"])
        self.assertEqual(calls[0][1], 2)
        self.assertEqual(calls[0][3], {"max_ranges": 3})
        self.assertEqual(calls[2][2], [6, 10])

    def test_lookup_requires_indexer_and_checks_holes(self):
        linker = self.make_linker()
        missing = linker._page_fragments(PoolName.INDEXER, ["p1"], [0], 1)[0].key
        linker.storage.exists_many.side_effect = lambda keys: [
            k != missing for k in keys
        ]
        transfers = [PoolTransfer(name=PoolName.KV, keys=["p0", "p1", "p2"])]
        self.assertEqual(linker.lookup("req", transfers), [1])
        linker.storage.exists_many.side_effect = lambda keys: [True] * len(keys)
        self.assertEqual(linker.lookup("req", transfers), [1, 2, 3])

    def test_window_policy_returns_sparse_boundaries(self):
        linker = self.make_linker()
        # Exercise trailing policy without relying on a particular model's SWA pool.
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=["p0", "p1", "p2", "p3"]),
            PoolTransfer(
                name=PoolName.INDEXER,
                keys=["p2", "p3"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        missing = linker._page_fragments(PoolName.INDEXER, ["p1"], [0], 0)[0].key
        linker.storage.exists_many.side_effect = lambda keys: [
            k != missing for k in keys
        ]
        # Preserve test-supplied policy rather than KV-derived sidecar ALL_PAGES.
        linker.pool_group.resolve_transfers = lambda value: value
        self.assertEqual(linker.lookup("req", transfers), [1, 4])

    def test_owner_ack_and_partial_load(self):
        linker = self.make_linker(independent_side_pool=True)
        transfer = PoolTransfer(
            name=PoolName.SWA, keys=["p0"], device_indices=torch.tensor([2, 3])
        )
        self.assertTrue(linker.load("req", [transfer]))
        self.assertEqual(linker.pending_loads["req"][0].name, PoolName.SWA)
        with self.assertRaisesRegex(RuntimeError, "already queued"):
            linker.load("req", [transfer])
        self.assertFalse(linker.cancel_queued_load("req"))
        linker.offload_owner = False
        linker._ready_event = Mock(side_effect=AssertionError("non-owner wrote"))
        self.assertTrue(
            linker.offload(
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["p0"],
                        device_indices=torch.tensor([2, 3]),
                    )
                ]
            )
        )
        self.assertTrue(linker.pop_completed_offload())

    def test_dsa_indexer_is_derived_from_logical_kv(self):
        linker = self.make_linker()
        self.assertFalse(
            linker.load("req", [PoolTransfer(name=PoolName.INDEXER, keys=["p0"])])
        )

    def test_ready_event_failure_preserves_pending_requests(self):
        linker = self.make_linker()
        linker.pending_loads = {"req": []}
        linker._ready_event = Mock(side_effect=RuntimeError("event failed"))
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            linker.start_layer_wise_loading()
        self.assertEqual(linker.pending_loads, {"req": []})
        self.assertEqual(linker.layer_done_counter.producer_index, -1)

    def test_page_key_location_mismatch_is_rejected(self):
        linker = self.make_linker()
        with self.assertRaisesRegex(ValueError, "equal length"):
            linker._page_fragments(PoolName.KV, ["p0", "p1"], [0], 0)

    def test_load_failure_resolves_counter_and_releases_completion(self):
        linker = self.make_linker()
        linker.device = "cpu"
        linker.load_queue = Queue()
        linker.completed_loads = Queue()
        index = linker.layer_done_counter.update_producer()
        linker.layer_done_counter.set_consumer(index)
        linker._run_transfers = Mock(side_effect=RuntimeError("injected get failure"))
        event = Mock()
        linker.load_queue.put((index, {"req": []}, event))
        linker.load_queue.put(None)
        with patch("torch.cuda.device", return_value=nullcontext()):
            linker._load_worker()
        self.assertEqual(linker.pop_completed_load(), ["req"])
        self.assertEqual(linker.load_queue.unfinished_tasks, 0)
        with self.assertRaisesRegex(RuntimeError, "OceanDisk layer-wise"):
            linker.layer_done_counter.wait_until(0)


if __name__ == "__main__":
    unittest.main()
