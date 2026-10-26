"""CPU tests for OceanDisk layout conversion and backend completion contracts."""

import gc
import inspect
import itertools
import tempfile
import threading
import unittest
from array import array
from contextlib import nullcontext
from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch
from weakref import WeakValueDictionary

import torch
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker import (
    OceanDiskBlockOffloader as OceanDiskDirectLinker,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
    OceanDiskRange,
    OceanDiskStorage,
    encode_key,
    gids_config_path,
    pack_io,
    read_connection_count,
    resolve_mr,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)

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
        for connections in ([8], [-1], [True], [0, 0]):
            with self.assertRaisesRegex(ValueError, "connection_indices"):
                OceanDiskStorage(
                    {"connection_indices": connections},
                    "cuda:0",
                    [],
                )

    def make_storage(self):
        storage = OceanDiskStorage.__new__(OceanDiskStorage)
        storage._lock = threading.Lock()
        storage._closed = False
        storage.device = SimpleNamespace(index=0)
        storage.regions = [(1000, 4096), (10000, 4096)]
        storage.connection_indices = (2, 5)
        storage._task_ids = itertools.count(1)
        trace = []

        class Metadata:
            def __init__(self, values):
                self.values = values
                self.shape = (
                    (len(values), len(values[0]))
                    if isinstance(values[0], list)
                    else (len(values),)
                )

            def to(self, _device):
                return self

            def __getitem__(self, selection):
                return type(self)(self.values[selection])

        fake_torch = Mock()
        fake_torch.cuda.device.side_effect = lambda *_: nullcontext()
        fake_torch.cuda.stream.side_effect = lambda *_: nullcontext()
        fake_torch.tensor.side_effect = lambda values, **_: Metadata(values)
        events = []

        def new_event():
            event = Mock()
            event.cuda_event = len(events) + 1
            event.record.side_effect = lambda *_: trace.append(
                ("record", event.cuda_event)
            )
            event.synchronize.side_effect = lambda: trace.append(
                ("wait", event.cuda_event)
            )
            events.append(event)
            return event

        fake_torch.cuda.Event.side_effect = new_event
        storage._torch = fake_torch
        storage._stream = Mock(stream_id=123)
        storage._output = Mock()
        storage._ops = Mock()
        self.od_utils = SimpleNamespace(
            batch_dispatch_get=Mock(
                side_effect=lambda *args: trace.append(("dispatch", args[0].values))
            ),
            batch_dispatch_put=Mock(
                side_effect=lambda *args: trace.append(("put", args[0].values))
            ),
        )
        modules = patch.dict("sys.modules", {"od_utils": self.od_utils})
        modules.start()
        self.addCleanup(modules.stop)
        return storage, trace, events

    @staticmethod
    def layer_ranges(pages, layer=0):
        return [
            OceanDiskRange(page + 1, 1000 + layer * 256 + page * 16, 16, layer * 512)
            for page in range(pages)
        ] + [
            OceanDiskRange(page + 1, 10000 + layer * 256 + page * 8, 8, layer * 256)
            for page in range(pages)
        ]

    def test_put_submits_all_layers_and_preserves_pairs_when_chunked(self):
        storage, trace, _ = self.make_storage()
        layers = [self.layer_ranges(3, layer) for layer in range(2)]
        storage.put_layers(layers, max_ranges=4)
        calls = self.od_utils.batch_dispatch_put.call_args_list
        self.assertEqual([c.args[0].shape for c in calls], [(2, 25), (2, 13)])
        for call in calls:
            args = call.args
            width = args[0].shape[1]
            self.assertEqual(args[1].values, [width, width])
            self.assertEqual(args[2].values, [0, 1])
            self.assertEqual(args[5:], ([], 123, 0, 0, 2, False, False))
            for row in args[0].values:
                count = (len(row) - 1) // 6
                keys = row[1 + count : 1 + count * 2]
                self.assertEqual(keys[: count // 2], keys[count // 2 :])
        self.assertEqual(calls[0].args[0].values[0][17:21], [512, 512, 512, 512])
        self.assertEqual([kind for kind, _ in trace], ["put", "put", "record", "wait"])
        storage._ops.kvcache_store_v2.assert_not_called()

    def test_put_dispatch_error_drains_submitted_work(self):
        storage, _, _ = self.make_storage()
        self.od_utils.batch_dispatch_put.side_effect = [
            None,
            RuntimeError("put failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "put failed"):
            storage.put_layers([self.layer_ranges(3)], max_ranges=4)
        storage._stream.synchronize.assert_called_once()

    def test_put_submits_all_layers_in_one_call(self):
        storage, trace, _ = self.make_storage()
        storage.put_layers([self.layer_ranges(2, layer) for layer in range(3)])
        self.od_utils.batch_dispatch_put.assert_called_once()
        self.assertEqual(
            self.od_utils.batch_dispatch_put.call_args.args[0].shape, (3, 25)
        )
        self.assertEqual([kind for kind, _ in trace], ["put", "record", "wait"])

    def test_descriptor_uses_paired_keys_and_registration_offsets(self):
        ranges = [
            OceanDiskRange(7, 1032, 16, 64),
            OceanDiskRange(8, 1048, 24, 128),
            OceanDiskRange(7, 5040, 8, 32),
            OceanDiskRange(8, 5048, 12, 64),
        ]
        row = pack_io(11, ranges, [(1000, 128), (5000, 256)], (2, 5))
        self.assertEqual(
            row,
            [
                11,
                0,
                0,
                1,
                1,
                7,
                8,
                7,
                8,
                32,
                48,
                40,
                48,
                64,
                128,
                32,
                64,
                16,
                24,
                8,
                12,
                5,
                2,
                5,
                2,
            ],
        )
        for bad in (
            ranges[:3],
            [ranges[0], ranges[3]],
            [ranges[0], OceanDiskRange(7, 5040, 8, -1)],
        ):
            with self.assertRaises(ValueError):
                pack_io(1, bad, [(1000, 128), (5000, 256)], (0,))
        for ptr, size in ((999, 1), (1120, 16), (1000, 0)):
            with self.assertRaises(ValueError):
                resolve_mr([(1000, 128)], ptr, size)

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
        linker._namespace = ("sglang-oceandisk-pool-pages", "test-model", 2, 1)
        linker._validate_dsa_layout()
        linker._closed = False
        linker.offload_queue = Queue()
        linker.offload_results = Queue()
        linker.device = "cuda:0"
        linker.storage = Mock()
        linker.max_ranges = 3
        linker.offload_results = Queue()
        return linker

    def test_fragment_keys_do_not_depend_on_gpu_slots(self):
        linker = self.make_linker()
        first = linker._page_fragments(PoolName.KV, ["p0", "p1"], [0, 2], 0)
        second = linker._page_fragments(PoolName.KV, ["p0", "p1"], [4, 6], 0)
        self.assertEqual([r.key for r in first], [r.key for r in second])
        self.assertNotEqual([r.ptr for r in first], [r.ptr for r in second])
        self.assertEqual([r.size for r in first], [6, 6])
        other_pool = linker._page_fragments(PoolName.INDEXER, ["p0"], [0], 0)
        other_layer = linker._page_fragments(PoolName.KV, ["p0"], [0], 1)
        self.assertEqual(first[0].key, other_pool[0].key)
        self.assertEqual(first[0].key, other_layer[0].key)
        self.assertEqual([r.storage_offset for r in first], [0, 0])
        self.assertEqual([r.storage_offset for r in other_layer], [512])

    def test_offsets_pad_each_physical_buffer_for_multiple_components(self):
        linker = self.make_linker(multi_component=True)
        first = linker._page_fragments(PoolName.KV, ["p0"], [0], 0)
        second = linker._page_fragments(PoolName.KV, ["p0"], [0], 1)
        self.assertEqual([r.storage_offset for r in first], [0, 1024])
        self.assertEqual([r.storage_offset for r in second], [512, 1536])

    def test_keys_share_dsa_namespaces_without_layout_fingerprinting(self):
        linker = self.make_linker()
        other_layout = self.make_linker(multi_component=True)
        key = linker._object_key("p0")
        self.assertEqual(key, other_layout._object_key("p0"))
        linker._validate_dsa_layout()
        self.assertEqual(key, linker._object_key("p0"))
        self.assertNotEqual(key, linker._object_key("p1"))
        linker._namespace = (*linker._namespace, "other-model")
        self.assertNotEqual(key, linker._object_key("p0"))

    def test_glm_page_uses_raw_local_stride_and_padded_remote_stride(self):
        linker = self.make_linker()
        for name, width, rows_are_pages in (
            (PoolName.KV, 1152, False),
            (PoolName.INDEXER, 8448, True),
        ):
            linker.pools[name] = DevicePoolEntry(
                name=name,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[
                    [
                        torch.zeros(
                            (128 if not rows_are_pages else 2, width), dtype=torch.uint8
                        )
                        for _ in range(2)
                    ]
                ],
                layer_mapping={0: 0, 1: 1},
                page_size=64,
                rows_are_pages=rows_are_pages,
            )
        linker._validate_dsa_layout()
        kv = linker._page_fragments(PoolName.KV, ["p0"], [64], 1)[0]
        indexer = linker._page_fragments(PoolName.INDEXER, ["p0"], [1], 1)[0]
        self.assertEqual((kv.size, kv.storage_offset), (73728, 73728))
        self.assertEqual((indexer.size, indexer.storage_offset), (8448, 8704))
        self.assertEqual(
            indexer.ptr, linker.pools[PoolName.INDEXER].buffer_meta[0][1][0] + 8448
        )

    def test_unpacked_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "packed=True"):
            self.make_linker(packed=False)

    def test_nonuniform_draft_io_counts_are_rejected(self):
        linker = self.make_linker()
        for entry in linker.pools.values():
            entry.layer_mapping = {0: (0, 1), 1: 1}
        with self.assertRaises(ValueError):
            linker._validate_dsa_layout()

    def test_lookup_probes_shared_dsa_keys_and_checks_holes(self):
        linker = self.make_linker()
        missing = linker._page_fragments(PoolName.INDEXER, ["p1"], [0], 1)[0].key
        linker.storage.exists_many.side_effect = lambda keys: [
            k != missing for k in keys
        ]
        transfers = [PoolTransfer(name=PoolName.KV, keys=["p0", "p1", "p2"])]
        self.assertEqual(linker.lookup("req", transfers), [1])
        linker.storage.exists_many.side_effect = lambda keys: [True] * len(keys)
        self.assertEqual(linker.lookup("req", transfers), [1, 2, 3])
        linker.storage.exists_many.assert_called_with(
            [linker._object_key(key) for key in ("p0", "p1", "p2")]
        )

    def test_page_key_location_mismatch_is_rejected(self):
        linker = self.make_linker()
        with self.assertRaises(ValueError):
            linker._page_fragments(PoolName.KV, ["p0", "p1"], [0], 0)

if __name__ == "__main__":
    unittest.main()
