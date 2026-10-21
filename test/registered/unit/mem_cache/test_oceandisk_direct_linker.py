"""CPU tests for OceanDisk layout conversion and backend completion contracts."""

import inspect
import itertools
import tempfile
import threading
import unittest
from array import array
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
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
    LayerWiseLoadCounter,
    LayerwiseSaveSession,
    OceanDiskDirectLinker,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
    OceanDiskRange,
    OceanDiskStorage,
    completion_key,
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

    def test_get_submits_all_layers_before_waiting(self):
        storage, trace, events = self.make_storage()
        ready = []
        layers = [self.layer_ranges(2, layer) for layer in range(3)]
        storage.get_layers(
            layers,
            max_ranges=8,
            on_layer_ready=lambda layer: (
                ready.append(layer),
                trace.append(("ready", layer)),
            ),
        )
        self.od_utils.batch_dispatch_get.assert_called_once()
        args = self.od_utils.batch_dispatch_get.call_args.args
        self.assertEqual(args[0].shape, (3, 25))
        self.assertEqual(args[1], 25)
        self.assertEqual(args[4], [1, 2, 3])
        self.assertEqual(args[5:], (123, 0, 0, 3, True, False))
        for row in args[0].values:
            self.assertEqual(row[1:5], [0, 0, 1, 1])
            self.assertEqual(row[5:9], [1, 2, 1, 2])
            self.assertEqual(row[17:21], [512, 512, 512, 512])
            self.assertEqual(row[21:25], [5, 2, 5, 2])
        self.assertEqual(ready, [0, 1, 2])
        self.assertEqual(
            [kind for kind, _ in trace],
            ["record"] * 3
            + ["dispatch", "wait", "ready", "wait", "ready", "wait", "ready"],
        )
        storage._ops.kvcache_load_v2.assert_not_called()

    def test_get_chunks_keep_pairs_and_wait_for_every_batch(self):
        storage, trace, events = self.make_storage()
        ready = []
        layers = [self.layer_ranges(5, layer) for layer in range(2)]
        storage.get_layers(
            layers,
            max_ranges=5,
            on_layer_ready=lambda layer: (
                ready.append(layer),
                trace.append(("ready", layer)),
            ),
        )
        calls = self.od_utils.batch_dispatch_get.call_args_list
        self.assertEqual(
            [call.args[0].shape for call in calls], [(2, 25), (2, 25), (2, 13)]
        )
        task_ids = []
        for call in calls:
            for row in call.args[0].values:
                task_ids.append(row[0])
                count = (len(row) - 1) // 6
                half = count // 2
                self.assertEqual(row[1 : 1 + half], [0] * half)
                self.assertEqual(row[1 + half : 1 + count], [1] * half)
                keys = row[1 + count : 1 + 2 * count]
                self.assertEqual(keys[:half], keys[half:])
        self.assertEqual(len(task_ids), len(set(task_ids)))
        first_wait = next(i for i, event in enumerate(trace) if event[0] == "wait")
        self.assertEqual(sum(event[0] == "dispatch" for event in trace[:first_wait]), 3)
        self.assertEqual(
            [event for event in trace if event[0] in ("wait", "ready")],
            [
                ("wait", 1),
                ("wait", 3),
                ("wait", 5),
                ("ready", 0),
                ("wait", 2),
                ("wait", 4),
                ("wait", 6),
                ("ready", 1),
            ],
        )

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

    def test_get_rejects_invalid_layers_before_dispatch(self):
        storage, _, _ = self.make_storage()
        for limit in (0, 1, True, 409601):
            with self.assertRaises(ValueError):
                storage.get_layers([self.layer_ranges(1)], max_ranges=limit)
        for layers in (
            [self.layer_ranges(1), self.layer_ranges(2)],
            [self.layer_ranges(1)[:1]],
            [
                self.layer_ranges(1)
                + [OceanDiskRange(8, 99999, 16), OceanDiskRange(8, 10000, 8)]
            ],
        ):
            with self.assertRaises(ValueError):
                storage.get_layers(layers)
        self.od_utils.batch_dispatch_get.assert_not_called()

    def test_legacy_direct_indexer_io_keeps_raw_offsets_and_pads_lengths(self):
        storage, trace, _ = self.make_storage()
        storage.regions = [(100000, 4 * 73728), (500000, 4 * 8448)]
        layers = [
            [
                OceanDiskRange(
                    page + 1,
                    100000 + (2 * layer + page) * 73728,
                    73728,
                    layer * 73728,
                )
                for page in range(2)
            ]
            + [
                OceanDiskRange(
                    page + 1,
                    500000 + (2 * layer + page) * 8448,
                    8448,
                    layer * 8704,
                )
                for page in range(2)
            ]
            for layer in range(2)
        ]
        ready = []
        storage.put_layers(layers, max_ranges=2)
        storage.get_layers(layers, max_ranges=2, on_layer_ready=ready.append)
        self.assertEqual(ready, [0, 1])
        for dispatch in (
            self.od_utils.batch_dispatch_put,
            self.od_utils.batch_dispatch_get,
        ):
            self.assertEqual(dispatch.call_count, 2)
            for page, call in enumerate(dispatch.call_args_list):
                for layer, row in enumerate(call.args[0].values):
                    self.assertEqual(row[1:3], [0, 1])
                    self.assertEqual(row[3:5], [page + 1, page + 1])
                    self.assertEqual(
                        row[5:7],
                        [(2 * layer + page) * 73728, (2 * layer + page) * 8448],
                    )
                    self.assertEqual(row[7:9], [layer * 73728, layer * 8704])
                    self.assertEqual(row[9:11], [73728, 8704])
        # Intentionally reproduce legacy overrun: the last wire range exceeds
        # the actual registered allocation. This test does not certify safety.
        last = self.od_utils.batch_dispatch_get.call_args.args[0].values[-1]
        self.assertEqual(last[6] + last[10] - storage.regions[1][1], 256)
        self.assertFalse(hasattr(storage, "_staging"))
        self.assertEqual(sum(kind == "wait" for kind, _ in trace), 5)

    def test_get_error_drains_before_releasing_metadata(self):
        storage, _, _ = self.make_storage()
        ready = Mock()
        self.od_utils.batch_dispatch_get.side_effect = [
            None,
            RuntimeError("dispatch failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            storage.get_layers(
                [self.layer_ranges(2), self.layer_ranges(2, 1)],
                max_ranges=2,
                on_layer_ready=ready,
            )
        ready.assert_not_called()
        storage._stream.synchronize.assert_called_once()

    def test_layerwise_put_waits_on_gpu_and_retains_metadata_until_final_wait(self):
        storage, trace, events = self.make_storage()
        prepared = storage.prepare_layerwise_put(
            [self.layer_ranges(2, layer) for layer in range(2)], max_ranges=2
        )
        self.assertEqual(len(prepared), 2)
        self.assertEqual([len(layer) for layer in prepared], [2, 2])
        ready = [Mock(), Mock()]
        for layer in range(2):
            storage.submit_put_layer(layer, prepared[layer], ready[layer])
        self.assertFalse(events)
        self.assertEqual(self.od_utils.batch_dispatch_put.call_count, 4)
        self.assertEqual(
            storage._stream.wait_event.call_args_list,
            [unittest.mock.call(event) for event in ready],
        )
        for call in self.od_utils.batch_dispatch_put.call_args_list:
            self.assertEqual(call.args[0].shape, (1, 13))
            self.assertEqual(call.args[5:], ([], 123, 0, 0, 1, False, False))
        self.assertEqual([kind for kind, _ in trace], ["put"] * 4)
        for event in ready:
            event.synchronize.assert_not_called()
        storage.wait_for_save()
        self.assertEqual([kind for kind, _ in trace][-2:], ["record", "wait"])

    def test_completion_marker_has_a_separate_key_and_is_published_in_both_pools(self):
        storage, _, _ = self.make_storage()
        storage._completion_buffer = SimpleNamespace(data_ptr=lambda: 1000)
        storage.publish_pages([7, 7, 8])
        rows = self.od_utils.batch_dispatch_put.call_args.args[0].values
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5:9], [completion_key(7), completion_key(8)] * 2)
        self.assertEqual(rows[0][17:21], [512] * 4)
        self.assertNotEqual(completion_key(7), 7)

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
        linker._layerwise_saved_keys = set()
        linker.storage = Mock()
        linker.max_ranges = 3
        linker.layer_done_counter = LayerWiseLoadCounter(2)
        linker.pending_loads = {}
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

    def test_load_submits_all_layers_with_kv_then_indexer_across_requests(self):
        linker = self.make_linker()
        transfers = []
        for key, slots in (("p0", [0, 1]), ("p1", [2, 3])):
            transfers.extend(
                linker.pool_group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=[key],
                            device_indices=torch.tensor(slots),
                        )
                    ]
                )
            )
        calls = []

        def load(layers, *, max_ranges, on_layer_ready):
            calls.append("load")
            self.assertEqual(max_ranges, 3)
            self.assertEqual(
                [[r.storage_offset for r in ranges] for ranges in layers],
                [[0, 0, 0, 0], [512, 512, 512, 512]],
            )
            for ranges in layers:
                self.assertEqual(
                    [r.key for r in ranges[:2]], [r.key for r in ranges[2:]]
                )
                self.assertEqual([r.size for r in ranges], [6, 6, 10, 10])
            for layer in range(len(layers)):
                on_layer_ready(layer)

        linker.storage.get_layers.side_effect = load
        linker.layer_done_counter.complete = lambda index, layer: calls.append(
            (index, layer)
        )
        linker._run_transfers(transfers, write=False, counter_index=7)
        self.assertEqual(calls, ["load", (7, 0), (7, 1)])
        linker.storage.get_layers.assert_called_once()
        linker.storage.put_layers.assert_not_called()
        linker._run_transfers(transfers, write=True)
        linker.storage.put_layers.assert_called_once()
        written = linker.storage.put_layers.call_args.args[0]
        self.assertEqual(len(written), 2)
        self.assertEqual([len(layer) for layer in written], [4, 4])

    def test_real_storage_events_resolve_linker_futures_in_layer_order(self):
        linker = self.make_linker()
        storage, trace, _ = TestOceanDiskStorage.make_storage(self)
        storage.regions = [
            (base, 16 * stride)
            for entry in linker.pools.values()
            for component in entry.buffer_meta
            for base, stride, _ in component
        ]
        linker.storage = storage
        index = linker.layer_done_counter.update_producer()
        complete = linker.layer_done_counter.complete

        def ready(batch, layer):
            self.assertEqual(trace[-1][0], "wait")
            complete(batch, layer)
            trace.append(("complete", layer))

        linker.layer_done_counter.complete = ready
        transfers = linker.pool_group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV, keys=["p0"], device_indices=torch.tensor([2, 3])
                )
            ]
        )
        linker._run_transfers(transfers, write=False, counter_index=index)
        self.od_utils.batch_dispatch_get.assert_called_once()
        rows = self.od_utils.batch_dispatch_get.call_args.args[0].values
        self.assertEqual([row[1:3] for row in rows], [[0, 2], [1, 3]])
        self.assertEqual(
            [event for event in trace if event[0] in ("wait", "complete")],
            [("wait", 1), ("complete", 0), ("wait", 2), ("complete", 1)],
        )
        self.assertTrue(
            all(future.done() for future in linker.layer_done_counter.futures[index])
        )

    def test_lookup_probes_shared_dsa_keys_and_checks_holes(self):
        linker = self.make_linker()
        missing = completion_key(
            linker._page_fragments(PoolName.INDEXER, ["p1"], [0], 1)[0].key
        )
        linker.storage.exists_many.side_effect = lambda keys: [
            k != missing for k in keys
        ]
        transfers = [PoolTransfer(name=PoolName.KV, keys=["p0", "p1", "p2"])]
        self.assertEqual(linker.lookup("req", transfers), [1])
        linker.storage.exists_many.side_effect = lambda keys: [True] * len(keys)
        self.assertEqual(linker.lookup("req", transfers), [1, 2, 3])

    def test_owner_ack_and_duplicate_load(self):
        linker = self.make_linker()
        transfer = PoolTransfer(
            name=PoolName.KV, keys=["p0"], device_indices=torch.tensor([2, 3])
        )
        self.assertTrue(linker.load("req", [transfer]))
        self.assertEqual(
            {t.name for t in linker.pending_loads["req"]},
            {PoolName.KV, PoolName.INDEXER},
        )
        with self.assertRaisesRegex(RuntimeError, "already queued"):
            linker.load("req", [transfer])
        self.assertFalse(linker.cancel_queued_load("req"))
        linker.offload_owner = False
        linker._ready_event = Mock(side_effect=AssertionError("non-owner wrote"))
        self.assertTrue(linker.offload([transfer]))
        self.assertTrue(linker.pop_completed_offload())

    def test_uc_offload_ack_does_not_repeat_a_completed_layerwise_save(self):
        linker = self.make_linker()
        linker.offload_owner = True
        linker._layerwise_saved_keys.add(linker._object_key("p0"))
        linker._ready_event = Mock()
        linker.device = "cuda:0"
        linker.offload_queue = Queue()
        transfer = PoolTransfer(
            name=PoolName.KV, keys=["p0"], device_indices=torch.tensor([2, 3])
        )
        self.assertTrue(linker.offload([transfer]))
        self.assertEqual(linker.num_completed_offloads(), 0)
        linker.offload_queue.put(None)
        with patch("torch.cuda.device", return_value=nullcontext()):
            linker._offload_worker()
        self.assertTrue(linker.pop_completed_offload())
        expanded = linker.pool_group.resolve_transfers([transfer])
        linker._run_transfers(expanded, write=True)
        linker.storage.put_layers.assert_not_called()
        linker.storage.publish_pages.assert_not_called()

    def test_saved_page_ack_cannot_overtake_an_earlier_failed_offload(self):
        linker = self.make_linker()
        linker.offload_owner = True
        linker.device = "cuda:0"
        linker.offload_queue = Queue()
        linker._ready_event = Mock()
        linker._layerwise_saved_keys.add(linker._object_key("saved"))
        for key in ("unsaved", "saved"):
            linker.offload(
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=[key],
                        device_indices=torch.tensor([2, 3]),
                    )
                ]
            )
        linker.storage.put_layers.side_effect = RuntimeError("first offload failed")
        self.assertEqual(linker.num_completed_offloads(), 0)
        linker.offload_queue.put(None)
        with patch("torch.cuda.device", return_value=nullcontext()):
            linker._offload_worker()
        self.assertFalse(linker.pop_completed_offload())
        self.assertTrue(linker.pop_completed_offload())
        linker.storage.put_layers.assert_called_once()

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
        with self.assertRaises(ValueError):
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


class TestOceanDiskLayerwiseSave(CustomTestCase):
    def make_session(self, owner=True):
        linker = Mock()
        linker.offload_owner = owner
        linker.num_layers = 2
        linker.max_ranges = 256
        linker._layerwise_saved_keys = set()
        linker.storage.prepare_layerwise_put.return_value = [["layer0"], ["layer1"]]
        return linker, LayerwiseSaveSession(linker, [[], []], [7, 8])

    def test_forward_saves_layers_without_waiting_and_publishes_only_at_end(self):
        linker, session = self.make_session()
        session.on_saved = Mock()
        with session:
            session.save_layer(0)
            session.save_layer(1)
            self.assertEqual(linker.storage.submit_put_layer.call_count, 2)
            linker.storage.wait_for_save.assert_not_called()
            linker.storage.publish_pages.assert_not_called()
            session.on_saved.assert_not_called()
            self.assertEqual(linker._layerwise_saved_keys, set())
        self.assertEqual(
            [call[0] for call in linker.storage.method_calls],
            [
                "prepare_layerwise_put",
                "submit_put_layer",
                "submit_put_layer",
                "wait_for_save",
                "publish_pages",
            ],
        )
        self.assertEqual(linker._layerwise_saved_keys, {7, 8})
        session.on_saved.assert_called_once()
        self.assertIsNone(session.prepared)

    def test_incomplete_forward_drains_but_does_not_publish(self):
        linker, session = self.make_session()
        with self.assertRaisesRegex(RuntimeError, "1/2"):
            with session:
                session.save_layer(0)
        linker.storage.wait_for_save.assert_called_once()
        linker.storage.publish_pages.assert_not_called()
        self.assertFalse(linker._layerwise_saved_keys)
        self.assertIsNone(session.prepared)

    def test_failed_forward_drains_and_preserves_the_original_exception(self):
        linker, session = self.make_session()
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with session:
                session.save_layer(0)
                raise ValueError("forward failed")
        linker.storage.wait_for_save.assert_called_once()
        linker.storage.publish_pages.assert_not_called()
        self.assertFalse(linker._layerwise_saved_keys)

    def test_publication_failure_does_not_mark_local_pages_saved(self):
        linker, session = self.make_session()
        linker.storage.publish_pages.side_effect = RuntimeError("publish failed")
        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            with session:
                session.save_layer(0)
                session.save_layer(1)
        self.assertFalse(linker._layerwise_saved_keys)

    def test_non_owner_runs_same_layer_lifecycle_without_native_puts(self):
        linker, session = self.make_session(owner=False)
        with session:
            session.save_layer(0)
            session.save_layer(1)
        self.assertFalse(linker.storage.method_calls)
        self.assertEqual(linker._layerwise_saved_keys, {7, 8})

    def make_planner(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = SimpleNamespace(
            page_size=64,
            tree_core=SimpleNamespace(is_eagle=False),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(256).reshape(2, 128)
            ),
        )
        wrapper.cache_linker = Mock()
        req = SimpleNamespace(
            origin_input_ids=array("q", range(65)),
            output_ids=array("q"),
            extra_key="tenant",
            cache_salt="salt",
            multimodal_inputs=None,
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0),
        )
        return wrapper, req

    @staticmethod
    def make_batch(req, tokens, end, mode="EXTEND"):
        return SimpleNamespace(
            reqs=[req],
            seq_lens_cpu=torch.tensor([end]),
            extend_lens=[len(tokens)],
            input_ids=torch.tensor(tokens),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            forward_mode=SimpleNamespace(name=mode, is_decode=lambda: mode == "DECODE"),
        )

    def test_prefill_and_chunk_resume_select_only_new_complete_pages(self):
        wrapper, req = self.make_planner()
        hash_calls = []

        def hashes(key, **kwargs):
            hash_calls.append(key)
            return [f"page{i}" for i in range(len(key) // 64)]

        with patch.dict(
            wrapper.prepare_layerwise_save.__globals__, {"get_storage_hash_str": hashes}
        ):
            session = wrapper.prepare_layerwise_save(
                self.make_batch(req, list(range(65)), 65)
            )
            transfer = wrapper.cache_linker.begin_layerwise_save.call_args.args[0][0]
            self.assertEqual(transfer.keys, ["page0"])
            self.assertEqual(transfer.device_indices.tolist(), list(range(64)))
            self.assertFalse(hasattr(req, "_uc_layerwise_saved_tokens"))
            session.on_saved()
            self.assertEqual(req._uc_layerwise_saved_tokens, 64)
            wrapper.prepare_layerwise_save(
                self.make_batch(req, list(range(65, 128)), 128)
            )
            transfer = wrapper.cache_linker.begin_layerwise_save.call_args.args[0][0]
            self.assertEqual(transfer.keys, ["page1"])
            self.assertEqual(transfer.device_indices.tolist(), list(range(64, 128)))
        self.assertEqual(list(hash_calls[0]), list(range(64)))
        self.assertEqual(list(hash_calls[1]), list(range(128)))
        self.assertEqual(
            (hash_calls[0].extra_key, hash_calls[0].cache_salt), ("tenant", "salt")
        )

    def test_decode_uses_actual_input_instead_of_unresolved_output_and_skips_partial_page(
        self,
    ):
        wrapper, req = self.make_planner()
        req.origin_input_ids = list(range(60))
        req.output_ids = [-1] * 4
        req._uc_layerwise_tokens = list(range(63))
        hasher = Mock(return_value=["decoded-page"])
        with patch.dict(
            wrapper.prepare_layerwise_save.__globals__, {"get_storage_hash_str": hasher}
        ):
            session = wrapper.prepare_layerwise_save(
                self.make_batch(req, [63], 64, "DECODE")
            )
            self.assertEqual(list(hasher.call_args.args[0]), list(range(64)))
            self.assertEqual(req._uc_layerwise_tokens, list(range(64)))
            session.on_saved()
            self.assertIsNone(
                wrapper.prepare_layerwise_save(self.make_batch(req, [64], 65, "DECODE"))
            )
            self.assertEqual(req._uc_layerwise_tokens, list(range(65)))
            self.assertEqual(hasher.call_count, 1)


if __name__ == "__main__":
    unittest.main()
