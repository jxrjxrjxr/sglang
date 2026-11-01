"""CPU regression tests for PUT completion -> READY -> scoped GET."""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake import conn as mooncake
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
    _linker_ready_end,
    _merge_linker_ranges,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker import (
    LayerWiseLoadCounter,
    LayerwiseSaveSession,
    OceanDiskDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinkerNotifyTransport(unittest.TestCase):
    def setUp(self):
        mgr = self.mgr = object.__new__(MooncakeKVManager)
        mgr.linker_notify = True
        mgr.linker_group_size, mgr.linker_num_layers = 2, 3
        mgr.kv_args = SimpleNamespace(page_size=4)
        mgr.linker_senders, mgr.linker_receivers = {}, {}
        mgr._staging_outstanding = defaultdict(int)
        mgr.request_status = {7: KVPoll.Transferring}
        mgr.failure_records, mgr.failure_lock = {}, threading.Lock()
        mgr.kv_status_msg_carries_reason = True
        mgr.kv_status_msg_tag = None
        mgr._prefill_unique_rank = lambda: 0
        self.peer = SimpleNamespace(
            is_dummy=False,
            required_dst_info_num=1,
            linker_start_page=0,
            decode_prefix_len=32,
            endpoint="127.0.0.1",
            dst_port=1234,
            mooncake_session_id="decode",
        )
        mgr.transfer_infos = {7: {"decode": self.peer}}
        sender = self.sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, 7
        sender.linker_lock = threading.RLock()
        sender.linker_ready, sender.linker_rdma_done = {}, False
        mgr.linker_senders[7] = sender
        receiver = self.receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room = mgr, 7
        receiver.session_id = "decode"
        receiver.linker_lock, receiver.linker_ready = threading.Lock(), {}
        receiver.linker_start_page, receiver.linker_end_page = 0, 8
        mgr.linker_receivers[7] = receiver
        self.messages = []

        def deliver(address, parts, **kwargs):
            self.messages.append(parts)
            if parts[0] == b"LINKER_READY" and 7 in mgr.linker_receivers:
                receiver.receive_linker_ready(parts)

        mgr._send_multipart_locked = Mock(side_effect=deliver)

    def _initialize_notify_manager(self, layer_ids, buffers, *, notify=True):
        self.mgr.disaggregation_mode = DisaggregationMode.DECODE
        args = SimpleNamespace(kv_layer_ids=layer_ids, kv_data_ptrs=buffers)
        with (
            patch.object(CommonKVManager, "__init__", return_value=None),
            patch.object(MooncakeKVManager, "init_engine"),
            patch.object(MooncakeKVManager, "register_buffer_to_engine"),
            patch.object(MooncakeKVManager, "start_decode_thread"),
            patch.object(
                mooncake,
                "get_observability",
                return_value=SimpleNamespace(enable_trace=False),
            ),
            envs.SGLANG_DISAGG_STAGING_BUFFER.override(False),
            envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS.override(50),
        ):
            self.mgr.__init__(
                args,
                DisaggregationMode.DECODE,
                SimpleNamespace(disaggregation_linker_notify=notify),
                is_mla_backend=True,
            )
        self.mgr.linker_senders[7] = self.sender
        self.mgr.linker_receivers[7] = self.receiver

    def test_dsa_positional_layers_publish_both_groups(self):
        # DSA reports positional KV buffers and no explicit kv_layer_ids.
        self._initialize_notify_manager([], list(range(78)))
        self.assertEqual(self.mgr.linker_num_layers, 78)
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.sender.notify_linker_ready(0, 8, 0, 50)
        self.assertEqual(self.receiver.linker_ready_until(0, 50, 0), 8)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.sender.notify_linker_ready(0, 8, 50, 78)
        self.assertEqual(self.receiver.linker_ready_until(50, 78, 0), 8)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)

    def test_explicit_layer_ids_and_native_mode(self):
        for notify, expected in ((True, 3), (False, 0)):
            with self.subTest(notify=notify):
                self._initialize_notify_manager(
                    [0, 0, 1, 1, 2, 2], list(range(6)), notify=notify
                )
                self.assertEqual(self.mgr.linker_num_layers, expected)

    def test_native_transfer_without_notify_completes_immediately(self):
        self.mgr.linker_notify = False
        self.mgr.linker_senders.clear()
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)
        self.assertFalse(any(m[0] == b"LINKER_READY" for m in self.messages))

    def test_rdma_success_waits_for_every_layer_group(self):
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.sender.notify_linker_ready(0, 8, 0, 2)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.sender.notify_linker_ready(0, 8, 2, 3)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)
        self.assertEqual(self.receiver.linker_ready_until(2, 3, 0), 8)

    def test_ready_before_rdma_keeps_transfer_pending(self):
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)

    def test_late_bootstrap_replays_completed_ranges(self):
        infos = self.mgr.transfer_infos.pop(7)
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.assertFalse(self.messages)
        self.mgr.transfer_infos[7] = infos
        self.sender.flush_linker_ready()
        self.assertEqual(self.receiver.linker_ready_until(0, 2, 0), 8)
        self.assertEqual(self.receiver.linker_ready_until(2, 3, 0), 8)

    def test_gap_and_duplicate_messages_do_not_publish_missing_pages(self):
        self.sender.notify_linker_ready(4, 8, 0, 3)
        self.assertEqual(self.receiver.linker_ready_until(0, 2, 0), 0)
        self.sender.notify_linker_ready(4, 8, 0, 3)
        self.assertEqual(self.receiver.linker_ready[(0, 2)], [(4, 8)])
        self.sender.notify_linker_ready(0, 4, 0, 3)
        self.assertEqual(self.receiver.linker_ready[(0, 2)], [(0, 8)])

    def test_non_aligned_hbm_prefix_clips_ready_messages(self):
        self.peer.linker_start_page = self.receiver.linker_start_page = 3
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.assertTrue(all(int(m[3]) == 3 for m in self.messages))
        self.assertEqual(self.receiver.linker_ready_until(0, 2, 3), 8)

    def test_all_hbm_hit_requires_no_storage_notification(self):
        self.peer.linker_start_page = 8
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)
        self.assertFalse(any(m[0] == b"LINKER_READY" for m in self.messages))

    def test_send_exception_becomes_native_failure(self):
        self.mgr._send_multipart_locked.side_effect = RuntimeError("disconnected")
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Failed)
        self.assertIn("disconnected", self.mgr.failure_records[7])

    def test_failure_is_not_promoted_by_late_completion(self):
        self.sender.fail_linker("PUT failed")
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Failed)

    def test_old_sender_cannot_publish_into_reused_room(self):
        self.sender.clear()
        self.mgr.request_status[7] = KVPoll.Transferring
        self.mgr.linker_senders[7] = object()
        self.sender.notify_linker_ready(0, 8, 0, 3)
        self.assertFalse(self.messages)

    def test_wrong_session_and_out_of_range_layer(self):
        with self.assertRaises(ValueError):
            self.receiver.receive_linker_ready(
                [b"LINKER_READY", b"7", b"old", b"0", b"8", b"0", b"2"]
            )
        self.receiver.receive_linker_ready(
            [b"LINKER_READY", b"7", b"decode", b"0", b"8", b"0", b"3"]
        )
        self.assertEqual(self.mgr.request_status[7], KVPoll.Failed)

    def test_metadata_parses_native_and_notification_messages(self):
        fields = [
            b"7", b"127.0.0.1", b"1234", b"decode", b"", b"", b"", b"1", b"32", b""
        ]
        self.assertIsNone(TransferInfo.from_zmq(fields).linker_start_page)
        info = TransferInfo.from_zmq(fields + [b"LINKER_NOTIFY", b"3", b"2", b"3"])
        self.assertEqual(
            (info.linker_start_page, info.linker_group_size, info.linker_num_layers),
            (3, 2, 3),
        )

    def test_ready_range_merge_preserves_gaps(self):
        ranges = _merge_linker_ranges([(0, 2), (6, 8)], 2, 4)
        self.assertEqual(ranges, [(0, 4), (6, 8)])
        self.assertEqual(_linker_ready_end(ranges, 0), 4)
        self.assertEqual(_linker_ready_end(ranges, 4), 4)
        self.assertEqual(_merge_linker_ranges(ranges, 4, 6), [(0, 8)])


class TestLinkerNotifyPrefix(unittest.TestCase):
    def setUp(self):
        self.wrapper = object.__new__(UnifiedCacheLinkerWrapper)
        self.backend = object.__new__(OceanDiskDirectLinker)
        self.backend.notify_enabled, self.backend.offload_owner = True, True
        self.backend.num_layers = 3
        self.backend._layerwise_receipts = {}
        self.backend._object_key = lambda key: key
        self.backend.lookup = Mock(return_value=[])
        self.wrapper.cache_linker = self.backend
        self.wrapper.cache = SimpleNamespace(
            page_size=4,
            tree_core=SimpleNamespace(is_eagle=False),
            _all_reduce_attn_groups=Mock(),
        )
        self.wrapper._components = [
            SimpleNamespace(
                build_external_linker_transfer=lambda phase, node, keys: PoolTransfer(
                    PoolName.KV, keys=keys
                )
            )
        ]
        self.sender = SimpleNamespace(
            notify_linker_ready=Mock(),
            fail_linker=Mock(),
            linker_page_bounds=Mock(return_value=None),
        )
        self.req = SimpleNamespace(
            rid="request",
            origin_input_ids=list(range(21)),
            extra_key=None,
            cache_salt=None,
            disagg_kv_sender=self.sender,
            _compute_max_prefix_len=lambda length: length - 1,
        )
        self.result = SimpleNamespace(device_indices=torch.arange(16))
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.unified_cache.unified_cache_linker."
                "get_storage_hash_str",
                side_effect=lambda key, **kw: [f"p{i}" for i in range(len(key) // 4)],
            )
        )

    def inspect(self):
        self.wrapper._prepare_handoff_prefix(self.req, self.result)

    def test_pending_receipt_blocks_ready_and_avoids_exists(self):
        receipt = Future()
        self.backend._layerwise_receipts = dict.fromkeys(
            [f"p{i}" for i in range(4)], receipt
        )
        self.inspect()
        self.backend.lookup.assert_not_called()
        self.sender.notify_linker_ready.assert_not_called()
        receipt.set_result(True)
        self.sender.notify_linker_ready.assert_called_once_with(0, 4, 0, 3)

    def test_completed_receipt_avoids_exists_and_repair(self):
        receipt = Future()
        receipt.set_result(True)
        self.backend._layerwise_receipts = dict.fromkeys(
            [f"p{i}" for i in range(4)], receipt
        )
        self.inspect()
        self.backend.lookup.assert_not_called()
        self.assertIsNone(self.req._uc_linker_repair_start)
        self.sender.notify_linker_ready.assert_called_once_with(0, 4, 0, 3)

    def test_missing_receipt_checks_once_and_repairs_missing_suffix(self):
        self.backend.lookup.return_value = [1, 2]
        self.inspect()
        self.inspect()
        self.backend.lookup.assert_called_once()
        self.assertEqual(self.req._uc_linker_repair_start, 2)
        self.sender.notify_linker_ready.assert_called_once_with(0, 2, 0, 3)

    def test_all_storage_hit_needs_no_put(self):
        self.backend.lookup.return_value = [1, 2, 3, 4]
        self.inspect()
        self.assertIsNone(self.req._uc_linker_repair_start)
        self.sender.notify_linker_ready.assert_called_once_with(0, 4, 0, 3)

    def test_pending_receipt_failure_fails_room(self):
        receipt = Future()
        self.backend._layerwise_receipts = dict.fromkeys(
            [f"p{i}" for i in range(4)], receipt
        )
        self.inspect()
        receipt.set_result(False)
        self.sender.fail_linker.assert_called_once()
        self.sender.notify_linker_ready.assert_not_called()

    def test_failed_receipt_cannot_be_overridden_by_exists(self):
        receipt = Future()
        receipt.set_result(False)
        self.backend._layerwise_receipts = dict.fromkeys(
            [f"p{i}" for i in range(4)], receipt
        )
        self.backend.lookup.return_value = [1, 2, 3, 4]
        self.inspect()
        self.sender.fail_linker.assert_called_once()
        self.sender.notify_linker_ready.assert_not_called()
        self.backend.lookup.assert_not_called()

    def test_repair_includes_cached_slots_and_keeps_source_lease(self):
        self.backend.lookup.return_value = [1, 2]
        self.inspect()
        allocator = object.__new__(PagedTokenToKVPoolAllocator)
        allocator.pin_pages_for_io = Mock(return_value="lease")
        allocator.unpin_pages_for_io = Mock()
        self.wrapper.cache.token_to_kv_pool_allocator = allocator
        self.wrapper.cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(24).reshape(1, 24)
        )
        self.wrapper.pending_layerwise_saves = []
        self.req.kv = SimpleNamespace(cache_protected_len=16, req_pool_idx=0)
        self.req._uc_layerwise_submitted_tokens = 16
        self.req.multimodal_inputs = None
        self.req.output_ids = []
        self.req.get_fill_ids = lambda: self.req.origin_input_ids
        session = SimpleNamespace(
            completion=Future(), aborted=False, prepare_details={}
        )
        self.backend.begin_layerwise_save = Mock(return_value=session)
        self.backend._layerwise_receipts.update(
            dict.fromkeys(["p2", "p3", "p4"], session.completion)
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND", is_decode=lambda: False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            reqs=[self.req], decoding_reqs=None,
            seq_lens_cpu=torch.tensor([21]), extend_lens=[5],
        )
        self.wrapper.prepare_layerwise_save(batch)
        transfer = self.backend.begin_layerwise_save.call_args.args[0][0]
        self.assertEqual(transfer.keys, ["p2", "p3", "p4"])
        self.assertEqual(transfer.device_indices.tolist(), list(range(8, 20)))
        self.sender.fail_linker.assert_not_called()
        allocator.pin_pages_for_io.assert_called_once()
        self.assertEqual(
            allocator.pin_pages_for_io.call_args.args[0].tolist(), [2, 3, 4]
        )
        session.on_submitted()
        self.assertIsNone(self.req._uc_linker_repair_start)
        self.assertEqual(self.req._uc_layerwise_submitted_tokens, 20)
        allocator.unpin_pages_for_io.assert_not_called()
        session.release_source()
        allocator.unpin_pages_for_io.assert_called_once_with("lease")

    def test_only_owner_probes_and_all_ranks_agree_on_repair(self):
        self.backend.offload_owner = False
        self.wrapper.cache._all_reduce_attn_groups.side_effect = (
            lambda value, op: value.copy_(torch.tensor([2, 4]))
        )
        self.inspect()
        self.assertEqual(self.req._uc_linker_repair_start, 2)
        self.backend.lookup.assert_not_called()
        self.sender.notify_linker_ready.assert_not_called()

    def test_empty_d_missing_prefix_still_enters_rank_consensus(self):
        self.sender.linker_page_bounds.return_value = (4, 4)
        self.inspect()
        self.backend.lookup.assert_not_called()
        self.wrapper.cache._all_reduce_attn_groups.assert_called_once()
        self.assertIsNone(self.req._uc_linker_repair_start)

    def test_owner_lookup_failure_still_enters_consensus(self):
        self.backend.lookup.side_effect = RuntimeError("EXISTS failed")
        self.inspect()
        self.wrapper.cache._all_reduce_attn_groups.assert_called_once()
        self.sender.fail_linker.assert_called_once()
        self.sender.notify_linker_ready.assert_not_called()

    def test_shared_session_binds_multiple_requests_without_duplicate_put(self):
        receipt = Future()
        self.backend._layerwise_receipts = {"p0": receipt, "p1": receipt}
        session = SimpleNamespace(completion=receipt)
        other = SimpleNamespace(
            disagg_kv_sender=SimpleNamespace(
                notify_linker_ready=Mock(), fail_linker=Mock()
            )
        )
        transfers = [PoolTransfer(PoolName.KV, keys=["p0", "p1"])] * 2
        self.wrapper._bind_handoff_saves(
            transfers, [(self.req, 8), (other, 8)], session
        )
        session.on_layer_group_ready(0, 2)
        self.sender.notify_linker_ready.assert_called_once_with(0, 2, 0, 2)
        other.disagg_kv_sender.notify_linker_ready.assert_called_once_with(0, 2, 0, 2)
        receipt.set_result(False)
        self.sender.fail_linker.assert_called_once()
        other.disagg_kv_sender.fail_linker.assert_called_once()


class TestLinkerNotifyIO(unittest.TestCase):
    def test_scoped_get_retains_original_storage_offsets(self):
        linker = object.__new__(OceanDiskDirectLinker)
        layers = [[("layer", i, i * 4096)] for i in range(4)]
        linker._transfer_layers = Mock(return_value=layers)
        linker.layer_done_counter = LayerWiseLoadCounter(4)
        index = linker.layer_done_counter.update_producer(2)
        linker.storage = SimpleNamespace(get_layers=Mock())

        def get_layers(selected, on_layer_ready, **kwargs):
            for local_layer in range(len(selected)):
                on_layer_ready(local_layer)

        linker.storage.get_layers.side_effect = get_layers
        with patch.dict("os.environ", {"SGLANG_OCEANDISK_PERF_TRACE": "0"}):
            linker._run_transfers(
                [], write=False, counter_index=index, layer_range=(2, 4)
            )
        self.assertEqual(linker.storage.get_layers.call_args.args[0], layers[2:4])
        self.assertTrue(
            all(
                f.done() and f.exception() is None
                for f in linker.layer_done_counter.futures[index]
            )
        )

    def run_save(self, fail_layer=None):
        linker = object.__new__(OceanDiskDirectLinker)
        linker.device, linker.num_layers, linker.notify_group_size = "cpu", 3, 2
        linker.offload_owner = True
        linker.offload_queue = Queue()
        linker._put_finished, linker._report_put_idle = Mock(), Mock()
        session = object.__new__(LayerwiseSaveSession)
        session.prepared = [[i] for i in range(3)]
        session.native_call_times, session.keys = [], [123]
        session.phase, session.next_layer = "prefill", 3
        session.put_failed = session.aborted = False
        session.completion = Future()
        events = []

        def submit(layer, prepared, ready):
            events.append(("put", layer))
            if layer == fail_layer:
                raise RuntimeError("PUT failed")
            return [0.0]

        linker.storage = SimpleNamespace(
            submit_put_layer=submit,
            record_save_event=lambda: SimpleNamespace(
                synchronize=lambda: events.append(("sync",))
            ),
        )
        session.on_layer_group_ready = lambda start, end: events.append(
            ("ready", start, end)
        )
        for layer in range(3):
            linker.offload_queue.put((session, layer, object()))
        linker.offload_queue.put(session)
        linker.offload_queue.put(None)
        with (
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.dict("os.environ", {"SGLANG_OCEANDISK_TRACE": "0"}),
        ):
            linker._offload_worker()
        return session, events

    def test_group_notification_follows_put_completion(self):
        session, events = self.run_save()
        self.assertEqual(
            events,
            [
                ("put", 0), ("put", 1), ("sync",), ("ready", 0, 2),
                ("put", 2), ("sync",), ("ready", 2, 3), ("sync",),
            ],
        )
        self.assertTrue(session.completion.result())

    def test_failed_group_is_never_notified(self):
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker",
            level="ERROR",
        ):
            session, events = self.run_save(fail_layer=1)
        self.assertFalse(any(e[0] == "ready" for e in events))
        self.assertFalse(session.completion.result())


if __name__ == "__main__":
    unittest.main()
