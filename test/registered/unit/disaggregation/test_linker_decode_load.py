"""CPU coverage for the OceanDisk prefix / Mooncake tail handoff."""

import importlib.util
import inspect
import json
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups import pd_disaggregation_hook
from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.decode_linker_mixin import (
    DecodeLinkerTransferMixin,
    LinkerLoadGatedKVReceiver,
    _linker_handoff_prefix_len,
)
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.disaggregation.mooncake.conn import MooncakeFailureExceptionMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker import (
    LayerWiseLoadCounter,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinkerDecodeLoad(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS.override(-1)
        )
        self.enterContext(envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS.override(50))
        self.enterContext(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(300))
        self.enterContext(envs.SGLANG_DISAGG_STAGING_BUFFER.override(False))
        self.hash_pages = self.enterContext(
            patch(
                "sglang.srt.mem_cache.unified_cache.unified_cache_linker."
                "get_storage_hash_str",
                side_effect=lambda key, prior_hash, page_size: [
                    f"page-{i}" for i in range(len(key) // page_size)
                ],
            )
        )
        self.reset_load_state()

    def reset_load_state(self):
        self.cache = SimpleNamespace(
            page_size=4,
            tree_core=SimpleNamespace(is_eagle=False),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(256).reshape(4, 64)
            ),
            _all_reduce_attn_groups=Mock(),
            dec_lock_ref=Mock(),
        )
        self.pending = {}
        self.batches = []
        self.completed = deque()
        self.counter = LayerWiseLoadCounter(3)
        self.backend = SimpleNamespace(
            layer_done_counter=self.counter,
            num_layers=3,
            lookup=Mock(side_effect=AssertionError("D must never issue EXISTS")),
            offload=Mock(side_effect=AssertionError("D must never issue PUT")),
            load=Mock(
                side_effect=lambda tag, transfers: (
                    self.pending.setdefault(tag, transfers) is transfers
                )
            ),
            start_layer_wise_loading=Mock(side_effect=self.start_batch),
            pop_completed_load=self.completed.popleft,
        )
        self.linker = object.__new__(UnifiedCacheLinkerWrapper)
        self.linker.cache = self.cache
        self.linker.cache_linker = self.backend
        self.linker._components = (
            SimpleNamespace(
                build_external_linker_transfer=Mock(
                    side_effect=lambda phase, node, keys: PoolTransfer(
                        PoolName.KV, keys=list(keys)
                    )
                ),
            ),
        )
        self.linker.handoff_transfers = {}
        self.linker.pending_handoff_loads = {}
        self.linker.completed_handoff_loads = {}
        self.linker.pending_loads = {}
        self.linker.hit_markers = {}
        self.linker.bypass_storage_match = True
        self.cache.linker = self.linker
        self.queue = SimpleNamespace(
            tree_cache=self.cache, scheduler=SimpleNamespace(chunked_prefill_size=8)
        )

    def request(self, rid="r", l1=4, prefix=24, slot=0):
        req = SimpleNamespace(
            rid=rid,
            origin_input_ids=list(range(29)),
            extra_key="model",
            cache_salt="tenant",
            prefix_indices=torch.arange(l1),
            kv=SimpleNamespace(req_pool_idx=slot),
            return_logprob=False,
            logprob_start_len=-1,
            time_stats=SimpleNamespace(
                decode_transfer_queue_entry_time=time.perf_counter()
            ),
        )
        req._compute_max_prefix_len = lambda length: Req._compute_max_prefix_len(
            req, length
        )
        return DecodeRequest(
            req=req,
            kv_receiver=SimpleNamespace(
                poll=Mock(return_value=KVPoll.Success), abort_notified=False,
                linker_ready_until=Mock(return_value=100),
                bootstrap_room=7,
                kv_mgr=SimpleNamespace(failure_records={}, failure_lock=threading.Lock()),
                conclude_state=None,
                clear=Mock(),
            ),
            linker_prefix_len=prefix,
            linker_loaded_len=l1,
        )

    def start_batch(self, layer_range=None):
        count = None if layer_range is None else layer_range[1] - layer_range[0]
        index = self.counter.update_producer(count)
        self.batches.append((index, list(self.pending)))
        self.pending.clear()
        return index

    def complete_batch(self, index=0, success=True):
        counter_index, tags = self.batches[index]
        if success:
            for layer in range(len(self.counter.futures[counter_index])):
                self.counter.complete(counter_index, layer)
        else:
            self.counter.complete(counter_index, 0)
            self.counter.fail(counter_index, RuntimeError("GIDS failed"))
        self.completed.append(tags)
        self.linker.drain_loads(1)

    def process(self, *requests):
        DecodeLinkerTransferMixin._process_linker_loads(self.queue, list(requests))

    def test_prefix_tail_alignment_and_fallback(self):
        req = self.request().req
        for tail, l1, expected in [
            (-1, 4, 20),
            (0, 4, 28),
            (5, 4, 20),
            (100, 4, 4),
            (8, 24, 24),
        ]:
            with (
                self.subTest(tail=tail, l1=l1),
                envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS.override(tail),
            ):
                self.assertEqual(_linker_handoff_prefix_len(req, l1, 4, 8), expected)
        req.origin_input_ids = list(range(28))
        with envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS.override(0):
            self.assertEqual(_linker_handoff_prefix_len(req, 0, 4, 8), 24)
        req.return_logprob, req.logprob_start_len = True, 11
        self.assertEqual(_linker_handoff_prefix_len(req, 0, 4, 8), 0)
        req.origin_input_ids = []
        self.assertEqual(_linker_handoff_prefix_len(req, 0, 4, 8), 0)

    def test_invalid_window_rejected(self):
        for chunk in (None, -1, 0, 3):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                _linker_handoff_prefix_len(self.request().req, 0, 4, chunk)

    def test_gate_preserves_nonterminal_states(self):
        dr = self.request()
        gate = LinkerLoadGatedKVReceiver(dr)
        for state in (
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
        ):
            dr.kv_receiver.poll.return_value = state
            self.assertEqual(gate.poll(), state)
        dr.kv_receiver.poll.return_value = KVPoll.Success
        self.assertEqual(gate.poll(), KVPoll.Transferring)
        dr.linker_loaded_len = dr.linker_prefix_len
        self.assertEqual(gate.poll(), KVPoll.Success)

    def test_terminal_states_wait_for_dma(self):
        for terminal in (KVPoll.Success, KVPoll.Failed):
            with self.subTest(terminal=terminal):
                dr = self.request()
                dr.linker_inflight_end = 12
                dr.kv_receiver.poll.return_value = terminal
                gate = LinkerLoadGatedKVReceiver(dr)
                self.assertEqual(gate.poll(), KVPoll.Transferring)
                dr.linker_inflight_end = 0
                dr.linker_loaded_len = dr.linker_prefix_len
                self.assertEqual(gate.poll(), terminal)

    def test_failure_and_abort_drain_before_release(self):
        for abort in (False, True):
            with self.subTest(abort=abort):
                self.reset_load_state()
                dr = self.request()
                self.process(dr)
                if abort:
                    dr.kv_receiver.abort_notified = True
                else:
                    dr.kv_receiver.poll.return_value = KVPoll.Failed
                self.process(dr)
                self.assertEqual(
                    LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Transferring
                )
                self.complete_batch()
                self.process(dr)
                self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)
                self.assertEqual(len(self.batches), 1)
                self.assertFalse(self.linker.handoff_transfers)

    def test_timeout_waits_for_inflight_dma(self):
        dr = self.request()
        dr.req.time_stats.decode_transfer_queue_entry_time -= 301
        dr.linker_inflight_end = 12
        gate = LinkerLoadGatedKVReceiver(dr)
        self.assertEqual(gate.poll(), KVPoll.Transferring)
        dr.linker_inflight_end = 0
        self.assertEqual(gate.poll(), KVPoll.Failed)
        self.process(dr)
        self.backend.lookup.assert_not_called()
        with self.assertRaisesRegex(KVTransferError, "timed out") as caught:
            MooncakeFailureExceptionMixin.failure_exception(dr.kv_receiver)
        self.assertFalse(caught.exception.is_from_another_rank)

    def test_timeout_in_scheduler_records_progress(self):
        dr = self.request()
        dr.req.time_stats.decode_transfer_queue_entry_time -= 301
        self.process(dr)
        reason = dr.kv_receiver.kv_mgr.failure_records[7]
        self.assertIn("timed out", reason)
        self.assertIn("loaded_tokens=4/24", reason)
        self.assertIn("loaded_layers=0", reason)
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)
        self.assertEqual(dr.kv_receiver.kv_mgr.failure_records[7], reason)

    def test_native_failure_reason_is_preserved(self):
        dr = self.request()
        dr.kv_receiver.kv_mgr.failure_records[7] = "RDMA tail failed"
        dr.kv_receiver.poll.return_value = KVPoll.Failed
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)
        self.assertEqual(dr.kv_receiver.kv_mgr.failure_records[7], "RDMA tail failed")

    def test_completed_prefix_leaves_tail_timeout_to_mooncake(self):
        dr = self.request()
        dr.linker_loaded_len = dr.linker_prefix_len
        dr.req.time_stats.decode_transfer_queue_entry_time -= 301
        dr.kv_receiver.poll.return_value = KVPoll.Transferring
        self.process(dr)
        gate = LinkerLoadGatedKVReceiver(dr)
        self.assertFalse(dr.linker_failed)
        self.assertEqual(gate.poll(), KVPoll.Transferring)
        dr.kv_receiver.poll.return_value = KVPoll.Success
        self.assertEqual(gate.poll(), KVPoll.Success)
        dr.kv_receiver.poll.return_value = KVPoll.Failed
        self.assertEqual(gate.poll(), KVPoll.Failed)

    def test_no_probe_due_reduces_once_and_preserves_cleanup(self):
        waiting = self.request("waiting")
        waiting.kv_receiver.linker_ready_until.return_value = 0
        finished = self.request("finished")
        finished.linker_loaded_len = finished.linker_prefix_len
        failed = self.request("failed")
        failed.kv_receiver.poll.return_value = KVPoll.Failed
        self.linker.handoff_transfers = {
            dr.req.rid: [] for dr in (waiting, finished, failed)
        }
        self.process(waiting, finished, failed)
        self.cache._all_reduce_attn_groups.assert_called_once()
        self.backend.lookup.assert_not_called()
        self.backend.load.assert_not_called()
        self.assertTrue(failed.linker_failed)
        self.assertFalse(waiting.linker_failed or finished.linker_failed)
        self.assertEqual(set(self.linker.handoff_transfers), {"waiting"})

    def test_native_fallback_does_not_probe_or_add_timeout(self):
        dr = self.request(l1=24, prefix=24)
        dr.req.time_stats.decode_transfer_queue_entry_time = 0
        self.process(dr)
        self.backend.lookup.assert_not_called()
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Success)

    def test_ready_prefix_loads_all_layers_in_one_get_and_caps_batches(self):
        # Every layer group READY over the whole prefix: one GET per request
        # covers all pages and layers instead of window x group pieces. A
        # batch holds about one longest prefix, so b waits one round.
        a, b = self.request(), self.request("b", slot=1)
        self.process(a, b)
        self.assertEqual(self.batches, [(0, ["r@1:0"])])
        tag, transfers = self.backend.load.call_args_list[0].args
        self.assertEqual(tag, "r@1:0")
        self.assertEqual(transfers[0].device_indices.tolist(), list(range(4, 24)))
        full = self.linker.handoff_transfers["r"][0]
        self.assertEqual(transfers[0].keys, full.keys[1:6])
        self.assertIsNone(full.device_indices)
        self.assertEqual(len(self.counter.futures[0]), 3)
        self.assertEqual((a.linker_inflight_end, b.linker_inflight_end), (24, 0))
        self.process(a, b)
        self.assertEqual(self.batches[1][1], ["b@1:0"])
        self.complete_batch()
        self.complete_batch(1)
        self.assertFalse(self.counter.futures)
        self.process(a, b)
        self.assertEqual((a.linker_loaded_len, b.linker_loaded_len), (24, 24))
        self.assertEqual(a.linker_loaded_layers, 0)
        self.assertEqual(len(self.batches), 2)
        self.assertFalse(self.linker.completed_handoff_loads)
        self.assertFalse(self.linker.pending_handoff_loads)
        self.assertFalse(self.linker.handoff_transfers)
        self.assertEqual(LinkerLoadGatedKVReceiver(a).poll(), KVPoll.Success)

    def test_all_layer_range_takes_rank_minimum(self):
        # Another rank is READY only up to page 3: the merged end is the MIN.
        self.cache._all_reduce_attn_groups.side_effect = lambda states, op: (
            states[:, 0].clamp_(max=3)
        )
        dr = self.request()
        self.process(dr)
        self.assertEqual(self.batches, [(0, ["r@1:0"])])
        self.assertEqual(dr.linker_inflight_end, 12)
        self.assertEqual(dr.linker_inflight_layers, 3)

    def test_decode_loop_drains_linker_even_when_retractions_block_polling(self):
        events = []
        scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            tree_cache=SimpleNamespace(
                check_hicache_events=lambda: events.append("drain")
            ),
            disagg_decode_transfer_queue=SimpleNamespace(
                resolve_deferred_releases=lambda: events.append("release")
            ),
            disagg_decode_prealloc_queue=SimpleNamespace(
                resume_retracted_reqs=lambda: [], retracted_queue=[object()]
            ),
            waiting_queue=[],
        )
        cfg = SimpleNamespace(
            disaggregation_linker_notify=True,
            disaggregation_decode_enable_offload_kvcache=False,
        )
        with patch.object(decode, "get_disagg", return_value=cfg):
            inspect.unwrap(
                decode.SchedulerDisaggregationDecodeMixin.process_decode_queue
            )(scheduler)
        self.assertEqual(events, ["drain", "release"])

    def test_hash_chain_includes_l1_and_request_namespace(self):
        dr = self.request()
        self.process(dr)
        key = self.hash_pages.call_args.args[0]
        self.assertEqual(list(key), dr.req.origin_input_ids[:24])
        self.assertEqual((key.extra_key, key.cache_salt), ("model", "tenant"))
        self.assertIsNone(self.hash_pages.call_args.kwargs["prior_hash"])
        self.complete_batch()
        self.process(dr)
        self.hash_pages.assert_called_once()

    def test_failure_injection_outside_gate_waits_for_dma(self):
        dr = self.request()
        dr.linker_inflight_end = 12
        queue = SimpleNamespace(
            queue=[dr],
            scheduler=SimpleNamespace(enable_decode_hicache=False),
            gloo_group=None,
            metadata_buffers=None,
        )
        with (
            patch.object(
                decode,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_linker_notify=True),
            ),
            patch.object(decode, "poll_and_all_reduce", return_value=[KVPoll.Failed]),
        ):
            self.assertEqual(
                decode.DecodeTransferQueue._poll_with_metadata_gate(queue),
                [KVPoll.Transferring],
            )
        self.assertTrue(dr.linker_failed)
        dr.linker_inflight_end = 0
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)

    def test_unready_range_waits_and_ready_part_loads_without_full_window(self):
        dr = self.request()
        dr.kv_receiver.linker_ready_until.return_value = 1  # == start: nothing
        self.process(dr)
        self.process(dr)
        self.backend.lookup.assert_not_called()
        self.backend.load.assert_not_called()
        self.assertEqual(self.cache._all_reduce_attn_groups.call_count, 2)
        # One READY page (less than the 2-page window) loads right away.
        dr.kv_receiver.linker_ready_until.return_value = 2
        self.process(dr)
        self.backend.load.assert_called_once()
        self.assertEqual(dr.linker_inflight_end, 8)

    def test_tp_peer_not_ready_prevents_load(self):
        self.cache._all_reduce_attn_groups.side_effect = (
            lambda states, op: states.zero_()
        )
        self.process(self.request())
        self.backend.load.assert_not_called()
        self.cache._all_reduce_attn_groups.assert_called_once()

    def test_layers_finish_before_advancing_token_window(self):
        dr = self.request()
        ready = {(0, 2): 100, (2, 3): None}  # None: not READY past start
        dr.kv_receiver.linker_ready_until.side_effect = lambda layer, end, start: (
            ready[(layer, end)] or start
        )
        with envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS.override(2):
            # Only group 0 is READY: fixed window x one group.
            self.process(dr)
            self.assertEqual(self.batches[0][1], ["r@1:0"])
            self.assertEqual(len(self.counter.futures[0]), 2)
            self.complete_batch()
            self.process(dr)
            self.assertEqual(dr.linker_loaded_len, 4)
            self.assertEqual(dr.linker_loaded_layers, 2)
            self.assertEqual(len(self.batches), 1)
            # Group 1 arrives: the same window, never a wider one.
            ready[(2, 3)] = 100
            self.process(dr)
            self.assertEqual(self.batches[1][1], ["r@1:2"])
            self.assertEqual(dr.linker_inflight_end, 12)
            self.assertEqual(len(self.counter.futures[1]), 1)
            self.complete_batch(1)
            self.process(dr)
            self.assertEqual(dr.linker_loaded_len, 12)
            self.assertEqual(dr.linker_loaded_layers, 0)
            # Both groups READY for the rest: one all-layer GET.
            self.assertEqual(self.batches[2][1], ["r@3:0"])
            self.assertEqual(dr.linker_inflight_end, 24)
            self.assertEqual(len(self.counter.futures[2]), 3)
        self.backend.lookup.assert_not_called()
        self.backend.offload.assert_not_called()

    def test_tp_peer_failure_is_sticky(self):
        dr = self.request()
        self.cache._all_reduce_attn_groups.side_effect = lambda states, op: (
            states.fill_(-1)
        )
        self.process(dr)
        self.assertTrue(dr.linker_failed)
        self.backend.lookup.assert_not_called()
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)
        self.assertIn("prefix transfer failed", dr.kv_receiver.kv_mgr.failure_records[7])

    def test_shared_counter_failure_and_normal_load_unlock(self):
        # Two 2-page ranges fit one batch (cap: the 6-page longest prefix).
        a, b = self.request(l1=16), self.request("b", l1=16, slot=1)
        self.process(a, b)
        self.linker.pending_loads["tree"] = (7, "lock")
        self.batches[0][1].append("tree")
        self.complete_batch(success=False)
        self.assertEqual(
            self.linker.completed_handoff_loads, {"r@4:0": False, "b@4:0": False}
        )
        self.cache.dec_lock_ref.assert_called_once_with(7, "lock")
        self.assertFalse(self.counter.futures)
        self.process(a, b)
        self.assertTrue(a.linker_failed and b.linker_failed)
        for dr in (a, b):
            self.assertIn("OceanDisk GET failed", dr.kv_receiver.kv_mgr.failure_records[7])
        self.assertEqual(a.linker_loaded_len, 16)
        self.assertEqual(len(self.batches), 1)

    def test_bypass_clears_stale_storage_hit(self):
        dr = self.request()
        self.linker.hit_markers[dr.req.rid] = object()
        result = object()
        self.assertIs(self.linker.match(None, dr.req, result), result)
        self.assertFalse(self.linker.hit_markers)
        self.backend.lookup.assert_not_called()

    def _preallocate_request(self, *, fake=False, l1=4):
        dr = self.request(prefix=0, l1=l1)
        dr.linker_loaded_len = 0
        req = dr.req
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST if fake else "127.0.0.1"
        req.finished_reason, req.output_ids = None, []
        req.sampling_params = SimpleNamespace(max_new_tokens=16)
        req.set_extend_range = Mock()
        req.time_stats.set_decode_transfer_queue_entry_time = Mock()
        dr.waiting_for_input = True
        pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 32), dtype=torch.int64),
            available_size=lambda: 10,
            alloc=lambda reqs: [0],
        )
        pool.write = lambda location, indices: pool.req_to_token.__setitem__(
            location, indices
        )
        pm = SimpleNamespace(
            prefix_indices=torch.arange(100, 100 + l1),
            l1_prefix_len=l1,
            decode_prefix_len=l1,
            restore_token_count=0,
        )
        allocator = SimpleNamespace(
            page_size=4, translate_kv_indices_for_transfer=lambda x: x
        )
        harness = SimpleNamespace(
            queue=[dr],
            pending_reqs=[],
            pp_size=1,
            _num_published_destinations=0,
            _resolve_pending_reqs=Mock(),
            _update_handshake_waiters=Mock(),
            _uses_swa_tail_prealloc=lambda: False,
            _swa_tail_len=lambda fill: 0,
            _allocatable_token_budgets=lambda **kw: 4096,
            _hicache_pending_restore_tokens=lambda: 0,
            _rebootstrap_prefill_len=lambda r: len(r.origin_input_ids),
            _match_prefix_and_lock=lambda r: pm,
            _pre_alloc_fill_len=lambda r: len(r.origin_input_ids),
            _required_alloc_tokens=lambda fill_len, prefix_len: fill_len - prefix_len,
            _radix_full_available=lambda: 4096,
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(reqs=[]),
                enable_priority_scheduling=False,
                enable_hisparse=False,
                enable_lora=False,
                enable_decode_hicache=False,
                chunked_prefill_size=8,
            ),
            tree_cache=self.cache,
            num_reserved_decode_tokens=0,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=SimpleNamespace(page_size=4),
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.DSA])
            ),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                available_size=lambda: 10, alloc=lambda: 0
            ),
            transfer_queue=SimpleNamespace(enable_staging=False),
        )
        harness._pre_alloc = lambda *args: decode.DecodePreallocQueue._pre_alloc(
            harness, *args
        )
        if fake:
            dr.kv_receiver = FakeKVReceiver(None, FAKE_BOOTSTRAP_HOST)
            dr.kv_receiver.init(0)
            dr.kv_receiver.send_metadata = Mock(wraps=dr.kv_receiver.send_metadata)
        else:
            dr.kv_receiver.send_metadata = Mock()
        cfg = SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_linker_notify=True,
        )
        with (
            patch.object(decode, "get_disagg", return_value=cfg),
            patch.object(
                decode,
                "alloc_for_decode_prealloc",
                return_value=torch.arange(100 + l1, 129),
            ) as allocate,
        ):
            ready, failed = decode.DecodePreallocQueue.pop_preallocated(harness)
        self.assertEqual((len(ready), len(failed)), (1, 0))
        return dr, allocate

    def test_prealloc_allocates_from_l1_but_rdma_starts_at_s(self):
        dr, allocate = self._preallocate_request()
        req = dr.req
        self.assertEqual(allocate.call_args.kwargs["total_prefix_len"], 4)
        self.assertEqual(allocate.call_args.kwargs["delta_len"], 25)
        self.assertEqual(req.kv.cache_protected_len, 4)
        self.assertEqual(req.prefix_indices.tolist(), [100, 101, 102, 103])
        self.assertEqual(dr.linker_prefix_len, 20)
        args, kwargs = dr.kv_receiver.send_metadata.call_args
        self.assertEqual(kwargs["decode_prefix_len"], 20)
        self.assertEqual(kwargs["linker_start_page"], 1)
        self.assertEqual(args[0].tolist(), [30, 31, 32])
        self.assertEqual(args[2][0].tolist(), list(range(25, 33)))

    def test_fake_warmup_skips_notify_metadata_and_loads(self):
        with envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS.override(0):
            for l1 in (0, 4):
                with self.subTest(l1=l1):
                    dr, allocate = self._preallocate_request(fake=True, l1=l1)
                    self.assertEqual(allocate.call_args.kwargs["total_prefix_len"], l1)
                    self.assertEqual(dr.req.kv.cache_protected_len, l1)
                    self.assertEqual(dr.linker_prefix_len, 0)
                    self.assertEqual(dr.linker_loaded_len, 0)
                    args, kwargs = dr.kv_receiver.send_metadata.call_args
                    self.assertEqual(kwargs, {"decode_prefix_len": l1})
                    self.assertEqual(args[0].tolist(), list(range(25 + l1 // 4, 33)))
                    self.process(dr)
                    self.backend.load.assert_not_called()
                    self.cache._all_reduce_attn_groups.assert_not_called()
                    self.assertEqual(
                        LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Success
                    )

    def test_dp_groups_progress_independently_with_different_queue_lengths(self):
        peer = TestLinkerDecodeLoad()
        peer.setUp()
        self.addCleanup(peer.doCleanups)
        a = self.request("a", l1=0, prefix=8)
        b = self.request("b", l1=4, prefix=12, slot=1)
        c = peer.request("c", l1=0, prefix=16)
        c.kv_receiver.linker_ready_until.return_value = 0
        self.process(a, b)
        peer.process(c)
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(len(peer.batches), 0)
        # Two columns per request: all-layer end page and window due.
        self.assertEqual(self.cache._all_reduce_attn_groups.call_args.args[0].numel(), 4)
        self.assertEqual(peer.cache._all_reduce_attn_groups.call_args.args[0].numel(), 2)
        self.process(a, b)  # b exceeded the batch cap and goes one round later
        self.complete_batch()
        self.complete_batch(1)
        self.process(a, b)
        self.assertEqual((a.linker_loaded_len, b.linker_loaded_len), (8, 12))
        self.assertEqual(LinkerLoadGatedKVReceiver(a).poll(), KVPoll.Success)
        self.assertEqual(c.linker_loaded_len, 0)
        c.kv_receiver.linker_ready_until.return_value = 100
        peer.process(c)
        peer.complete_batch(success=False)
        peer.process(c)
        self.assertTrue(c.linker_failed)
        self.assertFalse(a.linker_failed)
        self.assertEqual(LinkerLoadGatedKVReceiver(a).poll(), KVPoll.Success)

    def test_cache_consensus_stays_inside_each_attention_group(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        outer_tp = object()
        attention_groups = [object(), object()]
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
        ):
            for group, size in zip(attention_groups, (2, 1)):
                cache = SimpleNamespace(
                    attn_cp_group=None, attn_tp_group=group,
                    tp_group=outer_tp, tp_world_size=4,
                )
                UnifiedRadixCache._all_reduce_attn_groups(
                    cache, torch.ones(size), torch.distributed.ReduceOp.MIN
                )
        self.assertEqual([call.kwargs["group"] for call in reduce.call_args_list], attention_groups)

    def test_configuration_rejects_out_of_scope_combinations(self):
        defaults = dict(
            disaggregation_linker_notify=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="oceandisk",
            disaggregation_decode_enable_radix_cache=True,
            pp_size=1,
            dp_size=1,
            tp_size=4,
            enable_dp_attention=False,
            enable_hierarchical_cache=False,
            disaggregation_decode_retraction_backup=None,
            dcp_size=1,
            enable_hisparse=False,
            speculative_algorithm=None,
            attention_backend="dsa",
            enable_dsa_cache_layer_split=False,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_extra_slots=0,
        )
        cases = dict(
            disaggregation_mode="null",
            enable_dsa_cache_layer_split=True,
            disaggregation_decode_enable_offload_kvcache=True,
            disaggregation_transfer_backend="nixl",
            enable_unified_cache_external_linker=False,
            unified_cache_external_linker_backend="mooncake",
            disaggregation_decode_enable_radix_cache=False,
            pp_size=2,
            dp_size=2,
            enable_hierarchical_cache=True,
            disaggregation_decode_retraction_backup="host_pool",
        )
        for field, value in [(None, None), *cases.items()]:
            cfg = SimpleNamespace(**(defaults | ({field: value} if field else {})))
            with (
                self.subTest(field=field),
                patch.object(
                    pd_disaggregation_hook, "resolving_view", return_value=cfg
                ),
                patch.object(pd_disaggregation_hook, "resolved_view", return_value=cfg),
                patch.object(pd_disaggregation_hook, "declare_resolution"),
            ):
                if field:
                    with self.assertRaises(ValueError):
                        pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                else:
                    pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    cfg.enable_dp_attention = True
                    for tp, dp in ((4, 1), (4, 2), (8, 4)):
                        cfg.tp_size, cfg.dp_size = tp, dp
                        pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    for tp, dp in ((4, 3), (4, 0), (4, 8)):
                        cfg.tp_size, cfg.dp_size = tp, dp
                        with self.assertRaisesRegex(ValueError, "decode DP attention"):
                            pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    cfg.tp_size, cfg.dp_size = 4, 2
                    cfg.disaggregation_mode = "prefill"
                    with self.assertRaisesRegex(ValueError, "prefill dp-size=1"):
                        pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    cfg.disaggregation_mode, cfg.dp_size = "decode", 1
                    for env, value in [
                        (envs.SGLANG_DISAGG_STAGING_BUFFER, True),
                        (envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER, True),
                        (envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS, -2),
                        (envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS, 0),
                    ]:
                        with env.override(value), self.assertRaises(ValueError):
                            pd_disaggregation_hook.handle_pd_disaggregation(cfg)

                    cfg.disaggregation_mode = "prefill"
                    pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    cfg.enable_prefill_cp = False
                    with self.assertRaisesRegex(ValueError, "prefill interleave CP"):
                        pd_disaggregation_hook.handle_pd_disaggregation(cfg)
                    cfg.enable_prefill_cp = True
                    cfg.cp_strategy = "zigzag"
                    with self.assertRaisesRegex(ValueError, "prefill interleave CP"):
                        pd_disaggregation_hook.handle_pd_disaggregation(cfg)

    def test_check_script_requires_completed_load_for_exact_rid(self):
        path = (
            Path(__file__).resolve().parents[4]
            / "examples/runtime/engine/oceandisk_pd_tp4.py"
        )
        spec = importlib.util.spec_from_file_location("pd_check", path)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(path.parent), *sys.path]):
            spec.loader.exec_module(module)
        result = {
            "output": {"meta_info": {"id": "r", "cached_tokens_details": {"device": 4}}}
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "decode.log"
            log.write_text(
                '[OceanDisk trace] event=LOAD_QUEUED pid=1 {"rid":"r@1:0"}\n'
                '[OceanDisk trace] event=LOAD_DONE pid=1 {"rids":["rr@1"]}\n',
                encoding="utf-8",
            )
            self.assertEqual(module.cache_tiers(result, log), (4, False))
            with log.open("a", encoding="utf-8") as output:
                output.write(
                    "[OceanDisk trace] event=LOAD_DONE pid=1 "
                    + json.dumps({"rids": ["r@1:0", "b@1:0"]})
                    + "\n"
                )
            self.assertEqual(module.cache_tiers(result, log), (4, True))
            module.require_source(
                "mixed", result, log, expect_load=True, expect_hbm=True
            )
            with self.assertRaises(RuntimeError):
                module.require_source(
                    "repeat", result, log, expect_load=False, expect_hbm=True
                )


if __name__ == "__main__":
    unittest.main()
