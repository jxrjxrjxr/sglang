"""CPU coverage for the OceanDisk prefix / Mooncake tail handoff."""

import importlib.util
import inspect
import json
import sys
import tempfile
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
from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.decode_linker_mixin import (
    DecodeLinkerTransferMixin,
    LinkerLoadGatedKVReceiver,
    _linker_handoff_prefix_len,
)
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
        self.enterContext(envs.SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL.override(0))
        self.enterContext(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(300))
        self.enterContext(envs.SGLANG_DISAGG_STAGING_BUFFER.override(False))
        self.hash_pages = self.enterContext(
            patch(
                "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
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
            lookup=Mock(
                side_effect=lambda rid, transfers: list(
                    range(1, len(transfers[0].keys) + 1)
                )
            ),
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
        self.linker.handoff_lookups = {}
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
                poll=Mock(return_value=KVPoll.Success), abort_notified=False
            ),
            linker_prefix_len=prefix,
            linker_loaded_len=l1,
        )

    def start_batch(self):
        index = self.counter.update_producer()
        self.batches.append((index, list(self.pending)))
        self.pending.clear()
        return index

    def complete_batch(self, index=0, success=True):
        counter_index, tags = self.batches[index]
        if success:
            for layer in range(3):
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
                self.assertFalse(self.linker.handoff_lookups)

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
        waiting.linker_next_probe_time = time.perf_counter() + 10
        finished = self.request("finished")
        finished.linker_loaded_len = finished.linker_prefix_len
        failed = self.request("failed")
        failed.kv_receiver.poll.return_value = KVPoll.Failed
        self.linker.handoff_lookups = {
            dr.req.rid: [] for dr in (waiting, finished, failed)
        }
        self.process(waiting, finished, failed)
        self.cache._all_reduce_attn_groups.assert_called_once()
        self.backend.lookup.assert_not_called()
        self.backend.load.assert_not_called()
        self.assertTrue(failed.linker_failed)
        self.assertFalse(waiting.linker_failed or finished.linker_failed)
        self.assertEqual(set(self.linker.handoff_lookups), {"waiting"})

    def test_native_fallback_does_not_probe_or_add_timeout(self):
        dr = self.request(l1=24, prefix=24)
        dr.req.time_stats.decode_transfer_queue_entry_time = 0
        self.process(dr)
        self.backend.lookup.assert_not_called()
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Success)

    def test_windows_batch_and_preallocated_destinations(self):
        a, b = self.request(), self.request("b", slot=1)
        self.process(a, b)
        self.assertEqual(self.batches, [(0, ["r@1", "b@1"])])
        tag, transfers = self.backend.load.call_args_list[0].args
        self.assertEqual(tag, "r@1")
        self.assertEqual(transfers[0].device_indices.tolist(), list(range(4, 12)))
        full = self.linker.handoff_lookups["r"][0]
        self.assertEqual(transfers[0].keys, full.keys[1:3])
        self.assertIsNone(full.device_indices)
        self.process(a, b)
        self.assertEqual(len(self.batches), 1)
        self.complete_batch()
        self.assertFalse(self.counter.futures)
        self.process(a, b)
        self.assertEqual((a.linker_loaded_len, a.linker_inflight_end), (12, 20))
        self.assertEqual(self.batches[-1][1], ["r@3", "b@3"])
        self.complete_batch(1)
        self.process(a, b)
        self.assertEqual(a.linker_inflight_end, 24)
        self.complete_batch(2)
        self.process(a, b)
        self.assertEqual(a.linker_loaded_len, 24)
        self.assertFalse(self.linker.completed_handoff_loads)
        self.assertFalse(self.linker.pending_handoff_loads)
        self.assertFalse(self.linker.handoff_lookups)
        self.assertEqual(LinkerLoadGatedKVReceiver(a).poll(), KVPoll.Success)

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
            disaggregation_decode_linker_load=True,
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
                return_value=SimpleNamespace(disaggregation_decode_linker_load=True),
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

    def test_partial_visibility_and_probe_interval(self):
        dr = self.request()
        self.backend.lookup.return_value = [1]
        self.backend.lookup.side_effect = None
        with envs.SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL.override(10):
            self.process(dr)
            self.process(dr)
        self.backend.lookup.assert_called_once()
        self.backend.load.assert_not_called()
        self.assertEqual(self.cache._all_reduce_attn_groups.call_count, 3)

    def test_tp_peer_not_due_or_not_visible_prevents_load(self):
        for collective in (0, 1):
            with self.subTest(collective=collective):
                self.cache._all_reduce_attn_groups.reset_mock()
                calls = iter(range(2))
                self.cache._all_reduce_attn_groups.side_effect = lambda states, op: (
                    states.zero_() if next(calls) == collective else None
                )
                self.process(self.request())
                self.backend.load.assert_not_called()
                self.assertEqual(
                    self.cache._all_reduce_attn_groups.call_count, collective + 1
                )

    def test_tp_peer_failure_is_sticky(self):
        dr = self.request()
        self.cache._all_reduce_attn_groups.side_effect = lambda states, op: (
            states.fill_(-1)
        )
        self.process(dr)
        self.assertTrue(dr.linker_failed)
        self.backend.lookup.assert_not_called()
        self.assertEqual(LinkerLoadGatedKVReceiver(dr).poll(), KVPoll.Failed)

    def test_lookup_exception_becomes_request_failure(self):
        dr = self.request()
        self.backend.lookup.side_effect = RuntimeError("lookup failed")
        with self.assertLogs(
            "sglang.srt.disaggregation.decode_linker_mixin", level="ERROR"
        ):
            self.process(dr)
        self.assertTrue(dr.linker_failed)
        self.backend.load.assert_not_called()

    def test_shared_counter_failure_and_normal_load_unlock(self):
        a, b = self.request(), self.request("b", slot=1)
        self.process(a, b)
        self.linker.pending_loads["tree"] = (7, "lock")
        self.batches[0][1].append("tree")
        self.complete_batch(success=False)
        self.assertEqual(
            self.linker.completed_handoff_loads, {"r@1": False, "b@1": False}
        )
        self.cache.dec_lock_ref.assert_called_once_with(7, "lock")
        self.assertFalse(self.counter.futures)
        self.process(a, b)
        self.assertTrue(a.linker_failed and b.linker_failed)
        self.assertEqual(a.linker_loaded_len, 4)
        self.assertEqual(len(self.batches), 1)

    def test_bypass_clears_stale_storage_hit(self):
        dr = self.request()
        self.linker.hit_markers[dr.req.rid] = object()
        result = object()
        self.assertIs(self.linker.match(None, dr.req, result), result)
        self.assertFalse(self.linker.hit_markers)
        self.backend.lookup.assert_not_called()

    def test_prealloc_allocates_from_l1_but_rdma_starts_at_s(self):
        dr = self.request(prefix=0)
        req = dr.req
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
            prefix_indices=torch.arange(100, 104),
            l1_prefix_len=4,
            decode_prefix_len=4,
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
        dr.kv_receiver.send_metadata = Mock()
        cfg = SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_decode_linker_load=True,
        )
        with (
            patch.object(decode, "get_disagg", return_value=cfg),
            patch.object(
                decode, "alloc_for_decode_prealloc", return_value=torch.arange(104, 129)
            ) as allocate,
        ):
            ready, failed = decode.DecodePreallocQueue.pop_preallocated(harness)
        self.assertEqual((len(ready), len(failed)), (1, 0))
        self.assertEqual(allocate.call_args.kwargs["total_prefix_len"], 4)
        self.assertEqual(allocate.call_args.kwargs["delta_len"], 25)
        self.assertEqual(req.kv.cache_protected_len, 4)
        self.assertEqual(req.prefix_indices.tolist(), [100, 101, 102, 103])
        self.assertEqual(dr.linker_prefix_len, 20)
        args, kwargs = dr.kv_receiver.send_metadata.call_args
        self.assertEqual(kwargs["decode_prefix_len"], 20)
        self.assertEqual(args[0].tolist(), [30, 31, 32])
        self.assertEqual(args[2][0].tolist(), list(range(25, 33)))

    def test_configuration_rejects_out_of_scope_combinations(self):
        defaults = dict(
            disaggregation_decode_linker_load=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="oceandisk",
            disaggregation_decode_enable_radix_cache=True,
            pp_size=1,
            enable_dp_attention=False,
            enable_hierarchical_cache=False,
            disaggregation_decode_retraction_backup=None,
            dcp_size=1,
            enable_hisparse=False,
            speculative_algorithm=None,
            disaggregation_decode_extra_slots=0,
        )
        cases = dict(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            enable_unified_cache_external_linker=False,
            unified_cache_external_linker_backend="mooncake",
            disaggregation_decode_enable_radix_cache=False,
            pp_size=2,
            enable_dp_attention=True,
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
                    for env, value in [
                        (envs.SGLANG_DISAGG_STAGING_BUFFER, True),
                        (envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS, -2),
                        (envs.SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL, -0.1),
                    ]:
                        with env.override(value), self.assertRaises(ValueError):
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
                '[OceanDisk trace] event=LOAD_QUEUED pid=1 {"rid":"r@1"}\n[OceanDisk trace] event=LOAD_DONE pid=1 {"rids":["rr@1"]}\n',
                encoding="utf-8",
            )
            self.assertEqual(module.cache_tiers(result, log), (4, False))
            with log.open("a", encoding="utf-8") as output:
                output.write(
                    "[OceanDisk trace] event=LOAD_DONE pid=1 "
                    + json.dumps({"rids": ["r@1", "b@1"]})
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
