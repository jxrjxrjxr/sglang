"""CPU regressions for storage-backed PD admission and decode scheduling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
    _linker_handoff_prefix_len,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_req(rid="req", *, host_hit=0, output_ids=None):
    req = SimpleNamespace(
        rid=rid,
        bootstrap_host="prefill",
        origin_input_ids=list(range(7)),
        output_ids=[] if output_ids is None else output_ids,
        return_logprob=False,
        logprob_start_len=-1,
        prefix_indices=torch.tensor([10, 11]),
        host_hit_length=host_hit,
        host_loaded_length=0,
        last_node="resident",
        best_match_node="stored",
        lock_receipt="resident-lock",
        swa_prefix_lock_released=False,
        kv=SimpleNamespace(req_pool_idx=1, kv_committed_len=7),
        time_stats=MagicMock(),
        init_next_round_input=MagicMock(),
        set_extend_range=MagicMock(),
    )
    req._compute_max_prefix_len = lambda length: Req._compute_max_prefix_len(
        req, length
    )
    req.needs_host_load_back = lambda: req.host_hit_length > 0
    return req


def make_scheduler():
    scheduler = SchedulerDisaggregationDecodeMixin()
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.has_waiting_grammars.return_value = False
    scheduler.waiting_queue = []
    scheduler.enable_priority_scheduling = False
    scheduler.req_to_token_pool = SimpleNamespace(size=4)
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.tree_cache = MagicMock()
    scheduler.model_config = MagicMock()
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = MagicMock()
    scheduler.max_running_requests = 4
    scheduler.future_map = MagicMock()
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.enable_hisparse = False
    scheduler.enable_decode_hicache = False
    scheduler.enable_unified_cache_external_linker = True
    scheduler.batch_result_processor = MagicMock()
    scheduler.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch: batch
    )
    scheduler.scheduler_stage_metrics = None
    return scheduler


class TestLinkerDecode(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            disaggregation_transfer_backend="linker",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_decode_enable_offload_kvcache=False,
        )
        for module in (
            decode,
            "sglang.srt.disaggregation.utils",
            "sglang.srt.disaggregation.decode_hicache_mixin",
        ):
            target = module if isinstance(module, str) else module.__name__
            patcher = patch(target + ".get_disagg", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_prefix_cap_keeps_one_token_and_honors_logprob(self):
        req = make_req()
        self.assertEqual(_linker_handoff_prefix_len(req), 6)
        req.return_logprob = True
        req.logprob_start_len = 3
        self.assertEqual(_linker_handoff_prefix_len(req), 3)
        req.origin_input_ids = []
        self.assertEqual(_linker_handoff_prefix_len(req), 0)

    def test_add_hashes_once_and_probes_all_pages_without_dp_lookup(self):
        req = make_req()
        receiver = MagicMock()
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
        queue._create_receiver_and_enqueue = MagicMock(
            return_value=SimpleNamespace(kv_receiver=receiver)
        )
        queue._resolve_prefill_dp_rank = MagicMock(side_effect=AssertionError)
        queue.tree_cache = MagicMock(page_size=2)
        linker = queue.tree_cache.linker
        transfers = [object()]
        linker.build_handoff_lookup.return_value = transfers
        linker.cache_linker.lookup.side_effect = [[1, 2], [1, 2, 3]]
        queue.add(req)
        receiver.init.assert_called_once_with(0)
        probe = receiver.set_ready_probe.call_args.args[0]
        self.assertFalse(probe())
        self.assertTrue(probe())
        linker.build_handoff_lookup.assert_called_once_with(req, 3)
        self.assertEqual(linker.cache_linker.lookup.call_count, 2)
        linker.cache_linker.lookup.assert_called_with(req.rid, transfers)
        queue._resolve_prefill_dp_rank.assert_not_called()

    def test_short_prompt_needs_no_storage_probe(self):
        req = make_req()
        req.origin_input_ids = [1]
        receiver = MagicMock()
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._check_if_req_exceed_kv_capacity = lambda _: False
        queue._create_receiver_and_enqueue = lambda *a, **kw: SimpleNamespace(
            kv_receiver=receiver
        )
        queue.tree_cache = MagicMock(page_size=2)
        queue.add(req)
        self.assertTrue(receiver.set_ready_probe.call_args.args[0]())
        queue.tree_cache.linker.cache_linker.lookup.assert_not_called()

    def test_prefix_match_caps_only_linker(self):
        req = make_req()
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tree_cache = MagicMock()
        queue._build_decode_prefix_match = MagicMock()
        with patch.object(decode, "match_prefix_for_req") as match:
            queue._match_prefix_and_lock(req)
            self.assertEqual(match.call_args.args[2], list(range(6)))
            self.config.disaggregation_transfer_backend = "mooncake"
            queue._match_prefix_and_lock(req)
            self.assertEqual(match.call_args.args[2], list(range(7)))

    def test_commit_skips_aux_and_keeps_output_empty(self):
        req = make_req()
        receiver = MagicMock()
        dr = SimpleNamespace(req=req, kv_receiver=receiver)
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.metadata_buffers = MagicMock()
        queue._commit_transfer_to_req(dr)
        receiver.clear.assert_called_once_with()
        self.assertIsNone(dr.kv_receiver)
        self.assertEqual(req.output_ids, [])
        req.time_stats.set_wait_queue_entry_time.assert_called_once_with()
        queue.metadata_buffers.get_buf.assert_not_called()

    def test_waiting_loads_reserve_only_unmaterialized_hits(self):
        pending = make_req("pending", host_hit=4)
        loaded = make_req("loaded", host_hit=6)
        queue = DecodeHiCachePreallocMixin()
        queue.scheduler = SimpleNamespace(
            waiting_queue=[pending, loaded], enable_decode_hicache=False
        )
        queue.tree_cache = SimpleNamespace(
            linker=SimpleNamespace(has_hit=lambda rid: rid == "pending")
        )
        self.assertEqual(queue._hicache_pending_restore_tokens(), 4)
        self.config.disaggregation_transfer_backend = "mooncake"
        self.assertEqual(queue._hicache_pending_restore_tokens(), 0)

    def test_hicache_pending_reservations_unchanged(self):
        self.config.disaggregation_transfer_backend = "mooncake"
        queue = DecodeHiCachePreallocMixin()
        queue.scheduler = SimpleNamespace(enable_decode_hicache=True)
        queue.transfer_queue = SimpleNamespace(
            queue=[
                SimpleNamespace(
                    prefix_match=SimpleNamespace(restore_token_count=5),
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_restored_node=None,
                )
            ]
        )
        self.assertEqual(queue._hicache_pending_restore_tokens(), 5)

    def test_extend_rejects_missing_or_partial_load_back(self):
        for loaded, actual in (
            (None, "None"),
            ((torch.tensor([20, 21]), "restored"), "2"),
        ):
            with self.subTest(loaded=actual):
                scheduler = make_scheduler()
                req = make_req(host_hit=4)
                scheduler.tree_cache.init_load_back.return_value = loaded
                with self.assertRaisesRegex(
                    RuntimeError, f"req=req promised=4 loaded={actual}"
                ):
                    scheduler._build_linker_extend_batch([req])
                scheduler.tree_cache.inc_lock_ref.assert_not_called()
                scheduler.tree_cache.dec_lock_ref.assert_not_called()
                scheduler.tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_extend_installs_prefix_and_starts_one_batch_load(self):
        scheduler = make_scheduler()
        reqs = [make_req(str(i), host_hit=4) for i in range(2)]
        events = []
        locks = {"resident": 2, "restored": 0}
        table = torch.full((3, 7), -1, dtype=torch.int64)
        table[1:, :2] = torch.tensor([10, 11])
        table[1:, 6] = 90
        for i, req in enumerate(reqs, 1):
            req.kv.req_pool_idx = i

        def load(params):
            events.append("queue:" + params.req.rid)
            return torch.tensor([20, 21, 22, 23]), "restored"

        def inc(node):
            locks[node] += 1
            return SimpleNamespace(to_dec_params=lambda: "restored-lock")

        def dec(node, receipt, **kwargs):
            self.assertEqual(receipt, "resident-lock")
            locks[node] -= 1

        scheduler.tree_cache.init_load_back.side_effect = load
        scheduler.tree_cache.inc_lock_ref.side_effect = inc
        scheduler.tree_cache.dec_lock_ref.side_effect = dec
        scheduler.tree_cache.req_to_token_pool.write.side_effect = (
            lambda index, values: table.__setitem__(index, values)
        )
        scheduler.tree_cache.ready_to_load_host_cache.side_effect = (
            lambda: events.append("start") or 2
        )
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = lambda: events.append("prepare")
        with patch.object(decode.ScheduleBatch, "init_new", return_value=batch):
            self.assertIs(scheduler._build_linker_extend_batch(reqs), batch)
        self.assertEqual(events, ["queue:0", "queue:1", "start", "prepare"])
        self.assertEqual(locks, {"resident": 0, "restored": 2})
        self.assertEqual(batch.hicache_consumer_index, 2)
        for req in reqs:
            self.assertEqual(req.host_loaded_length, 4)
            self.assertEqual(req.prefix_indices.tolist(), [10, 11, 20, 21, 22, 23])
            self.assertEqual(
                table[req.kv.req_pool_idx].tolist(), [10, 11, 20, 21, 22, 23, 90]
            )
            req.set_extend_range.assert_called_once_with(6, 7)
            self.assertEqual(req.lock_receipt, "restored-lock")

    def test_hicache_restore_uses_same_lock_handoff(self):
        req = make_req(host_hit=4)
        req.swa_prefix_lock_released = True
        pm = DecodePrefixMatch(req.prefix_indices, 4, 0, "resident")
        dr = SimpleNamespace(
            req=req,
            prefix_match=pm,
            hicache_restored_node="restored",
            hicache_restore_lock_receipt="restored-lock",
            hicache_restored_kv_indices=torch.tensor([20, 21, 22, 23]),
        )
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = MagicMock()
        queue._commit_hicache_local_restore_to_req(dr)
        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            "resident", "resident-lock", skip_swa=True
        )
        self.assertEqual(req.prefix_indices.tolist(), [10, 11, 20, 21, 22, 23])
        self.assertEqual(req.last_node, "restored")
        self.assertEqual(req.lock_receipt, "restored-lock")
        self.assertFalse(req.swa_prefix_lock_released)
        self.assertIsNone(dr.hicache_restored_node)
        self.assertIsNone(dr.hicache_restore_lock_receipt)

    def test_shared_capacity_gate_splits_fresh_and_resumed_requests(self):
        scheduler = make_scheduler()
        fresh = make_req("fresh")
        resumed = make_req("resumed", output_ids=[8])
        waiting = make_req("waiting")
        scheduler.waiting_queue = [fresh, resumed, waiting]
        scheduler.max_running_requests = 2
        running = MagicMock()
        running.batch_size.return_value = 0
        extend, prebuilt = MagicMock(), MagicMock()
        scheduler._build_linker_extend_batch = MagicMock(return_value=extend)
        with patch.object(
            decode.ScheduleBatch, "init_new", return_value=prebuilt
        ) as init:
            self.assertEqual(
                scheduler.get_new_prebuilt_batch(running), (prebuilt, extend)
            )
        scheduler._build_linker_extend_batch.assert_called_once_with([fresh])
        self.assertEqual(init.call_args.args[0], [resumed])
        self.assertEqual(scheduler.waiting_queue, [waiting])
        prebuilt.prepare_for_prebuilt.assert_called_once_with()
        prebuilt.process_prebuilt.assert_called_once_with(scheduler.future_map)

    def test_other_backend_uses_prebuilt_even_without_output(self):
        self.config.disaggregation_transfer_backend = "mooncake"
        scheduler = make_scheduler()
        req = make_req()
        scheduler.waiting_queue = [req]
        running = MagicMock()
        running.batch_size.return_value = 0
        batch = MagicMock()
        scheduler._build_linker_extend_batch = MagicMock()
        with patch.object(decode.ScheduleBatch, "init_new", return_value=batch):
            self.assertEqual(scheduler.get_new_prebuilt_batch(running), (batch, None))
        scheduler._build_linker_extend_batch.assert_not_called()

    def test_empty_and_full_capacity_return_two_empty_batches(self):
        scheduler = make_scheduler()
        running = MagicMock()
        self.assertEqual(scheduler.get_new_prebuilt_batch(running), (None, None))
        scheduler.waiting_queue = [make_req()]
        running.batch_size.return_value = scheduler.max_running_requests
        self.assertEqual(scheduler.get_new_prebuilt_batch(running), (None, None))
        self.assertEqual(len(scheduler.waiting_queue), 1)

    def test_extend_has_priority_without_advancing_decode(self):
        scheduler = make_scheduler()
        running, extend = MagicMock(), MagicMock()
        scheduler.get_new_prebuilt_batch = MagicMock(return_value=(None, extend))
        plan = scheduler.get_next_disagg_decode_batch_to_run(running)
        self.assertIs(plan.batch_to_run, extend)
        self.assertIs(plan.running_batch, running)
        scheduler.update_running_batch.assert_not_called()

    def test_previous_extend_is_filtered_and_merged_before_selection(self):
        scheduler = make_scheduler()
        previous, running = MagicMock(), MagicMock()
        previous.forward_mode.is_extend.return_value = True
        previous.is_empty.return_value = False
        previous.batch_size.side_effect = [2, 1]
        running.is_empty.return_value = False
        scheduler.last_batch = previous
        scheduler.get_new_prebuilt_batch = MagicMock(return_value=(None, None))
        scheduler.get_next_disagg_decode_batch_to_run(running)
        previous.filter_batch.assert_called_once_with()
        running.merge_batch.assert_called_once_with(previous)
        self.assertFalse(running.batch_is_full)
        scheduler.update_running_batch.assert_called_once_with(running)

    def test_load_events_are_drained_before_retraction_early_return(self):
        scheduler = make_scheduler()
        scheduler.disagg_decode_transfer_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue.resume_retracted_reqs.return_value = []
        scheduler.disagg_decode_prealloc_queue.retracted_queue = [make_req()]
        scheduler.process_decode_queue()
        scheduler.tree_cache.check_hicache_events.assert_called_once_with()
        scheduler.disagg_decode_prealloc_queue.pop_preallocated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
