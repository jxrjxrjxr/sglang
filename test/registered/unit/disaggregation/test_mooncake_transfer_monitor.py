import logging
import os
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.mooncake.transfer_monitor import TransferMonitor
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.mooncake.transfer_monitor"


class StopWorker(BaseException):
    pass


class TestTransferMonitor(unittest.TestCase):
    def setUp(self):
        self.clock = patch(f"{MODULE}.time.monotonic", return_value=0.0).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def chunk(room=1):
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([1, 2]),
            index_slice=slice(0, 2),
            is_last_chunk=True,
            prefill_aux_index=0,
            state_indices=[[1, 2]],
        )

    def test_defaults_are_disabled(self):
        for setting in (
            envs.SGLANG_MOONCAKE_TRANSFER_TRACE,
            envs.SGLANG_MOONCAKE_TRANSFER_WARN_SECONDS,
            envs.SGLANG_MOONCAKE_TRANSFER_TIMEOUT_SECONDS,
            envs.SGLANG_MOONCAKE_TRANSFER_FAIL_FAST,
        ):
            with patch.dict(os.environ):
                os.environ.pop(setting.name, None)
                self.assertFalse(setting.get())

    def test_trace_identifies_queue_destination_and_state_pages(self):
        monitor = TransferMonitor(2, True, 0, 0, False)
        chunk = self.chunk()
        with self.assertLogs(MODULE, logging.INFO) as logs:
            monitor.start(chunk, "QUEUED")
            self.clock.return_value = 2
            monitor.start(chunk, "STATE/dsa", "peer:1234", 2000)
            self.clock.return_value = 3
            monitor.end(chunk, 0)
        self.assertEqual(len(logs.output), 4)
        self.assertIn("phase=QUEUED", logs.output[0])
        self.assertIn("elapsed_ms=2000.000", logs.output[1])
        self.assertIn("rank=2 phase=STATE/dsa dst=peer:1234", logs.output[2])
        self.assertIn("pages=2000", logs.output[2])
        self.assertIn("elapsed_ms=1000.000 rc=0", logs.output[3])
        self.assertFalse(monitor._stages)

    def test_later_chunk_does_not_reset_active_deadline(self):
        monitor = TransferMonitor(0, False, 0, 10, True)
        monitor.start(self.chunk(), "KV")
        self.clock.return_value = 9
        monitor.start(self.chunk(), "QUEUED")
        self.clock.return_value = 10
        with self.assertLogs(MODULE, logging.ERROR):
            with self.assertRaisesRegex(RuntimeError, "phase=KV.*TIMEOUT"):
                monitor.check()

    def test_phase_change_resets_deadline_and_completion_removes_it(self):
        monitor = TransferMonitor(0, False, 0, 10, True)
        chunk = self.chunk()
        monitor.start(chunk, "KV")
        self.clock.return_value = 9
        monitor.start(chunk, "STATE/dsa")
        self.clock.return_value = 10
        monitor.check()
        monitor.end(chunk, 0)
        self.clock.return_value = 100
        monitor.check()
        self.assertFalse(monitor._stages)

    def test_warning_and_timeout_report_once_without_cancelling(self):
        monitor = TransferMonitor(0, False, 3, 10, False)
        chunk = self.chunk()
        monitor.start(chunk, "AUX")
        with self.assertLogs(MODULE, logging.WARNING) as logs:
            for now in (3, 4, 10, 11, 100):
                self.clock.return_value = now
                monitor.check()
        self.assertEqual(len(logs.output), 2)
        self.assertIn("event=STALL", logs.output[0])
        self.assertIn("event=TIMEOUT", logs.output[1])
        self.assertIs(monitor._stages[id(chunk)].chunk, chunk)

    def test_trace_only_does_not_enable_deadlines(self):
        monitor = TransferMonitor(0, True, 0, 0, True)
        with self.assertLogs(MODULE, logging.INFO):
            monitor.start(self.chunk(), "KV")
        self.clock.return_value = 1000000
        with self.assertNoLogs(MODULE, logging.WARNING):
            monitor.check()

    def test_finished_stage_is_not_reported_from_an_old_snapshot(self):
        monitor = TransferMonitor(0, False, 0, 10, True)
        chunk = self.chunk()
        monitor.start(chunk, "KV")
        self.clock.return_value = 20
        # Completion wins the lock before the scheduler checks the deadline.
        thread = threading.Thread(target=monitor.end, args=(chunk, 0))
        thread.start()
        thread.join()
        with self.assertNoLogs(MODULE, logging.WARNING):
            monitor.check()

    def make_worker(self, monitor):
        manager = object.__new__(MooncakeKVManager)
        manager.transfer_monitor = monitor
        manager.enable_trace = manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager.bootstrap_port = 1234
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {1: KVPoll.WaitingForInput}
        manager.check_status = lambda room: manager.request_status[room]
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.session_failures = defaultdict(int)
        manager.req_to_decode_prefix_len = {}
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 4
        manager._prefill_unique_rank = lambda: 0
        manager._get_dsa_cache_transfer_skip_flags = lambda info: (False, False)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100],
            state_types=[StateType.DSA],
            state_data_ptrs=[[100]],
            state_item_lens=[[8]],
            state_dim_per_tensor=[[]],
            state_layer_ids=[[]],
        )
        request = SimpleNamespace(
            room=1,
            is_dummy=False,
            mooncake_session_id="peer:1234",
            dst_kv_indices=np.array([10, 11]),
            dst_device_kv_indices=None,
            dst_state_indices=[[10, 11]],
            endpoint="peer",
            dst_port=1234,
            required_dst_info_num=1,
        )
        manager.transfer_infos = {1: {"peer:1234": request}}
        manager.decode_kv_args_table = {
            "peer:1234": SimpleNamespace(
                requires_dcp_relayout=False,
                dst_kv_ptrs=[200],
                dst_kv_layer_ids=[],
                dst_kv_item_len=8,
                dst_attn_tp_size=4,
                dst_aux_ptrs=[300],
                dst_state_data_ptrs=[[200]],
                dst_state_item_lens=[[8]],
                dst_state_dim_per_tensor=[[]],
                dst_state_layer_ids=[[]],
            )
        }
        manager.send_kvcache = MagicMock(return_value=0)
        manager._send_kvcache_generic = MagicMock(return_value=0)
        manager.send_aux = MagicMock(return_value=0)
        manager.conclude_transfer = MagicMock(
            side_effect=lambda **kw: manager.request_status.update({1: kw["status"]})
        )
        manager.conclude_failure = MagicMock(
            side_effect=lambda **kw: manager.request_status.update({1: KVPoll.Failed})
        )
        patch(
            "sglang.srt.disaggregation.mooncake.conn.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=False),
        ).start()
        return manager

    def run_worker(self, manager, chunk):
        if manager.transfer_monitor is not None:
            manager.transfer_monitor.start(chunk, "QUEUED")
        queue = MagicMock()
        queue.get.side_effect = [chunk, StopWorker()]
        with self.assertRaises(StopWorker):
            manager.transfer_worker(queue, MagicMock())

    def test_real_worker_records_kv_indexer_aux_and_conclude_in_order(self):
        monitor = TransferMonitor(0, True, 0, 0, False)
        manager = self.make_worker(monitor)
        with self.assertLogs(MODULE, logging.INFO) as logs:
            self.run_worker(manager, self.chunk())
        begins = [
            line.split("phase=")[1].split()[0]
            for line in logs.output
            if "event=BEGIN" in line
        ]
        self.assertEqual(
            begins, ["QUEUED", "DEQUEUE", "KV", "STATE/dsa", "AUX", "CONCLUDE"]
        )
        manager.send_kvcache.assert_called_once()
        manager._send_kvcache_generic.assert_called_once()
        manager.send_aux.assert_called_once()
        self.assertEqual(manager.request_status[1], KVPoll.Success)
        self.assertFalse(monitor._stages)

    def test_disabled_monitor_does_not_read_clock_or_change_transfer(self):
        manager = self.make_worker(None)
        self.clock.side_effect = AssertionError("disabled monitor read the clock")
        with self.assertNoLogs(MODULE, logging.INFO):
            self.run_worker(manager, self.chunk())
        self.assertEqual(manager.request_status[1], KVPoll.Success)
        manager.send_aux.assert_called_once()

    def test_worker_exception_reaches_scheduler_without_releasing_room(self):
        monitor = TransferMonitor(0, False, 0, 0, True)
        manager = self.make_worker(monitor)
        manager._send_kvcache_generic.side_effect = RuntimeError(
            "injected indexer failure"
        )
        queue = MagicMock()
        queue.get.return_value = self.chunk()
        with self.assertLogs(MODULE, logging.ERROR) as logs:
            with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
                manager.transfer_worker(queue, MagicMock())
        self.assertIn("phase=STATE/dsa", logs.output[0])
        sender = SimpleNamespace(kv_mgr=manager, conclude_state=None, bootstrap_room=1)
        with self.assertRaisesRegex(
            RuntimeError, "injected indexer failure.*stopping prefill"
        ):
            MooncakeKVSender.poll(sender)
        self.assertEqual(manager.request_status[1], KVPoll.WaitingForInput)
        self.assertEqual(manager._staging_outstanding[1], 1)
        manager.conclude_failure.assert_not_called()
        manager.send_aux.assert_not_called()

    def test_scheduler_times_out_while_each_transfer_stage_is_blocked(self):
        for method, phase in (
            ("send_kvcache", "KV"),
            ("_send_kvcache_generic", "STATE/dsa"),
            ("send_aux", "AUX"),
        ):
            with self.subTest(phase=phase):
                self.clock.return_value = 0
                monitor = TransferMonitor(0, False, 0, 10, True)
                manager = self.make_worker(monitor)
                entered, release = threading.Event(), threading.Event()
                errors = []

                def blocked(*args, **kwargs):
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test did not release the blocked transfer")
                    return 0

                def worker():
                    try:
                        self.run_worker(manager, self.chunk())
                    except BaseException as error:
                        errors.append(error)

                getattr(manager, method).side_effect = blocked
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    self.assertTrue(entered.wait(2))
                    self.clock.return_value = 10
                    sender = SimpleNamespace(
                        kv_mgr=manager, conclude_state=None, bootstrap_room=1
                    )
                    with self.assertLogs(MODULE, logging.ERROR):
                        with self.assertRaisesRegex(
                            RuntimeError, f"phase={phase}.*TIMEOUT"
                        ):
                            MooncakeKVSender.poll(sender)
                    self.assertEqual(manager.request_status[1], KVPoll.WaitingForInput)
                    self.assertEqual(manager._staging_outstanding[1], 1)
                    manager.conclude_failure.assert_not_called()
                finally:
                    release.set()
                    thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
                self.assertFalse(monitor._stages)

    def test_nonzero_returns_keep_existing_failure_path(self):
        for stage in ("send_kvcache", "_send_kvcache_generic", "send_aux"):
            with self.subTest(stage=stage):
                monitor = TransferMonitor(0, False, 0, 10, True)
                manager = self.make_worker(monitor)
                getattr(manager, stage).return_value = -1
                self.run_worker(manager, self.chunk())
                self.assertEqual(manager.request_status[1], KVPoll.Failed)
                self.assertFalse(monitor._stages)
                if stage != "send_aux":
                    manager.send_aux.assert_not_called()


if __name__ == "__main__":
    unittest.main()
