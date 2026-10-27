import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.fake import FakeKVManager, FakeKVSender
from sglang.srt.disaggregation.linker import LinkerKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    KVClassType,
    TransferBackend,
    _apply_metadata_gate,
    _is_linker_handoff,
    get_kv_class,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLinkerKVReceiver(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "300",
                "SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL": "0.01",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        clock = patch(
            "sglang.srt.disaggregation.linker.conn.time.time", return_value=10.0
        )
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.receiver = LinkerKVReceiver(Mock(), "prefill:8998", bootstrap_room=42)
        self.probe = Mock(return_value=False)
        self.receiver.set_ready_probe(self.probe)

    def test_waits_for_init_and_storage_readiness(self):
        self.assertEqual(self.receiver.poll(), KVPoll.Bootstrapping)
        self.probe.assert_not_called()
        self.receiver.init(0)
        self.assertFalse(self.receiver.bootstrap_done)
        self.assertEqual(self.receiver.poll(), KVPoll.Bootstrapping)
        self.probe.assert_called_once_with()

    def test_ready_then_metadata_completes_without_reprobing(self):
        self.receiver.init(0)
        self.probe.return_value = True
        self.assertEqual(self.receiver.poll(), KVPoll.WaitingForInput)
        self.clock.return_value = 400.0
        self.assertEqual(self.receiver.poll(), KVPoll.WaitingForInput)
        self.receiver.send_metadata([])
        self.assertEqual(self.receiver.poll(), KVPoll.Success)
        self.assertEqual(self.receiver.poll(), KVPoll.Success)
        self.probe.assert_called_once_with()

    def test_probe_is_rate_limited(self):
        self.receiver.init(0)
        self.assertEqual(self.receiver.poll(), KVPoll.Bootstrapping)
        self.clock.return_value = 10.005
        self.assertEqual(self.receiver.poll(), KVPoll.Bootstrapping)
        self.assertEqual(self.probe.call_count, 1)
        self.clock.return_value = 10.02
        self.assertEqual(self.receiver.poll(), KVPoll.Bootstrapping)
        self.assertEqual(self.probe.call_count, 2)

    def test_timeout_is_terminal_and_reports_reason(self):
        self.receiver.init(0)
        self.clock.return_value = 310.0
        self.assertEqual(self.receiver.poll(), KVPoll.Failed)
        self.probe.return_value = True
        self.assertEqual(self.receiver.poll(), KVPoll.Failed)
        self.probe.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "42.*timed out after 300.0s"):
            self.receiver.failure_exception()

    def test_probe_exception_is_terminal_and_reports_reason(self):
        self.receiver.init(0)
        self.probe.side_effect = ValueError("lookup failed")
        self.assertEqual(self.receiver.poll(), KVPoll.Failed)
        self.assertEqual(self.receiver.poll(), KVPoll.Failed)
        self.probe.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "42: lookup failed"):
            self.receiver.failure_exception()

    def test_abort_does_not_probe(self):
        self.receiver.init(0)
        self.receiver.abort()
        self.assertEqual(self.receiver.poll(), KVPoll.Failed)
        self.probe.assert_not_called()


class TestLinkerTransferBackend(unittest.TestCase):
    def test_backend_classes(self):
        for class_type, expected in (
            (KVClassType.MANAGER, FakeKVManager),
            (KVClassType.SENDER, FakeKVSender),
            (KVClassType.RECEIVER, LinkerKVReceiver),
            (KVClassType.BOOTSTRAP_SERVER, CommonKVBootstrapServer),
        ):
            with self.subTest(class_type=class_type):
                self.assertIs(
                    get_kv_class(TransferBackend.LINKER, class_type), expected
                )

    def test_linker_and_warmup_skip_metadata_gate(self):
        cfg = SimpleNamespace(disaggregation_transfer_backend="linker")
        req = SimpleNamespace(bootstrap_host="prefill")
        warmup = SimpleNamespace(bootstrap_host=FAKE_BOOTSTRAP_HOST)
        polls = [int(KVPoll.Success), int(KVPoll.Success)]
        with patch("sglang.srt.disaggregation.utils.get_disagg", return_value=cfg):
            self.assertTrue(_is_linker_handoff(req))
            self.assertFalse(_is_linker_handoff(warmup))
            _apply_metadata_gate(
                polls,
                [SimpleNamespace(req=req), SimpleNamespace(req=warmup)],
                None,
            )
        self.assertEqual(polls, [int(KVPoll.Success), int(KVPoll.Success)])

    def test_other_backend_still_waits_for_metadata(self):
        cfg = SimpleNamespace(disaggregation_transfer_backend="mooncake")
        req = SimpleNamespace(bootstrap_host="prefill")
        room = Mock()
        room.__getitem__ = Mock(return_value=Mock(item=Mock(return_value=0)))
        polls = [int(KVPoll.Success)]
        with patch("sglang.srt.disaggregation.utils.get_disagg", return_value=cfg):
            self.assertFalse(_is_linker_handoff(req))
            _apply_metadata_gate(
                polls,
                [SimpleNamespace(req=req, metadata_buffer_index=0)],
                SimpleNamespace(bootstrap_room=room),
            )
        self.assertEqual(polls, [int(KVPoll.Transferring)])


class TestLinkerDisaggregationArgs(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="linker",
            enable_unified_cache_external_linker=True,
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_decode_retraction_backup="cpu_tensor",
            enable_dp_attention=False,
            enable_hisparse=False,
            speculative_algorithm=None,
            pp_size=1,
            dcp_size=1,
            disaggregation_decode_extra_slots=0,
        )
        for name in ("resolving_view", "resolved_view"):
            view = patch(
                f"sglang.srt.arg_groups.pd_disaggregation_hook.{name}",
                return_value=self.cfg,
            )
            view.start()
            self.addCleanup(view.stop)
        resolution = patch(
            "sglang.srt.arg_groups.pd_disaggregation_hook.declare_resolution"
        )
        resolution.start()
        self.addCleanup(resolution.stop)
        env = patch.dict(
            os.environ,
            {"SGLANG_DISAGG_STAGING_BUFFER": "false", "SGLANG_RUST_SERVER": "false"},
        )
        env.start()
        self.addCleanup(env.stop)

    def test_supported_prefill_and_decode(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                self.cfg.disaggregation_mode = mode
                handle_pd_disaggregation(self.cfg)

    def test_rejects_unsupported_configuration(self):
        for field, value, message in (
            ("enable_unified_cache_external_linker", False, "external-linker"),
            ("disaggregation_decode_enable_radix_cache", False, "radix-cache"),
            ("enable_dp_attention", True, "dp-attention"),
            ("pp_size", 2, "pp-size 1"),
            ("disaggregation_decode_retraction_backup", "host_pool", "host_pool"),
        ):
            with self.subTest(field=field):
                original = getattr(self.cfg, field)
                setattr(self.cfg, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    handle_pd_disaggregation(self.cfg)
                setattr(self.cfg, field, original)


if __name__ == "__main__":
    unittest.main()
