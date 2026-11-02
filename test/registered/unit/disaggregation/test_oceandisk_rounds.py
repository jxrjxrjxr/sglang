"""Regression tests for the TPOT semantics copied from run_test_rounds_v6.py."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "oceandisk_rounds", ROOT / "examples/runtime/engine/Round/run_rounds.py"
)
rounds = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rounds
SPEC.loader.exec_module(rounds)


def delta(**fields):
    return {"choices": [{"delta": fields}]}


class TestOceanDiskRounds(unittest.TestCase):
    def stream(self, outputs, completion_tokens, tokenizer=None, include_usage=True):
        clock = [0.0]
        events = [(0.5, ""), (1.0, delta(role="assistant"))] + outputs
        events += [(7.0, {"choices": [{"delta": {}, "finish_reason": "stop"}]})]
        if include_usage:
            events += [(9.0, {"choices": [], "usage": {
                "completion_tokens": completion_tokens, "prompt_tokens": 16050,
                "prompt_tokens_details": {"cached_tokens": 16000},
            }})]
        events += [(10.0, "data: [DONE]")]

        class Response:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_lines(self):
                for timestamp, data in events:
                    clock[0] = timestamp
                    line = "data: " + json.dumps(data) if isinstance(data, dict) else data
                    yield line.encode()

        if tokenizer is None:
            tokenizer = Mock()
            tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
        bench = rounds.SGLangBatchBenchmark(
            "http://unused", "model", tokenizer=tokenizer, request_usage=include_usage
        )
        with patch.object(bench.session, "post", return_value=Response()), \
                patch.object(rounds.time, "perf_counter", side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            return bench.send_chat_request([{"role": "user", "content": "prompt"}], max_tokens=16)

    def test_v6_total_latency_minus_first_sse_with_local_output_count(self):
        result = self.stream([
            (2.0, delta(reasoning_content="think")),
            (3.0, delta(content="abc")), (5.0, delta(content="def")),
        ], completion_tokens=100)
        self.assertTrue(result.success)
        self.assertEqual(result.ttft_ms, 1000)
        self.assertEqual(result.total_latency_ms, 10000)
        self.assertEqual(result.output_tokens, 6)
        self.assertEqual(result.tpot_ms, 1800)  # (10000 - 1000) / (6 - 1), as in v6.
        self.assertEqual(result.response, "abcdef")
        self.assertEqual(result.cached_tokens, 16000)

    def test_v6_short_output_is_none_and_one_frame_can_have_many_tokens(self):
        self.assertIsNone(self.stream([], 0).tpot_ms)
        self.assertIsNone(self.stream([(2.0, delta(content="a"))], 1).tpot_ms)
        self.assertEqual(self.stream([(2.0, delta(content="abc"))], 3).tpot_ms, 4500)

    def test_v6_does_not_require_usage(self):
        result = self.stream([(2.0, delta(content="a")), (5.0, delta(content="bc"))],
                             None, include_usage=False)
        self.assertEqual(result.tpot_ms, 4500)

    def batch(self, requests):
        bench = rounds.SGLangBatchBenchmark("http://unused", "model")
        with patch.object(bench, "get_server_metrics", return_value={}), \
                patch.object(bench, "send_chat_request", side_effect=requests), \
                contextlib.redirect_stdout(io.StringIO()):
            result, _ = bench.run_batch_round([[] for _ in requests], len(requests), max_workers=1)
        return result

    def test_batch_uses_valid_successful_request_means(self):
        result = self.batch([
            rounds.SingleRequestResult(success=True, tpot_ms=10),
            rounds.SingleRequestResult(success=True, tpot_ms=30),
            rounds.SingleRequestResult(success=True),
            rounds.SingleRequestResult(success=False, tpot_ms=999),
        ])
        self.assertEqual(result.successful_requests, 3)
        self.assertEqual(result.failed_requests, 1)
        self.assertEqual(result.avg_tpot_ms, 20)
        self.assertEqual(result.max_tpot_ms, 30)
        self.assertEqual(result.min_tpot_ms, 10)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rounds.SGLangBatchBenchmark("http://unused", "model")._print_batch_result(result)
        self.assertIn("Avg TPOT:   20.00 ms | Max TPOT:   30.00 ms", output.getvalue())

    def test_v6_no_valid_tpot_reports_zero(self):
        for requests in ([rounds.SingleRequestResult(success=True)],
                         [rounds.SingleRequestResult(success=False)]):
            result = self.batch(requests)
            self.assertEqual(result.avg_tpot_ms, 0)
            self.assertEqual(result.max_tpot_ms, 0)
            self.assertEqual(result.min_tpot_ms, 0)

    def test_summary_table_and_json(self):
        results = [self.batch([rounds.SingleRequestResult(success=True, tpot_ms=12.5)]),
                   self.batch([rounds.SingleRequestResult(success=True)])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rounds.json"
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_rounds.py", "--model", "model", "--output-json", str(path)]), \
                    patch.object(rounds.SGLangBatchBenchmark, "run_conversation", return_value=(results, [])), \
                    contextlib.redirect_stdout(output):
                rounds.main()
            data = json.loads(path.read_text())
        self.assertEqual(data["results"][0]["avg_tpot_ms"], 12.5)
        self.assertEqual(data["results"][0]["max_tpot_ms"], 12.5)
        self.assertEqual(data["results"][0]["min_tpot_ms"], 12.5)
        self.assertEqual(data["results"][1]["avg_tpot_ms"], 0)
        self.assertEqual(data["results"][1]["max_tpot_ms"], 0)
        self.assertEqual(data["results"][1]["min_tpot_ms"], 0)
        text = output.getvalue()
        self.assertIn("Average TPOT (across rounds): 12.50 ms/token", text)
        table = text.split("Detailed Results (per round):", 1)[1]
        self.assertIn("  Avg TPOT |   Max TPOT", table)
        self.assertIn("     12.50 |      12.50", table)
        self.assertIn("      0.00 |       0.00", table)


if __name__ == "__main__":
    unittest.main()
