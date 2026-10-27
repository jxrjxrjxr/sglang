"""CPU-only checks: python -m unittest discover -s examples/runtime/engine -p test_oceandisk_offline_batch_chat.py"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import oceandisk_offline_batch_chat as chat


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, prompt, **kwargs):
        return list(prompt.encode("utf-8"))


class Engine:
    def __init__(self, remote=False):
        self.remote = remote
        self.calls = []
        self.flushes = 0

    def flush_cache(self):
        self.flushes += 1
        return SimpleNamespace(success=True)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "text": f"{'different' if self.remote else 'answer'}-{len(self.calls)}-{index}",
                "output_ids": [len(self.calls), index, int(self.remote)],
                "meta_info": {
                    "completion_tokens": 3,
                    "cached_tokens_details": {
                        "host": 64 if self.remote and len(self.calls) > 1 else 0
                    },
                    "finish_reason": {"type": "length" if index == 0 else "stop"},
                },
            }
            for index in range(len(kwargs["input_ids"]))
        ]


class BatchChatTests(unittest.TestCase):
    def test_five_conversations_three_rounds_replay_identical_inputs(self):
        cases = [
            {"context": f"article-{i}", "input": "question", "answers": ["answer"]}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with (
                patch.object(chat, "REFERENCE_PATH", Path(directory) / "base.json"),
                patch.object(chat, "REPORT_PATH", Path(directory) / "report.json"),
            ):
                base, remote = Engine(), Engine(remote=True)
                reference = chat.run_chat(base, Tokenizer(), cases, {})
                report = chat.run_chat(remote, Tokenizer(), cases, {}, reference)
                self.assertEqual([len(batch) for batch in report["rounds"]], [5, 5, 5])
                self.assertEqual(base.flushes, 4)
                self.assertEqual(remote.flushes, 4)
                self.assertEqual(
                    [c["input_ids"] for c in base.calls],
                    [c["input_ids"] for c in remote.calls],
                )
                self.assertEqual(len({c["cache_salt"] for c in remote.calls}), 1)
                self.assertNotEqual(
                    base.calls[0]["cache_salt"], remote.calls[0]["cache_salt"]
                )
                self.assertEqual(base.calls[0]["sampling_params"]["temperature"], 0)
                self.assertIn("answer-1-0", reference["rounds"][1][0]["prompt"])
                self.assertNotIn("different-1-0", report["rounds"][1][0]["prompt"])
                self.assertEqual(
                    [r[0]["cache_check"] for r in report["rounds"]],
                    ["COLD", "REMOTE_HIT", "REMOTE_HIT"],
                )
                self.assertFalse(report["rounds"][0][0]["text_equal"])
                self.assertTrue(report["rounds"][0][0]["truncated"])
                saved = json.loads(chat.REPORT_PATH.read_text(encoding="utf-8"))
                self.assertEqual(len(saved["rounds"]), 3)

    def test_replay_rejects_changed_tokenization(self):
        reference = {"rounds": [[{"prompt": "text", "input_ids": [999]}] * 5]}
        with (
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "token IDs differ"),
        ):
            chat.run_chat(
                Engine(True),
                Tokenizer(),
                [{"context": "a", "input": "q"}] * 5,
                {},
                reference,
            )

    def test_flush_failure_stops_test(self):
        engine = SimpleNamespace(
            flush_cache=lambda: SimpleNamespace(success=False, message="busy")
        )
        with self.assertRaisesRegex(RuntimeError, "busy"):
            chat.flush_local_cache(engine)

    def test_scores_do_not_treat_missing_gold_as_a_pass(self):
        self.assertIsNone(chat.score_answer("anything", {})["gold_contained"])
        score = chat.score_answer(
            "analysis</think>厦门大学。", {"answers": ["厦门大学"]}
        )
        self.assertTrue(score["gold_exact"])
        self.assertEqual(score["token_f1"], 1)

    def test_token_difference_includes_shorter_output(self):
        self.assertEqual(
            chat.first_token_difference([1, 2], [1]),
            {"index": 1, "reference": 2, "actual": None},
        )
        self.assertIsNone(chat.first_token_difference(None, [1]))

    def test_dataset_must_have_enough_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 5"):
                chat.read_cases(path)


if __name__ == "__main__":
    unittest.main()
