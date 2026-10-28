"""CPU-only regression checks for the two-round batch script."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import oceandisk_offline_batch_chat as chat


class BatchChatTests(unittest.TestCase):
    def test_identical_batches_flush_and_save_separate_results(self):
        cases = [
            {"context": f"article-{i}", "input": f"question-{i}"} for i in range(5)
        ]
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kw: json.dumps(messages),
            encode=lambda prompt, **kw: list(prompt.encode()),
        )
        engine = Mock()
        calls = []
        flush_count = 0

        def flush():
            nonlocal flush_count
            flush_count += 1
            return SimpleNamespace(success=True)

        def generate(**kwargs):
            self.assertEqual(flush_count, len(calls) + 1)
            calls.append(kwargs)
            return [
                {
                    "text": "different"
                    if len(calls) == 2 and i == 0
                    else f"answer-{i}",
                    "output_ids": [i, len(calls) if i == 0 else 0],
                    "meta_info": {
                        "cached_tokens_details": {"host": 64 if len(calls) == 2 else 0},
                        "finish_reason": {"type": "length" if i == 1 else "stop"},
                    },
                }
                for i in range(5)
            ]

        engine.generate.side_effect = generate
        engine.flush_cache.side_effect = flush
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(chat, "OUTPUT_DIR", Path(directory)),
            patch.object(chat.time, "sleep") as sleep,
            redirect_stdout(io.StringIO()) as console,
        ):
            rounds = chat.run_two_rounds(engine, tokenizer, cases)
            first, second = rounds
            self.assertEqual(list(Path(directory).glob("*.json")), [])
            self.assertEqual(first["round"], 1)
            self.assertEqual(second["round"], 2)
            self.assertEqual(len(first["results"]), 5)
            self.assertNotIn("text_equal", second["summary"])
            self.assertNotIn("comparison", second)
            self.assertEqual(second["summary"]["host_hits_without_hbm"], 5)
            self.assertEqual(second["summary"]["truncated_outputs"], 2)
            self.assertNotIn("token_ids_equal", console.getvalue())
            for number in (1, 2):
                for index, ids in enumerate(calls[number - 1]["input_ids"]):
                    self.assertIn(
                        f"QA {index + 1}: input_tokens={len(ids)}",
                        console.getvalue(),
                    )
            sleep.assert_called_once_with(chat.INTER_ROUND_SLEEP)
            for number, expected in ((1, "answer-0"), (2, "different")):
                text = (
                    Path(directory) / f"oceandisk_batch_round{number}.txt"
                ).read_text(encoding="utf-8")
                self.assertEqual(text.count("===== QA "), 5)
                self.assertIn(f"===== QA 1 =====\n{expected}", text)
        self.assertEqual(calls[0]["input_ids"], calls[1]["input_ids"])
        prefixes = []
        for ids in calls[0]["input_ids"]:
            prompt = bytes(ids).decode()
            self.assertRegex(prompt, r"^[0-9a-f]{32}\n")
            prefixes.append(prompt.split("\n", 1)[0])
        self.assertEqual(len(set(prefixes)), 5)
        self.assertEqual(calls[0]["cache_salt"], calls[1]["cache_salt"])
        self.assertEqual(len(set(calls[0]["cache_salt"])), 5)
        self.assertEqual(flush_count, 3)

    def test_flush_failure_stops_test(self):
        engine = SimpleNamespace(
            flush_cache=lambda: SimpleNamespace(success=False, message="busy")
        )
        with self.assertRaisesRegex(RuntimeError, "busy"):
            chat.flush_local_cache(engine)

    def test_read_first_n_cases_ignores_blank_lines(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(chat, "BATCH_SIZE", 2),
        ):
            path = Path(directory) / "qa.jsonl"
            path.write_text('\n{"input":"a"}\n{"input":"b"}\n{"input":"c"}\n')
            self.assertEqual([c["input"] for c in chat.read_cases(path)], ["a", "b"])

    def test_dataset_must_have_enough_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            path.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "at least 5"):
                chat.read_cases(path)


if __name__ == "__main__":
    unittest.main()
