"""Compare batched multi-turn QA against an uncached reference run."""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from itertools import islice
from pathlib import Path

# Edit these values here; no batch/round command-line arguments are needed.
BATCH_SIZE = 5
NUM_ROUNDS = 3
MAX_NEW_TOKENS = 256
REFERENCE_PATH = Path("oceandisk_batch_chat_reference.json")
REPORT_PATH = Path("oceandisk_batch_chat_report.json")
FOLLOWUPS = (
    "请根据前面的文章再次确认上题的答案，只输出答案。",
    "请引用原文中支持上题答案的句子，并给出答案。",
)


def read_cases(path):
    if BATCH_SIZE < 1 or NUM_ROUNDS < 1:
        raise ValueError("BATCH_SIZE and NUM_ROUNDS must be positive.")
    with path.open(encoding="utf-8") as source:
        cases = [
            json.loads(line)
            for line in islice((x for x in source if x.strip()), BATCH_SIZE)
        ]
    if len(cases) != BATCH_SIZE:
        raise ValueError(f"Dataset needs at least {BATCH_SIZE} nonempty QA records.")
    return cases


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def flush_local_cache(engine):
    # OceanDisk reset drains its queued I/O before clearing the local cache.
    result = engine.flush_cache()
    if not result.success:
        raise RuntimeError(f"Cache flush failed: {result.message}")
    print("Local cache flushed; queued OceanDisk I/O drained.", flush=True)


def answer_text(text):
    return text.rsplit("</think>", 1)[-1].strip()


def normalize(text):
    return "".join(
        c
        for c in text.casefold()
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def score_answer(text, case):
    """Diagnostic answer scores, not an official benchmark evaluator."""
    golds = case.get("answers") or []
    if isinstance(golds, str):
        golds = [golds]
    golds = [str(gold) for gold in golds if str(gold).strip()]
    if not golds:
        return {"gold_exact": None, "gold_contained": None, "token_f1": None}
    answer = normalize(answer_text(text))

    def tokens(value):
        return Counter(re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", value.casefold()))

    predicted = tokens(answer_text(text))
    best_f1 = 0.0
    for gold in golds:
        expected = tokens(gold)
        overlap = sum((predicted & expected).values())
        total = sum(predicted.values()) + sum(expected.values())
        best_f1 = max(best_f1, 2 * overlap / total if total else 0.0)
    return {
        "gold_exact": any(answer == normalize(gold) for gold in golds),
        "gold_contained": any(normalize(gold) in answer for gold in golds),
        "token_f1": best_f1,
    }


def first_token_difference(reference, actual):
    if reference is None or actual is None:
        return None
    for position in range(max(len(reference), len(actual))):
        left = reference[position] if position < len(reference) else None
        right = actual[position] if position < len(actual) else None
        if left != right:
            return {"index": position, "reference": left, "actual": right}
    return None


def run_batch(engine, prompts, tokenizer, salt):
    input_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    started = time.perf_counter()
    outputs = engine.generate(
        input_ids=input_ids,
        sampling_params={
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        cache_salt=salt,
    )
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError("Engine returned an unexpected number of batch outputs.")
    print(f"Batch response latency: {elapsed:.3f}s (not TTFT/TBT)", flush=True)
    return input_ids, outputs


def run_chat(engine, tokenizer, cases, settings, reference=None):
    remote = reference is not None
    histories = [
        [
            {
                "role": "system",
                "content": "根据文章回答问题，使用中文，简短回答，不要输出分析。",
            },
            {
                "role": "user",
                "content": f"阅读文章：\n{case['context']}\n\n问题：{case['input']}\n请只给出答案。",
            },
        ]
        for case in cases
    ]
    report = {
        "settings": settings,
        "mode": "oceandisk" if remote else "base",
        "rounds": [],
    }
    salt = f"batch-chat-{uuid.uuid4().hex}"
    print(f"Cache salt: {salt} (does not change model inputs)", flush=True)
    flush_local_cache(engine)
    for round_index in range(NUM_ROUNDS):
        print(
            f"\n=== Round {round_index + 1}/{NUM_ROUNDS}, batch={BATCH_SIZE} ===",
            flush=True,
        )
        prompts = (
            [item["prompt"] for item in reference["rounds"][round_index]]
            if remote
            else [
                tokenizer.apply_chat_template(
                    h, tokenize=False, add_generation_prompt=True
                )
                for h in histories
            ]
        )
        # Reject a tokenizer mismatch before submitting any requests in this round.
        if remote:
            for prompt, expected in zip(
                prompts, reference["rounds"][round_index], strict=True
            ):
                if (
                    tokenizer.encode(prompt, add_special_tokens=False)
                    != expected["input_ids"]
                ):
                    raise ValueError(
                        "Reference input token IDs differ; rebuild the base reference."
                    )
        input_ids, outputs = run_batch(engine, prompts, tokenizer, salt)
        records = []
        for index, (case, prompt, ids, output) in enumerate(
            zip(cases, prompts, input_ids, outputs, strict=True)
        ):
            meta = output["meta_info"]
            details = meta.get("cached_tokens_details") or {}
            device = int(details.get("device", 0))
            restored = int(details.get("host", 0)) + int(details.get("storage", 0))
            record = {
                "case": index + 1,
                "prompt": prompt,
                "input_ids": ids,
                "text": output["text"],
                "output_ids": output.get("output_ids"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "cached_tokens": meta.get("cached_tokens", 0),
                "cache_details": details,
                "finish_reason": meta.get("finish_reason"),
                "truncated": (meta.get("finish_reason") or {}).get("type") == "length",
                "scores": score_answer(output["text"], case),
            }
            if remote:
                expected = reference["rounds"][round_index][index]
                record.update(
                    text_equal=output["text"] == expected["text"],
                    token_ids_equal=(
                        expected["output_ids"] == output["output_ids"]
                        if expected["output_ids"] is not None
                        and output.get("output_ids") is not None
                        else None
                    ),
                    reference_text=expected["text"],
                    reference_truncated=expected["truncated"],
                    reference_scores=expected["scores"],
                    first_token_difference=first_token_difference(
                        expected["output_ids"], output.get("output_ids")
                    ),
                    cache_check=(
                        "HBM_HIT"
                        if device
                        else "COLD"
                        if round_index == 0 and restored == 0
                        else "REMOTE_HIT"
                        if restored > 0
                        else "REMOTE_MISS"
                    ),
                )
            print(
                f"Case {index + 1}: {json.dumps({k: v for k, v in record.items() if k not in ('prompt', 'input_ids', 'output_ids')}, ensure_ascii=False)}",
                flush=True,
            )
            records.append(record)
            if not remote:
                histories[index].append(
                    {"role": "assistant", "content": output["text"]}
                )
                histories[index].append(
                    {"role": "user", "content": FOLLOWUPS[round_index % len(FOLLOWUPS)]}
                )
        report["rounds"].append(records)
        # Also drain the last round before Engine.shutdown; outside response timing.
        flush_local_cache(engine)
        write_json(REPORT_PATH if remote else REFERENCE_PATH, report)
    return report


def main():
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=os.getenv(
            "DATASET_PATH", "/home/z00591349/dataset/multifieldqa_zh.jsonl"
        ),
    )
    args = parser.parse_args()
    cases = read_cases(args.dataset_path)
    settings = {
        "batch_size": BATCH_SIZE,
        "num_rounds": NUM_ROUNDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_path": args.model_path,
        "tp_size": args.tp_size,
        "page_size": args.page_size,
        "dataset_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    remote = args.enable_unified_cache_external_linker
    if remote and not REFERENCE_PATH.exists():
        raise ValueError(
            "Run run_oceandisk_offline_batch_chat.sh base first to build the reference."
        )
    reference = (
        json.loads(REFERENCE_PATH.read_text(encoding="utf-8")) if remote else None
    )
    if reference is not None and (
        reference["settings"] != settings or len(reference["rounds"]) != NUM_ROUNDS
    ):
        raise ValueError(
            "Reference settings differ or reference is incomplete; run base first."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    args.enable_cache_report = True
    if not remote:
        args.disable_radix_cache = True
    engine = sgl.Engine(server_args=ServerArgs.from_cli_args(args))
    try:
        report = run_chat(engine, tokenizer, cases, settings, reference)
        records = [item for batch in report["rounds"] for item in batch]
        summary = {
            "requests": len(records),
            "truncated": sum(r["truncated"] for r in records),
        }
        if remote:
            summary.update(
                text_equal=sum(r["text_equal"] for r in records),
                cache_checks=dict(Counter(r["cache_check"] for r in records)),
                cache_expectations_met=all(
                    item["cache_check"]
                    == ("COLD" if round_index == 0 else "REMOTE_HIT")
                    for round_index, batch in enumerate(report["rounds"])
                    for item in batch
                ),
            )
        summary["gold_contained"] = sum(
            r["scores"]["gold_contained"] is True for r in records
        )
        summary["scored_requests"] = sum(
            r["scores"]["gold_contained"] is not None for r in records
        )
        report["summary"] = summary
        write_json(REPORT_PATH if remote else REFERENCE_PATH, report)
        print(f"\nSummary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        print(
            f"Saved: {(REPORT_PATH if remote else REFERENCE_PATH).resolve()}",
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
