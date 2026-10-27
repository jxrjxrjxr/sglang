"""Run identical QA batches twice, flushing HBM between the two rounds."""

import argparse
import json
import os
import time
import uuid
from itertools import islice
from pathlib import Path

from oceandisk_offline_tp8 import build_prompt

# Edit these values here, rather than passing batch options at launch.
BATCH_SIZE = 5
MAX_NEW_TOKENS = 256
INTER_ROUND_SLEEP = 10
OUTPUT_DIR = Path(".")


def read_cases(path):
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive.")
    with path.open(encoding="utf-8") as source:
        cases = [
            json.loads(line)
            for line in islice((x for x in source if x.strip()), BATCH_SIZE)
        ]
    if len(cases) != BATCH_SIZE:
        raise ValueError(f"Dataset needs at least {BATCH_SIZE} nonempty QA records.")
    return cases


def flush_local_cache(engine):
    # OceanDisk reset drains queued I/O before clearing the local tree/cache.
    result = engine.flush_cache()
    if not result.success:
        raise RuntimeError(f"Cache flush failed: {result.message}")
    print("HBM cache flushed; queued OceanDisk I/O drained.", flush=True)


def run_two_rounds(engine, tokenizer, cases):
    # Generate once per case; both rounds must reuse the exact same prompt.
    prompts = [f"{uuid.uuid4().hex}\n{build_prompt(tokenizer, case)}" for case in cases]
    input_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    run_id = uuid.uuid4().hex
    # Keep each case's cache identity unchanged between the two rounds.
    salts = [f"batch-qa-{run_id}-{i}" for i in range(len(cases))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds = []
    flush_local_cache(engine)
    for number in (1, 2):
        print(f"\n=== Round {number}/2, batch={len(cases)} ===", flush=True)
        rids = [f"batch-{run_id}-r{number}-qa{i + 1}" for i in range(len(cases))]
        started = time.perf_counter()
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params={
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            cache_salt=salts,
            rid=rids,
        )
        elapsed = time.perf_counter() - started
        if len(outputs) != len(cases):
            raise RuntimeError("Engine returned an unexpected number of batch outputs.")
        records = []
        for index, (case, prompt, ids, output) in enumerate(
            zip(cases, prompts, input_ids, outputs, strict=True)
        ):
            meta = output["meta_info"]
            details = meta.get("cached_tokens_details") or {}
            record = {
                "case": index + 1,
                "rid": rids[index],
                "prompt_tokens": len(ids),
                "question": case["input"],
                "answers": case.get("answers"),
                "prompt": prompt,
                "input_ids": ids,
                "text": output["text"],
                "meta_info": meta,
                "truncated": (meta.get("finish_reason") or {}).get("type") == "length",
                "device_cached_tokens": int(details.get("device", 0)),
                "host_cached_tokens": int(details.get("host", 0)),
            }
            records.append(record)
            print(
                f"QA {index + 1}: input_tokens={len(ids)} "
                f"host_hit_tokens={record['host_cached_tokens']} hbm_hit_tokens={record['device_cached_tokens']} "
                f"truncated={record['truncated']}\nOutput: {output['text']!r}",
                flush=True,
            )
        result = {
            "run_id": run_id,
            "round": number,
            "batch_size": len(cases),
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_latency_s": elapsed,
            "results": records,
        }
        if number == 2:
            result["summary"] = {
                "cases": len(cases),
                "host_hits_without_hbm": sum(
                    r["host_cached_tokens"] > 0 and r["device_cached_tokens"] == 0
                    for r in records
                ),
                "truncated_outputs": sum(
                    r["truncated"]
                    for batch in (rounds[0]["results"], records)
                    for r in batch
                ),
            }
            print(f"Summary: {json.dumps(result['summary'])}", flush=True)
        text_path = OUTPUT_DIR / f"oceandisk_batch_round{number}.txt"
        text_path.write_text(
            "\n\n".join(
                f"===== QA {record['case']} =====\n{record['text']}"
                for record in records
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Saved: {text_path.resolve()}\nBatch latency: {elapsed:.3f}s (not TTFT/TBT)",
            flush=True,
        )
        rounds.append(result)
        # Round 1 prepares the load-back test; round 2 drains I/O before shutdown.
        # Save results first so they remain available even if flushing fails.
        flush_local_cache(engine)
        if number == 1:
            print(f"Waiting {INTER_ROUND_SLEEP}s before round 2...", flush=True)
            time.sleep(INTER_ROUND_SLEEP)
    return rounds


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
    if (
        not args.enable_unified_cache_external_linker
        or args.unified_cache_external_linker_backend != "oceandisk"
    ):
        raise ValueError("This test requires the OceanDisk Unified Cache Linker.")
    cases = read_cases(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    args.enable_cache_report = True
    engine = sgl.Engine(server_args=ServerArgs.from_cli_args(args))
    try:
        run_two_rounds(engine, tokenizer, cases)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
