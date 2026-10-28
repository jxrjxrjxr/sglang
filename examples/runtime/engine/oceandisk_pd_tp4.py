"""OceanDisk PD checks and benchmark through the router, with the offline prompts.

Prompts come from build_prompt() in oceandisk_offline_tp8.py and the same
dataset, so answers line up with the offline TP8 runs. As in
oceandisk_offline_batch_chat.py, each prompt starts with a uuid so pages
cached by earlier runs cannot be hit.
"""

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from oceandisk_offline_tp8 import build_prompt

# Sampling of oceandisk_offline_tp8.py, with room for the thinking output.
SAMPLING_PARAMS = {
    "temperature": 0,
    "top_p": 0.95,
    "max_new_tokens": 600,
    "stream_interval": 1,
}
CONCURRENT_REQUESTS = 8
LATENCY_PROMPT_TOKENS = [4096, 16384, 32768, 60000]
THROUGHPUT_PROMPT_TOKENS = 8192
THROUGHPUT_CONCURRENCY = [1, 4, 8, 16]


def generate(url, prompt, sampling_params):
    """Stream one /generate request, timed as oceandisk_offline_tp8.py does."""
    started = time.perf_counter()
    arrivals = []
    completion_tokens = 0
    output = None
    payload = {"text": prompt, "sampling_params": sampling_params, "stream": True}
    with requests.post(f"{url}/generate", json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            output = json.loads(line[len(b"data:") :])
            if "error" in output:
                raise RuntimeError(f"Generation failed: {output['error']}")
            count = output["meta_info"]["completion_tokens"]
            if count > completion_tokens:
                arrivals.append((time.perf_counter(), count - completion_tokens))
                completion_tokens = count
    if output is None:
        raise RuntimeError("Generation returned no output.")
    single_token_chunks = all(count == 1 for _, count in arrivals)
    first = arrivals[0] if arrivals else (None, 0)
    return {
        "output": output,
        "latency": time.perf_counter() - started,
        "ttft": (first[0] - started) * 1000 if first[1] == 1 else None,
        "tbts": [(b[0] - a[0]) * 1000 for a, b in zip(arrivals, arrivals[1:])]
        if single_token_chunks
        else [],
    }


def summary(values, reduce=statistics.mean):
    values = [value for value in values if value is not None]
    return f"{reduce(values):.3f}" if values else "N/A"


def print_result(label, result, answers):
    meta = result["output"]["meta_info"]
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(f"Total latency: {result['latency']:.3f}s (full response)")
    print(f"TTFT:          {summary([result['ttft']])} ms")
    print(f"TBT mean:      {summary(result['tbts'])} ms")
    print(f"Prompt tokens: {meta.get('prompt_tokens')}")
    print(f"Output tokens: {meta.get('completion_tokens')}")
    print(f"Cached tokens: {meta.get('cached_tokens')}")
    print(f"Cache details: {meta.get('cached_tokens_details') or {}}")
    print(f"Output:        {result['output']['text']!r}")
    print(f"Answers:       {answers}")


def cache_tiers(result):
    """D's cached prefix as (HBM, OceanDisk). Direct Linker currently reports
    remote restoration through the host tier."""
    details = result["output"]["meta_info"].get("cached_tokens_details") or {}
    remote = int(details.get("host", 0)) + int(details.get("storage") or 0)
    return int(details.get("device", 0)), remote


def require_source(label, result, expect_hbm):
    device, remote = cache_tiers(result)
    if (device > 0, remote > 0) != (expect_hbm, not expect_hbm):
        expected = "HBM hits only" if expect_hbm else "OceanDisk load-back only"
        raise RuntimeError(
            f"{label}: device={device}, host+storage={remote}; expected {expected}."
        )


def flush_decode_cache(decode_url):
    """Clear D's local cache; D refuses while a request is still finishing."""
    for _ in range(10):
        response = requests.post(f"{decode_url}/flush_cache")
        if response.status_code == 200:
            return
        time.sleep(2)
    raise RuntimeError(f"D cache flush failed: {response.text.strip()}")


def check(args, tokenizer, rows):
    url = args.router_url
    head = uuid.uuid4().hex
    prompt = f"{head}\n{build_prompt(tokenizer, rows[0])}"
    answers = rows[0].get("answers")

    first = generate(url, prompt, SAMPLING_PARAMS)
    print_result("FIRST REQUEST - D LOADS FROM OCEANDISK", first, answers)
    require_source("First request", first, expect_hbm=False)
    time.sleep(2)
    second = generate(url, prompt, SAMPLING_PARAMS)
    print_result("SECOND REQUEST - D HBM CACHE TEST", second, answers)
    require_source("Second request", second, expect_hbm=True)
    flush_decode_cache(args.decode_url)
    print("D local cache cleared; requesting the same prompt from OceanDisk.")
    third = generate(url, prompt, SAMPLING_PARAMS)
    print_result("THIRD REQUEST - D OCEANDISK LOAD-BACK TEST", third, answers)
    require_source("Third request", third, expect_hbm=False)

    # Fresh lines at once form one linker EXTEND batch on D.
    batch = rows[1 : 1 + CONCURRENT_REQUESTS]
    prompts = [f"{uuid.uuid4().hex}\n{build_prompt(tokenizer, row)}" for row in batch]
    with ThreadPoolExecutor(len(prompts)) as pool:
        results = list(pool.map(lambda p: generate(url, p, SAMPLING_PARAMS), prompts))
    for line, (row, result) in enumerate(zip(batch, results), start=2):
        print_result(f"CONCURRENT - DATASET LINE {line}", result, row.get("answers"))
        if cache_tiers(result)[1] <= 0:
            raise RuntimeError(f"Line {line}: D did not load from OceanDisk.")

    # Line 1's context is in D's HBM; the appended context only in OceanDisk.
    line = 2 + CONCURRENT_REQUESTS
    extra = rows[line - 1]
    context = f"{rows[0]['context']}\n\n{extra['context']}"
    joined = {"context": context, "input": extra["input"]}
    # Same uuid as line 1, so the two prompts share line 1's prefix.
    joined_prompt = f"{head}\n{build_prompt(tokenizer, joined)}"
    result = generate(url, joined_prompt, SAMPLING_PARAMS)
    print_result(f"LINE 1 + LINE {line} CONTEXT", result, extra.get("answers"))
    device, remote = cache_tiers(result)
    if device < first["output"]["meta_info"]["cached_tokens"] // 2 or remote <= 0:
        raise RuntimeError(
            f"Extended prompt: device={device}, host+storage={remote}; "
            "expected line 1 from HBM and the rest from OceanDisk."
        )
    print("\nAll checks passed; compare each Output with its Answers.")


def make_prompt(tokenizer, rows, start, prompt_tokens):
    """Contexts from line `start` on, cut to about prompt_tokens. Each prompt
    starts from an unused line, so prompts share no prefix pages."""
    ids, row = [], start
    while len(ids) < prompt_tokens:
        text = rows[row % len(rows)]["context"] + "\n\n"
        ids += tokenizer.encode(text, add_special_tokens=False)
        row += 1
    context = tokenizer.decode(ids[: max(1, prompt_tokens - 100)])
    data = {"context": context, "input": rows[start]["input"]}
    return f"{uuid.uuid4().hex}\n{build_prompt(tokenizer, data)}"


def print_row(*cells):
    print("".join(f"{cell:>14}" for cell in cells))


def bench(args, tokenizer, rows):
    url = args.router_url
    start_lines = iter(range(len(rows)))
    params = {"temperature": 0, "ignore_eos": True, "stream_interval": 1}

    # Clearing D before the reload leaves only the handoff: OceanDisk or RDMA.
    params["max_new_tokens"] = 128
    print("\n===== Single request (ms): cold, then again after clearing D =====")
    print_row("target tokens", "prompt tokens", "cold TTFT", "reload TTFT")
    for prompt_tokens in LATENCY_PROMPT_TOKENS:
        prompt = make_prompt(tokenizer, rows, next(start_lines), prompt_tokens)
        cold = generate(url, prompt, params)
        flush_decode_cache(args.decode_url)
        reload = generate(url, prompt, params)
        sent = cold["output"]["meta_info"]["prompt_tokens"]
        ttfts = summary([cold["ttft"]]), summary([reload["ttft"]])
        print_row(prompt_tokens, sent, *ttfts)

    # Max TBT shows how long linker EXTEND batches stall running decodes on D.
    params["max_new_tokens"] = 256
    print(f"\n===== Concurrency (ms): {THROUGHPUT_PROMPT_TOKENS}-token prompts =====")
    print_row("concurrency", "TTFT p50", "TBT mean", "TBT max", "tokens/s")
    for concurrency in THROUGHPUT_CONCURRENCY:
        prompts = [
            make_prompt(tokenizer, rows, next(start_lines), THROUGHPUT_PROMPT_TOKENS)
            for _ in range(2 * concurrency)
        ]
        started = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(lambda p: generate(url, p, params), prompts))
        rate = sum(r["output"]["meta_info"]["completion_tokens"] for r in results)
        rate /= time.perf_counter() - started
        ttft = summary([r["ttft"] for r in results], statistics.median)
        tbts = [tbt for r in results for tbt in r["tbts"]]
        print_row(concurrency, ttft, summary(tbts), summary(tbts, max), f"{rate:.1f}")


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "bench"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--router-url", default="http://127.0.0.1:31345")
    parser.add_argument("--decode-url", default="http://127.0.0.1:31001")
    args = parser.parse_args()
    print(f"Model:       {args.model_path}")
    print(f"Dataset:     {args.dataset_path}")
    with args.dataset_path.open(encoding="utf-8") as dataset:
        rows = [json.loads(line) for line in dataset if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    (check if args.mode == "check" else bench)(args, tokenizer, rows)


if __name__ == "__main__":
    main()
