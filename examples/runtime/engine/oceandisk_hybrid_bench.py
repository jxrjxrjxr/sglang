"""OceanDisk benchmark against a running hybrid (non-PD) server.

Runs the same bench() as oceandisk_pd_tp4.py, so both tables have the same
layout. The server is not started here: launch it first, then run
run_oceandisk_hybrid_bench.sh with its address. Reload TTFT clears the
server's local cache, so the prefix comes back from OceanDisk.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import oceandisk_pd_tp4
import requests


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--model-path", help="Local tokenizer path; defaults to the server's own."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--latency-prompt-tokens",
        type=int,
        nargs="+",
        default=oceandisk_pd_tp4.LATENCY_PROMPT_TOKENS,
    )
    parser.add_argument(
        "--latency-output-tokens",
        type=int,
        default=oceandisk_pd_tp4.LATENCY_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--throughput-prompt-tokens",
        type=int,
        default=oceandisk_pd_tp4.THROUGHPUT_PROMPT_TOKENS,
    )
    parser.add_argument(
        "--throughput-concurrency",
        type=int,
        nargs="+",
        default=oceandisk_pd_tp4.THROUGHPUT_CONCURRENCY,
    )
    parser.add_argument(
        "--throughput-output-tokens",
        type=int,
        default=oceandisk_pd_tp4.THROUGHPUT_OUTPUT_TOKENS,
    )
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    try:
        requests.get(f"{url}/health", timeout=10).raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"Server at {url} is not healthy: {error}")
    model_path = args.model_path
    if model_path is None:
        info = requests.get(f"{url}/model_info", timeout=10).json()
        model_path = info.get("tokenizer_path") or info["model_path"]
    print(f"mode=bench backend=hybrid url={url}")
    print(f"Model:       {model_path}")
    print(f"Dataset:     {args.dataset_path}")
    with args.dataset_path.open(encoding="utf-8") as dataset:
        rows = [json.loads(line) for line in dataset if line.strip()]
    # One fresh line per prompt: each latency prompt, 2 x each concurrency.
    needed = len(args.latency_prompt_tokens) + 2 * sum(args.throughput_concurrency)
    if len(rows) < needed:
        raise SystemExit(f"Dataset has {len(rows)} lines; this bench needs {needed}.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    # bench() reads these module settings when it runs.
    oceandisk_pd_tp4.LATENCY_PROMPT_TOKENS = args.latency_prompt_tokens
    oceandisk_pd_tp4.LATENCY_OUTPUT_TOKENS = args.latency_output_tokens
    oceandisk_pd_tp4.THROUGHPUT_PROMPT_TOKENS = args.throughput_prompt_tokens
    oceandisk_pd_tp4.THROUGHPUT_CONCURRENCY = args.throughput_concurrency
    oceandisk_pd_tp4.THROUGHPUT_OUTPUT_TOKENS = args.throughput_output_tokens
    # One server serves requests and owns the cache that reload clears.
    oceandisk_pd_tp4.bench(
        SimpleNamespace(router_url=url, decode_url=url), tokenizer, rows
    )


if __name__ == "__main__":
    main()
