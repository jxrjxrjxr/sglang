"""OceanDisk benchmark against a running hybrid (non-PD) server.

Runs the same bench() as oceandisk_pd_tp4.py, so both tables have the same
layout. The server is not started here: launch it first, for example with
run_oceandisk_online_tp8.sh, then pass its address. Reload TTFT clears the
server's local cache, so the prefix comes back from OceanDisk.

python3 examples/runtime/engine/oceandisk_hybrid_bench.py --host 127.0.0.1 --port 9101 2>&1 | tee hybrid_bench.log
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import requests
from oceandisk_pd_tp4 import bench


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--model-path", help="Local tokenizer path; defaults to the server's own."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/home/z00591349/dataset/multifieldqa_zh.jsonl"),
    )
    args = parser.parse_args()
    # The yellow-zone proxy does not forward intranet addresses.
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(name, None)
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    # One server serves requests and owns the cache that reload clears.
    bench(SimpleNamespace(router_url=url, decode_url=url), tokenizer, rows)


if __name__ == "__main__":
    main()
