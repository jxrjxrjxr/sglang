"""Single-request P-side profile on a mixed (non-PD) server with OceanDisk.

Each cycle: random prefix -> seed request (P writes the prefix to OceanDisk
layer by layer) -> wait for the PUTs -> flush the GPU radix cache -> one
request = same prefix + new tail, so the prefix is read back from OceanDisk.
The last cycle is wrapped in /start_profile and /stop_profile; earlier cycles
are warmup. A fresh random prefix every cycle keeps old keys on the box from
raising the hit rate.

All settings come from environment variables; see run_profile_client.sh.
"""

import json
import os
import random
import time

import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:31010")
INPUT_LEN = int(os.environ.get("INPUT_LEN", "4096"))
# 3712 = 58 pages of 64 = 90.6% of 4096; must be a multiple of page size.
PREFIX_LEN = int(os.environ.get("PREFIX_LEN", "3712"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "1"))
WARMUP = int(os.environ.get("WARMUP", "2"))
SETTLE_S = float(os.environ.get("SETTLE_S", "5"))
STOP_DELAY_S = float(os.environ.get("STOP_DELAY_S", "2"))
# CUDA_PROFILER -> nsys (server must run under nsys)
# CPU,GPU       -> torch profiler *.trace.json.gz (server without nsys)
ACTIVITIES = os.environ.get("ACTIVITIES", "CUDA_PROFILER").split(",")
PROFILE_DIR = os.environ.get("PROFILE_DIR", "/tmp/p_profile")
PROFILE_PREFIX = os.environ.get("PROFILE_PREFIX", "p_mixed_4k")
# GLM vocab is ~151k with special tokens at the top; stay well below them.
TOKEN_LOW = int(os.environ.get("TOKEN_LOW", "1000"))
TOKEN_HIGH = int(os.environ.get("TOKEN_HIGH", "100000"))

rng = random.Random()


def tokens(n):
    return [rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(n)]


def generate(input_ids, max_new_tokens):
    start = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=3600,
    )
    r.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, r.json()["meta_info"]


def post(path, body=None):
    r = requests.post(f"{BASE_URL}{path}", json=body, timeout=3600)
    print(f"{path}: {r.status_code} {r.text.strip()}", flush=True)
    r.raise_for_status()


def report(tag, elapsed_ms, meta):
    prompt = meta.get("prompt_tokens", 0)
    cached = meta.get("cached_tokens", 0)
    hit = cached / prompt if prompt else 0.0
    print(
        f"[{tag}] e2e={elapsed_ms:.2f} ms prompt={prompt} cached={cached} "
        f"hit={hit:.1%}",
        flush=True,
    )
    print(f"[{tag}] meta_info={json.dumps(meta, ensure_ascii=False)}", flush=True)


def main():
    print(
        f"input={INPUT_LEN} prefix={PREFIX_LEN} "
        f"target_hit={PREFIX_LEN / INPUT_LEN:.1%} output={OUTPUT_LEN} "
        f"warmup={WARMUP} activities={ACTIVITIES}",
        flush=True,
    )
    for cycle in range(WARMUP + 1):
        profiled = cycle == WARMUP
        tag = "profile" if profiled else f"warmup{cycle}"
        prefix = tokens(PREFIX_LEN)
        tail = tokens(INPUT_LEN - PREFIX_LEN)

        elapsed_ms, meta = generate(prefix, 1)
        report(f"{tag} seed", elapsed_ms, meta)
        time.sleep(SETTLE_S)
        post("/flush_cache")

        if profiled:
            post(
                "/start_profile",
                {
                    "output_dir": PROFILE_DIR,
                    "activities": ACTIVITIES,
                    "profile_prefix": PROFILE_PREFIX,
                    "record_shapes": True,
                },
            )
        elapsed_ms, meta = generate(prefix + tail, OUTPUT_LEN)
        if profiled:
            # Layer-wise PUTs of the new tail outlive the response.
            time.sleep(STOP_DELAY_S)
            post("/stop_profile")
        report(tag, elapsed_ms, meta)


if __name__ == "__main__":
    main()
