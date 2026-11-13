"""T2: EXIST while GET (and optionally PUT) is running -- false misses.

Production on prefill TP1: the scheduler thread calls EXIST while the same
rank's load worker is still GETting. gids_exist launches its own kernels on a
private non-blocking stream and polls the same QPs, so the two can take each
other's completions. Its result is a uint32, so the "< 0 means error" branch
can never fire: a wrong answer comes back as a plain miss.

Every key queried here was written and verified beforehand, so any EXIST
answer below --pages is a false miss.

    python t2_exist_get_overlap.py --mode full --seconds 60   # control
    python t2_exist_get_overlap.py --mode none --seconds 60
    python t2_exist_get_overlap.py --mode none --with-put     # + a PUT thread (mixed mode)
"""

import argparse
import secrets
import sys
import threading
import time
from collections import Counter

import torch

from gids_probe_lib import Gate, Gids, Worker, add_common_args, log, make_keys, serial_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--mode", choices=Gate.MODES, default="none")
    parser.add_argument("--with-put", action="store_true")
    args = parser.parse_args()
    token = args.run_token or secrets.token_hex(8)
    n = args.pages
    gids = Gids(args.device, args.layers, 2 * n, args.kv_bytes, args.index_bytes, args.timeout)
    main_stream = torch.cuda.Stream(device=gids.device)
    getter = Worker("get-worker", gids.device)
    putter = Worker("put-worker", gids.device)
    exister = Worker("exist-worker", gids.device)  # plays the scheduler thread
    gate = Gate(args.mode, putter.stream)

    pre_keys = make_keys(token, "pre", n)
    slots, put_slots = list(range(n)), list(range(n, 2 * n))
    gids.fill(pre_keys, slots, main_stream)
    gids.put(pre_keys, slots, main_stream).wait("preload PUT")
    getter.call(lambda: gids.get(pre_keys, slots, getter.stream).wait("warm GET"))
    warm_keys = make_keys(token, "warm", n)  # fresh keys: never re-PUT an existing key here
    putter.call(lambda: gids.put(warm_keys, put_slots, putter.stream).wait("warm PUT"))
    if exister.call(lambda: gids.exist(pre_keys, "warm EXIST")) != n:
        log("RESULT test=T2 verdict=BASELINE_FAIL (preload not visible; run T0)")
        return 1

    stop = threading.Event()
    deadline = time.monotonic() + args.seconds

    def exist_loop():
        calls, answers = 0, Counter()
        while time.monotonic() < deadline:
            answers[gate.exist(gids, pre_keys, f"EXIST #{calls}")] += 1
            calls += 1
            time.sleep(0.0002)  # yield the gate in `full` mode
        stop.set()
        return calls, answers

    def get_loop():
        gets = bad = 0
        while not stop.is_set():
            def issue():
                gids.clear(slots, getter.stream)
                return gids.get(pre_keys, slots, getter.stream)

            gate.io(f"GET #{gets}", issue)
            bad += gids.verify(pre_keys, slots, getter.stream)["bad"]
            gets += 1
            time.sleep(0.0002)
        return gets, bad

    def put_loop():
        puts = 0
        while not stop.is_set():
            keys = make_keys(token, f"put{puts}", n)

            def issue():
                gids.fill(keys, put_slots, putter.stream)
                return gids.put(keys, put_slots, putter.stream)

            gate.io(f"PUT #{puts}", issue)
            puts += 1
            time.sleep(0.0002)
        return puts

    futures = [getter.submit(get_loop)]
    if args.with_put:
        futures.append(putter.submit(put_loop))
    calls, answers = exister.submit(exist_loop).result()
    gets, get_bad = futures[0].result()
    puts = futures[1].result() if args.with_put else 0

    false_miss = calls - answers.get(n, 0)
    settled = gids.exist(pre_keys, "EXIST(after)")  # quiet: must be n
    healthy = serial_health(gids, token, "health-after", n, main_stream)
    log(
        f"[T2] mode={args.mode} exist_calls={calls} false_miss_calls={false_miss} "
        f"answers(prefix->count)={dict(sorted(answers.items()))} | "
        f"overlapped_gets={gets} get_bad_pages={get_bad} | puts={puts} | "
        f"exist_after_quiet={settled}/{n}"
    )
    ok = false_miss == 0 and get_bad == 0 and settled == n and healthy
    log(
        f"RESULT test=T2 mode={args.mode} with_put={args.with_put} "
        f"verdict={'PASS' if ok else 'FAIL'} false_miss_calls={false_miss}/{calls} "
        f"get_bad_pages={get_bad} exist_after_quiet={settled}/{n} "
        f"health_after={'OK' if healthy else 'FAIL'} token={token}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
