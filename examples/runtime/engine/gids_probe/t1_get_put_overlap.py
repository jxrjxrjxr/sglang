"""T1: GET and PUT at the same time in one process (the decisive test).

Production on prefill TP0: the load worker GETs on the get stream while the
offload worker PUTs on the put stream; nothing keeps the two kernels apart.
Both kernels poll the same 8 QPs of a connection (find_qp = rcs[lane % 8])
and consume completions by "read cons_idx, spin, atomicAdd" with no owner
check, so they can take each other's completions.

Each round:
  1. a GET thread keeps reading a preloaded, verified page set and checks
     every byte;
  2. meanwhile a PUT thread writes a fresh page set (layerwise, 78 calls);
  3. after both stop: EXIST + GET + verify the fresh pages, and re-check the
     preloaded ones. A miss is re-checked after --settle seconds to tell
     "finished late" (completion stolen, write landed later) from "lost".

Run three times and compare (same arguments, separate processes):
    python t1_get_put_overlap.py --mode full     # control: GIDS' contract
    python t1_get_put_overlap.py --mode launch   # CPU lock like _put_io_lock
    python t1_get_put_overlap.py --mode none     # production GET vs PUT
"""

import argparse
import secrets
import sys
import threading
import time

import torch

from gids_probe_lib import Gate, Gids, Worker, add_common_args, log, make_keys, serial_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--mode", choices=Gate.MODES, default="none")
    parser.add_argument("--settle", type=float, default=3.0)
    args = parser.parse_args()
    token = args.run_token or secrets.token_hex(8)
    n = args.pages
    gids = Gids(args.device, args.layers, 2 * n, args.kv_bytes, args.index_bytes, args.timeout)
    main_stream = torch.cuda.Stream(device=gids.device)
    getter = Worker("get-worker", gids.device)
    putter = Worker("put-worker", gids.device)
    gate = Gate(args.mode, putter.stream)

    pre_slots, new_src = list(range(n)), list(range(n, 2 * n))
    loop_dst, check_dst = list(range(n)), list(range(n, 2 * n))
    pre_keys = make_keys(token, "pre", n)

    # Preload serially and warm both worker threads (thread_local GIDS buffers).
    gids.fill(pre_keys, pre_slots, main_stream)
    gids.put(pre_keys, pre_slots, main_stream).wait("preload PUT")
    if gids.exist(pre_keys, "preload EXIST") != n:
        log("RESULT test=T1 verdict=BASELINE_FAIL (preload not visible; run T0)")
        return 1
    getter.call(lambda: gids.get(pre_keys, loop_dst, getter.stream).wait("warm GET"))
    warm_keys = make_keys(token, "warm", n)  # fresh keys: never re-PUT an existing key here
    putter.call(lambda: gids.put(warm_keys, new_src, putter.stream).wait("warm PUT"))
    if not serial_health(gids, token, "health-before", n, main_stream):
        log("RESULT test=T1 verdict=BASELINE_FAIL")
        return 1

    totals = dict(get_bad=0, put_late=0, put_lost=0, put_bad=0, pre_lost=0)
    for r in range(args.rounds):
        keys = make_keys(token, f"t1r{r}", n)
        put_done = threading.Event()

        def get_loop():
            gets = bad = untouched = 0
            while True:
                def issue():
                    gids.clear(loop_dst, getter.stream)
                    return gids.get(pre_keys, loop_dst, getter.stream)

                gate.io(f"r{r} GET(preloaded)", issue)
                check = gids.verify(pre_keys, loop_dst, getter.stream)
                gets += 1
                bad += check["bad"]
                untouched += check["untouched"]
                if put_done.is_set():
                    return gets, bad, untouched
                time.sleep(0.0002)  # yield the gate so `full` mode cannot starve the PUT

        def put_once():
            def issue():
                gids.fill(keys, new_src, putter.stream)  # reuse source slots, like freed KV pages
                return gids.put(keys, new_src, putter.stream)

            started = time.perf_counter()
            try:
                gate.io(f"r{r} PUT(fresh)", issue)
            finally:
                put_done.set()
            return (time.perf_counter() - started) * 1000

        get_future = getter.submit(get_loop)
        time.sleep(0.002)  # let the first GET get going before the PUT starts
        put_ms = putter.submit(put_once).result()
        gets, get_bad, get_untouched = get_future.result()

        # Quiet now: check the fresh pages and the preloaded ones serially.
        hit = gids.exist(keys, f"r{r} EXIST(fresh)")
        late = lost = 0
        if hit < n:
            time.sleep(args.settle)
            hit2 = gids.exist(keys, f"r{r} EXIST(fresh, settled)")
            late, lost = hit2 - hit, n - hit2
            hit = hit2
        gids.clear(check_dst, main_stream)
        if hit:
            gids.get(keys[:hit], check_dst[:hit], main_stream).wait(f"r{r} GET(fresh)")
        put_check = gids.verify(keys[:hit], check_dst[:hit], main_stream)
        pre_hit = gids.exist(pre_keys, f"r{r} EXIST(preloaded)")

        totals["get_bad"] += get_bad
        totals["put_late"] += late
        totals["put_lost"] += lost
        totals["put_bad"] += put_check["bad"]
        totals["pre_lost"] += n - pre_hit
        anomaly = any((get_bad, late, lost, put_check["bad"], pre_hit < n))
        flag = "  <-- ANOMALY" if anomaly else ""
        log(
            f"[T1 r{r:02d}] mode={args.mode} overlapped_gets={gets} "
            f"get_bad_pages={get_bad}(untouched={get_untouched}) put_ms={put_ms:.1f} | "
            f"fresh exist={hit}/{n} late={late} lost={lost} data_bad={put_check['bad']} | "
            f"preloaded exist={pre_hit}/{n}{flag}"
        )

    healthy = serial_health(gids, token, "health-after", n, main_stream)
    anomalies = sum(totals.values())
    verdict = "PASS" if anomalies == 0 and healthy else "FAIL"
    log(
        f"RESULT test=T1 mode={args.mode} verdict={verdict} rounds={args.rounds} pages={n} "
        + " ".join(f"{k}={v}" for k, v in totals.items())
        + f" health_after={'OK' if healthy else 'FAIL'} token={token}"
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
