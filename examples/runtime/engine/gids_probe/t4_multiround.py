"""T4: multi-round chat replay -- the sglang rhythm without sglang.

C conversations take turns, round after round, like the rounds script. For
one request the "scheduler" (main thread) does what UnifiedCacheLinker does:

    EXIST(prefix)  ->  GET hit pages on the load worker, wait (forward needs
    them), verify  ->  "forward" (sleep)  ->  queue PUT of this turn's new
    pages on the offload worker WITHOUT waiting  ->  next request.

So request k+1's EXIST/GET runs while request k's PUT is still going, exactly
as in production. The prefix grows by --grow pages per round, so later rounds
GET more and overlap more.

A miss is an ANOMALY only if the PUT of those pages had already reported
completion before the EXIST (otherwise it is counted as pending, which is
expected). Pages are never recomputed/re-PUT here, so one loss shows up in
every later round of that conversation: watch the round where it first appears.

    python t4_multiround.py --mode full   # control
    python t4_multiround.py --mode none
"""

import argparse
import secrets
import sys
import time

import torch

from gids_probe_lib import Gate, Gids, Worker, add_common_args, log, make_keys, serial_health


class Conversation:
    def __init__(self, index):
        self.index = index
        self.done_keys = []  # PUT reported complete
        self.pending = None  # (future, keys) of the latest PUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--convs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--grow", type=int, default=8, help="new pages per conv per round")
    parser.add_argument("--forward-ms", type=float, default=30.0)
    parser.add_argument("--mode", choices=Gate.MODES, default="none")
    args = parser.parse_args()
    token = args.run_token or secrets.token_hex(8)
    g, C, R = args.grow, args.convs, args.rounds
    slots = max(C * g, R * g)  # region 0: g source slots per conv; region 1: longest prefix
    gids = Gids(args.device, args.layers, slots, args.kv_bytes, args.index_bytes, args.timeout)
    main_stream = torch.cuda.Stream(device=gids.device)
    loader = Worker("load-worker", gids.device)
    offloader = Worker("offload-worker", gids.device)
    gate = Gate(args.mode, offloader.stream)
    if not serial_health(gids, token, "health-before", g, main_stream):
        log("RESULT test=T4 verdict=BASELINE_FAIL")
        return 1
    warm = make_keys(token, "warm", g)  # PUT first: never GET a key that was not written
    offloader.call(lambda: gids.put(warm, range(g), offloader.stream).wait("warm PUT"))
    loader.call(lambda: gids.get(warm, range(g), loader.stream).wait("warm GET"))

    convs = [Conversation(c) for c in range(C)]
    totals = dict(anomaly_pages=0, data_bad=0, pending_at_lookup=0)
    for r in range(R):
        row = dict(lookups=0, expected=0, hit=0, anomaly=0, pending=0, bad=0, get_ms=0.0)
        for conv in convs:
            # The previous turn's PUT may or may not have finished: like
            # pending_offload_at_lookup in production.
            if conv.pending is not None and conv.pending[0].done():
                conv.pending[0].result()
                conv.done_keys += conv.pending[1]
                conv.pending = None
            prefix = list(conv.done_keys)
            if conv.pending is not None:
                prefix += conv.pending[1]
                row["pending"] += 1
            hit = gate.exist(gids, prefix, f"r{r} c{conv.index} EXIST")
            row["lookups"] += 1
            row["expected"] += len(conv.done_keys)
            row["hit"] += min(hit, len(conv.done_keys))
            if hit < len(conv.done_keys):
                row["anomaly"] += len(conv.done_keys) - hit
            # GET only pages whose PUT finished: a page still being written may
            # already EXIST but be incomplete, which would be a false data_bad.
            load_n = min(hit, len(conv.done_keys))
            if load_n:
                dst = list(range(load_n))
                keys = prefix[:load_n]

                def load():
                    def issue():
                        gids.clear(dst, loader.stream)
                        return gids.get(keys, dst, loader.stream)

                    started = time.perf_counter()
                    gate.io(f"r{r} c{conv.index} GET", issue)
                    elapsed = (time.perf_counter() - started) * 1000
                    return elapsed, gids.verify(keys, dst, loader.stream)["bad"]

                get_ms, bad = loader.submit(load).result()  # forward waits for its KV
                row["get_ms"] += get_ms
                row["bad"] += bad
            time.sleep(args.forward_ms / 1000)  # forward
            if conv.pending is not None:
                # Source pages are freed only after their PUT completes.
                conv.pending[0].result()
                conv.done_keys += conv.pending[1]
            new_keys = make_keys(token, f"c{conv.index}r{r}", g)
            src = list(range(conv.index * g, (conv.index + 1) * g))

            def offload(keys=new_keys, src=src, tag=f"r{r} c{conv.index} PUT"):
                def issue():
                    gids.fill(keys, src, offloader.stream)
                    return gids.put(keys, src, offloader.stream)

                gate.io(tag, issue)

            conv.pending = (offloader.submit(offload), new_keys)
        totals["anomaly_pages"] += row["anomaly"]
        totals["data_bad"] += row["bad"]
        totals["pending_at_lookup"] += row["pending"]
        rate = row["hit"] / row["expected"] if row["expected"] else 1.0
        flag = "  <-- ANOMALY" if row["anomaly"] or row["bad"] else ""
        log(
            f"[T4 r{r:02d}] mode={args.mode} lookups={row['lookups']} "
            f"expected_hit_pages={row['expected']} hit_pages={row['hit']} ({rate:.1%}) "
            f"missing_after_put_done={row['anomaly']} get_data_bad={row['bad']} "
            f"put_pending_at_lookup={row['pending']} get_ms_sum={row['get_ms']:.0f}{flag}"
        )

    for conv in convs:  # drain
        if conv.pending is not None:
            conv.pending[0].result()
            conv.done_keys += conv.pending[1]
    final_missing = 0
    for conv in convs:
        final_missing += len(conv.done_keys) - gids.exist(conv.done_keys, f"final c{conv.index}")
    healthy = serial_health(gids, token, "health-after", g, main_stream)
    ok = totals["anomaly_pages"] == 0 and totals["data_bad"] == 0 and final_missing == 0 and healthy
    log(
        f"RESULT test=T4 mode={args.mode} verdict={'PASS' if ok else 'FAIL'} "
        + " ".join(f"{k}={v}" for k, v in totals.items())
        + f" final_prefix_missing={final_missing} health_after={'OK' if healthy else 'FAIL'} token={token}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
