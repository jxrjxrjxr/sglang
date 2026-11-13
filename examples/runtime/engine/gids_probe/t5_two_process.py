"""T5: writer and reader in two processes (prefill TP0 PUTs, TP1 EXISTs).

Each process has its own GIDS connections and QPs, so the CQE-sharing problem
of T1/T2 cannot happen across processes. The writer reports a page set only
after its PUT completed; the reader (other GPU) then runs EXIST + GET + verify.
Expected: clean. A failure here would mean "PUT completed" does not imply
"visible to another client", which points at the box rather than at QP sharing.

    python t5_two_process.py --devices 0,1 --rounds 30
    python t5_two_process.py --devices 0,1 --reader-get-load   # reader also GETs nonstop
"""

import argparse
import multiprocessing as mp
import secrets
import sys
import threading


def writer(args, token, out_q):
    import torch

    from gids_probe_lib import Gids, make_keys

    gids = Gids(args.writer, args.layers, args.pages, args.kv_bytes, args.index_bytes, args.timeout, stagger=True)
    stream = torch.cuda.Stream(device=gids.device)
    slots = list(range(args.pages))
    out_q.put(("ready", None))
    for r in range(args.rounds):
        keys = make_keys(token, f"t5r{r}", args.pages)
        gids.fill(keys, slots, stream)
        gids.put(keys, slots, stream).wait(f"writer r{r} PUT")
        out_q.put(("put_done", (r, keys)))
    out_q.put(("end", None))


def reader(args, token, in_q, result_q):
    import torch

    from gids_probe_lib import Gids, Worker, log, make_keys

    n = args.pages
    gids = Gids(args.reader, args.layers, 2 * n, args.kv_bytes, args.index_bytes, args.timeout, stagger=True)
    stream = torch.cuda.Stream(device=gids.device)
    slots = list(range(n))
    stop = threading.Event()
    background = None
    if args.reader_get_load:
        # Keep the reader's own GET path busy, like TP1's load worker.
        filler = make_keys(token, "reader-own", n)
        own = list(range(n, 2 * n))
        gids.fill(filler, own, stream)
        gids.put(filler, own, stream).wait("reader preload PUT")
        getter = Worker("get-worker", gids.device)

        def loop():
            while not stop.is_set():
                gids.get(filler, own, getter.stream).wait("reader background GET")

        background = getter.submit(loop)
    totals = dict(miss=0, bad=0)
    while True:
        kind, payload = in_q.get()
        if kind == "end":
            break
        r, keys = payload
        hit = gids.exist(keys, f"reader r{r} EXIST")
        gids.clear(slots, stream)
        if hit:
            gids.get(keys[:hit], slots[:hit], stream).wait(f"reader r{r} GET")
        bad = gids.verify(keys[:hit], slots[:hit], stream)["bad"]
        totals["miss"] += n - hit
        totals["bad"] += bad
        flag = "  <-- ANOMALY" if hit < n or bad else ""
        log(f"[T5 r{r:02d}] reader exist={hit}/{n} data_bad={bad}{flag}")
    stop.set()
    if background is not None:
        background.result()
    result_q.put(totals)


def main():
    from gids_probe_lib import add_common_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--devices", default="0,1", help="writer,reader GPU indices")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--reader-get-load", action="store_true")
    args = parser.parse_args()
    args.writer, args.reader = (int(x) for x in args.devices.split(","))
    token = args.run_token or secrets.token_hex(8)
    ctx = mp.get_context("spawn")
    to_reader, from_writer, results = ctx.Queue(), ctx.Queue(), ctx.Queue()
    pw = ctx.Process(target=writer, args=(args, token, from_writer))
    pr = ctx.Process(target=reader, args=(args, token, to_reader, results))
    pr.start()
    pw.start()
    import queue

    def died():
        return [p.exitcode for p in (pw, pr) if p.exitcode not in (None, 0)]

    while True:  # relay writer -> reader; a child hang exits with code 3
        try:
            kind, payload = from_writer.get(timeout=1.0)
        except queue.Empty:
            if died():
                break
            continue
        if kind == "ready":
            continue
        to_reader.put((kind, payload))
        if kind == "end":
            break
    totals = dict(miss=-1, bad=-1)
    while pr.is_alive() or not results.empty():
        try:
            totals = results.get(timeout=1.0)
            break
        except queue.Empty:
            if died():
                break
    for p in (pw, pr):
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    ok = pw.exitcode == 0 and pr.exitcode == 0 and not totals["miss"] and not totals["bad"]
    print(
        f"RESULT test=T5 verdict={'PASS' if ok else 'FAIL'} reader_get_load={args.reader_get_load} "
        f"missing_pages={totals['miss']} data_bad={totals['bad']} "
        f"writer_exit={pw.exitcode} reader_exit={pr.exitcode} token={token}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
