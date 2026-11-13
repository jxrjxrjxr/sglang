"""Condense one .nsys-rep into a few hundred lines of text, per rank.

The report itself is too big to leave the yellow zone; this text is what gets
pasted back for analysis. Usage:

    python3 nsys_summary.py p_mixed_4k.nsys-rep > summary.txt

It runs `nsys export --type sqlite` once (next to the report) and then only
reads the sqlite file. Sections:

  1 per-GPU overview      span, GPU busy (union), kernel/memcpy totals
  2 top kernels           aggregated over ranks: avg / min / max per rank
  3 streams               busiest streams on every GPU (GET/PUT vs compute)
  4 keyword kernels       kernels matching KEYWORDS (GIDS etc.) per GPU
  5 GPU0 timeline         busy segments split at gaps > GAP_MS
  6 GPU gaps              longest idle gaps per GPU and what the rank's main
                          thread (the one issuing most CUDA calls) was doing
  7 NVTX ranges           aggregated over ranks, digits folded to '#'
  8 CUDA runtime          main thread's costliest CUDA API calls per rank

Times are ms relative to the first GPU activity in the capture.
"""

import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict

TOP = int(os.environ.get("TOP", "15"))
GAP_MS = float(os.environ.get("GAP_MS", "1.0"))
GAPS_PER_GPU = int(os.environ.get("GAPS_PER_GPU", "10"))
STREAMS_PER_GPU = int(os.environ.get("STREAMS_PER_GPU", "6"))
SEGMENTS = int(os.environ.get("SEGMENTS", "40"))
KEYWORDS = os.environ.get("KEYWORDS", "gids,dispatch,kvcache,oceandisk,nccl,deep_ep")
NAME_WIDTH = int(os.environ.get("NAME_WIDTH", "70"))

COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def pid_of(global_id):
    return (global_id >> 24) & 0xFFFFFF


def short(name, width=NAME_WIDTH):
    name = name or "?"
    return name if len(name) <= width else name[: width - 3] + "..."


def fold(name):
    return re.sub(r"\d+", "#", name or "?")


def union_ms(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total / 1e6


def open_db(path):
    if path.endswith(".nsys-rep"):
        db = path[: -len(".nsys-rep")] + ".sqlite"
        if not os.path.exists(db):
            subprocess.run(
                ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                 "-o", db, path],
                check=True,
            )
        path = db
    return sqlite3.connect(path)


def main():
    conn = open_db(sys.argv[1])
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(conn.execute("SELECT id, value FROM StringIds"))

    def rows(table, sql):
        return list(conn.execute(sql)) if table in have else []

    kernels = [
        (s, e, dev, stream, pid_of(gp), strings.get(name, str(name)))
        for s, e, dev, stream, gp, name in rows(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "SELECT start, end, deviceId, streamId, globalPid, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL",
        )
    ]
    if not kernels:
        print("no kernels captured: was /start_profile sent with CUDA_PROFILER?")
        return
    copies = rows(
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "SELECT start, end, deviceId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY",
    )
    memsets = rows(
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_MEMSET",
    )
    runtime = [
        (s, e, gt, strings.get(n, str(n)))
        for s, e, gt, n in rows(
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME",
        )
    ]
    nvtx = []
    if "NVTX_EVENTS" in have:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(NVTX_EVENTS)")}
        text_id = "textId" if "textId" in cols else "NULL"
        for s, e, text, tid, gt in conn.execute(
            f"SELECT start, end, text, {text_id}, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND end > start"
        ):
            nvtx.append((s, e, gt, text or strings.get(tid, "?")))
    osrt = [
        (s, e, gt, strings.get(n, str(n)))
        for s, e, gt, n in rows(
            "OSRT_API", "SELECT start, end, globalTid, nameId FROM OSRT_API"
        )
    ]

    t0 = min(k[0] for k in kernels)

    def ms(t):
        return (t - t0) / 1e6

    by_dev = defaultdict(list)
    for k in kernels:
        by_dev[k[2]].append(k)
    devs = sorted(by_dev)
    dev_pid = {d: Counter(k[4] for k in by_dev[d]).most_common(1)[0][0] for d in devs}
    # Main thread of a rank = the thread issuing most CUDA runtime calls.
    calls_per_tid = Counter(r[2] for r in runtime)
    main_tid = {}
    for gt, n in calls_per_tid.most_common():
        main_tid.setdefault(pid_of(gt), gt)

    busy_iv = defaultdict(list)
    for s, e, d, *_ in kernels:
        busy_iv[d].append((s, e))
    for s, e, d, *_ in copies:
        busy_iv[d].append((s, e))
    for s, e, d in memsets:
        busy_iv[d].append((s, e))

    def on_main(records, d):
        tid = main_tid.get(dev_pid[d])
        return [r for r in records if r[2] == tid]

    print(f"# nsys summary: {os.path.basename(sys.argv[1])}")
    print(f"GPUs={len(devs)} kernels={len(kernels)} memcpy={len(copies)} "
          f"cuda_api={len(runtime)} nvtx={len(nvtx)} osrt={len(osrt)} gap_ms={GAP_MS}")

    # 1 overview
    print("\n## 1 per-GPU overview (ms)")
    print("gpu pid      first    last     span     busy     busy%  kernels kern_sum  copies(MB by kind)")
    for d in devs:
        ks = by_dev[d]
        first, last = min(k[0] for k in ks), max(k[1] for k in ks)
        span = (last - first) / 1e6
        busy = union_ms(busy_iv[d])
        mb = defaultdict(float)
        for s, e, dd, kind, nbytes in copies:
            if dd == d:
                mb[COPY_KIND.get(kind, str(kind))] += (nbytes or 0) / 1e6
        mbs = " ".join(f"{k}={v:.1f}" for k, v in sorted(mb.items())) or "-"
        pct = busy / span * 100 if span else 0
        ksum = sum(k[1] - k[0] for k in ks) / 1e6
        print(f"{d:<3} {dev_pid[d]:<8} {ms(first):<8.2f} {ms(last):<8.2f} {span:<8.2f} "
              f"{busy:<8.2f} {pct:<6.1f} {len(ks):<7} {ksum:<9.2f} {mbs}")

    # 2 top kernels across ranks
    print(f"\n## 2 top {TOP} kernels (per-rank total ms: avg / min / max; calls per rank)")
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s, e, d, st, p, name in kernels:
        per[name][d][0] += e - s
        per[name][d][1] += 1
    table = []
    for name, dd in per.items():
        totals = [dd[d][0] / 1e6 if d in dd else 0.0 for d in devs]
        calls = sum(v[1] for v in dd.values()) / len(devs)
        table.append((sum(totals) / len(devs), min(totals), max(totals), calls, name))
    table.sort(reverse=True)
    print("avg      min      max      calls   kernel")
    for avg, lo, hi, calls, name in table[:TOP]:
        print(f"{avg:<8.2f} {lo:<8.2f} {hi:<8.2f} {calls:<7.1f} {short(name)}")

    # 3 streams
    print(f"\n## 3 busiest {STREAMS_PER_GPU} streams per GPU")
    print("gpu stream  busy     n      first    last     top kernel in stream")
    for d in devs:
        st = defaultdict(list)
        for k in by_dev[d]:
            st[k[3]].append(k)
        busy_of = {sid: union_ms([(k[0], k[1]) for k in ks]) for sid, ks in st.items()}
        for sid in sorted(st, key=busy_of.get, reverse=True)[:STREAMS_PER_GPU]:
            ks = st[sid]
            name_t = Counter()
            for k in ks:
                name_t[k[5]] += k[1] - k[0]
            top = name_t.most_common(1)[0][0]
            print(f"{d:<3} {sid:<7} {busy_of[sid]:<8.2f} {len(ks):<6} "
                  f"{ms(min(k[0] for k in ks)):<8.2f} {ms(max(k[1] for k in ks)):<8.2f} {short(top, 50)}")

    # 4 keyword kernels
    words = [w.strip().lower() for w in KEYWORDS.split(",") if w.strip()]
    print(f"\n## 4 keyword kernels ({','.join(words)}), top 5 per GPU")
    print("gpu n      total    first    last     kernel")
    for d in devs:
        agg = defaultdict(list)
        for k in by_dev[d]:
            if any(w in k[5].lower() for w in words):
                agg[k[5]].append(k)
        ranked = sorted(agg.items(), key=lambda kv: -sum(k[1] - k[0] for k in kv[1]))
        for name, ks in ranked[:5]:
            print(f"{d:<3} {len(ks):<6} {sum(k[1] - k[0] for k in ks) / 1e6:<8.2f} "
                  f"{ms(min(k[0] for k in ks)):<8.2f} {ms(max(k[1] for k in ks)):<8.2f} {short(name, 55)}")

    # 5 GPU0 timeline
    d0 = devs[0]
    segs = []
    for k in sorted(by_dev[d0]):
        if segs and k[0] - segs[-1][1] <= GAP_MS * 1e6:
            segs[-1][1] = max(segs[-1][1], k[1])
            segs[-1][2].append(k)
        else:
            segs.append([k[0], k[1], [k]])
    print(f"\n## 5 GPU{d0} timeline: {len(segs)} busy segments (split at gaps > {GAP_MS} ms), first {SEGMENTS}")
    print("seg start    end      dur      kernels  streams  top kernels by time")
    for i, (s, e, ks) in enumerate(segs[:SEGMENTS]):
        name_t = Counter()
        for k in ks:
            name_t[k[5]] += k[1] - k[0]
        tops = [n for n, _ in name_t.most_common(2)]
        print(f"{i:<3} {ms(s):<8.2f} {ms(e):<8.2f} {(e - s) / 1e6:<8.2f} {len(ks):<8} "
              f"{len({k[3] for k in ks}):<8} " + " | ".join(short(t, 40) for t in tops))

    # 6 gaps with main-thread context
    print(f"\n## 6 longest GPU idle gaps (> {GAP_MS} ms), top {GAPS_PER_GPU} per GPU; "
          "context = that rank's main thread")
    print("gpu start    dur      NVTX at gap middle             | longest CUDA call (ms)              | longest OS call (ms)")
    for d in devs:
        gaps, end = [], None
        for s, e in sorted(busy_iv[d]):
            if end is not None and s - end > GAP_MS * 1e6:
                gaps.append((end, s))
            end = e if end is None else max(end, e)
        gaps.sort(key=lambda g: g[0] - g[1])
        main_nvtx, main_rt, main_os = on_main(nvtx, d), on_main(runtime, d), on_main(osrt, d)
        for gs, ge in gaps[:GAPS_PER_GPU]:
            mid = (gs + ge) // 2
            cover = [r for r in main_nvtx if r[0] <= mid <= r[1]]
            inner = min(cover, key=lambda r: r[1] - r[0])[3] if cover else "-"
            cols = []
            for recs in (main_rt, main_os):
                hit = [r for r in recs if r[0] < ge and r[1] > gs]
                r = max(hit, key=lambda r: r[1] - r[0]) if hit else None
                cols.append(f"{short(r[3], 26)} {(r[1] - r[0]) / 1e6:.2f}" if r else "-")
            print(f"{d:<3} {ms(gs):<8.2f} {(ge - gs) / 1e6:<8.2f} {short(inner, 30):<30} "
                  f"| {cols[0]:<34} | {cols[1]}")

    # 7 NVTX
    if nvtx:
        print(f"\n## 7 top {TOP} NVTX ranges (digits folded; per-rank total ms avg / max, calls per rank)")
        ranks = set(dev_pid.values())
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for s, e, gt, name in nvtx:
            if pid_of(gt) in ranks:
                agg[fold(name)][pid_of(gt)][0] += e - s
                agg[fold(name)][pid_of(gt)][1] += 1
        table = []
        for name, pp in agg.items():
            totals = [v[0] / 1e6 for v in pp.values()]
            table.append((sum(totals) / len(devs), max(totals),
                          sum(v[1] for v in pp.values()) / len(devs), name))
        table.sort(reverse=True)
        print("avg      max      calls   range")
        for avg, hi, calls, name in table[:TOP]:
            print(f"{avg:<8.2f} {hi:<8.2f} {calls:<7.1f} {short(name)}")

    # 8 CUDA runtime on the main thread
    if runtime:
        print("\n## 8 main thread CUDA API, top 3 by total per rank")
        print("gpu total    n      max      call")
        for d in devs:
            agg = defaultdict(list)
            for s, e, gt, name in on_main(runtime, d):
                agg[name].append(e - s)
            for name, durs in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:3]:
                print(f"{d:<3} {sum(durs) / 1e6:<8.2f} {len(durs):<6} {max(durs) / 1e6:<8.2f} {short(name, 40)}")


if __name__ == "__main__":
    main()
