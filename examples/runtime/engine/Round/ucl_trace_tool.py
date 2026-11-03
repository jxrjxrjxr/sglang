#!/usr/bin/env python3
"""Merge and analyze UCLinker trace files written by ``ucl_trace.py``.

    python ucl_trace_tool.py report  RUN_DIR [--wait 3]   # report.md + trace.json + summary.json
    python ucl_trace_tool.py merge   RUN_DIR [-o trace.json]
    python ucl_trace_tool.py compare RUN_DIR_A RUN_DIR_B

trace.json opens in chrome://tracing or ui.perfetto.dev. Naming follows what
L1 CPU events can prove: "worker observed" completion is not device
completion, and Python spans are not GPU execution intervals.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

STATE_ORDER = (
    "等待READY",
    "READY已到·未提交",
    "本rank就绪·等其他rank",
    "load进行中",
    "prefix完成·等传输/放行",
    "失败处理中",
)
HINT = "READY已满足·被inflight限制（候选优化点）"


# ----------------------------------------------------------------- loading


class TraceFile:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.identity = {}
        self.events = []
        self.stats = []
        self.last_flush = None
        self.dropped = 0
        self.ended = False
        self.bad_lines = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.bad_lines += 1  # torn last line of a killed process
                    continue
                kind = rec.get("type")
                if kind is None:
                    if rec["n"] == "process.identity":
                        self.identity = rec.get("f") or {}
                    self.events.append(rec)
                elif kind == "header":
                    self.header = rec
                elif kind == "stat":
                    self.stats.append(rec)
                elif kind == "flush":
                    self.last_flush = rec["t"]
                    self.dropped = rec.get("dropped", 0)
                elif kind == "end":
                    self.ended = True
                    self.last_flush = rec["t"]
                    self.dropped = rec.get("dropped", 0)
        self.host = self.header.get("host", "?")
        self.role = self.identity.get("role", "?")
        self.label = self.identity.get("label", os.path.basename(path))

    @property
    def name(self):
        return f"{self.role} {self.label}@{self.host}"


class Run:
    def __init__(self, directory):
        self.dir = directory
        paths = sorted(
            os.path.join(directory, p)
            for p in os.listdir(directory)
            if p.endswith(".jsonl")
        )
        if not paths:
            sys.exit(f"No *.jsonl trace files in {directory}")
        self.files = [TraceFile(p) for p in paths]
        # One perf clock per host: use one anchor per host so processes on the
        # same host keep their exact relative order (no per-process jitter).
        self.hosts = sorted({f.host for f in self.files})
        offsets = {}
        for f in self.files:
            h = f.header
            if f.host not in offsets and h:
                offsets[f.host] = h["wall_anchor_ns"] - h["perf_anchor_ns"]
        self.cross_host = len(self.hosts) > 1
        self.events = []
        for index, f in enumerate(self.files):
            offset = f.offset = offsets.get(f.host, 0)
            for e in f.events:
                e["src"] = index
                e["w"] = e["t"] + offset
                self.events.append(e)
        self.events.sort(key=lambda e: e["w"])
        self.t0 = self.events[0]["w"] if self.events else 0
        self._index_requests()

    def _index_requests(self):
        self.room_rid = {}
        for e in self.events:
            room = (e.get("f") or {}).get("room")
            if (
                room is not None
                and e.get("r")
                and e["n"]
                in (
                    "req.arrive",
                    "pd.prealloc",
                    "linker.plan",
                )
            ):
                self.room_rid.setdefault(room, e["r"])

    def rid_of(self, e):
        r = e.get("r")
        if r:
            return r.split("@", 1)[0]
        room = (e.get("f") or {}).get("room")
        return self.room_rid.get(room)

    def ms(self, w):
        return (w - self.t0) / 1e6


# ----------------------------------------------------------------- helpers


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(q / 100 * len(values) + 0.5)) - 1))
    return values[k]


def fmt(v, digits=1):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def dist(values):
    return dict(
        n=len(values),
        p50=pct(values, 50),
        p99=pct(values, 99),
        max=max(values) if values else None,
        sum=sum(values) if values else 0.0,
    )


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f_(e, key, default=None):
    return (e.get("f") or {}).get(key, default)


def merge_range(ranges, start, end):
    """Union [start, end) into sorted disjoint ranges (adjacent ranges join)."""
    out = []
    for a, b in sorted(list(ranges) + [(start, end)]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def covered_until(ranges, start):
    for a, b in ranges:
        if a <= start < b:
            return b
    return start


# ----------------------------------------------------------- completeness


def completeness(run):
    rows, problems = [], []
    for f in run.files:
        status = "完整" if f.ended else "无结束标记"
        if f.dropped:
            status += f"，丢弃 {f.dropped}"
            problems.append(f"{f.name} 丢弃了 {f.dropped} 条事件：相关统计偏小。")
        if not f.ended:
            problems.append(
                f"{f.name} 没有结束标记（进程仍在运行或被 kill）："
                "最后一次刷盘之后的事件缺失。"
            )
        if f.bad_lines:
            status += f"，损坏行 {f.bad_lines}"
        last = None if f.last_flush is None else run.ms(f.last_flush + f.offset) / 1000
        rows.append(
            [
                f.role,
                f.label,
                f.host,
                f.header.get("pid", "?"),
                len(f.events),
                fmt(last, 1),
                status,
            ]
        )
    pairs = [
        ("od.get.enqueue", "od.get.done", lambda e: (e["src"], f_(e, "batch"))),
        ("uc.save.begin", "uc.save.consumed", lambda e: (e["src"], f_(e, "session"))),
        ("linker.load_submit", "linker.load_applied", lambda e: (e["src"], e.get("r"))),
        ("client.send", "client.end", lambda e: e.get("r")),
    ]
    unclosed = []
    for start, end, key in pairs:
        opened = {}
        for e in run.events:
            if e["n"] == start:
                opened[key(e)] = e
            elif e["n"] == end:
                opened.pop(key(e), None)
        if opened:
            unclosed.append([start, end, len(opened)])
    return rows, problems, unclosed


# ------------------------------------------------------------------ client


def client_section(run):
    reqs = {}
    for e in run.events:
        if e["n"].startswith("client."):
            reqs.setdefault(e["r"], []).append(e)
    server_rids = {e.get("r") for e in run.events if e["n"] == "req.arrive"}
    per_round = defaultdict(lambda: defaultdict(list))
    detail = {}
    for rid, evs in reqs.items():
        send = next((e for e in evs if e["n"] == "client.send"), None)
        if send is None:
            continue
        tag = f_(send, "tag", rid)
        rnd = tag.split("-", 1)[0]
        sse = [e for e in evs if e["n"] == "client.sse"]
        content = [e for e in sse if f_(e, "content")]
        end = next((e for e in evs if e["n"] == "client.end"), None)
        bucket = per_round[rnd]
        bucket["n"].append(1)
        if end is not None and f_(end, "ok"):
            bucket["ok"].append(1)
        if sse:
            bucket["first_frame"].append((sse[0]["w"] - send["w"]) / 1e6)
        if content:
            first = (content[0]["w"] - send["w"]) / 1e6
            bucket["first_content"].append(first)
            gaps = [(b["w"] - a["w"]) / 1e6 for a, b in zip(content, content[1:])]
            bucket["gap"].extend(gaps)
            detail[rid] = dict(first_content_ms=first, max_gap_ms=max(gaps, default=0))
        if end is not None:
            bucket["e2e"].append((end["w"] - send["w"]) / 1e6)
        bucket["matched"].append(int(rid in server_rids))
    rows, summary = [], {}
    for rnd in sorted(per_round, key=lambda r: int(r.lstrip("r") or 0)):
        b = per_round[rnd]
        ff, fc, gap, e2e = (
            dist(b[k]) for k in ("first_frame", "first_content", "gap", "e2e")
        )
        summary[rnd] = dict(first_frame=ff, first_content=fc, gap=gap, e2e=e2e)
        rows.append(
            [
                rnd,
                f"{len(b['ok'])}/{len(b['n'])}",
                fmt(ff["p50"]),
                fmt(fc["p50"]),
                fmt(fc["max"]),
                fmt(gap["p50"]),
                fmt(gap["p99"]),
                fmt(gap["max"]),
                fmt(e2e["p50"]),
                f"{sum(b['matched'])}/{len(b['matched'])}",
            ]
        )
    return rows, summary, detail, bool(reqs)


# --------------------------------------------------- decode request states


def decode_states(run):
    """Per-request mutually exclusive D-side state segments plus hints."""
    plans = {}
    for e in run.events:
        if e["n"] == "linker.plan":
            plans.setdefault(e["r"], e)  # first rank to plan = representative
    out = {}
    for rid, plan in plans.items():
        src, room = plan["src"], f_(plan, "room")
        evs = [
            e
            for e in run.events
            if e["src"] == src
            and e["w"] >= plan["w"]
            and (run.rid_of(e) == rid or (room is not None and f_(e, "room") == room))
        ]
        f = run.files[src]
        cutoff = None if f.last_flush is None else f.last_flush + f.offset
        out[rid] = _walk_states(plan, evs, cutoff)
    return out


def _walk_states(plan, evs, cutoff=None):
    """``cutoff``: the process's last confirmed flush; an unfinished request
    stays in its last state until then (a stall leaves no further events)."""
    page = f_(plan, "page", 1)
    window = f_(plan, "window_pages", 1)
    group = f_(plan, "group", 1)
    prefix_pages = f_(plan, "prefix_len", 0) // page
    loaded_len, loaded_layers = f_(plan, "loaded_len", 0), 0
    num_layers = 0
    for e in evs:
        layers = f_(e, "layers")
        if e["n"] in ("notify.recv", "linker.load_submit") and layers:
            num_layers = max(num_layers, layers[1])
    num_layers = num_layers or group
    ready = defaultdict(list)
    inflight = None  # (start_page, end_page, layer, layer_end)
    decide = (None, None)
    released = None  # the event that ended the D-side interval
    outcome = "open"  # released | aborted | open
    failed = False
    segments, hints = [], []

    def key(layer):
        return (layer, min(layer + group, num_layers))

    def window_ready(start, layer):
        end = min(start + window, prefix_pages)
        return start < end and covered_until(ready[key(layer)], start) >= end

    def state():
        if failed:
            return "失败处理中"
        if inflight is not None:
            return "load进行中"
        if loaded_len // page >= prefix_pages:
            return "prefix完成·等传输/放行"
        start = loaded_len // page
        if window_ready(start, loaded_layers):
            if decide == (1, 0):
                return "本rank就绪·等其他rank"
            return "READY已到·未提交"
        return "等待READY"

    def hint():
        if inflight is None:
            return False
        start, end, layer, layer_end = inflight
        if layer_end >= num_layers:
            return end < prefix_pages and window_ready(end, 0)
        return window_ready(start, layer_end)

    last_w = plan["w"]

    def advance(w):
        if w > last_w:
            segments.append((last_w, w, state()))
            if hint():
                hints.append((last_w, w))

    for e in evs:
        w = e["w"]
        advance(w)
        last_w = w
        n = e["n"]
        if n == "notify.recv":
            (a, b), (layer, layer_end) = f_(e, "pages"), f_(e, "layers")
            ready[(layer, layer_end)] = merge_range(ready[(layer, layer_end)], a, b)
        elif n == "linker.load_submit":
            (a, b), (layer, layer_end) = f_(e, "pages"), f_(e, "layers")
            inflight = (a, b, layer, layer_end)
        elif n == "linker.load_applied":
            inflight = None
            loaded_len = f_(e, "loaded_len", loaded_len)
            loaded_layers = f_(e, "loaded_layers", loaded_layers)
            failed = failed or not f_(e, "ok", True)
        elif n == "linker.decide":
            decide = (f_(e, "local"), f_(e, "merged"))
            failed = failed or -1 in decide
            loaded_len = f_(e, "loaded_len", loaded_len)
            loaded_layers = f_(e, "loaded_layers", loaded_layers)
        elif n in ("pd.transferred", "req.abort"):
            released = e
            outcome = "released" if n == "pd.transferred" else "aborted"
            break
    open_ms = 0.0
    if released is None and cutoff is not None and cutoff > last_w:
        open_ms = (cutoff - last_w) / 1e6
        advance(cutoff)
    return dict(
        plan=plan,
        segments=segments,
        hints=hints,
        released=released,
        outcome=outcome,
        open_ms=open_ms,  # time in the last state up to the capture cutoff
        last=released or (evs[-1] if evs else plan),
        prefix_pages=prefix_pages,
        num_layers=num_layers,
    )


def state_section(states):
    totals = defaultdict(list)
    per_rid = []
    for rid, s in states.items():
        acc = defaultdict(float)
        for a, b, name in s["segments"]:
            acc[name] += (b - a) / 1e6
        hint_ms = sum((b - a) / 1e6 for a, b in s["hints"])
        total = sum(acc.values())
        for name in STATE_ORDER:
            totals[name].append(acc.get(name, 0.0))
        totals[HINT].append(hint_ms)
        per_rid.append((total, rid, acc, hint_ms, s))
    per_rid.sort(reverse=True)
    return totals, per_rid


# ------------------------------------------------------------ storage ops


def storage_section(run):
    by_thread = defaultdict(list)
    for e in run.events:
        if e["n"].startswith("od."):
            by_thread[(e["src"], e["th"])].append(e)
    consumed = {}
    for e in run.events:
        if e["n"] == "uc.load.consumed":
            consumed.setdefault((e["src"], f_(e, "batch")), e["w"])
    enqueued = {}
    for e in run.events:
        if e["n"] == "od.get.enqueue":
            enqueued[(e["src"], f_(e, "batch"))] = e
    get = defaultdict(list)
    put = defaultdict(list)
    native = defaultdict(list)
    native_wait = defaultdict(list)
    slow_batches = []
    for (src, _), evs in by_thread.items():
        cur = None
        last_native_end = None
        for e in evs:
            n = e["n"]
            if n == "od.native":
                op = f_(e, "op")
                native[op].append(e["d"] / 1e6)
                native_wait[op].append(f_(e, "wait_us", 0) / 1e3)
                last_native_end = e["w"] + e["d"]
                if cur is not None:
                    cur["native"] += e["d"] / 1e6
            elif n == "od.get.dequeue":
                enq = enqueued.get((src, f_(e, "batch")))
                cur = dict(
                    op="get", batch=f_(e, "batch"), deq=e["w"], native=0.0, enq=enq
                )
                if enq is not None:
                    get["入队→worker取出"].append((e["w"] - enq["w"]) / 1e6)
            elif n == "od.get.dep_ready" and cur is not None and cur["op"] == "get":
                cur["dep"] = e["w"]
                get["取出→前置依赖就绪"].append((e["w"] - cur["deq"]) / 1e6)
            elif n == "od.get.done" and cur is not None and cur["op"] == "get":
                get["native调用合计"].append(cur["native"])
                if last_native_end is not None and "dep" in cur:
                    get["最后native返回→worker观察完成"].append(
                        (e["w"] - max(last_native_end, cur["dep"])) / 1e6
                    )
                done_seen = consumed.get((src, cur["batch"]))
                if done_seen is not None:
                    get["worker观察完成→scheduler消费"].append(
                        (done_seen - e["w"]) / 1e6
                    )
                if cur["enq"] is not None:
                    total = (e["w"] - cur["enq"]["w"]) / 1e6
                    get["入队→worker观察完成"].append(total)
                    slow_batches.append((total, src, cur["batch"], cur["enq"]))
                cur = None
            elif n == "od.put.dequeue":
                cur = dict(op="put", deq=e["w"], native=0.0)
            elif n == "od.put.dep_ready" and cur is not None and cur["op"] == "put":
                put["取出→前置依赖就绪(每层)"].append((e["w"] - cur["deq"]) / 1e6)
            elif n == "od.put.group_done":
                if last_native_end is not None:
                    put["层组最后native返回→worker观察完成"].append(
                        (e["w"] - last_native_end) / 1e6
                    )
    put_enq = {}
    for e in run.events:
        key = (e["src"], f_(e, "session"), f_(e, "layer"))
        if e["n"] == "od.put.enqueue":
            put_enq[key] = e["w"]
        elif e["n"] == "od.put.dequeue" and key in put_enq:
            put["入队→worker取出(每层)"].append((e["w"] - put_enq[key]) / 1e6)
    sess = {}
    for e in run.events:
        key = (e["src"], f_(e, "session"))
        if e["n"] == "uc.save.begin":
            sess[key] = dict(begin=e["w"])
        elif key in sess:
            if e["n"] == "od.put.session_done":
                sess[key]["done"] = e["w"]
                put["session开始→worker观察完成"].append(
                    (e["w"] - sess[key]["begin"]) / 1e6
                )
            elif e["n"] == "uc.save.consumed" and "done" in sess[key]:
                put["session完成→scheduler回收"].append(
                    (e["w"] - sess[key]["done"]) / 1e6
                )
    slow_batches.sort(key=lambda x: x[0], reverse=True)
    return get, put, native, native_wait, slow_batches[:10]


# ----------------------------------------------------------------- notify


def notify_section(run):
    """P send -> D receive delay, paired per receiver (``dst`` session) in
    FIFO order. A key sent by more than one P process cannot be attributed to
    one sender, so it is counted as ambiguous instead of mixing rank skew in.
    Receives with no earlier send record are counted as unmatched (the send
    is recorded before the send call, so this means missing data).
    Returns (delays, group_to_ready, ambiguous, unmatched)."""

    def key(e):
        return (
            f_(e, "room"),
            f_(e, "dst"),
            tuple(f_(e, "pages")),
            tuple(f_(e, "layers")),
        )

    senders = defaultdict(set)
    for e in run.events:
        if e["n"] == "notify.send":
            senders[key(e)].add(e["src"])
    sent, delays, group_to_ready = defaultdict(list), [], []
    ambiguous = unmatched = 0
    for e in run.events:
        if e["n"] == "notify.send":
            sent[key(e)].append(e)
        elif e["n"] == "notify.recv":
            k = key(e)
            if len(senders.get(k, ())) > 1:
                ambiguous += 1
                continue
            if not sent.get(k):
                unmatched += 1
                continue
            s = sent[k].pop(0)
            if run.files[s["src"]].host == run.files[e["src"]].host:
                delays.append((e["w"] - s["w"]) / 1e6)
    last_group = {}
    for e in run.events:
        if e["n"] == "od.put.group_done":
            last_group[e["src"]] = e["w"]
        elif e["n"] == "notify.ready" and e["src"] in last_group:
            gap = (e["w"] - last_group[e["src"]]) / 1e6
            if gap >= 0:
                group_to_ready.append(gap)
    return delays, group_to_ready, ambiguous, unmatched


# -------------------------------------------------------------- scheduler


def scheduler_section(run):
    stats = defaultdict(lambda: [0, 0, 0])
    for f in run.files:
        for s in f.stats:
            entry = stats[(f.role, s["n"])]
            entry[0] += s["c"]
            entry[1] += s["s"]
            entry[2] = max(entry[2], s["m"])
    batches = defaultdict(list)
    slow = []
    for e in run.events:
        if e["n"] == "sched.run_batch":
            role = run.files[e["src"]].role
            batches[(role, f_(e, "mode"))].append(e["d"] / 1e6)
        elif "d" in e and (e["n"].startswith("sched.") or e["n"] == "linker.consensus"):
            slow.append(e)
    slow.sort(key=lambda e: e["d"], reverse=True)
    return stats, batches, slow[:20]


# ----------------------------------------------------------------- report


def build_report(run):
    lines = [
        f"# UCLinker trace report: {os.path.basename(os.path.normpath(run.dir))}",
        "",
    ]
    clock = (
        "跨机：各主机内部用同一 perf 时钟；主机间按各自 wall 时钟锚点粗对齐，"
        "未校准，跨机差值和重叠不作结论。"
        if run.cross_host
        else "单机：所有进程共用同一 perf 时钟（CLOCK_MONOTONIC），可直接比较。"
    )
    span_ms = run.ms(run.events[-1]["w"]) if run.events else 0
    lines += [
        f"- 主机：{', '.join(run.hosts)}；时钟域：{clock}",
        f"- 事件：{len(run.events)}，时间跨度 {span_ms / 1000:.1f} s",
        "",
    ]
    summary = {}

    rows, problems, unclosed = completeness(run)
    lines += ["## 1. 数据完整性", ""]
    lines += [
        table(["角色", "进程", "主机", "pid", "事件数", "最后刷盘 s", "状态"], rows),
        "",
    ]
    if unclosed:
        lines += [
            "未闭合的操作（开始有记录、结束没有；可能仍在进行、卡住或丢失）：",
            "",
        ]
        lines += [table(["开始事件", "结束事件", "数量"], unclosed), ""]
    lines += [f"- {p}" for p in problems] or ["- 未发现丢弃或缺失的结束标记。"]
    lines += [""]
    summary["incomplete"] = bool(problems or unclosed)

    crows, csum, cdetail, has_client = client_section(run)
    lines += ["## 2. 客户端（毫秒）", ""]
    if has_client:
        lines += [
            "首帧 = 第一个 SSE data 帧（rounds.json 旧 TTFT 口径）；首内容 = 第一个带 content 的帧；"
            "帧间隔是相邻带内容帧的间隔，不等于逐 token TBT。",
            "",
            table(
                [
                    "轮",
                    "成功",
                    "首帧 p50",
                    "首内容 p50",
                    "首内容 max",
                    "帧间隔 p50",
                    "帧间隔 p99",
                    "帧间隔 max",
                    "端到端 p50",
                    "rid 匹配服务端",
                ],
                crows,
            ),
            "",
        ]
    else:
        lines += ["没有客户端事件（客户端未开启 SGLANG_UCL_TRACE）。", ""]
    summary["client"] = csum

    states = decode_states(run)
    totals, per_rid = state_section(states)
    lines += ["## 3. D 侧请求状态（毫秒）", ""]
    if states:
        lines += [
            "状态互斥，按事件逐段划分，加起来等于从 D 预分配到放行（或 abort）的时间；"
            "未结束的请求停在最后状态，计到该进程最后一次刷盘（仍在进行）。"
            "并行进度看 trace.json 的“请求并行轨道”。最后一列是提示，可与 load进行中 重叠。",
            "",
        ]
        rows = []
        for name in STATE_ORDER + (HINT,):
            d = dist(totals[name])
            rows.append([name, fmt(d["sum"]), fmt(d["p50"]), fmt(d["max"])])
        lines += [table(["状态", "合计", "每请求 p50", "每请求 max"], rows), ""]
        lines += ["最慢 10 个请求：", ""]
        rows = []
        for total, rid, acc, hint_ms, s in per_rid[:10]:
            end = outcome_text(s)
            parts = "，".join(f"{k} {acc[k]:.0f}" for k in STATE_ORDER if acc.get(k))
            first = cdetail.get(rid, {}).get("first_content_ms")
            rows.append([rid, fmt(total), parts, fmt(hint_ms), fmt(first), end])
        lines += [
            table(["rid", "D 侧合计", "分解", "提示", "客户端首内容", "结局"], rows),
            "",
        ]
    else:
        lines += ["没有 linker.plan 事件（D 未开启 notify 或未开启 trace）。", ""]
    summary["decode_states"] = {k: dist(v) for k, v in totals.items()}

    get, put, native, native_wait, slow_batches = storage_section(run)
    lines += [
        "## 4. OceanDisk 操作（毫秒）",
        "",
        "“worker观察完成”是 worker 线程等到 CUDA event 后的时间，不等于设备完成时间。",
        "",
    ]
    rows = [["GET " + k, *dist_cells(v)] for k, v in get.items()]
    rows += [["PUT " + k, *dist_cells(v)] for k, v in put.items()]
    rows += [[f"native {op} 调用", *dist_cells(v)] for op, v in native.items()]
    rows += [
        [f"native {op} 进入前等待（锁/上下文）", *dist_cells(v)]
        for op, v in native_wait.items()
    ]
    if rows:
        lines += [table(["阶段", "次数", "p50", "p99", "max"], rows), ""]
    if slow_batches:
        lines += ["最慢 10 个 GET batch：", ""]
        lines += [
            table(
                ["入队→观察完成", "进程", "batch", "tags", "页数"],
                [
                    [
                        fmt(t),
                        run.files[src].label,
                        b,
                        len(f_(enq, "tags", [])),
                        sum(f_(enq, "pages", [])),
                    ]
                    for t, src, b, enq in slow_batches
                ],
            ),
            "",
        ]
    summary["get"] = {k: dist(v) for k, v in get.items()}
    summary["put"] = {k: dist(v) for k, v in put.items()}

    delays, group_to_ready, ambiguous, unmatched = notify_section(run)
    lines += ["## 5. notify（毫秒）", ""]
    rows = [["P 层组完成被观察→notify.ready", *dist_cells(group_to_ready)]]
    if run.cross_host:
        rows.append(["P 发送→D 接收", "跨机未校准，不计算", "", "", ""])
    else:
        rows.append(
            ["P 发送→D 接收（含发送调用，同一接收方按顺序配对）", *dist_cells(delays)]
        )
    lines += [table(["阶段", "次数", "p50", "p99", "max"], rows), ""]
    if ambiguous:
        lines += [
            f"- {ambiguous} 条接收无法归属到唯一发送进程（多个 P 进程向同一接收方"
            "发送了相同范围），未计入传递延迟。",
            "",
        ]
    if unmatched:
        lines += [
            f"- {unmatched} 条接收找不到更早的发送记录（P 侧事件丢失或 P 未开 trace），"
            "未计入传递延迟。",
            "",
        ]

    stats, batches, slow = scheduler_section(run)
    lines += [
        "## 6. 调度",
        "",
        "run_batch 是 CPU 侧耗时（overlap 模式下主要是提交），不是 GPU 执行区间。",
        "",
    ]
    rows = [[role, mode, *dist_cells(v)] for (role, mode), v in sorted(batches.items())]
    if rows:
        lines += [table(["角色", "batch 类型", "次数", "p50", "p99", "max"], rows), ""]
    rows = [
        [role, name, c, fmt(s / 1e6), fmt(s / c / 1e3 if c else 0), fmt(m / 1e6, 2)]
        for (role, name), (c, s, m) in sorted(stats.items())
    ]
    if rows:
        lines += [
            table(["角色", "阶段", "调用次数", "合计 ms", "平均 us", "max ms"], rows),
            "",
        ]
    if slow:
        lines += ["最慢的调度调用（超过阈值才单独记录）：", ""]
        lines += [
            table(
                ["时间 s", "进程", "名称", "耗时 ms"],
                [
                    [
                        fmt(run.ms(e["w"]) / 1000, 3),
                        run.files[e["src"]].label,
                        e["n"],
                        fmt(e["d"] / 1e6, 2),
                    ]
                    for e in slow
                ],
            ),
            "",
        ]

    lines += ["## 7. 未完成与异常", ""]
    issues = []
    for rid, s in states.items():
        if s["outcome"] == "open":
            issues.append(
                [
                    rid,
                    f"D 未放行（停在 {s['segments'][-1][2]} {s['open_ms']:.0f} ms，"
                    "仍在进行）"
                    if s["open_ms"] and s["segments"]
                    else "D 未放行",
                    s["last"]["n"],
                    fmt(run.ms(s["last"]["w"]) / 1000, 3),
                ]
            )
    for e in run.events:
        if e["n"] in ("req.abort", "notify.fail", "notify.recv_invalid") or (
            e["n"] in ("od.get.done", "linker.load_applied", "client.end")
            and f_(e, "ok") is False
        ):
            issues.append(
                [
                    run.rid_of(e) or "-",
                    e["n"],
                    str(f_(e, "reason") or f_(e, "error") or ""),
                    fmt(run.ms(e["w"]) / 1000, 3),
                ]
            )
    if issues:
        lines += [table(["rid", "情况", "最后事件/原因", "时间 s"], issues[:50]), ""]
    else:
        lines += ["无。", ""]
    return "\n".join(lines) + "\n", summary


def outcome_text(s):
    if s["outcome"] == "released":
        return "已放行"
    if s["outcome"] == "aborted":
        return "abort"
    tail = f"，最后状态已持续 {s['open_ms']:.0f}（仍在进行）" if s["open_ms"] else ""
    return f"未放行，最后事件 {s['last']['n']}{tail}"


def dist_cells(values):
    d = dist(values)
    return [d["n"], fmt(d["p50"], 2), fmt(d["p99"], 2), fmt(d["max"], 2)]


# --------------------------------------------------------------- timeline


def build_trace(run):
    out = []
    us = lambda w: (w - run.t0) / 1000  # noqa: E731
    for index, f in enumerate(run.files):
        out.append(dict(ph="M", name="process_name", pid=index, args=dict(name=f.name)))
    for e in run.events:
        args = dict(e.get("f") or {})
        if e.get("r"):
            args["rid"] = e["r"]
        item = dict(
            name=e["n"], pid=e["src"], tid=str(e["th"]), ts=us(e["w"]), args=args
        )
        if "d" in e:
            item.update(ph="X", dur=e["d"] / 1000)
        else:
            item.update(ph="i", s="t")
        out.append(item)
    out += request_tracks(run, decode_states(run), len(run.files), us)
    return dict(traceEvents=out, displayTimeUnit="ms")


LANES = (
    "客户端",
    "P 调度",
    "P PUT",
    "P READY",
    "D READY",
    "D 状态",
    "D 提示",
    "D load",
    "D 传输/放行",
    "D 调度",
)


def request_tracks(run, states, pid, us):
    """One group of lanes per request, joined through rid, room, load tag,
    GET batch and PUT session, so READY, load, transfer and release of
    concurrent requests line up without hunting across threads."""
    role = lambda e: run.files[e["src"]].role  # noqa: E731
    owner = {}  # (rid, role) -> representative src (first scheduler seen)
    first = {}
    for e in run.events:
        rid = run.rid_of(e)
        if rid is None:
            for r in f_(e, "rids") or ():
                first.setdefault(r, e["w"])
            continue
        first.setdefault(rid, e["w"])
        if e["n"] in ("req.arrive", "linker.plan"):
            owner.setdefault((rid, role(e)), e["src"])
    for rid, s in states.items():
        owner[(rid, "decode")] = s["plan"]["src"]
    order = {rid: i for i, rid in enumerate(sorted(first, key=first.get))}
    lane_of = {name: i for i, name in enumerate(LANES)}
    out, used = [], set()

    def tid(rid, lane):
        t = order[rid] * len(LANES) + lane_of[lane] if rid in order else None
        if t is not None and t not in used:
            used.add(t)
            out.append(
                dict(
                    ph="M",
                    name="thread_name",
                    pid=pid,
                    tid=t,
                    args=dict(name=f"{rid} · {lane}"),
                )
            )
        return t

    def add(rid, lane, name, a, b=None, **args):
        t = tid(rid, lane)
        if t is None:
            return
        item = dict(name=name, pid=pid, tid=t, ts=us(a), args=args)
        if b is None:
            item.update(ph="i", s="t")
        else:
            item.update(ph="X", dur=max(b - a, 0) / 1000)
        out.append(item)

    def mine(e, rid, r):
        return owner.get((rid, r)) == e["src"]

    sends, loads, batches, sessions, prealloc = {}, {}, {}, {}, {}
    for e in run.events:
        n, rid = e["n"], run.rid_of(e)
        if n == "client.send":
            sends[e["r"]] = e
        elif n == "client.sse" and f_(e, "content") and e["r"] in sends:
            if not sends[e["r"]].get("_first"):
                sends[e["r"]]["_first"] = True
                add(e["r"], "客户端", "首内容", e["w"])
        elif n == "client.end" and e["r"] in sends:
            a = sends.pop(e["r"])
            add(e["r"], "客户端", f_(a, "tag", e["r"]), a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "sched.run_batch":
            for r in f_(e, "rids") or ():
                lane = "P 调度" if role(e) == "prefill" else "D 调度"
                if mine(e, r, role(e)):
                    add(r, lane, f"run_batch {f_(e, 'mode')}", e["w"], e["w"] + e["d"])
        elif n == "uc.save.begin":
            for r in f_(e, "rids") or ():
                if mine(e, r, "prefill"):
                    sessions.setdefault((e["src"], f_(e, "session")), []).append(
                        (r, e["w"])
                    )
        elif n in ("od.put.session_done", "uc.save.consumed"):
            for r, begin in sessions.get((e["src"], f_(e, "session")), ()):
                if n == "od.put.session_done":
                    add(r, "P PUT", "PUT session", begin, e["w"], ok=f_(e, "ok"))
                else:
                    add(r, "P PUT", "scheduler 回收", e["w"])
        elif n in ("notify.ready", "notify.send", "notify.rdma_done", "notify.fail"):
            if rid and role(e) == "prefill" and mine(e, rid, "prefill"):
                add(rid, "P READY", n, e["w"], pages=f_(e, "pages"))
        elif n in ("notify.recv", "notify.recv_invalid"):
            if rid and mine(e, rid, "decode"):
                add(
                    rid,
                    "D READY",
                    n,
                    e["w"],
                    pages=f_(e, "pages"),
                    layers=f_(e, "layers"),
                )
        elif n == "linker.load_submit" and mine(e, rid, "decode"):
            loads[(e["src"], e["r"])] = e
        elif n == "linker.load_applied" and (e["src"], e["r"]) in loads:
            a = loads.pop((e["src"], e["r"]))
            add(rid, "D load", e["r"].split("@", 1)[-1], a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "od.get.enqueue":
            batches[(e["src"], f_(e, "batch"))] = e
        elif n == "od.get.done" and (e["src"], f_(e, "batch")) in batches:
            a = batches.pop((e["src"], f_(e, "batch")))
            for tag in f_(a, "tags") or ():
                r = tag.split("@", 1)[0]
                if mine(a, r, "decode"):
                    name = f"GET batch {f_(e, 'batch')}"
                    add(r, "D load", name, a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "pd.prealloc" and mine(e, rid, "decode"):
            prealloc[rid] = e
        elif n in ("pd.transferred", "req.abort") and mine(e, rid, "decode"):
            if rid not in prealloc:
                continue
            a = prealloc.pop(rid)
            label = "传输完成·放行" if n == "pd.transferred" else "abort"
            add(rid, "D 传输/放行", label, a["w"], e["w"])
        elif n == "pd.conclude" and rid and mine(e, rid, role(e)):
            lane = "D 传输/放行" if role(e) == "decode" else "P READY"
            add(rid, lane, f"conclude {f_(e, 'status')}", e["w"])

    # Still open at the capture cutoff: draw to the owner's last flush, named
    # as in progress (trace only; report distributions use completed ones).
    def cutoff(e):
        f = run.files[e["src"]]
        return None if f.last_flush is None else f.last_flush + f.offset

    for (_, tag), a in loads.items():
        if cutoff(a) is not None:
            name = f"{tag.split('@', 1)[-1]} 进行中"
            add(run.rid_of(a), "D load", name, a["w"], cutoff(a))
    for (_, batch), a in batches.items():
        for tag in f_(a, "tags") or ():
            r = tag.split("@", 1)[0]
            if mine(a, r, "decode") and cutoff(a) is not None:
                add(r, "D load", f"GET batch {batch} 进行中", a["w"], cutoff(a))
    for rid, a in prealloc.items():
        f = run.files[a["src"]]
        if f.last_flush is not None:
            add(
                rid,
                "D 传输/放行",
                "未放行（仍在进行）",
                a["w"],
                f.last_flush + f.offset,
            )
    for rid, s in states.items():
        for a, b, name in s["segments"]:
            add(rid, "D 状态", name, a, b)
        for a, b in s["hints"]:
            add(rid, "D 提示", HINT, a, b)
    out.insert(
        0, dict(ph="M", name="process_name", pid=pid, args=dict(name="请求并行轨道"))
    )
    return out


# -------------------------------------------------------------------- CLI


def cmd_report(args):
    if args.wait:
        time.sleep(args.wait)  # let live servers write their next flush
    run = Run(args.dir)
    out = args.out or args.dir
    os.makedirs(out, exist_ok=True)
    text, summary = build_report(run)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(out, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(build_trace(run), f)
    print(f"report: {os.path.join(out, 'report.md')}")
    print(
        f"timeline: {os.path.join(out, 'trace.json')} (chrome://tracing or ui.perfetto.dev)"
    )


def cmd_merge(args):
    run = Run(args.dir)
    path = args.output or os.path.join(args.dir, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_trace(run), f)
    print(path)


def _load_summary(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "summary.json")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
        return build_report(Run(path))[1]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_rows(a, b):
    rows = []

    def add(label, x, y):
        if x is None and y is None:
            return
        delta = "-"
        if x not in (None, 0) and y is not None:
            delta = f"{(y - x) / x * 100:+.1f}%"
        rows.append([label, fmt(x), fmt(y), delta])

    for rnd in sorted(
        set(a.get("client", {})) | set(b.get("client", {})),
        key=lambda r: int(r.lstrip("r") or 0),
    ):
        ca, cb = a.get("client", {}).get(rnd, {}), b.get("client", {}).get(rnd, {})
        for metric, label in (
            ("first_content", "首内容"),
            ("first_frame", "首帧"),
            ("gap", "帧间隔"),
        ):
            for q in ("p50", "max"):
                add(
                    f"{rnd} {label} {q}",
                    ca.get(metric, {}).get(q),
                    cb.get(metric, {}).get(q),
                )
    for section, label in (
        ("decode_states", "D状态合计"),
        ("get", "GET"),
        ("put", "PUT"),
    ):
        keys = set(a.get(section, {})) | set(b.get(section, {}))
        for k in sorted(keys):
            field = "sum" if section == "decode_states" else "p50"
            add(
                f"{label} {k} {field}",
                a.get(section, {}).get(k, {}).get(field),
                b.get(section, {}).get(k, {}).get(field),
            )
    return rows


def cmd_compare(args):
    a, b = _load_summary(args.a), _load_summary(args.b)
    for name, s in ((args.a, a), (args.b, b)):
        if s.get("incomplete"):
            print(f"注意：{name} 的数据不完整，见其 report.md 第 1 节。")
    print(table(["指标（ms）", "A", "B", "B 相对 A"], compare_rows(a, b)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="report.md + trace.json + summary.json")
    p.add_argument("dir")
    p.add_argument("--out", help="output directory (default: the run directory)")
    p.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="seconds to wait first so running servers flush",
    )
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("merge", help="write only the Chrome/Perfetto trace")
    p.add_argument("dir")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_merge)
    p = sub.add_parser("compare", help="compare two runs (dirs or summary.json)")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
