"""Unit tests for the UCLinker trace recorder and its offline report tool."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
RECORDER = ROOT / "python/sglang/srt/observability/ucl_trace.py"
TOOL = ROOT / "examples/runtime/engine/Round/ucl_trace_tool.py"


def load(path, name, env):
    with patch.dict(os.environ, env, clear=False):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


tool = load(TOOL, "ucl_trace_tool", {})


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


MS = 1_000_000


def write_trace(directory, name, role, events, host="h1", ended=True, extra=""):
    """Write one synthetic process file; event times are in milliseconds."""
    lines = [
        dict(type="header", host=host, pid=name, perf_anchor_ns=0, wall_anchor_ns=0)
    ]
    lines.append(dict(t=0, n="process.identity", th=1, f=dict(role=role, label=name)))
    for item in events:
        t, n, r, f = item[:4]
        record = dict(t=int(t * MS), n=n, th=item[5] if len(item) > 5 else 1)
        if r is not None:
            record["r"] = r
        if f:
            record["f"] = f
        if len(item) > 4 and item[4] is not None:
            record["d"] = int(item[4] * MS)
        lines.append(record)
    lines.append(dict(type="flush", t=int(300 * MS), events=len(events), dropped=0))
    if ended:
        lines.append(dict(type="end", t=int(300 * MS), dropped=0))
    with open(os.path.join(directory, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n" + extra)


RID = "r1-0-abcd"
ROOM = 7


def notify_run(directory, *, merged_wait=False):
    """One request: 2 windows x 2 layer groups of 50 layers; page=64."""
    plan = dict(
        room=ROOM,
        prefix_len=2 * 16 * 64,
        loaded_len=0,
        page=64,
        window_pages=16,
        group=50,
    )
    g0, g1 = [0, 50], [50, 61]
    d = [
        (0, "req.arrive", RID, dict(room=ROOM, input_len=3000)),
        (0, "pd.prealloc", RID, dict(room=ROOM, decode_prefix_len=2048)),
        (0, "linker.plan", RID, plan),
        (10, "notify.recv", None, dict(room=ROOM, pages=[0, 32], layers=g0)),
    ]
    if merged_wait:
        # Local rank is ready at 10 ms but the consensus allows it at 15 ms.
        d.append(
            (
                10,
                "linker.decide",
                RID,
                dict(local=1, merged=0, loaded_len=0, loaded_layers=0),
            )
        )
        d.append(
            (
                15,
                "linker.decide",
                RID,
                dict(local=1, merged=1, loaded_len=0, loaded_layers=0),
            )
        )
    d += [
        (16, "linker.load_submit", f"{RID}@0:0", dict(pages=[0, 16], layers=g0)),
        (
            16,
            "od.get.enqueue",
            None,
            dict(batch=0, tags=[f"{RID}@0:0"], pages=[16], layers=g0),
        ),
        (18, "od.get.dequeue", None, dict(batch=0), None, 2),
        (19, "od.get.dep_ready", None, dict(batch=0), None, 2),
        (19, "od.native", None, dict(op="get", wait_us=0), 4, 2),
        (20, "notify.recv", None, dict(room=ROOM, pages=[0, 32], layers=g1)),
        (40, "od.get.layer_ready", None, dict(layer=49), None, 2),
        (41, "od.get.done", None, dict(batch=0, ok=True), None, 2),
        (45, "uc.load.consumed", f"{RID}@0:0", dict(batch=0, ok=True)),
        (
            46,
            "linker.load_applied",
            f"{RID}@0:0",
            dict(ok=True, loaded_len=0, loaded_layers=50),
        ),
        (50, "linker.load_submit", f"{RID}@0:50", dict(pages=[0, 16], layers=g1)),
        (
            70,
            "linker.load_applied",
            f"{RID}@0:50",
            dict(ok=True, loaded_len=1024, loaded_layers=0),
        ),
        (72, "linker.load_submit", f"{RID}@16:0", dict(pages=[16, 32], layers=g0)),
        (
            90,
            "linker.load_applied",
            f"{RID}@16:0",
            dict(ok=True, loaded_len=1024, loaded_layers=50),
        ),
        (92, "linker.load_submit", f"{RID}@16:50", dict(pages=[16, 32], layers=g1)),
        (
            110,
            "linker.load_applied",
            f"{RID}@16:50",
            dict(ok=True, loaded_len=2048, loaded_layers=0),
        ),
        (115, "linker.kv_poll", RID, dict(state="Success")),
        (120, "pd.transferred", RID, dict(room=ROOM)),
        (121, "sched.run_batch", None, dict(mode="PREBUILT", bs=1, rids=[RID]), 0.5),
    ]
    p = [
        (-50, "req.arrive", RID, dict(room=ROOM, input_len=3000)),
        (8, "od.put.group_done", None, dict(session=1, layer_start=0, layer_end=50)),
        (9, "notify.ready", None, dict(room=ROOM, pages=[0, 32], layers=g0)),
        (9.5, "notify.send", None, dict(room=ROOM, pages=[0, 32], layers=g0)),
        (19, "notify.send", None, dict(room=ROOM, pages=[0, 32], layers=g1)),
    ]
    c = [
        (-60, "client.send", RID, dict(tag="r1-0")),
        (125, "client.sse", RID, dict(content=False)),
        (130, "client.sse", RID, dict(content=True)),
        (140, "client.sse", RID, dict(content=True)),
        (141, "client.end", RID, dict(ok=True)),
    ]
    write_trace(directory, "decode", "decode", d)
    write_trace(directory, "prefill", "prefill", p)
    write_trace(directory, "client", "client", c)


class TestRecorder(unittest.TestCase):
    def test_disabled_without_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": ""}
            module = load(RECORDER, "ucl_off", env)
            self.assertFalse(module.ON)
            module = load(
                RECORDER,
                "ucl_off2",
                {"SGLANG_UCL_TRACE": "0", "SGLANG_UCL_TRACE_DIR": tmp},
            )
            self.assertFalse(module.ON)
            self.assertEqual(os.listdir(tmp), [])

    def test_records_events_stats_and_end_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_on",
                {
                    "SGLANG_UCL_TRACE": "1",
                    "SGLANG_UCL_TRACE_DIR": tmp,
                    "SGLANG_UCL_TRACE_SLOW_MS": "0",
                },
            )
            self.assertTrue(module.ON)
            module.set_identity("decode", "scheduler_TP0", tp_rank=0)
            module.emit("req.arrive", "rid-1", room=3)
            module.span("od.native", module.now() - 5000, op="get")
            for _ in range(3):
                module.stat("sched.process_queue", module.now(), slow_ns=None)
            module.stat("linker.consensus", module.now() - 10**7)
            module.close()
            module.emit("after.close")  # ignored, must not reopen or raise
            (name,) = os.listdir(tmp)
            text = read_text(os.path.join(tmp, name))
            records = [json.loads(x) for x in text.splitlines()]
            kinds = [r.get("type", r.get("n")) for r in records]
            self.assertEqual(kinds[0], "header")
            self.assertEqual(kinds[-1], "end")
            names = [r["n"] for r in records if "type" not in r]
            self.assertEqual(names[:3], ["process.identity", "req.arrive", "od.native"])
            self.assertIn("linker.consensus", names)  # slow call kept as a span
            stat = next(
                r
                for r in records
                if r.get("n") == "sched.process_queue" and r.get("type") == "stat"
            )
            self.assertEqual(stat["c"], 3)
            self.assertGreaterEqual(
                next(r for r in records if r.get("n") == "od.native")["d"], 5000
            )
            run = tool.Run(tmp)
            self.assertEqual(run.files[0].role, "decode")
            self.assertTrue(run.files[0].ended)

    def test_round_client_finds_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": tmp}
            rounds = load(
                ROOT / "examples/runtime/engine/Round/run_rounds.py",
                "rounds_traced",
                env,
            )
            self.assertIsNotNone(rounds.ucl_trace)
            self.assertTrue(rounds.UCL_CLIENT_RID)

    def test_bounded_buffer_counts_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_small",
                {
                    "SGLANG_UCL_TRACE": "1",
                    "SGLANG_UCL_TRACE_DIR": tmp,
                    "SGLANG_UCL_TRACE_MAX_BUFFERED": "2",
                    "SGLANG_UCL_TRACE_FLUSH_S": "60",
                },
            )
            for i in range(5):
                module.emit("e", str(i))
            module.close()
            run = tool.Run(tmp)
            self.assertEqual(run.files[0].dropped, 3)
            _, problems, _ = tool.completeness(run)
            self.assertTrue(any("丢弃" in p for p in problems))


class TestReport(unittest.TestCase):
    def test_states_hint_and_storage_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            run = tool.Run(tmp)
            states = tool.decode_states(run)
            acc = {}
            for a, b, name in states[RID]["segments"]:
                acc[name] = acc.get(name, 0) + (b - a) / MS
            self.assertAlmostEqual(acc["等待READY"], 10)
            self.assertAlmostEqual(acc["READY已到·未提交"], 6 + 4 + 2 + 2)
            self.assertAlmostEqual(acc["load进行中"], 30 + 20 + 18 + 18)
            self.assertAlmostEqual(acc["prefix完成·等传输/放行"], 10)
            self.assertAlmostEqual(sum(acc.values()), 120)
            # The next window/layer group was READY while the current load ran:
            # 20-46 (group 1), 50-70 (window 1 group 0), 72-90 (window 1 group 1).
            hint = sum((b - a) / MS for a, b in states[RID]["hints"])
            self.assertAlmostEqual(hint, 26 + 20 + 18)
            get, _, native, _, _ = tool.storage_section(run)
            self.assertEqual(get["入队→worker取出"], [2.0])
            self.assertEqual(get["取出→前置依赖就绪"], [1.0])
            self.assertEqual(get["最后native返回→worker观察完成"], [18.0])
            self.assertEqual(get["worker观察完成→scheduler消费"], [4.0])
            self.assertEqual(native["get"], [4.0])
            delays, group_to_ready = tool.notify_section(run)
            self.assertEqual(delays, [0.5, 1.0])
            self.assertEqual(group_to_ready, [1.0])

    def test_rank_wait_is_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp, merged_wait=True)
            states = tool.decode_states(tool.Run(tmp))
            acc = {}
            for a, b, name in states[RID]["segments"]:
                acc[name] = acc.get(name, 0) + (b - a) / MS
            self.assertAlmostEqual(acc["本rank就绪·等其他rank"], 5)

    def test_report_files_client_columns_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main(["report", tmp])
            text = read_text(os.path.join(tmp, "report.md"))
            self.assertIn("首内容 = 第一个带 content 的帧", text)
            self.assertIn("| r1 | 1/1 | 185.0 | 190.0 |", text)
            trace = json.loads(read_text(os.path.join(tmp, "trace.json")))
            names = {e["args"]["name"] for e in trace["traceEvents"] if e["ph"] == "M"}
            self.assertIn("D 请求状态", names)
            summary = json.loads(read_text(os.path.join(tmp, "summary.json")))
            self.assertFalse(summary["incomplete"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                tool.main(["compare", tmp, tmp])
            self.assertIn("r1 首内容 p50 | 190.0 | 190.0 | +0.0%", out.getvalue())

    def test_killed_process_and_unreleased_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(
                room=1,
                prefix_len=1024,
                loaded_len=0,
                page=64,
                window_pages=16,
                group=50,
            )
            write_trace(
                tmp,
                "decode",
                "decode",
                [
                    (0, "linker.plan", "r1-1-x", plan),
                    (
                        5,
                        "linker.load_submit",
                        "r1-1-x@0:0",
                        dict(pages=[0, 16], layers=[0, 50]),
                    ),
                    (
                        5,
                        "od.get.enqueue",
                        None,
                        dict(batch=3, tags=["r1-1-x@0:0"], pages=[16], layers=[0, 50]),
                    ),
                ],
                ended=False,
                extra='{"t": 9, "n": "torn',
            )
            run = tool.Run(tmp)
            rows, problems, unclosed = tool.completeness(run)
            self.assertIn("损坏行 1", rows[0][-1])
            self.assertTrue(any("没有结束标记" in p for p in problems))
            self.assertEqual(
                {r[0] for r in unclosed}, {"od.get.enqueue", "linker.load_submit"}
            )
            text, summary = tool.build_report(run)
            self.assertTrue(summary["incomplete"])
            self.assertIn("| r1-1-x | D 未放行 | linker.load_submit |", text)

    def test_cross_host_notify_delay_is_not_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            path = os.path.join(tmp, "prefill.jsonl")
            write_text(path, read_text(path).replace('"host": "h1"', '"host": "h2"'))
            run = tool.Run(tmp)
            self.assertTrue(run.cross_host)
            self.assertEqual(tool.notify_section(run)[0], [])
            self.assertIn("跨机未校准", tool.build_report(run)[0])


if __name__ == "__main__":
    unittest.main()
