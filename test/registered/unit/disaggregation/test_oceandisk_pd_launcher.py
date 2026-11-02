"""Launcher lifecycle tests using fake services; no model, GPU or network required."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "examples/runtime/engine/run_oceandisk_pd_tp4.sh"


class TestOceanDiskPDStop(unittest.TestCase):
    def setUp(self):
        self.code = (
            SCRIPT.read_text(encoding="utf-8")
            .split('if [ "$mode" = stop ]; then\n', 1)[1]
            .split("<<'PY'\n", 1)[1]
            .split("\nPY\n", 1)[0]
        )
        self.psutil = Mock(
            CONN_LISTEN="LISTEN", STATUS_ZOMBIE="zombie", NoSuchProcess=ProcessLookupError
        )
        self.psutil.net_connections.return_value = []
        self.psutil.wait_procs.return_value = ([], [])

    def process(self, pid, command="sglang::tokenizer"):
        proc = Mock(pid=pid)
        proc.cmdline.return_value = [command]
        proc.children.return_value = []
        return proc

    def connection(self, pid, port=31002, status="LISTEN", ip="127.0.0.1"):
        return SimpleNamespace(
            pid=pid, laddr=SimpleNamespace(ip=ip, port=port), status=status
        )

    def run_stop(self):
        with patch.dict(sys.modules, psutil=self.psutil), patch("builtins.print"):
            exec(compile(self.code, str(SCRIPT), "exec"), {})

    def test_current_listeners_and_children_are_stopped_once(self):
        parents = [self.process(pid) for pid in (101, 102, 103)]
        child = self.process(201, "sglang::scheduler")
        parents[0].children.return_value = [child]
        self.psutil.Process.side_effect = {p.pid: p for p in parents}.__getitem__
        connections = [
            self.connection(p.pid, port)
            for p, port in zip(parents, (31002, 31001, 31345))
        ]
        self.psutil.net_connections.return_value = connections + connections[:1]
        self.run_stop()
        for proc in [*parents, child]:
            proc.terminate.assert_called_once_with()
            proc.kill.assert_not_called()
        for proc in parents:
            proc.children.assert_called_with(recursive=True)

    def test_other_ports_and_non_listeners_are_ignored(self):
        self.psutil.net_connections.return_value = [
            self.connection(101, port=9000),
            self.connection(102, status="ESTABLISHED"),
            self.connection(103, ip="127.0.0.2"),
        ]
        self.run_stop()
        self.psutil.Process.assert_not_called()

    def test_all_listeners_are_checked_before_signalling(self):
        parent = self.process(101)
        for other in (None, 102):
            with self.subTest(other=other):
                self.psutil.net_connections.return_value = [
                    self.connection(101), self.connection(other, port=31345)
                ]
                self.psutil.Process.side_effect = lambda pid: (
                    parent if pid == 101 else self.process(pid, "unrelated-server")
                )
                with self.assertRaises(SystemExit):
                    self.run_stop()
                parent.terminate.assert_not_called()

    def test_exited_service_and_child_do_not_interrupt_cleanup(self):
        parent, child = self.process(101), self.process(201)
        parent.children.return_value = [child]
        child.terminate.side_effect = ProcessLookupError
        self.psutil.net_connections.return_value = [
            self.connection(102), self.connection(101)
        ]
        self.psutil.Process.side_effect = [ProcessLookupError, parent]
        self.run_stop()
        parent.terminate.assert_called_once_with()

    def test_term_timeout_kills_remaining_processes(self):
        parent = self.process(101)
        self.psutil.net_connections.return_value = [self.connection(101)]
        self.psutil.Process.return_value = parent
        self.psutil.wait_procs.side_effect = [([], [parent]), ([parent], [])]
        self.run_stop()
        parent.kill.assert_called_once_with()
        self.assertEqual(self.psutil.wait_procs.call_args_list[0].kwargs, {"timeout": 5})
        self.assertEqual(self.psutil.wait_procs.call_args_list[1].kwargs, {"timeout": 5})

    def test_remaining_process_is_reported_as_failure(self):
        parent = self.process(101)
        self.psutil.net_connections.return_value = [self.connection(101)]
        self.psutil.Process.return_value = parent
        self.psutil.wait_procs.return_value = ([], [parent])
        with self.assertRaisesRegex(SystemExit, "Processes still running: \\[101\\]"):
            self.run_stop()

    def test_zombie_or_reaped_process_is_already_stopped(self):
        for status in ("zombie", ProcessLookupError):
            with self.subTest(status=status):
                parent = self.process(101)
                parent.status.side_effect = [status]
                self.psutil.net_connections.return_value = [self.connection(101)]
                self.psutil.Process.return_value = parent
                self.psutil.wait_procs.return_value = ([], [parent])
                self.run_stop()


@unittest.skipUnless(
    os.name == "posix" and shutil.which("sh") and shutil.which("setsid"),
    "requires POSIX sh and setsid",
)
class TestOceanDiskPDLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pd-launcher-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        for name in ("bin", "model", "gids", "calls"):
            (self.base / name).mkdir()
        (self.base / "model/config.json").write_text(
            json.dumps({"max_position_embeddings": 200000})
        )
        self.env = dict(
            os.environ,
            PATH=str(self.base / "bin") + os.pathsep + os.environ["PATH"],
            MODEL_PATH=str(self.base / "model"),
            GIDS_DIR=str(self.base / "gids"),
            CALLS=str(self.base / "calls"),
            LOG_DIR=str(self.base / "logs"),
            TRANSFER_BACKEND="notify",
        )
        self.write_tool(
            "curl",
            r"""#!/bin/sh
case "$*" in
    *31002*) role=prefill ;;
    *31001*) role=decode ;;
    *) role=router ;;
esac
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "$CALLS/$role.child" ] && exit 0
    sleep 0.05
done
exit 1
""",
        )
        self.write_tool(
            "python3",
            r"""#!/bin/sh
if [ "$1" = - ]; then cat >/dev/null; exit 0; fi
role=client
if [ "$1" = -m ]; then
    role=router
    previous=
    for arg do
        if [ "$previous" = --disaggregation-mode ]; then role=$arg; fi
        previous=$arg
    done
fi
printf '%s\n' "$$" > "$CALLS/$role.pid"
printf '%s\n' "$@" > "$CALLS/$role.args"
if [ "${FAIL_START_ROLE:-}" = "$role" ]; then exit 7; fi
if [ "$role" = client ]; then
    echo client-output
    touch "$CALLS/client.started"
    if [ "${CLIENT_HANG:-0}" = 0 ]; then exit "${CLIENT_EXIT:-0}"; fi
fi
if [ "${IGNORE_TERM:-0}" = 1 ]; then trap '' TERM; fi
# A child that outlives the group leader exercises group cleanup.
sleep 60 &
printf '%s\n' "$!" > "$CALLS/$role.child"
if [ "${FAIL_ROLE:-}" = "$role" ]; then
    while [ ! -f "$CALLS/client.started" ]; do sleep 0.05; done
    exit 7
fi
wait
""",
        )

    def write_tool(self, name, text):
        path = self.base / "bin" / name
        path.write_text(text)
        path.chmod(0o755)

    def start(self, mode="check", **env):
        for path in (self.base / "calls").iterdir():
            path.unlink()
        process = subprocess.Popen(
            ["sh", str(SCRIPT), mode],
            env=dict(self.env, **env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.stop_process, process)
        return process

    def stop_process(self, process):
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        # Ensure failing tests cannot leave fake service groups behind.
        for path in (self.base / "calls").glob("*.pid"):
            try:
                os.killpg(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def assert_groups_stopped(self):
        groups = {
            path.read_text().strip() for path in (self.base / "calls").glob("*.pid")
        }
        deadline = time.monotonic() + 2
        while True:
            # Ignore zombies waiting for the container's init to reap them.
            result = subprocess.run(
                ["ps", "-eo", "pgid=,stat="], capture_output=True, text=True, check=True
            )
            live = [
                line
                for line in result.stdout.splitlines()
                if line.split()[0] in groups and not line.split()[1].startswith("Z")
            ]
            if not live or time.monotonic() >= deadline:
                self.assertFalse(live)
                return
            time.sleep(0.05)

    def finish(self, process):
        stdout, stderr = process.communicate(timeout=15)
        self.assert_groups_stopped()
        prefill = (self.base / "calls/prefill.args").read_text().splitlines()
        decode = (self.base / "calls/decode.args").read_text().splitlines()
        self.assertIn("--disaggregation-linker-notify", prefill)
        self.assertIn("--disaggregation-linker-notify", decode)
        self.assertIn("--enable-prefill-cp", prefill)
        self.assertEqual(prefill[prefill.index("--cp-strategy") + 1], "interleave")
        self.assertNotIn("--enable-prefill-cp", decode)
        self.assertNotIn("--cp-strategy", decode)
        return process.returncode, stdout, stderr

    def test_normal_exit_and_client_error_preserve_status_and_output(self):
        for status in (0, 7):
            with self.subTest(status=status):
                code, stdout, stderr = self.finish(self.start(CLIENT_EXIT=str(status)))
                self.assertEqual(code, status, stderr)
                self.assertIn("client-output", stdout)

    def test_start_returns_with_services_running_and_no_client(self):
        process = self.start("start")
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        for port in (31345, 31002, 31001):
            self.assertIn(f"http://127.0.0.1:{port}", stdout)
        self.assertFalse((self.base / "calls/client.started").exists())
        pids = (self.base / "logs/server.pids").read_text().splitlines()
        self.assertEqual(len(pids), 3)
        for role, pid in zip(("prefill", "decode", "router"), pids):
            self.assertEqual(pid, (self.base / f"calls/{role}.pid").read_text().strip())
            os.killpg(int(pid), signal.SIGHUP)
        time.sleep(0.1)
        for pid in pids:
            state = subprocess.check_output(
                ["ps", "-o", "stat=", "-p", pid], text=True
            )
            self.assertFalse(state.strip().startswith("Z"), state)

    def test_start_failure_stops_started_services(self):
        for role in ("prefill", "decode", "router"):
            with self.subTest(role=role):
                (self.base / "logs").mkdir(exist_ok=True)
                (self.base / "logs/server.pids").write_text("42\n")
                process = self.start("start", FAIL_START_ROLE=role)
                stdout, _ = process.communicate(timeout=15)
                self.assertNotEqual(process.returncode, 0)
                self.assertNotIn("Ready:", stdout)
                self.assertFalse((self.base / "logs/server.pids").exists())
                self.assert_groups_stopped()

    def test_each_backend_exit_stops_blocked_client_and_other_groups(self):
        for role in ("prefill", "decode", "router"):
            with self.subTest(role=role):
                (self.base / "calls/client.started").unlink(missing_ok=True)
                code, _, stderr = self.finish(
                    self.start(FAIL_ROLE=role, CLIENT_HANG="1")
                )
                self.assertNotEqual(code, 0)
                self.assertIn("stopping experiment", stderr)

    def test_signals_escalate_for_unresponsive_service_and_children(self):
        for sig, status in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig):
                process = self.start(CLIENT_HANG="1", IGNORE_TERM="1")
                deadline = time.monotonic() + 5
                while not (self.base / "calls/client.child").exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                process.send_signal(sig)
                code, _, _ = self.finish(process)
                self.assertEqual(code, status)

    def test_check_and_bench_use_same_supervision(self):
        for mode in ("check", "bench"):
            with self.subTest(mode=mode):
                code, stdout, stderr = self.finish(self.start(mode))
                self.assertEqual(code, 0, stderr)
                self.assertIn("client-output", stdout)
                self.assertIn(
                    "--decode-log", (self.base / "calls/client.args").read_text()
                )


if __name__ == "__main__":
    unittest.main()
