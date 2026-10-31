"""Launcher lifecycle tests using fake services; no model, GPU or network required."""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "examples/runtime/engine/run_oceandisk_pd_tp4.sh"


@unittest.skipUnless(
    os.name == "posix" and shutil.which("sh") and shutil.which("setsid"),
    "requires POSIX sh and setsid",
)
class TestOceanDiskPDLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pd-launcher-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        for name in ("bin", "model", "gids", "rounds", "calls"):
            (self.base / name).mkdir()
        (self.base / "model/config.json").write_text(
            json.dumps({"max_position_embeddings": 200000})
        )
        (self.base / "rounds/run_rounds.py").touch()
        (self.base / "rounds/doupocangqiong.txt").touch()
        self.env = dict(
            os.environ,
            PATH=str(self.base / "bin") + os.pathsep + os.environ["PATH"],
            MODEL_PATH=str(self.base / "model"),
            GIDS_DIR=str(self.base / "gids"),
            ROUNDS_DIR=str(self.base / "rounds"),
            CALLS=str(self.base / "calls"),
            LOG_DIR=str(self.base / "logs"),
            TRANSFER_BACKEND="chunkload",
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

    def start(self, mode="rounds", **env):
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
        return process.returncode, stdout, stderr

    def test_normal_exit_and_client_error_preserve_status_and_output(self):
        for status in (0, 7):
            with self.subTest(status=status):
                code, stdout, stderr = self.finish(self.start(CLIENT_EXIT=str(status)))
                self.assertEqual(code, status, stderr)
                self.assertIn("client-output", stdout)
                self.assertIn(
                    "client-output", (self.base / "logs/rounds.log").read_text()
                )

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
