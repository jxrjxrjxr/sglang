"""Expand the real standalone scripts without starting GPU services."""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
ENGINE = ROOT / "examples/runtime/engine"
SHELL = shutil.which("sh")
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "GIDS_DIR", "SGLANG_HOST_IP",
    "SGLANG_DISAGGREGATION_QUEUE_SIZE", "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
    "SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "SGL_DG_USE_NVRTC",
)


@unittest.skipUnless(SHELL, "requires a POSIX shell")
class TestOceanDiskNotifyEntrypoints(unittest.TestCase):
    def expand(self, role, **overrides):
        env = {"PATH": os.environ["PATH"], "LOCAL_IP": "141.111.65.134",
               "GIDS_DIR": "/configs/local", "OD_NAMESPACE": "test-shared",
               "MOONCAKE_IB_DEVICES": "mlx5_3,mlx5_5",
               "PREFILL_IP": "141.111.65.132", "DECODE_IP": "141.111.65.134"}
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
        env.update(overrides)
        env = {k: v for k, v in env.items() if v is not None}
        # A shell function intercepts the sole Python command and records only
        # the selected environment, without executing Python or reading secrets.
        capture = "python() { printf '%s\\0' \"$@\"; printf '__ENV__\\0'; "
        capture += "printf '%s\\0' " + " ".join('"${' + k + '-}"' for k in ENV_KEYS)
        capture += '; }; . "$1"'
        result = subprocess.run(
            [SHELL, "-c", capture, "capture", str(ENGINE / f"run_oceandisk_{role}.sh")],
            env=env, capture_output=True, timeout=10,
        )
        return result

    def parse(self, role, **overrides):
        result = self.expand(role, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        values = result.stdout.decode().split("\0")
        split = values.index("__ENV__")
        return values[:split], dict(zip(ENV_KEYS, values[split + 1:]))

    def test_single_and_two_node_topologies(self):
        for tp, dp, base, ip in ((4, 2, 4, "141.111.65.134"), (8, 4, 0, "141.111.65.134")):
            with self.subTest(tp=tp, dp=dp):
                args, env = self.parse("decode", TP_SIZE=str(tp), DP_SIZE=str(dp), BASE_GPU_ID=str(base), LOCAL_IP=ip)
                value = lambda name: args[args.index(name) + 1]
                self.assertEqual(value("--tensor-parallel-size"), str(tp))
                self.assertEqual(value("--data-parallel-size"), str(dp))
                self.assertEqual(value("--base-gpu-id"), str(base))
                self.assertEqual(value("--dist-init-addr"), ip + ":31501")
                self.assertEqual(value("--chunked-prefill-size"), "16384")
                self.assertEqual(value("--cuda-graph-max-bs"), "128")
                self.assertIn("--enable-dp-attention", args)
                self.assertIn("--disaggregation-decode-enable-radix-cache", args)
                config = json.loads(value("--hicache-storage-backend-extra-config"))
                self.assertEqual(config, {"namespace": "test-shared", "layerwise_save": True})
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                self.assertEqual(env["GIDS_DIR"], "/configs/local/")
                self.assertEqual(env["SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS"], "8192")
                self.assertEqual(env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"], "8")
        args, env = self.parse("prefill", TP_SIZE="8", LOCAL_IP="141.111.65.132")
        self.assertIn("--enable-prefill-cp", args)
        self.assertIn("--disable-prefill-cuda-graph", args)
        self.assertEqual(args[args.index("--dist-init-addr") + 1], "141.111.65.132:31500")
        self.assertEqual(env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"], "1")

    def test_namespace_is_required_and_controls_are_overridable(self):
        for role in ("prefill", "decode"):
            for key in ("OD_NAMESPACE", "LOCAL_IP", "GIDS_DIR", "MOONCAKE_IB_DEVICES"):
                with self.subTest(role=role, missing=key):
                    self.assertNotEqual(self.expand(role, **{key: None}).returncode, 0)
        config = '{"namespace":"test-shared","connection_indices":[0],"layerwise_save":true}'
        args, env = self.parse("decode", OD_STORAGE_CONFIG=config, CHUNKED_PREFILL_SIZE="8192", SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS="0")
        self.assertEqual(args[args.index("--hicache-storage-backend-extra-config") + 1], config)
        self.assertEqual(env["SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS"], "0")

    def test_router_uses_remote_prefill_bootstrap_and_decode(self):
        args, _ = self.parse("router")
        start = args.index("--prefill")
        self.assertEqual(args[start + 1:start + 3], ["http://141.111.65.132:31002", "31145"])
        self.assertEqual(args[args.index("--decode") + 1], "http://141.111.65.134:31001")

    def test_shell_syntax(self):
        for name in ("run_oceandisk_prefill.sh", "run_oceandisk_decode.sh", "run_oceandisk_router.sh", "Round/run_rounds.sh"):
            result = subprocess.run([SHELL, "-n", str(ENGINE / name)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
