"""Linker decode consumes storage pages without scheduling writes."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache import registry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLinkerReadOnlyCache(unittest.TestCase):
    def test_only_linker_decode_disables_both_write_paths(self):
        for mode, backend in (
            ("decode", "linker"),
            ("prefill", "linker"),
            ("null", "linker"),
            ("decode", "mooncake"),
            ("decode", "fake"),
        ):
            with self.subTest(mode=mode, backend=backend):
                self.check_write_paths(mode, backend)

    def check_write_paths(self, mode, backend):
        fake_components = MagicMock()
        fake_components.ComponentType.FULL = "full"
        fake_radix = MagicMock()
        fake_backend = MagicMock()
        cache = fake_radix.UnifiedRadixCache.return_value
        cache.components = {"full": object()}
        cache.write_through_threshold = 1
        cache.linker.cache_linker.layerwise_save_enabled = True
        allocator = MagicMock()
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=allocator,
        )
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=type("DeepseekV2ForCausalLM", (), {})()),
            register_hicache_layer_transfer_counter=MagicMock(),
            layerwise_cache_linker=None,
        )
        ctx = SimpleNamespace(
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hierarchical_cache=False,
            tp_worker=worker,
        )
        disagg = SimpleNamespace(
            disaggregation_mode=mode,
            disaggregation_transfer_backend=backend,
            disaggregation_decode_retraction_backup="cpu_tensor",
        )
        memory = SimpleNamespace(
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="oceandisk",
        )
        with (
            patch.object(registry, "get_disagg", return_value=disagg),
            patch.object(registry, "get_memory", return_value=memory),
            patch.object(registry, "use_mlx", return_value=False),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.mem_cache.unified_cache.components": fake_components,
                    "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
                    "sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker": (
                        fake_backend
                    ),
                },
            ),
        ):
            result = registry._create_unified_radix_cache(ctx, object(), params)

        self.assertIs(result, cache)
        readonly = mode == "decode" and backend == "linker"
        self.assertEqual(cache.write_through_threshold, sys.maxsize if readonly else 1)
        self.assertIs(worker.layerwise_cache_linker, None if readonly else cache.linker)
        counter = cache.linker.layer_done_counter
        worker.register_hicache_layer_transfer_counter.assert_called_once_with(counter)
        kvcache = allocator.get_kvcache.return_value
        kvcache.register_layer_transfer_counter.assert_called_once_with(counter)


if __name__ == "__main__":
    unittest.main()
