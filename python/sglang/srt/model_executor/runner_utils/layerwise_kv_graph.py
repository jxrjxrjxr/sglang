"""Per-layer cache hooks at breakable CUDA graph replay boundaries."""

from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.runtime_context import get_memory


class LayerwiseKVGraphHooks:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self._replaying = False
        self._saver = None

    @classmethod
    def create(cls, model_runner):
        memory = get_memory()
        if (
            not memory.enable_unified_cache_external_linker
            or memory.unified_cache_external_linker_backend != "oceandisk"
        ):
            return None
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            memory.hicache_storage_backend_extra_config
        )
        if not config.get("layerwise_save", True):
            return None
        if not any(
            base.__name__ == "DeepseekV2ForCausalLM"
            for base in type(model_runner.model).__mro__
        ):
            return None
        return cls(model_runner)

    @contextmanager
    def replay_session(self, saver):
        self._replaying = True
        self._saver = saver
        try:
            yield
        finally:
            self._saver = None
            self._replaying = False

    @eager_on_graph(True)
    def wait_for_layer(self, layer: int):
        # Capture and warmup must neither wait for live requests nor issue PUTs.
        if self._replaying:
            counter = self.model_runner.token_to_kv_pool.layer_transfer_counter
            if counter is not None:
                counter.wait_until(layer)

    @eager_on_graph(True)
    def save_layer(self, layer: int):
        # The captured callback binds to this runner, never a request's session.
        if self._saver is not None:
            self._saver.save_layer(layer)
