"""Real CUDA/GIDS roundtrip through OceanDiskDirectLinker, without a model.

Run from the repository root with SGLang installed:
    export CUDA_VISIBLE_DEVICES=3
    export OPEXTENSION=/opt/oceandisk/libop_extension.so
    export PYTHONPATH=/opt/oceandisk:$PWD/python:$PYTHONPATH
    python examples/runtime/engine/test_oceandisk_linker_roundtrip.py \
        --gids-config /opt/oceandisk/config/gids_config.txt

od_utils must be importable. LIBGIDS follows the native library's loading rules.
Each run uses fresh object keys. It does not delete objects from GIDS.
"""

import argparse
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NUM_LAYERS = 3
NUM_PAGES = 5
PAGE_SIZE = 64
KV_WIDTH = 16
INDEXER_WIDTH = 8


def make_pool_group(torch, device):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        DevicePoolEntry,
        DevicePoolGroup,
    )

    rows = (2 * NUM_PAGES + 2) * PAGE_SIZE
    buffers = {
        name: [
            torch.empty((rows, width), dtype=torch.uint8, device=device)
            for _ in range(NUM_LAYERS)
        ]
        for name, width in ((PoolName.KV, KV_WIDTH), (PoolName.INDEXER, INDEXER_WIDTH))
    }
    entries = [
        DevicePoolEntry(
            name=name,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[layers],
            layer_mapping={layer: layer for layer in range(NUM_LAYERS)},
            page_size=PAGE_SIZE,
            rows_are_pages=False,
        )
        for name, layers in buffers.items()
    ]
    return DevicePoolGroup(
        entries, NUM_LAYERS, PAGE_SIZE, rank_replicated=True
    ), buffers


def make_linker(pool_group, config):
    from sglang.srt.mem_cache.storage.oceandisk import oceandisk_direct_linker as module

    params = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        page_size=PAGE_SIZE,
        attn_tp_cache_group=None,
        tp_cache_group=None,
        attn_cp_size=1,
        attn_cp_rank=0,
        pp_size=1,
        pp_rank=0,
    )
    memory = SimpleNamespace(hicache_storage_backend_extra_config=json.dumps(config))
    model = SimpleNamespace(model_path="oceandisk-roundtrip-test")
    # Replace model setup only; constructor, workers and storage are real.
    with (
        patch.object(
            module, "resolve_hybrid_device_pool_group", return_value=pool_group
        ),
        patch.object(module, "get_memory", return_value=memory),
        patch.object(module, "get_model", return_value=model),
    ):
        return module.OceanDiskDirectLinker(None, params, components={})


def run_roundtrip(torch, linker, buffers, run_id, *, max_ranges):
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    label = f"max_ranges={max_ranges}"
    linker.max_ranges = max_ranges
    keys = [
        hashlib.sha256(f"{run_id}:{label}:{page}".encode()).hexdigest()
        for page in range(NUM_PAGES)
    ]
    source = torch.tensor(
        [
            page * PAGE_SIZE + token
            for page in range(1, 2 * NUM_PAGES, 2)
            for token in range(PAGE_SIZE)
        ],
        device=linker.device,
    )
    target = source + PAGE_SIZE
    expected = {}
    for pool_index, (name, layers) in enumerate(buffers.items()):
        expected[name] = []
        for layer, buffer in enumerate(layers):
            pattern = torch.arange(
                buffer.numel(), device=buffer.device, dtype=torch.int32
            )
            pattern = (pattern + pool_index * 71 + layer * 29 + max_ranges) % 251
            buffer.copy_(pattern.to(torch.uint8).reshape_as(buffer))
            expected[name].append(buffer[source].cpu())

    write = PoolTransfer(name=PoolName.KV, keys=keys, device_indices=source)
    read = PoolTransfer(name=PoolName.KV, keys=keys, device_indices=target)
    if linker.lookup("cold", [read]):
        raise AssertionError(f"{label}: fresh keys unexpectedly hit.")
    if not linker.offload([write]):
        raise AssertionError(f"{label}: offload was not queued.")
    linker.offload_queue.join()
    if not linker.pop_completed_offload():
        raise AssertionError(f"{label}: offload failed; see worker log.")

    for layers in buffers.values():
        for buffer in layers:
            buffer.fill_(0xFF)
    hits = linker.lookup("restore", [read])
    if hits != list(range(1, NUM_PAGES + 1)):
        raise AssertionError(f"{label}: expected all pages to exist, got {hits}.")

    rid = f"restore-{max_ranges}"
    if not linker.load(rid, [read]):
        raise AssertionError(f"{label}: load was not queued.")
    batch = linker.start_layer_wise_loading()
    linker.layer_done_counter.set_consumer(batch)
    for layer in range(NUM_LAYERS):
        linker.layer_done_counter.wait_until(layer)
        for name, layers in buffers.items():
            actual = layers[layer][target].cpu()
            torch.testing.assert_close(actual, expected[name][layer], rtol=0, atol=0)
            untouched = torch.ones(
                layers[layer].shape[0], dtype=torch.bool, device=linker.device
            )
            untouched[target] = False
            if not torch.all(layers[layer][untouched] == 0xFF).item():
                raise AssertionError(
                    f"{label}: {name} layer {layer} overwrote other slots."
                )
        print(f"PASS {label}: layer {layer}, KV + Indexer exact match", flush=True)
    linker.load_queue.join()
    if linker.pop_completed_load() != [rid]:
        raise AssertionError(f"{label}: unexpected completed request IDs.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gids-config", type=Path, default=os.getenv("GIDS_CONFIG"))
    parser.add_argument("--op-extension", default=os.getenv("OPEXTENSION"))
    parser.add_argument("--connection-indices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.gids_config is not None:
        path = args.gids_config.resolve()
        if path.name == "gids_config.txt":
            os.environ["USE_SINGLE_PORT"] = "1"
        elif path.name == "gids_config_0.txt":
            os.environ["USE_SINGLE_PORT"] = "0"
        else:
            parser.error(
                "Config must be gids_config.txt or gids_config_0.txt (visible device 0)."
            )
        os.environ["GIDS_DIR"] = str(path.parent) + os.sep

    import torch

    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    run_id = uuid.uuid4().hex
    config = {
        "namespace": f"oceandisk-roundtrip-{run_id}",
        "connection_indices": args.connection_indices,
    }
    if args.op_extension:
        config["op_extension"] = args.op_extension
    pool_group, buffers = make_pool_group(torch, device)
    linker = make_linker(pool_group, config)
    try:
        print(
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '(unset)')}; "
            f"device={device}; layers={NUM_LAYERS}; pages={NUM_PAGES}",
            flush=True,
        )
        run_roundtrip(torch, linker, buffers, run_id, max_ranges=2 * NUM_PAGES)
        run_roundtrip(torch, linker, buffers, run_id, max_ranges=4)
        print(
            "PASS: real OceanDisk Linker + Storage roundtrip (single GPU).", flush=True
        )
    finally:
        linker.close()


if __name__ == "__main__":
    main()
