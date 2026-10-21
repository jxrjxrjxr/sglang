"""Real CUDA/GIDS roundtrip through OceanDiskDirectLinker, without a model.

Run from the repository root with SGLang installed:
    export CUDA_VISIBLE_DEVICES=3
    export OPEXTENSION=/opt/oceandisk/libop_extension.so
    export PYTHONPATH=/opt/oceandisk:$PWD/python:$PYTHONPATH
    python examples/runtime/engine/test_oceandisk_linker_roundtrip.py \
        --gids-config /opt/oceandisk/config/gids_config.txt

od_utils must be importable. LIBGIDS follows the native library's loading rules.
Each run uses fresh object keys. It does not delete objects from GIDS.
Default: compare batch Put with direct per-layer operator calls. Direct mode
bypasses only the batch Put wrapper and synchronizes after each operator call.
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NUM_LAYERS = 3
NUM_PAGES = 5
PAGE_SIZE = 64
# Byte widths: BF16 MLA (512 + 64) * 2; FP8 index (128) + FP32 scale (4).
KV_WIDTH = 1152
INDEXER_WIDTH = 132


def print_runtime(linker):
    import od_utils

    from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
        gids_config_path,
    )

    print(f"Linker source: {inspect.getfile(type(linker))}", flush=True)
    print(f"Storage source: {inspect.getfile(type(linker.storage))}", flush=True)
    print(f"od_utils: {od_utils.__file__}", flush=True)
    print(
        f"OPEXTENSION: {os.getenv('OPEXTENSION', '(unset; CLI override possible)')}",
        flush=True,
    )
    print(
        f"LIBGIDS: {os.getenv('LIBGIDS', '(unset; native compiled default)')}",
        flush=True,
    )
    path = gids_config_path(os.environ["GIDS_DIR"], 0, os.getenv("USE_SINGLE_PORT"))
    print(f"GIDS config: {path.resolve()}", flush=True)
    maps = Path("/proc/self/maps")
    if maps.exists():
        libraries = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if "/" in line
                and any(
                    name in line.lower()
                    for name in ("gids", "op_extension", "od_utils")
                )
            }
        )
        print(f"Mapped native libraries: {json.dumps(libraries)}", flush=True)
    print(f"Key namespace: {json.dumps(linker._namespace)}", flush=True)
    for mr, (ptr, size) in enumerate(linker.storage.regions):
        print(f"Registered MR={mr}: base={ptr:#x}, bytes={size}", flush=True)


def print_metadata(operation, row_index, row):
    count = (len(row) - 1) // 6
    fields = (
        "mr_ids",
        "keys",
        "local_offsets",
        "storage_offsets",
        "sizes",
        "connections",
    )
    metadata = {"task_id": row[0], "io_count": count}
    metadata.update(
        {
            field: row[1 + column * count : 1 + (column + 1) * count]
            for column, field in enumerate(fields)
        }
    )
    print(f"{operation} row={row_index}: {json.dumps(metadata)}", flush=True)


@contextmanager
def trace_native_calls(storage, put_mode="batch"):
    import od_utils

    native_put = od_utils.batch_dispatch_put
    native_get = od_utils.batch_dispatch_get
    native_exist = storage._ops.kvcache_exist
    native_store = storage._ops.kvcache_store_v2

    def put(*args):
        lengths = args[1].tolist()
        indices = args[2].tolist()
        print(
            f"PUT mode={put_mode}: shape={tuple(args[0].shape)}, lengths={lengths}, indices={indices}, stream={args[6]}, device={args[7]}",
            flush=True,
        )
        for index in indices:
            print_metadata("PUT", index, args[3][index].tolist()[: lengths[index]])
        if put_mode == "direct":
            for index in indices:
                row = args[0][index, : lengths[index]]
                task_id = int(args[3][index][0].item())
                status = native_store(row, args[4], task_id)
                storage._stream.synchronize()
                print(
                    f"PUT direct row={index}: task={task_id}, raw_status={status!r}",
                    flush=True,
                )
                storage._check_status(status, "direct put")
            return None
        result = native_put(*args)
        print(
            f"PUT wrapper returned {result!r} (not a per-IO success result)", flush=True
        )
        return result

    def get(*args):
        print(
            f"GET batch: shape={tuple(args[0].shape)}, stream={args[5]}, device={args[6]}",
            flush=True,
        )
        for index, row in enumerate(args[2].tolist()):
            print_metadata("GET", index, row[: args[1]])
        return native_get(*args)

    def exists(metadata, output):
        row = metadata.cpu().tolist()
        result = native_exist(metadata, output)
        count = (len(row) - 1) // 2
        print(
            f"EXIST task={row[0]} keys={row[1 : 1 + count]} connections={row[1 + count :]} raw_result={result!r}",
            flush=True,
        )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(od_utils, "batch_dispatch_put", new=put))
        stack.enter_context(patch.object(od_utils, "batch_dispatch_get", new=get))
        stack.enter_context(patch.object(storage._ops, "kvcache_exist", new=exists))
        yield


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


def run_roundtrip(torch, linker, buffers, run_id, *, max_ranges, put_mode):
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    label = f"put={put_mode}, max_ranges={max_ranges}"
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
    print(
        f"\nCASE {label}: source page slots={source[::PAGE_SIZE].tolist()}; "
        f"target page slots={target[::PAGE_SIZE].tolist()}",
        flush=True,
    )
    print(f"Page hashes: {json.dumps(keys)}", flush=True)
    print("STAGE: cold lookup", flush=True)
    if linker.lookup("cold", [read]):
        raise AssertionError(f"{label}: fresh keys unexpectedly hit.")
    print("STAGE: offload", flush=True)
    if not linker.offload([write]):
        raise AssertionError(f"{label}: offload was not queued.")
    linker.offload_queue.join()
    if not linker.pop_completed_offload():
        raise AssertionError(f"{label}: offload failed; see worker log.")
    print(
        "STAGE: offload queue and CUDA event finished; checking existence", flush=True
    )

    for layers in buffers.values():
        for buffer in layers:
            buffer.fill_(0xFF)
    hits = linker.lookup("restore", [read])
    print(f"Restorable prefix boundaries: {hits}", flush=True)
    if hits != list(range(1, NUM_PAGES + 1)):
        raise AssertionError(f"{label}: expected all pages to exist, got {hits}.")

    rid = f"restore-{put_mode}-{max_ranges}"
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
    parser.add_argument(
        "--put-mode",
        choices=("compare", "batch", "direct"),
        default="compare",
        help="Compare both Put paths (default), or run one path only; Get remains batch.",
    )
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
        if args.op_extension:
            print(
                f"Requested op extension: {Path(args.op_extension).resolve()}",
                flush=True,
            )
        print_runtime(linker)
        modes = ("batch", "direct") if args.put_mode == "compare" else (args.put_mode,)
        results = []
        for max_ranges in (2 * NUM_PAGES, 4):
            for mode in modes:
                label = f"put={mode}, max_ranges={max_ranges}"
                with trace_native_calls(linker.storage, put_mode=mode):
                    try:
                        run_roundtrip(
                            torch,
                            linker,
                            buffers,
                            run_id,
                            max_ranges=max_ranges,
                            put_mode=mode,
                        )
                    except AssertionError as error:
                        results.append((label, str(error)))
                        print(f"FAIL {label}: {error}", flush=True)
                    else:
                        results.append((label, None))
                    finally:
                        linker.reset()
        print("\nCOMPARISON SUMMARY", flush=True)
        for label, error in results:
            print(
                f"{'FAIL' if error is not None else 'PASS'} {label}"
                + (f": {error}" if error is not None else ""),
                flush=True,
            )
        if any(error is not None for _, error in results):
            raise AssertionError(
                "OceanDisk roundtrip failed; see comparison summary and native call logs."
            )
        print(
            "PASS: real OceanDisk Linker + Storage roundtrip (single GPU).", flush=True
        )
    finally:
        linker.close()


if __name__ == "__main__":
    main()
