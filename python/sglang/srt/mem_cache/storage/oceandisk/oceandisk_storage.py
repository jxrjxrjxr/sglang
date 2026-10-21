# SPDX-License-Identifier: Apache-2.0
"""OceanDisk direct-GPU storage with a reserved HiCache interface."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

if TYPE_CHECKING:
    import torch

MAX_GIDS_CONNECTIONS = 8  # od/gids_kernel.cu: MAX_CONN_NUM
MAX_GIDS_RANGES = 409600  # od/gids_kernel.cu: MAX_BLOCK_NUM


def gids_config_path(config_dir, device_index: int, use_single_port=None) -> Path:
    """Match gids_kernel.cu: unset USE_SINGLE_PORT defaults to single-port."""
    single = use_single_port is None or use_single_port == "1"
    filename = "gids_config.txt" if single else f"gids_config_{device_index}.txt"
    return Path(config_dir) / filename


def read_connection_count(path: Path) -> int:
    """Fail before native init, which otherwise logs bad config and may succeed."""
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "ip_config" not in line:
            continue
        name, sep, value = line.partition("=")
        fields = [field.strip() for field in value.split(",")]
        if (
            not raw.startswith("ip_config")
            or len(raw.encode()) >= 256
            or not sep
            or name.strip() != "ip_config"
            or len(fields) != 6
        ):
            raise ValueError(f"Invalid GIDS ip_config at {path}:{number}.")
        try:
            port, gid, roce_port, qp_count = map(int, fields[2:])
        except ValueError as error:
            raise ValueError(
                f"Invalid GIDS connection numbers at {path}:{number}."
            ) from error
        if (
            not fields[0]
            or not fields[1]
            or len(value.split(",")[0].lstrip().encode()) >= 32
            or len(value.split(",")[1].lstrip().encode()) >= 32
            or not 0 < port <= 65535
            or not 0 <= gid < 256
            or roce_port < 0
            or qp_count < 0
        ):
            raise ValueError(f"Invalid GIDS connection fields at {path}:{number}.")
        count += 1
    if not 1 <= count <= MAX_GIDS_CONNECTIONS:
        raise ValueError(
            f"OceanDisk requires 1..{MAX_GIDS_CONNECTIONS} ip_config entries: {path}."
        )
    return count


@dataclass(frozen=True)
class OceanDiskRange:
    key: int
    ptr: int
    size: int
    storage_offset: int = 0


def encode_key(identity: tuple) -> int:
    """Encode an object identity as a positive int64."""
    data = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    return (
        int.from_bytes(hashlib.sha256(data).digest()[:8], "big") & ((1 << 63) - 1)
    ) or 1


def resolve_mr(regions: list[tuple[int, int]], ptr: int, size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("OceanDisk ranges must have positive length.")
    for mr_id, (base, length) in enumerate(regions):
        if base <= ptr and ptr + size <= base + length:
            return mr_id, ptr - base
    raise ValueError(f"OceanDisk range ptr={ptr:#x}, size={size} is not registered.")


def pack_io(
    task_id: int,
    ranges: list[OceanDiskRange],
    regions: list[tuple[int, int]],
    connection_indices: tuple[int, ...],
) -> list[int]:
    """Column-major ABI used by od/cache_controller.py, with remote offsets."""
    half = len(ranges) // 2
    if (
        not ranges
        or len(ranges) % 2
        or [r.key for r in ranges[:half]] != [r.key for r in ranges[half:]]
    ):
        raise ValueError("OceanDisk requires matching KV and Indexer halves.")
    mr_offsets = [resolve_mr(regions, item.ptr, item.size) for item in ranges]
    if any(item.storage_offset < 0 for item in ranges):
        raise ValueError("OceanDisk storage offsets must be nonnegative.")
    return [
        task_id,
        *[mr for mr, _ in mr_offsets],
        *[item.key for item in ranges],
        *[offset for _, offset in mr_offsets],
        *[item.storage_offset for item in ranges],
        *[item.size for item in ranges],
        *[connection_indices[item.key % len(connection_indices)] for item in ranges],
    ]


class OceanDiskStorage(HiCacheStorage):
    """Serialize native I/O on one CUDA stream."""

    _registration_lock = threading.Lock()
    _initialized = False
    _registered_tensors: list = []

    def __init__(
        self,
        config: dict,
        device: torch.device,
        buffers: list[torch.Tensor],
        *,
        staging_bytes: int = 0,
    ):
        connections = config.get("connection_indices", [0])
        if (
            not isinstance(connections, list)
            or not connections
            or any(
                type(index) is not int or not 0 <= index < MAX_GIDS_CONNECTIONS
                for index in connections
            )
            or len(set(connections)) != len(connections)
        ):
            raise ValueError(
                "OceanDisk connection_indices must be distinct integers in [0, 7]."
            )
        self.connection_indices = tuple(connections)
        self._closed = False
        self._lock = threading.Lock()
        self._task_ids = itertools.count(1)

        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The OceanDisk direct linker requires CUDA buffers.")
        self._staging = (
            torch.empty(staging_bytes, dtype=torch.uint8, device=self.device)
            if staging_bytes
            else None
        )
        buffers = list(buffers)
        if self._staging is not None:
            buffers.append(self._staging)
        self.regions = []
        self._region_views = []
        for buffer in buffers:
            if buffer.device != self.device:
                raise ValueError(
                    "OceanDisk buffers must all be on the linker's CUDA device."
                )
            storage = buffer.untyped_storage()
            region = (int(storage.data_ptr()), int(storage.nbytes()))
            if region[1] and region not in self.regions:
                self.regions.append(region)
                if self._staging is not None:
                    self._region_views.append(
                        torch.empty(0, dtype=torch.uint8, device=self.device).set_(
                            storage, 0, (region[1],), (1,)
                        )
                    )
        if not self.regions:
            raise ValueError("OceanDisk has no GPU allocations to register.")
        if self._staging is not None:
            self._staging_mr = self.regions.index(
                (self._staging.data_ptr(), self._staging.numel())
            )

        library = config.get("op_extension") or os.environ.get("OPEXTENSION")
        if not library:
            raise ValueError(
                "Set OceanDisk op_extension or OPEXTENSION to libop_extension.so."
            )
        # LIBGIDS must be set before load_library: the library constructor dlopens it.
        for key, env in (("core_library", "LIBGIDS"), ("config_dir", "GIDS_DIR")):
            if config.get(key):
                os.environ[env] = os.fspath(config[key])
        if not os.environ.get("GIDS_DIR"):
            raise ValueError(
                "Set OceanDisk config_dir or GIDS_DIR before initialization."
            )
        # Native code concatenates the directory and filename.
        os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")
        if "use_single_port" in config:
            if type(config["use_single_port"]) is not bool:
                raise ValueError("OceanDisk use_single_port must be a boolean.")
            os.environ["USE_SINGLE_PORT"] = "1" if config["use_single_port"] else "0"
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        config_path = gids_config_path(
            os.environ["GIDS_DIR"], device_index, os.environ.get("USE_SINGLE_PORT")
        )
        connection_count = read_connection_count(config_path)
        if any(index >= connection_count for index in self.connection_indices):
            raise ValueError(
                f"OceanDisk connection_indices refer outside {connection_count} connections "
                f"in {config_path}."
            )

        with self._registration_lock, torch.cuda.device(self.device):
            if type(self)._initialized:
                raise RuntimeError(
                    "OceanDisk process-global state is already initialized; restart the "
                    "process rather than reinitializing registered GPU memory."
                )
            torch.ops.load_library(library)
            self._ops = torch.ops.my_ops
            self._stream = torch.cuda.Stream(device=self.device)
            self._output = torch.zeros(2, dtype=torch.int64, device=self.device)
            addresses, lengths = zip(*self.regions)
            tensor = torch.tensor(
                [*addresses, *lengths], dtype=torch.int64, device=self.device
            )
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            # Even failed init may have registered memory: retain it and prohibit retry.
            type(self)._initialized = True
            type(self)._registered_tensors = list(buffers)
            with torch.cuda.stream(self._stream):
                try:
                    status = self._ops.kvcache_init(tensor, self._output)
                finally:
                    self._stream.synchronize()
            self._check_status(status, "init")

    @staticmethod
    def _check_status(status, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"OceanDisk {operation} returned status {status!r}.")

    def exists_many(self, keys: list[int]) -> list[bool]:
        """Singleton probes avoid mistaking a total hit count for a prefix."""
        torch = self._torch
        result = []
        with (
            self._lock,
            torch.cuda.device(self.device),
            torch.cuda.stream(self._stream),
        ):
            self._ensure_open()
            for key in keys:
                conn = self.connection_indices[key % len(self.connection_indices)]
                metadata = torch.tensor(
                    [next(self._task_ids), key, conn],
                    dtype=torch.int64,
                    device=self.device,
                )
                try:
                    status = self._ops.kvcache_exist(metadata, self._output)
                finally:
                    self._stream.synchronize()
                if status not in (0, 1):
                    raise RuntimeError(
                        f"OceanDisk singleton exist returned {status!r}."
                    )
                result.append(status == 1)
        return result

    def _pack_layer_batches(self, layers, max_ranges):
        if type(max_ranges) is not int or not 2 <= max_ranges <= MAX_GIDS_RANGES:
            raise ValueError(f"OceanDisk max_ranges must be in [2, {MAX_GIDS_RANGES}].")
        if not layers:
            return []
        count = len(layers[0])
        if count % 2 or any(len(layer) != count for layer in layers):
            raise ValueError("OceanDisk layers require equal, even IO counts.")
        half = count // 2
        batch_pages = max_ranges // 2
        return [
            [
                pack_io(
                    next(self._task_ids),
                    ranges[start : min(start + batch_pages, half)]
                    + ranges[half + start : half + min(start + batch_pages, half)],
                    self.regions,
                    self.connection_indices,
                )
                for ranges in layers
            ]
            for start in range(0, half, batch_pages)
        ]

    def _stage_batch(self, batch, *, write):
        """Pad native I/O without accessing bytes outside the source allocation."""
        rows, copies = [], []
        cursor = 0
        for original in batch:
            row = list(original)
            count = (len(row) - 1) // 6
            layer_copies = []
            for index in range(count):
                size = row[1 + 4 * count + index]
                if size % 512 == 0:
                    continue
                padded_size = (size + 511) & ~511
                if (
                    self._staging is None
                    or cursor + padded_size > self._staging.numel()
                ):
                    raise ValueError(
                        "OceanDisk padded batch exceeds registered staging memory."
                    )
                mr = row[1 + index]
                local_offset = row[1 + 2 * count + index]
                source = self._region_views[mr][local_offset : local_offset + size]
                padded = self._staging[cursor : cursor + padded_size]
                payload = padded[:size]
                if write:
                    padded.zero_()
                    payload.copy_(source)
                else:
                    layer_copies.append((source, payload))
                row[1 + index] = self._staging_mr
                row[1 + 2 * count + index] = cursor
                row[1 + 4 * count + index] = padded_size
                cursor += padded_size
            rows.append(row)
            copies.append(layer_copies)
        return rows, copies

    def get_layers(self, layers, *, max_ranges=256, on_layer_ready=None) -> None:
        import od_utils

        torch = self._torch
        with (
            self._lock,
            torch.cuda.device(self.device),
            torch.cuda.stream(self._stream),
        ):
            self._ensure_open()
            descriptors = self._pack_layer_batches(layers, max_ranges)
            submitted = []
            staged = self._staging is not None
            try:
                for batch_index, batch in enumerate(descriptors):
                    if staged:
                        batch, copies = self._stage_batch(batch, write=False)
                    cpu = torch.tensor(batch, dtype=torch.int64, device="cpu")
                    gpu = cpu.to(self.device)
                    events = [torch.cuda.Event() for _ in layers]
                    for event in events:
                        event.record(self._stream)
                    submitted.append((cpu, gpu, events))
                    od_utils.batch_dispatch_get(
                        gpu,
                        gpu.shape[1],
                        cpu,
                        self._output,
                        [event.cuda_event for event in events],
                        self._stream.stream_id,
                        self.device.index,
                        0,
                        len(layers),
                        True,
                        False,
                    )
                    if staged:
                        for layer, event in enumerate(events):
                            event.synchronize()
                            for target, payload in copies[layer]:
                                target.copy_(payload)
                            restored = torch.cuda.Event()
                            restored.record(self._stream)
                            restored.synchronize()
                            if batch_index == len(descriptors) - 1 and on_layer_ready:
                                on_layer_ready(layer)
                        submitted.clear()
                if not staged:
                    for layer in range(len(layers)):
                        for _, _, events in submitted:
                            events[layer].synchronize()
                        if on_layer_ready is not None:
                            on_layer_ready(layer)
            except BaseException:
                self._stream.synchronize()
                raise

    def put_layers(self, layers, *, max_ranges=256) -> None:
        import od_utils

        torch = self._torch
        with (
            self._lock,
            torch.cuda.device(self.device),
            torch.cuda.stream(self._stream),
        ):
            self._ensure_open()
            descriptors = self._pack_layer_batches(layers, max_ranges)
            submitted = []
            try:
                for batch in descriptors:
                    if self._staging is not None:
                        batch, _ = self._stage_batch(batch, write=True)
                    cpu = torch.tensor(batch, dtype=torch.int64, device="cpu")
                    gpu = cpu.to(self.device)
                    lengths = torch.tensor(
                        [len(row) for row in batch], dtype=torch.int64, device="cpu"
                    )
                    indices = torch.tensor(
                        list(range(len(batch))), dtype=torch.int64, device="cpu"
                    )
                    submitted.append((cpu, gpu, lengths, indices))
                    od_utils.batch_dispatch_put(
                        gpu,
                        lengths,
                        indices,
                        cpu,
                        self._output,
                        [],
                        self._stream.stream_id,
                        self.device.index,
                        0,
                        len(layers),
                        False,
                        False,
                    )
                    if self._staging is not None:
                        event = torch.cuda.Event()
                        event.record(self._stream)
                        event.synchronize()
                        submitted.clear()
                if self._staging is None:
                    event = torch.cuda.Event()
                    event.record(self._stream)
                    event.synchronize()
            except BaseException:
                self._stream.synchronize()
                raise

    @staticmethod
    def _unsupported_hicache() -> NotImplementedError:
        return NotImplementedError(
            "OceanDiskStorage currently supports UnifiedCacheLinker range I/O only; "
            "HiCache host-cache integration is not implemented."
        )

    # Abstract HiCache methods remain unsupported by the UC adapter.
    def get(self, key, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def exists(self, key):
        raise self._unsupported_hicache()

    # Override base no-ops to reject unsupported host-cache operations.
    def batch_get_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def register_mem_pool_host(self, mem_pool_host):
        raise self._unsupported_hicache()

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        raise self._unsupported_hicache()

    def clear(self):
        raise self._unsupported_hicache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OceanDisk storage is closed.")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                with self._torch.cuda.device(self.device):
                    self._stream.synchronize()
                self._closed = True
        # No close/deregister operator is exposed. Keep allocations pinned by
        # Python references for the same lifetime as GIDS's process-global MRs.
