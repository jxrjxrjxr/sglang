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


def read_io_alignment(config: dict) -> int:
    alignment = config.get("io_alignment", 512)
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("OceanDisk io_alignment must be a positive integer.")
    return alignment


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
    if not connection_indices:
        raise ValueError("OceanDisk needs at least one connection index.")
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

    def __init__(self, config: dict, device: torch.device, buffers: list[torch.Tensor]):
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
        self.io_alignment = read_io_alignment(config)
        self._closed = False
        self._lock = threading.Lock()
        self._task_ids = itertools.count(1)

        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The OceanDisk direct linker requires CUDA buffers.")
        self.regions = []
        for buffer in buffers:
            if buffer.device != self.device:
                raise ValueError(
                    "OceanDisk buffers must all be on the linker's CUDA device."
                )
            storage = buffer.untyped_storage()
            region = (int(storage.data_ptr()), int(storage.nbytes()))
            if region[1] and region not in self.regions:
                self.regions.append(region)
        if not self.regions:
            raise ValueError("OceanDisk has no GPU allocations to register.")

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

    def _io(self, ranges: list[OceanDiskRange], *, write: bool, max_ranges=256) -> None:
        if type(max_ranges) is not int or not 0 < max_ranges <= MAX_GIDS_RANGES:
            raise ValueError(f"OceanDisk max_ranges must be in [1, {MAX_GIDS_RANGES}].")
        if not ranges:
            return
        for item in ranges:
            if item.storage_offset % self.io_alignment or item.size % self.io_alignment:
                raise ValueError(
                    f"OceanDisk range size/remote offset must align to {self.io_alignment} "
                    "bytes; padding GPU reads/writes beyond the payload is not supported."
                )
        torch = self._torch
        with (
            self._lock,
            torch.cuda.device(self.device),
            torch.cuda.stream(self._stream),
        ):
            self._ensure_open()
            op = self._ops.kvcache_store_v2 if write else self._ops.kvcache_load_v2
            # GIDS requires equal IO lengths within each get dispatch.
            groups = (
                [ranges]
                if write
                else [
                    list(group)
                    for _, group in itertools.groupby(
                        sorted(ranges, key=lambda item: item.size),
                        key=lambda item: item.size,
                    )
                ]
            )
            # Resolve every range before submitting any part of the layer.
            descriptors = [
                pack_io(
                    next(self._task_ids),
                    group[start : start + max_ranges],
                    self.regions,
                    self.connection_indices,
                )
                for group in groups
                for start in range(0, len(group), max_ranges)
            ]
            metadata_buffers = []  # Keep CUDA tensors alive until the event completes.
            try:
                for descriptor in descriptors:
                    task_id = descriptor[0]
                    metadata = torch.tensor(
                        descriptor,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    metadata_buffers.append(metadata)
                    status = op(metadata, self._output, task_id)
                    self._check_status(status, "put" if write else "get")
                event = torch.cuda.Event()
                event.record(self._stream)
                event.synchronize()
            except BaseException:
                # Already submitted work must drain before metadata is released.
                self._stream.synchronize()
                raise

    def get_ranges(self, ranges: list[OceanDiskRange], *, max_ranges=256) -> None:
        self._io(ranges, write=False, max_ranges=max_ranges)

    def put_ranges(self, ranges: list[OceanDiskRange], *, max_ranges=256) -> None:
        self._io(ranges, write=True, max_ranges=max_ranges)

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
