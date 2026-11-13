"""Minimal GIDS PUT/GET/EXIST wrapper for standalone probes (no sglang).

Every native call mirrors production in
python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_storage.py:
  * metadata layout: one row per layer,
    [task, mr x2n, key x2n, local_off x2n, remote_off x2n, size x2n, conn x2n]
    with the KV half first and the Indexer half second, same keys;
  * PUT: one batch_dispatch_put per layer, need_record=False
    (submit_put_layer, the layerwise_save path);
  * GET: one batch_dispatch_get for all layers, need_record=True
    (get_layers);
  * EXIST: kvcache_exist([task, keys..., conns...]) (exists_many);
  * one process-wide task-id counter and one shared `output` tensor;
  * connection routing copied from connection_router.py (DPU hash ring,
    round robin inside a DPU group).

Each page is filled with an int32 stamp derived from (key, layer, pool), so a
GET can tell "correct data" from "never written" (sentinel) and from
"someone else's page".
"""

import hashlib
import itertools
import os
import queue
import sys
import threading
import time
from bisect import bisect_left
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import torch

LAYERS = 78
KV_BYTES = 73728
INDEX_BYTES = 8448
SENTINEL = int(np.array([0xA5A5A5A5], dtype=np.uint32).view(np.int32)[0])


def log(*parts):
    print(*parts, flush=True)


def add_common_args(parser):
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--pages", type=int, default=32, help="pages per PUT/GET")
    parser.add_argument("--kv-bytes", type=int, default=KV_BYTES)
    parser.add_argument("--index-bytes", type=int, default=INDEX_BYTES)
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="seconds before an op is a hang"
    )
    parser.add_argument("--run-token", default=None)


def make_keys(token, tag, count):
    """Fresh positive int64 keys; a new token never collides with old data."""
    keys = []
    for index in range(count):
        digest = hashlib.sha256(f"{token}:{tag}:{index}".encode()).digest()
        keys.append((int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1)
    return keys


# ---------------------------------------------------------------- watchdog --
# A stolen or lost CQE makes a GIDS kernel spin forever (no timeout in
# _gids_request_cpl_poll). Every wait registers here; if one exceeds the
# timeout we print what was in flight and exit with code 3.

_inflight = {}
_inflight_lock = threading.Lock()
_watchdog_timeout = [60.0]


class watch:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        with _inflight_lock:
            _inflight[id(self)] = (self.what, time.monotonic())
        return self

    def __exit__(self, *exc):
        with _inflight_lock:
            _inflight.pop(id(self), None)
        return False


def _watchdog():
    while True:
        time.sleep(1.0)
        now = time.monotonic()
        with _inflight_lock:
            items = list(_inflight.values())
        stuck = [(what, now - t0) for what, t0 in items if now - t0 > _watchdog_timeout[0]]
        if stuck:
            log("HANG detected; in-flight ops:")
            for what, t0 in items:
                log(f"  {what}: {now - t0:.1f}s")
            log(f"RESULT verdict=HANG stuck={stuck[0][0]!r}")
            sys.stdout.flush()
            os._exit(3)


def start_watchdog(timeout):
    _watchdog_timeout[0] = timeout
    threading.Thread(target=_watchdog, daemon=True, name="watchdog").start()


class Pending:
    """A submitted GIDS op: its final CUDA event plus tensors it must keep alive."""

    def __init__(self, event, keepalive):
        self.event = event
        self.keepalive = keepalive

    def done(self):
        return self.event.query()

    def wait(self, what):
        with watch(what):
            while not self.event.query():
                time.sleep(0.0002)
        self.keepalive = None


# ------------------------------------------------------------------ router --
class Router:
    """Copy of oceandisk/connection_router.py."""

    def __init__(self, config_path):
        try:
            from xxhash3_py import xxh3_64_hexdigest
        except ImportError:
            from xxhash import xxh3_64_hexdigest
        self.groups = {}
        index = 0
        for raw in Path(config_path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or not line.startswith("ip_config"):
                continue
            fields = [part.strip() for part in line.split("=", 1)[1].split(",")]
            self.groups.setdefault(int(fields[5]), []).append(index)
            index += 1
        if not self.groups:
            raise ValueError(f"no ip_config lines in {config_path}")
        self.count = index
        self.ring = sorted(
            (int(xxh3_64_hexdigest(f"DPU-{dpu}-V-{v}".encode()), 16), dpu)
            for dpu in self.groups
            for v in range(50)
        )
        self.coordinates = [c for c, _ in self.ring]
        self.next = [0] * len(self.ring)
        self.lock = threading.Lock()

    def select(self, key):
        slot = bisect_left(self.coordinates, key % (1 << 64)) % len(self.ring)
        conns = self.groups[self.ring[slot][1]]
        with self.lock:
            conn = conns[self.next[slot] % len(conns)]
            self.next[slot] += 1
        return conn


def gids_config_path(device):
    single = os.environ.get("USE_SINGLE_PORT") in (None, "1")
    name = "gids_config.txt" if single else f"gids_config_{device}.txt"
    return Path(os.environ["GIDS_DIR"]) / name


# -------------------------------------------------------------------- gids --
class Gids:
    """One process, one GPU, one GIDS init (GIDS cannot re-init in-process).

    GPU memory per pool: two regions (0 = PUT source, 1 = GET destination),
    each [layers][slots][page bytes]. Region 1 is reset to SENTINEL before GETs.
    """

    def __init__(self, device, layers, slots, kv_bytes, index_bytes, timeout, stagger=False):
        for name in ("OPEXTENSION", "GIDS_DIR"):
            if not os.environ.get(name):
                raise SystemExit(f"set {name} (see run_probe.sh)")
        os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")
        start_watchdog(timeout)
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        self.layers, self.slots = layers, slots
        self.sizes = tuple((size + 511) & ~511 for size in (kv_bytes, index_bytes))
        for size in self.sizes:
            assert size % 4 == 0
        self.buffers = [
            torch.full((2 * layers * slots * size + 512,), 0xA5, dtype=torch.uint8, device=self.device)
            for size in self.sizes
        ]
        self.views = [
            buf[: 2 * layers * slots * size].view(torch.int32).view(2, layers, slots, size // 4)
            for buf, size in zip(self.buffers, self.sizes)
        ]
        config = gids_config_path(device)
        self.router = Router(config)
        torch.ops.load_library(os.environ["OPEXTENSION"])
        import od_utils

        self.od = od_utils
        self.ops = torch.ops.my_ops
        self.output = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.task_ids = itertools.count(1)
        registration = torch.tensor(
            [b.data_ptr() for b in self.buffers] + [b.numel() for b in self.buffers],
            dtype=torch.int64,
            device=self.device,
        )
        if stagger:
            time.sleep(device * 2)  # production staggers connection setup by GPU
        torch.cuda.synchronize(self.device)
        status = self.ops.kvcache_init(registration, self.output)
        torch.cuda.synchronize(self.device)
        if status != 0:
            raise RuntimeError(f"kvcache_init returned {status}")
        log(
            f"[init] device={device} config={config} connections={self.router.count} "
            f"dpu_groups={ {k: v for k, v in sorted(self.router.groups.items())} } "
            f"layers={layers} slots={slots} page_io_bytes={self.sizes}"
        )

    # -- data stamps -------------------------------------------------------
    def _stamps(self, keys, pool):
        k = np.asarray(keys, dtype=np.uint64)[None, :]
        layer = np.arange(self.layers, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            mixed = (
                k * np.uint64(0x9E3779B97F4A7C15)
                + layer * np.uint64(0xBF58476D1CE4E5B9)
                + np.uint64(pool + 1) * np.uint64(0x94D049BB133111EB)
            )
        stamps = (mixed >> np.uint64(33)).astype(np.int32)  # in [0, 2^31): never SENTINEL
        return torch.from_numpy(stamps).to(self.device, non_blocking=False)

    def _index(self, slots):
        return torch.tensor(list(slots), dtype=torch.long, device=self.device)

    def fill(self, keys, slots, stream):
        """Write each page's stamps into the PUT source region."""
        index = self._index(slots)
        with torch.cuda.stream(stream):
            for pool, view in enumerate(self.views):
                view[0][:, index, :] = self._stamps(keys, pool)[:, :, None]

    def clear(self, slots, stream):
        index = self._index(slots)
        with torch.cuda.stream(stream):
            for view in self.views:
                view[1][:, index, :] = SENTINEL

    def verify(self, keys, slots, stream):
        """Compare GET destination pages with the stamps of `keys`."""
        if not keys:
            return dict(bad=0, untouched=0, bad_keys=[])
        index = self._index(slots)
        page_ok = None
        untouched = None
        with torch.cuda.stream(stream):
            for pool, view in enumerate(self.views):
                data = view[1][:, index, :]
                ok = (data == self._stamps(keys, pool)[:, :, None]).all(dim=2).all(dim=0)
                blank = (data == SENTINEL).all(dim=2).all(dim=0)
                page_ok = ok if page_ok is None else page_ok & ok
                untouched = blank if untouched is None else untouched & blank
            page_ok = page_ok.cpu().tolist()
            untouched = untouched.cpu().tolist()
        bad = [i for i, ok in enumerate(page_ok) if not ok]
        return dict(
            bad=len(bad),
            untouched=sum(1 for i in bad if untouched[i]),
            bad_keys=[keys[i] for i in bad[:3]],
        )

    # -- native metadata ---------------------------------------------------
    def _rows(self, keys, slots, region):
        n = len(keys)
        conns = np.asarray([self.router.select(k) for k in keys], dtype=np.int64)
        k = np.asarray(keys, dtype=np.int64)
        s = np.asarray(list(slots), dtype=np.int64)
        rows = np.empty((self.layers, 1 + 12 * n), dtype=np.int64)
        for layer in range(self.layers):
            row = rows[layer]
            row[0] = next(self.task_ids)
            mr, key, local, remote, size, conn = row[1:].reshape(6, 2 * n)
            for pool, nbytes in enumerate(self.sizes):
                part = slice(pool * n, (pool + 1) * n)
                mr[part] = pool
                key[part] = k
                local[part] = ((region * self.layers + layer) * self.slots + s) * nbytes
                remote[part] = layer * nbytes
                size[part] = nbytes
                conn[part] = conns
        return rows

    def _upload(self, rows, stream):
        cpu = torch.from_numpy(rows).pin_memory()
        gpu = torch.empty(cpu.shape, dtype=torch.int64, device=self.device)
        with torch.cuda.stream(stream):
            gpu.copy_(cpu, non_blocking=True)
        return cpu, gpu

    # -- the three operations ---------------------------------------------
    def put(self, keys, slots, stream):
        """Layerwise PUT exactly like submit_put_layer: one native call per layer."""
        cpu, gpu = self._upload(self._rows(keys, slots, 0), stream)
        width = gpu.shape[1]
        lengths = torch.tensor([width], dtype=torch.int64)
        indices = torch.tensor([0], dtype=torch.int64)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            for layer in range(self.layers):
                self.od.batch_dispatch_put(
                    gpu[layer : layer + 1], lengths, indices, cpu[layer : layer + 1],
                    self.output, [], stream.stream_id, self.device.index,
                    0, 1, False, False,
                )
            event = torch.cuda.Event()
            event.record(stream)
        return Pending(event, (cpu, gpu, lengths, indices))

    def get(self, keys, slots, stream):
        """GET of all layers in one call, like get_layers()."""
        cpu, gpu = self._upload(self._rows(keys, slots, 1), stream)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            events = [torch.cuda.Event() for _ in range(self.layers)]
            for event in events:
                event.record(stream)  # materialize handles; native re-records per layer
            self.od.batch_dispatch_get(
                gpu, gpu.shape[1], cpu, self.output,
                [event.cuda_event for event in events],
                stream.stream_id, self.device.index, 0, self.layers, True, False,
            )
            done = torch.cuda.Event()
            done.record(stream)
        return Pending(done, (cpu, gpu, events))

    def exist(self, keys, what="EXIST"):
        """Prefix-hit count, like exists_many(); synchronous."""
        if not keys:
            return 0
        conns = [self.router.select(k) for k in keys]
        meta = torch.tensor([next(self.task_ids), *keys, *conns], dtype=torch.int64)
        meta = meta.to(self.device)
        with watch(what), torch.cuda.device(self.device):
            hit = self.ops.kvcache_exist(meta, self.output)
        if not isinstance(hit, int) or not 0 <= hit <= len(keys):
            raise RuntimeError(f"kvcache_exist returned {hit!r} for {len(keys)} keys")
        return hit


# --------------------------------------------------------------- execution --
class Worker:
    """A persistent thread with its own stream (like the load / offload workers).

    Persistent on purpose: GIDS allocates large thread_local staging buffers on
    a thread's first call and never frees them.
    """

    def __init__(self, name, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def _run(self):
        torch.cuda.set_device(self.device)
        with torch.cuda.stream(self.stream):
            while True:
                fn, future = self.q.get()
                if fn is None:
                    return
                try:
                    future.set_result(fn())
                except BaseException as error:  # surfaced by future.result()
                    future.set_exception(error)

    def submit(self, fn):
        future = Future()
        self.q.put((fn, future))
        return future

    def call(self, fn):
        return self.submit(fn).result()


class Gate:
    """How GIDS ops from different threads are ordered.

    none   : no ordering (what GET vs PUT looks like in production today)
    launch : a CPU lock held only while *submitting* (what _put_io_lock does);
             EXIST additionally syncs the PUT stream under the lock, as
             exists_many() does. Kernels can still overlap on the GPU.
    full   : the lock is held from submission until the GPU op completes, so
             at most one GIDS kernel runs at a time (GIDS' original contract).
    """

    MODES = ("none", "launch", "full")

    def __init__(self, mode, put_stream=None):
        assert mode in self.MODES
        self.mode = mode
        self.put_stream = put_stream
        self.lock = threading.Lock()

    def io(self, what, issue):
        if self.mode == "full":
            with self.lock:
                pending = issue()
                pending.wait(what)
        elif self.mode == "launch":
            with self.lock:
                pending = issue()
            pending.wait(what)
        else:
            pending = issue()
            pending.wait(what)

    def exist(self, gids, keys, what="EXIST"):
        if self.mode == "none":
            return gids.exist(keys, what)
        with self.lock:
            if self.mode == "launch" and self.put_stream is not None:
                with watch(what + " put_sync"):
                    self.put_stream.synchronize()
            return gids.exist(keys, what)


def serial_health(gids, token, tag, pages, stream):
    """put -> exist -> get -> verify with nothing else running."""
    keys = make_keys(token, tag, pages)
    slots = list(range(pages))
    gids.fill(keys, slots, stream)
    gids.put(keys, slots, stream).wait(f"{tag} PUT")
    hit = gids.exist(keys, f"{tag} EXIST")
    gids.clear(slots, stream)
    if hit:
        gids.get(keys[:hit], slots[:hit], stream).wait(f"{tag} GET")
    check = gids.verify(keys[:hit], slots[:hit], stream)
    ok = hit == pages and check["bad"] == 0
    log(f"[{tag}] serial put/exist/get: exist_hit={hit}/{pages} data_bad={check['bad']} -> {'OK' if ok else 'FAIL'}")
    return ok
