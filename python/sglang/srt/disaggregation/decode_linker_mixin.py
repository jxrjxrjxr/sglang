"""Decode-side OceanDisk prefix loads alongside the Mooncake RDMA tail."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs
from sglang.srt.observability import ucl_trace

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _linker_handoff_prefix_len(req: Req, l1: int, page: int, chunk: int) -> int:
    if chunk is None or chunk < page:
        raise ValueError(
            "Decode Linker loading requires chunked_prefill_size >= page_size."
        )
    cacheable = req._compute_max_prefix_len(len(req.origin_input_ids)) // page * page
    tail = envs.SGLANG_DISAGGREGATION_LINKER_RDMA_TAIL_TOKENS.get()
    if tail == -1:
        tail = chunk
    return max(l1, (cacheable - tail) // page * page)


def _record_linker_failure(dr: DecodeRequest, reason: str) -> None:
    receiver = dr.kv_receiver
    with receiver.kv_mgr.failure_lock:
        receiver.kv_mgr.failure_records.setdefault(receiver.bootstrap_room, reason)
    dr.linker_failed = True


def _check_linker_load_timeout(dr: DecodeRequest, now: float) -> bool:
    if (
        not dr.linker_inflight_end
        and dr.linker_loaded_len < dr.linker_prefix_len
        and now - dr.req.time_stats.decode_transfer_queue_entry_time
        >= envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
    ):
        elapsed = now - dr.req.time_stats.decode_transfer_queue_entry_time
        _record_linker_failure(
            dr,
            f"OceanDisk prefix load timed out after {elapsed:.1f}s: "
            f"loaded_tokens={dr.linker_loaded_len}/{dr.linker_prefix_len}, "
            f"loaded_layers={dr.linker_loaded_layers}",
        )
        return True
    return False


class LinkerLoadGatedKVReceiver:
    """Hold terminal states until all ranks have drained the prefix DMA."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        dr = self.decode_req
        poll = dr.kv_receiver.poll()
        if ucl_trace.ON and getattr(dr, "_ucl_kv_poll", None) != poll:
            dr._ucl_kv_poll = poll
            ucl_trace.emit(
                "linker.kv_poll", dr.req.rid, state=getattr(poll, "name", poll)
            )
        if dr.linker_prefix_len <= len(dr.req.prefix_indices):
            return poll
        if poll == KVPoll.Failed or dr.kv_receiver.abort_notified:
            dr.linker_failed = True
        if dr.linker_inflight_end:
            return KVPoll.Transferring
        _check_linker_load_timeout(dr, time.perf_counter())
        if dr.linker_failed:
            # Peers may only observe the failed rank-consensus bit. Preserve any
            # local GET/timeout/transport reason before reporting their progress.
            _record_linker_failure(
                dr,
                "OceanDisk prefix transfer failed: "
                f"loaded_tokens={dr.linker_loaded_len}/{dr.linker_prefix_len}, "
                f"loaded_layers={dr.linker_loaded_layers}",
            )
            return KVPoll.Failed
        if poll == KVPoll.Success and dr.linker_loaded_len < dr.linker_prefix_len:
            return KVPoll.Transferring
        return poll


class DecodeLinkerTransferMixin:
    """Load notified layer groups into the existing chunkload windows."""

    def _process_linker_loads(self, decode_reqs: list[DecodeRequest]) -> None:
        active = [
            dr
            for dr in decode_reqs
            if dr.linker_prefix_len > len(dr.req.prefix_indices)
        ]
        if not active:
            return
        linker = self.tree_cache.linker
        page = self.tree_cache.page_size
        window = self.scheduler.chunked_prefill_size // page
        num_layers = linker.cache_linker.num_layers
        group = envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS.get()
        now = time.perf_counter()

        for dr in active:
            if not dr.linker_inflight_end:
                continue
            start = dr.linker_loaded_len // page
            tag = f"{dr.req.rid}@{start}:{dr.linker_loaded_layers}"
            success = linker.completed_handoff_loads.pop(tag, None)
            if success is None:
                continue
            if success:
                dr.linker_loaded_layers = dr.linker_inflight_layers
                if dr.linker_loaded_layers == num_layers:
                    dr.linker_loaded_len = dr.linker_inflight_end
                    dr.linker_loaded_layers = 0
            else:
                _record_linker_failure(
                    dr,
                    "OceanDisk GET failed: "
                    f"tokens={dr.linker_loaded_len}:{dr.linker_inflight_end}, "
                    f"layers={dr.linker_loaded_layers}:{dr.linker_inflight_layers}",
                )
            if ucl_trace.ON:
                ucl_trace.emit(
                    "linker.load_applied",
                    tag,
                    ok=success,
                    loaded_len=dr.linker_loaded_len,
                    loaded_layers=dr.linker_loaded_layers,
                )
            dr.linker_inflight_end = dr.linker_inflight_layers = 0

        # Only CPU bookkeeping and rank consensus run here; EXISTS is never
        # issued on decode, including while another request has GET in flight.
        # Per request: [all-layer end page, fixed-window due]. When every layer
        # group is READY over [start, end), one GET loads all layers of that
        # whole range (e.g. history READY at prefill admission). Otherwise the
        # fixed window x one layer group keeps a stable window across groups.
        # Failure is -1 in both columns; MIN makes all three rank-consistent.
        states = []
        for dr in active:
            failed = (
                dr.linker_failed
                or dr.kv_receiver.poll() == KVPoll.Failed
                or dr.kv_receiver.abort_notified
                or _check_linker_load_timeout(dr, now)
            )
            full_end = due = 0
            if not failed and not dr.linker_inflight_end:
                start = dr.linker_loaded_len // page
                last = dr.linker_prefix_len // page
                layer = dr.linker_loaded_layers
                ready = dr.kv_receiver.linker_ready_until
                if layer == 0:
                    full_end = min(
                        [last]
                        + [
                            ready(g, min(g + group, num_layers), start)
                            for g in range(0, num_layers, group)
                        ]
                    )
                    full_end = full_end if full_end > start else 0
                end = min(start + window, last)
                due = int(
                    start < end
                    and ready(layer, min(layer + group, num_layers), start) >= end
                )
            states.append([-1, -1] if failed else [full_end, due])
        local_states = states
        states = torch.tensor(states, dtype=torch.int, device="cpu")
        consensus_start = ucl_trace.now() if ucl_trace.ON else 0
        self.tree_cache._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MIN)
        if ucl_trace.ON:
            ucl_trace.stat("linker.consensus", consensus_start, reqs=len(active))
            _trace_linker_decisions(active, local_states, states.tolist(), page, window)

        # Bound one GET batch to about the longest request's prefix, the size a
        # prefill restore already issues in a single GET. The first request of
        # a batch always goes; later ones wait a round. The order and the
        # consensus are identical on every rank, so the deferral is too.
        cap = max(window, *(dr.linker_prefix_len // page for dr in active))
        batches, batch_pages = {}, {}
        for dr, (full_end, due) in zip(active, states.tolist()):
            if full_end < 0 or due < 0:
                dr.linker_failed = True
            elif full_end > 0 or due == 1:
                start = dr.linker_loaded_len // page
                if full_end > 0:
                    end, layer, layer_end = full_end, 0, num_layers
                else:
                    end = min(start + window, dr.linker_prefix_len // page)
                    layer = dr.linker_loaded_layers
                    layer_end = min(layer + group, num_layers)
                pages = batch_pages.get((layer, layer_end), 0)
                if pages and pages + end - start > cap:
                    continue
                batch_pages[(layer, layer_end)] = pages + end - start
                linker.build_handoff_transfers(dr.req, dr.linker_prefix_len // page)
                indices = self.tree_cache.req_to_token_pool.req_to_token[
                    dr.req.kv.req_pool_idx, start * page : end * page
                ]
                batches.setdefault((layer, layer_end), []).append(
                    (f"{dr.req.rid}@{start}:{layer}", start, end, indices)
                )
                if ucl_trace.ON:
                    ucl_trace.emit(
                        "linker.load_submit",
                        f"{dr.req.rid}@{start}:{layer}",
                        pages=[start, end],
                        layers=[layer, layer_end],
                    )
                dr.linker_inflight_end = end * page
                dr.linker_inflight_layers = layer_end
            if not dr.linker_inflight_end and (
                dr.linker_failed or dr.linker_loaded_len == dr.linker_prefix_len
            ):
                linker.handoff_transfers.pop(dr.req.rid, None)
        # The same rank-consistent candidates produce the same ordered batches.
        for layer_range, loads in sorted(batches.items()):
            linker.queue_handoff_loads(loads, layer_range=layer_range)


def _trace_linker_decisions(active, local_states, global_states, page, window):
    """Emit one event per request whenever its local/consensus decision changes.

    local/merged: -1 failed, 0 cannot advance, 1 a load is due (all-layer
    range or next window/layer group). local=1 with merged=0 means another
    rank is not ready yet. ends: [local, merged] all-layer end page (0: none).
    """
    for dr, local_row, merged_row in zip(active, local_states, global_states):
        local, merged = (
            -1 if min(row) < 0 else int(any(row)) for row in (local_row, merged_row)
        )
        start = dr.linker_loaded_len // page
        key = (
            local,
            merged,
            local_row[0],
            merged_row[0],
            dr.linker_loaded_len,
            dr.linker_loaded_layers,
            dr.linker_inflight_end,
        )
        if getattr(dr, "_ucl_decision", None) == key:
            continue
        dr._ucl_decision = key
        ucl_trace.emit(
            "linker.decide",
            dr.req.rid,
            local=local,
            merged=merged,
            ends=[local_row[0], merged_row[0]],
            loaded_len=dr.linker_loaded_len,
            loaded_layers=dr.linker_loaded_layers,
            inflight=bool(dr.linker_inflight_end),
            window=[start, min(start + window, dr.linker_prefix_len // page)],
        )
