"""Decode-side OceanDisk prefix loads alongside the Mooncake RDMA tail."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs

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


def _linker_load_timed_out(dr: DecodeRequest, now: float) -> bool:
    return (
        not dr.linker_inflight_end
        and dr.linker_loaded_len < dr.linker_prefix_len
        and now - dr.req.time_stats.decode_transfer_queue_entry_time
        >= envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
    )


class LinkerLoadGatedKVReceiver:
    """Hold terminal states until all ranks have drained the prefix DMA."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        dr = self.decode_req
        poll = dr.kv_receiver.poll()
        if dr.linker_prefix_len <= len(dr.req.prefix_indices):
            return poll
        if poll == KVPoll.Failed or dr.kv_receiver.abort_notified:
            dr.linker_failed = True
        if dr.linker_inflight_end:
            return KVPoll.Transferring
        if _linker_load_timed_out(dr, time.perf_counter()):
            dr.linker_failed = True
        if dr.linker_failed:
            return KVPoll.Failed
        if poll == KVPoll.Success and dr.linker_loaded_len < dr.linker_prefix_len:
            return KVPoll.Transferring
        return poll


class DecodeLinkerTransferMixin:
    """Probe one window per request and merge all ready windows into one load."""

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
        now = time.perf_counter()
        interval = envs.SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL.get()

        for dr in active:
            if not dr.linker_inflight_end:
                continue
            tag = f"{dr.req.rid}@{dr.linker_loaded_len // page}"
            success = linker.completed_handoff_loads.pop(tag, None)
            if success is not None:
                if success:
                    dr.linker_loaded_len = dr.linker_inflight_end
                else:
                    dr.linker_failed = True
                dr.linker_inflight_end = 0

        # Local clocks, receiver failures and load results can differ by rank.
        # Agree on candidates before issuing any request-dependent collective.
        states = []
        for dr in active:
            failed = (
                dr.linker_failed
                or dr.kv_receiver.poll() == KVPoll.Failed
                or dr.kv_receiver.abort_notified
                or _linker_load_timed_out(dr, now)
            )
            due = (
                not dr.linker_inflight_end
                and dr.linker_loaded_len < dr.linker_prefix_len
                and now >= dr.linker_next_probe_time
            )
            states.append(-1 if failed else int(due))
        states = torch.tensor(states, dtype=torch.int, device="cpu")
        self.tree_cache._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MIN)
        if not (states == 1).any():
            for dr, state in zip(active, states.tolist()):
                dr.linker_failed |= state < 0
                if not dr.linker_inflight_end and (
                    dr.linker_failed or dr.linker_loaded_len == dr.linker_prefix_len
                ):
                    linker.handoff_lookups.pop(dr.req.rid, None)
            return
        for i, dr in enumerate(active):
            if states[i] != 1:
                continue
            dr.linker_next_probe_time = now + interval
            start = dr.linker_loaded_len // page
            end = min(start + window, dr.linker_prefix_len // page)
            try:
                transfers = linker.build_handoff_lookup(
                    dr.req, dr.linker_prefix_len // page
                )
                transfers = [replace(t, keys=t.keys[start:end]) for t in transfers]
                hits = linker.cache_linker.lookup(dr.req.rid, transfers)
                states[i] = int(end - start in hits)
            except Exception:
                logger.exception("Linker handoff lookup failed for rid=%s", dr.req.rid)
                states[i] = -1
        self.tree_cache._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MIN)

        loads = []
        for dr, state in zip(active, states.tolist()):
            if state < 0:
                dr.linker_failed = True
            elif state == 1:
                start = dr.linker_loaded_len // page
                end = min(start + window, dr.linker_prefix_len // page)
                indices = self.tree_cache.req_to_token_pool.req_to_token[
                    dr.req.kv.req_pool_idx, start * page : end * page
                ]
                loads.append((f"{dr.req.rid}@{start}", start, end, indices))
                dr.linker_inflight_end = end * page
            if not dr.linker_inflight_end and (
                dr.linker_failed or dr.linker_loaded_len == dr.linker_prefix_len
            ):
                linker.handoff_lookups.pop(dr.req.rid, None)
        linker.queue_handoff_loads(loads)
