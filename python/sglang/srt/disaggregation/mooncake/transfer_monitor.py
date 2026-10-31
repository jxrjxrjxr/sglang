"""Opt-in phase diagnostics. A deadline never implies that RDMA has drained."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.utils import TransferKVChunk

logger = logging.getLogger(__name__)


@dataclass
class _TransferStage:
    chunk: TransferKVChunk
    label: str
    started: float
    warned: bool = False
    timed_out: bool = False


class TransferMonitor:
    def __init__(
        self, rank: int, trace: bool, warn_after: float, timeout: float, fail_fast: bool
    ):
        self.rank = rank
        self.trace = trace
        self.warn_after = warn_after
        self.timeout = timeout
        self.fail_fast = fail_fast
        self._stages: dict[int, _TransferStage] = {}
        self._lock = threading.Lock()
        self._next_check = 0.0
        self._worker_error: str | None = None

    def start(
        self,
        chunk: TransferKVChunk,
        phase: str,
        destination: str = "-",
        pages: int | None = None,
    ):
        # One entry per chunk: enqueuing a later chunk must not reset an active
        # transfer's deadline. Retaining chunk also prevents id reuse.
        self.end(chunk)
        label = (
            f"room={chunk.room} rank={self.rank} phase={phase} dst={destination} "
            f"range={chunk.index_slice.start}:{chunk.index_slice.stop} "
            f"pages={len(chunk.prefill_kv_indices) if pages is None else pages} "
            f"last={int(chunk.is_last_chunk)}"
        )
        with self._lock:
            self._stages[id(chunk)] = _TransferStage(chunk, label, time.monotonic())
        if self.trace:
            logger.info("[PDXFER] %s event=BEGIN", label)

    def end(self, chunk: TransferKVChunk, rc=None):
        with self._lock:
            stage = self._stages.pop(id(chunk), None)
        if stage is not None and self.trace:
            logger.info(
                "[PDXFER] %s event=END elapsed_ms=%.3f rc=%s",
                stage.label,
                (time.monotonic() - stage.started) * 1000,
                rc,
            )

    def failed(self, chunk: TransferKVChunk | None, error: Exception):
        with self._lock:
            stage = self._stages.pop(id(chunk), None)
            label = (
                stage.label if stage is not None else f"rank={self.rank} phase=WORKER"
            )
            self._worker_error = f"[PDXFER] {label} event=EXCEPTION error={error!r}"
            message = self._worker_error
        logger.error(message, exc_info=True)

    def check(self):
        # Called by the scheduler, independently of the blocking transfer worker.
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + 1.0
        messages = []
        with self._lock:
            fatal = self._worker_error if self.fail_fast else None
            for stage in self._stages.values():
                elapsed = now - stage.started
                if self.timeout > 0 and elapsed >= self.timeout:
                    message = (
                        f"[PDXFER] {stage.label} event=TIMEOUT elapsed_s={elapsed:.3f}"
                    )
                    if not stage.timed_out:
                        messages.append((logging.ERROR, message))
                        stage.timed_out = True
                        stage.warned = True
                    if self.fail_fast:
                        fatal = fatal or message
                elif (
                    self.warn_after > 0
                    and elapsed >= self.warn_after
                    and not stage.warned
                ):
                    messages.append(
                        (
                            logging.WARNING,
                            f"[PDXFER] {stage.label} event=STALL elapsed_s={elapsed:.3f}",
                        )
                    )
                    stage.warned = True
        # Never hold the progress lock across logging or network operations.
        for level, message in messages:
            logger.log(level, message)
        if fatal is not None:
            # The scheduler's fatal-exception path stops the instance. Marking
            # just this room Failed could free buffers while RDMA still writes.
            raise RuntimeError(f"{fatal}; stopping prefill instance (fail-fast)")
