import logging
import time
from typing import Callable, Optional

from sglang.srt.disaggregation.base.conn import BaseKVManager, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class LinkerKVReceiver(FakeKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.bootstrap_room = bootstrap_room
        self.init_time: Optional[float] = None
        self.ready_probe: Optional[Callable[[], bool]] = None
        self.next_probe_time = 0.0
        self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        self.probe_interval = envs.SGLANG_DISAGGREGATION_LINKER_PROBE_INTERVAL.get()
        self.failure_reason: Optional[str] = None

    def init(self, prefill_dp_rank: int):
        self.init_time = time.time()

    def set_ready_probe(self, probe: Callable[[], bool]):
        self.ready_probe = probe

    def poll(self) -> KVPoll:
        if self.conclude_state is not None or self.bootstrap_done:
            return super().poll()
        if self.init_time is None:
            return KVPoll.Bootstrapping

        now = time.time()
        elapsed = now - self.init_time
        if elapsed >= self.waiting_timeout:
            self.failure_reason = (
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
                "waiting for Linker pages in KVPoll.Bootstrapping"
            )
            logger.warning_once(
                "Some requests timed out waiting for Linker pages in "
                "KVPoll.Bootstrapping. If a greater mean TTFT is acceptable, you can "
                "'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' "
                "(10 minutes) to relax the timeout condition."
            )
            self.conclude_state = KVPoll.Failed
            return self.conclude_state

        if self.ready_probe is not None and now >= self.next_probe_time:
            self.next_probe_time = now + self.probe_interval
            try:
                self.bootstrap_done = self.ready_probe()
            except Exception as exc:
                self.failure_reason = (
                    f"Linker readiness probe failed for request "
                    f"{self.bootstrap_room}: {exc}"
                )
                self.conclude_state = KVPoll.Failed
                return self.conclude_state
        return super().poll()

    def failure_exception(self):
        raise RuntimeError(self.failure_reason or "Linker KV receiver failed")
