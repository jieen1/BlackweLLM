"""D3: Lightweight request-level tracing for BlackweLLM.

Records span-level timing for each request's lifecycle:
  admission_wait → prefill → decode rounds → finish

Design constraints:
  - <1% throughput overhead (no allocations on hot path when disabled)
  - Configurable sampling rate (default: only trace slow requests)
  - No external dependencies (no OTel, no prometheus_client)
  - Thread-safe (engine thread writes, asyncio thread reads)

Usage:
    from server.tracing import tracer

    # At admission:
    tracer.request_admitted(request_id, slot, prompt_len)

    # At prefill completion:
    tracer.prefill_done(request_id, prefill_ms)

    # Each decode round:
    tracer.decode_round(request_id, round_idx, tokens_committed, round_ms)

    # At finish:
    tracer.request_finished(request_id, finish_reason)

    # Query:
    tracer.get_slow_requests(threshold_ms=5000)
    tracer.get_trace(request_id)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RequestTrace:
    """Complete lifecycle trace for one request."""

    request_id: str
    slot: int = -1
    prompt_len: int = 0
    admitted_at: float = 0.0
    prefill_done_at: float = 0.0
    finished_at: float = 0.0
    finish_reason: str = ""
    prefill_ms: float = 0.0
    total_rounds: int = 0
    total_tokens: int = 0
    decode_rounds: list[tuple[int, int, float]] = field(default_factory=list)
    # (round_idx, tokens_committed, round_ms)

    @property
    def admission_wait_ms(self) -> float:
        if self.admitted_at and self.prefill_done_at:
            return 0.0  # admission is instant in current design
        return 0.0

    @property
    def total_ms(self) -> float:
        if self.finished_at and self.admitted_at:
            return (self.finished_at - self.admitted_at) * 1000
        return 0.0

    @property
    def decode_ms(self) -> float:
        return sum(r[2] for r in self.decode_rounds)

    @property
    def avg_round_ms(self) -> float:
        if not self.decode_rounds:
            return 0.0
        return self.decode_ms / len(self.decode_rounds)

    @property
    def tokens_per_sec(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return self.total_tokens / (self.decode_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "slot": self.slot,
            "prompt_len": self.prompt_len,
            "total_ms": round(self.total_ms, 1),
            "prefill_ms": round(self.prefill_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "total_rounds": self.total_rounds,
            "total_tokens": self.total_tokens,
            "avg_round_ms": round(self.avg_round_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "finish_reason": self.finish_reason,
        }


class RequestTracer:
    """Lightweight request tracer with configurable sampling and slow-request capture."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        sample_rate: float = 0.0,
        slow_threshold_ms: float = 10000.0,
        max_traces: int = 256,
        max_rounds_per_trace: int = 500,
    ) -> None:
        self._lock = threading.Lock()
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.slow_threshold_ms = slow_threshold_ms
        self.max_traces = max_traces
        self.max_rounds_per_trace = max_rounds_per_trace

        self._active: dict[str, RequestTrace] = {}
        self._completed: deque[RequestTrace] = deque(maxlen=max_traces)
        self._slow: deque[RequestTrace] = deque(maxlen=max_traces)
        self._request_counter = 0
        self._sample_counter = 0

    def _should_trace(self) -> bool:
        """Determine if this request should be traced (sampling)."""
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        self._sample_counter += 1
        return (self._sample_counter % max(1, int(1.0 / self.sample_rate))) == 0

    def request_admitted(self, request_id: str, slot: int, prompt_len: int) -> None:
        """Called when a request is admitted to a slot (starts prefill)."""
        if not self.enabled:
            return
        with self._lock:
            self._request_counter += 1
            # Always create trace entry (needed for slow-request detection)
            trace = RequestTrace(
                request_id=request_id,
                slot=slot,
                prompt_len=prompt_len,
                admitted_at=time.perf_counter(),
            )
            self._active[request_id] = trace

    def prefill_done(self, request_id: str, prefill_ms: float) -> None:
        """Called when prefill completes for a request."""
        if not self.enabled:
            return
        with self._lock:
            trace = self._active.get(request_id)
            if trace is None:
                return
            trace.prefill_done_at = time.perf_counter()
            trace.prefill_ms = prefill_ms

    def decode_round(
        self, request_id: str, round_idx: int, tokens_committed: int, round_ms: float
    ) -> None:
        """Called after each MTP verify/commit round for a request."""
        if not self.enabled:
            return
        with self._lock:
            trace = self._active.get(request_id)
            if trace is None:
                return
            trace.total_rounds += 1
            trace.total_tokens += tokens_committed
            if len(trace.decode_rounds) < self.max_rounds_per_trace:
                trace.decode_rounds.append((round_idx, tokens_committed, round_ms))

    def request_finished(self, request_id: str, finish_reason: str) -> None:
        """Called when a request completes (stop/length/error)."""
        if not self.enabled:
            return
        with self._lock:
            trace = self._active.pop(request_id, None)
            if trace is None:
                return
            trace.finished_at = time.perf_counter()
            trace.finish_reason = finish_reason
            self._completed.append(trace)
            if trace.total_ms >= self.slow_threshold_ms:
                self._slow.append(trace)

    def get_trace(self, request_id: str) -> dict | None:
        """Get trace for a specific request (active or completed)."""
        with self._lock:
            trace = self._active.get(request_id)
            if trace:
                return trace.to_dict()
            for t in self._completed:
                if t.request_id == request_id:
                    return t.to_dict()
        return None

    def get_slow_requests(self, limit: int = 20) -> list[dict]:
        """Get recent slow requests (above threshold)."""
        with self._lock:
            traces = list(self._slow)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get most recent completed traces."""
        with self._lock:
            traces = list(self._completed)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_stats(self) -> dict:
        """Get aggregate tracing statistics."""
        with self._lock:
            completed = list(self._completed)
        if not completed:
            return {"total_requests": 0, "active": len(self._active)}
        total_ms_list = [t.total_ms for t in completed if t.total_ms > 0]
        tps_list = [t.tokens_per_sec for t in completed if t.tokens_per_sec > 0]
        prefill_list = [t.prefill_ms for t in completed if t.prefill_ms > 0]
        return {
            "total_requests": self._request_counter,
            "active": len(self._active),
            "completed": len(completed),
            "slow_count": len(self._slow),
            "avg_total_ms": (
                round(sum(total_ms_list) / len(total_ms_list), 1) if total_ms_list else 0
            ),
            "p95_total_ms": (
                round(sorted(total_ms_list)[int(len(total_ms_list) * 0.95)], 1)
                if total_ms_list
                else 0
            ),
            "avg_tokens_per_sec": round(sum(tps_list) / len(tps_list), 1) if tps_list else 0,
            "avg_prefill_ms": (
                round(sum(prefill_list) / len(prefill_list), 1) if prefill_list else 0
            ),
        }

    def render_prometheus(self, model_name: str = "qwen3.6-27b") -> str:
        """Render tracing stats as Prometheus gauges."""
        stats = self.get_stats()
        lines = [
            "# HELP blackwellm:trace_total_requests Total traced requests",
            "# TYPE blackwellm:trace_total_requests counter",
            f'blackwellm:trace_total_requests{{model_name="{model_name}"}} '
            f'{stats["total_requests"]}',
            "# HELP blackwellm:trace_active_requests Currently active traced requests",
            "# TYPE blackwellm:trace_active_requests gauge",
            f'blackwellm:trace_active_requests{{model_name="{model_name}"}} {stats["active"]}',
            "# HELP blackwellm:trace_slow_requests Requests exceeding slow threshold",
            "# TYPE blackwellm:trace_slow_requests counter",
            f'blackwellm:trace_slow_requests{{model_name="{model_name}"}} {stats["slow_count"]}',
            "# HELP blackwellm:trace_avg_total_ms Average request total time",
            "# TYPE blackwellm:trace_avg_total_ms gauge",
            f'blackwellm:trace_avg_total_ms{{model_name="{model_name}"}} {stats["avg_total_ms"]}',
            "# HELP blackwellm:trace_avg_tokens_per_sec Average decode throughput per request",
            "# TYPE blackwellm:trace_avg_tokens_per_sec gauge",
            f'blackwellm:trace_avg_tokens_per_sec{{model_name="{model_name}"}} '
            f"{stats['avg_tokens_per_sec']}",
        ]
        return "\n".join(lines)


# Global tracer instance
tracer = RequestTracer(
    enabled=True,
    sample_rate=0.0,  # default: only slow-request detection, no per-round tracing
    slow_threshold_ms=10000.0,
)
