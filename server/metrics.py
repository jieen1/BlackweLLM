"""Prometheus-style request metrics for the BlackweLLM server.

Hand-rolled (no ``prometheus_client`` dependency) to match the existing
hand-rolled ``/metrics`` endpoint in ``server/app.py`` and to honour the
repo's "no new dependencies" rule. Metrics use the runtime's own
``blackwellm:*`` namespace so dashboards do not imply a vLLM serving path.

What this module records (all measured at the request layer, so every value
is real, not estimated):

Performance:
- ``blackwellm:e2e_request_latency_seconds``      end-to-end latency per request
- ``blackwellm:time_to_first_token_seconds``      streaming time-to-first-token
- ``blackwellm:request_time_per_output_token_seconds``  (e2e - ttft) / (gen - 1)
- ``blackwellm:request_prompt_tokens``            prompt-length distribution
- ``blackwellm:request_generation_tokens``        generation-length distribution
- ``blackwellm:prompt_tokens_total`` / ``blackwellm:generation_tokens_total``  throughput

Reliability:
- ``blackwellm:request_success_total``            labelled by endpoint + finish_reason
- ``blackwellm:request_errors_total``             labelled by endpoint + status code

Thread-safety: every mutation takes ``_LOCK``. In practice all recording
happens on the asyncio event-loop thread, but the lock is cheap insurance
against the engine thread's callbacks.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()

# Stable histogram bucket boundaries for this server.
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0, 120.0, 300.0)
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
PROMPT_TOKEN_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 65536, 262144)
GENERATION_TOKEN_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 65536)


class _Histogram:
    """Cumulative-bucket histogram. ``series`` maps a label tuple to a list of
    ``len(buckets)`` cumulative counts plus ``[sum, count]``."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.series: dict[tuple, list[float]] = {}

    def observe(self, value: float, labels: tuple = ()) -> None:
        with _LOCK:
            entry = self.series.get(labels)
            if entry is None:
                entry = [0.0] * (len(self.buckets) + 2)
                self.series[labels] = entry
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    entry[i] += 1
            entry[-2] += value  # sum
            entry[-1] += 1  # count


class _Counter:
    def __init__(self) -> None:
        self.series: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, labels: tuple = ()) -> None:
        with _LOCK:
            self.series[labels] = self.series.get(labels, 0.0) + amount


# -- global metric instances -------------------------------------------------
E2E_LATENCY = _Histogram(LATENCY_BUCKETS)
TTFT = _Histogram(TTFT_BUCKETS)
TPOT = _Histogram(TPOT_BUCKETS)
PROMPT_TOKENS_HIST = _Histogram(PROMPT_TOKEN_BUCKETS)
GENERATION_TOKENS_HIST = _Histogram(GENERATION_TOKEN_BUCKETS)
PROMPT_TOKENS_TOTAL = _Counter()
GENERATION_TOKENS_TOTAL = _Counter()
REQUEST_SUCCESS = _Counter()  # labels: (endpoint, finish_reason)
REQUEST_ERRORS = _Counter()  # labels: (endpoint, status_code)


def record_request(
    endpoint: str,
    prompt_tokens: int,
    generation_tokens: int,
    finish_reason: str,
    e2e_seconds: float,
    ttft_seconds: float | None = None,
) -> None:
    """Record one completed inference request (streaming or not)."""
    ep = (endpoint,)
    PROMPT_TOKENS_HIST.observe(float(prompt_tokens), ep)
    GENERATION_TOKENS_HIST.observe(float(generation_tokens), ep)
    PROMPT_TOKENS_TOTAL.inc(float(prompt_tokens), ep)
    GENERATION_TOKENS_TOTAL.inc(float(generation_tokens), ep)
    E2E_LATENCY.observe(e2e_seconds, ep)
    REQUEST_SUCCESS.inc(1.0, (endpoint, finish_reason))
    if ttft_seconds is not None and generation_tokens > 1:
        TTFT.observe(ttft_seconds, ep)
        TPOT.observe((e2e_seconds - ttft_seconds) / (generation_tokens - 1), ep)


def record_error(endpoint: str, status_code: int) -> None:
    REQUEST_ERRORS.inc(1.0, (endpoint, str(status_code)))


def _fmt(value: float) -> str:
    # Counters/sums are integral-ish; keep ints clean, floats with precision.
    if value == int(value):
        return str(int(value))
    return f"{value:.6g}"


def _render_histogram(
    lines: list[str], name: str, help_text: str, model_name: str, hist: _Histogram
) -> None:
    if not hist.series:
        return
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} histogram")
    for labels, entry in sorted(hist.series.items()):
        base = f'model_name="{model_name}"'
        for key, value in zip(("endpoint",), labels):
            base += f',{key}="{value}"'
        cumulative = 0.0
        for bound, count in zip(hist.buckets, entry[: len(hist.buckets)]):
            cumulative = count  # entries are already cumulative
            lines.append(f'{name}_bucket{{{base},le="{bound}"}} {_fmt(cumulative)}')
        lines.append(f'{name}_bucket{{{base},le="+Inf"}} {_fmt(entry[-1])}')
        lines.append(f"{name}_sum{{{base}}} {_fmt(entry[-2])}")
        lines.append(f"{name}_count{{{base}}} {_fmt(entry[-1])}")


def _render_counter(
    lines: list[str],
    name: str,
    help_text: str,
    model_name: str,
    counter: _Counter,
    label_names: tuple[str, ...],
) -> None:
    if not counter.series:
        return
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for labels, value in sorted(counter.series.items()):
        parts = [f'model_name="{model_name}"']
        for key, val in zip(label_names, labels):
            parts.append(f'{key}="{val}"')
        lines.append(f"{name}{{{','.join(parts)}}} {_fmt(value)}")


def render(model_name: str) -> list[str]:
    """Render all app-layer request metrics as Prometheus exposition lines."""
    lines: list[str] = []
    _render_histogram(
        lines,
        "blackwellm:e2e_request_latency_seconds",
        "End-to-end request latency in seconds (request received -> response complete).",
        model_name,
        E2E_LATENCY,
    )
    _render_histogram(
        lines,
        "blackwellm:time_to_first_token_seconds",
        "Streaming time to first generated token in seconds.",
        model_name,
        TTFT,
    )
    _render_histogram(
        lines,
        "blackwellm:request_time_per_output_token_seconds",
        "Mean time per output token in seconds ((e2e - ttft) / (generation_tokens - 1)).",
        model_name,
        TPOT,
    )
    _render_histogram(
        lines,
        "blackwellm:request_prompt_tokens",
        "Distribution of prompt length in tokens.",
        model_name,
        PROMPT_TOKENS_HIST,
    )
    _render_histogram(
        lines,
        "blackwellm:request_generation_tokens",
        "Distribution of generation length in tokens.",
        model_name,
        GENERATION_TOKENS_HIST,
    )
    _render_counter(
        lines,
        "blackwellm:prompt_tokens_total",
        "Total prompt tokens processed.",
        model_name,
        PROMPT_TOKENS_TOTAL,
        ("endpoint",),
    )
    _render_counter(
        lines,
        "blackwellm:generation_tokens_total",
        "Total generation tokens produced.",
        model_name,
        GENERATION_TOKENS_TOTAL,
        ("endpoint",),
    )
    _render_counter(
        lines,
        "blackwellm:request_success_total",
        "Total successful requests by endpoint and finish reason.",
        model_name,
        REQUEST_SUCCESS,
        ("endpoint", "finish_reason"),
    )
    _render_counter(
        lines,
        "blackwellm:request_errors_total",
        "Total rejected/failed requests by endpoint and status code.",
        model_name,
        REQUEST_ERRORS,
        ("endpoint", "code"),
    )
    return lines


# ---------------------------------------------------------------------------
# D2: Runtime-internal metrics (MTP acceptance, prefix cache, KV usage)
# ---------------------------------------------------------------------------


def _accept_buckets() -> tuple[int, ...]:
    """0..K, derived from the real speculative depth rather than written out.

    This was a literal ``(0, ..., 8)`` while ``NUM_SPECULATIVE_TOKENS`` was 15
    and ``ServerEngine.stats`` kept its own 5-wide list -- three numbers for one
    quantity, none of them agreeing. A healthy DFlash round accepts most of its
    drafts, so the buckets that mattered were exactly the ones that did not
    exist: the better acceptance got, the smaller the fraction recorded.

    Imported lazily because ``dflash_constants`` is a runtime module and this
    one is imported by the torch-free CI job.
    """
    try:
        from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS
    except Exception:  # pragma: no cover - keeps /metrics alive if runtime moves
        NUM_SPECULATIVE_TOKENS = 15
    return tuple(range(NUM_SPECULATIVE_TOKENS + 1))


MTP_ACCEPT_BUCKETS = _accept_buckets()  # 0..K accepted tokens

# MTP acceptance per round (histogram of num_accepted per verify round)
mtp_acceptance_histogram = _Histogram(MTP_ACCEPT_BUCKETS)

# Prefix cache counters
_prefix_cache_hits = 0
_prefix_cache_misses = 0
_prefix_cache_hit_depth_sum = 0  # cumulative blocks matched on hits

# Per-slot KV usage (gauge: fraction of blocks_per_slot used)
_slot_kv_usage: dict[int, float] = {}


def record_mtp_acceptance(num_accepted: int) -> None:
    """Record one MTP verify round's acceptance count."""
    mtp_acceptance_histogram.observe(float(num_accepted))


def record_prefix_cache_hit(depth_blocks: int) -> None:
    """Record a prefix cache hit with the number of blocks matched."""
    global _prefix_cache_hits, _prefix_cache_hit_depth_sum
    with _LOCK:
        _prefix_cache_hits += 1
        _prefix_cache_hit_depth_sum += depth_blocks


def record_prefix_cache_miss() -> None:
    """Record a prefix cache miss (cold start)."""
    global _prefix_cache_misses
    with _LOCK:
        _prefix_cache_misses += 1


def record_slot_kv_usage(slot: int, used_blocks: int, total_blocks: int) -> None:
    """Record per-slot KV cache utilization."""
    with _LOCK:
        _slot_kv_usage[slot] = used_blocks / max(total_blocks, 1)


#: Phase 0 KV capacity snapshot (`.omx/plans/qwen38-dynamic-context-vllm-plan.md`
#: Phase 0 -- "显存、页池、前缀缓存、admission 的状态必须可观测、可诊断").
#: Populated by the /metrics handler from the backend's
#: ``kv_capacity_snapshot()`` so the Prometheus surface and the startup log
#: report the same numbers. Key names mirror
#: :meth:`runtime.model.qwen36_slots.Qwen36SlotPool.capacity_snapshot`.
_qwen_kv_capacity: dict[str, float] = {}


def record_qwen_kv_capacity(snapshot: dict[str, int]) -> None:
    """Record the Qwen KV capacity snapshot for the /metrics gauges."""
    with _LOCK:
        _qwen_kv_capacity.clear()
        _qwen_kv_capacity.update(snapshot)


def _render_qwen_kv_capacity(lines: list[str], model_name: str) -> None:
    with _LOCK:
        snap = dict(_qwen_kv_capacity)
    if not snap:
        return
    keys = [
        ("qwen_kv_pool_bytes", "Total Qwen KV tensor storage bytes (formula)"),
        ("qwen_kv_pool_bytes_measured", "Total Qwen KV tensor storage bytes (measured)"),
        ("qwen_kv_scratch_row_bytes", "Qwen scratch-row KV bytes (reclaim target)"),
        ("qwen_kv_total_bundles", "Physical KV pages allocated (slots + scratch)"),
        ("qwen_kv_pages_per_slot", "Logical pages per slot"),
        ("qwen_kv_slots", "Configured slot count"),
        ("qwen_kv_full_attention_layers", "Full-attention layers feeding the KV pool"),
        ("qwen_kv_gdn_layers", "GDN/recurrent layers"),
        ("qwen_kv_bundle_bytes", "Bytes in one lock-step backbone/MTP KV bundle"),
        ("qwen_kv_mtp_pool_bytes", "MTP KV tensor storage included in the pool total"),
        ("qwen_kv_free_bundles", "Currently reclaimable Qwen KV bundles"),
        ("qwen_kv_live_bundles", "Currently live Qwen KV bundles"),
        ("qwen_kv_cached_bundles", "Cached refcount-zero Qwen KV bundles"),
        (
            "qwen_kv_request_reserved_bundles",
            "Unmaterialized KV bundles promised to admitted requests",
        ),
        ("qwen_kv_watermark_bundles", "Emergency/COW bundles withheld from admission"),
    ]
    for key, help_text in keys:
        lines.append(f"# HELP blackwellm:{key} {help_text}")
        lines.append(f"# TYPE blackwellm:{key} gauge")
        lines.append(f'blackwellm:{key}{{model_name="{model_name}"}} {snap.get(key, -1)}')


def render_d2_metrics(model_name: str = "qwen3.6-27b") -> str:
    """Render D2 metrics in Prometheus exposition format."""
    lines: list[str] = []
    # MTP acceptance
    _render_histogram(
        lines,
        "blackwellm:mtp_accepted_tokens",
        "MTP accepted tokens per verify round",
        model_name,
        mtp_acceptance_histogram,
    )
    # Prefix cache
    with _LOCK:
        hits = _prefix_cache_hits
        misses = _prefix_cache_misses
        depth_sum = _prefix_cache_hit_depth_sum
    lines.append("# HELP blackwellm:prefix_cache_hits_total Prefix cache hit count")
    lines.append("# TYPE blackwellm:prefix_cache_hits_total counter")
    lines.append(f"blackwellm:prefix_cache_hits_total {hits}")
    lines.append("# HELP blackwellm:prefix_cache_misses_total Prefix cache miss count")
    lines.append("# TYPE blackwellm:prefix_cache_misses_total counter")
    lines.append(f"blackwellm:prefix_cache_misses_total {misses}")
    if hits > 0:
        lines.append("# HELP blackwellm:prefix_cache_avg_hit_depth Average blocks matched on hit")
        lines.append("# TYPE blackwellm:prefix_cache_avg_hit_depth gauge")
        lines.append(f"blackwellm:prefix_cache_avg_hit_depth {depth_sum / hits:.1f}")
    # Per-slot KV usage
    with _LOCK:
        slot_usage = dict(_slot_kv_usage)
    if slot_usage:
        lines.append("# HELP blackwellm:slot_kv_usage_fraction Per-slot KV cache utilization")
        lines.append("# TYPE blackwellm:slot_kv_usage_fraction gauge")
        for slot, frac in sorted(slot_usage.items()):
            lines.append(f'blackwellm:slot_kv_usage_fraction{{slot="{slot}"}} {frac:.3f}')
    return "\n".join(lines)
