"""Warm-daemon subsystem of bfdiag: a hot, model-loaded process that diagnostic
snippets are executed against, so the GPU cold start (weights + draft model +
CUDA Graph capture + autotune, several minutes) is paid once per day instead
of once per experiment.

Modules:
    protocol: newline-delimited JSON wire format for the Unix domain socket.
    provider: pluggable ``EngineProvider`` (real Laguna engine vs. a
        CPU-only fake for GPU-free development and testing).
    session: the concrete "what must be reset" checklist and reset routine
        for the real Laguna + DFlash engine.
    canary: the fixed-prompt greedy self-check that runs before every
        ``exec`` to catch cross-experiment state contamination.
    server: the ``Daemon`` -- Unix socket server, single-instance flock,
        FIFO worker thread, timeout/taint/restart policy.
    client: thin socket client + ``bf daemon``/``bf exec``/``bf repl``
        implementations.
    queue: ``bf submit`` -- FIFO submission with Cartesian-product env sweeps.
    cli: ``register(subparsers)`` entry point auto-mounted by ``bfdiag/cli.py``.
"""
