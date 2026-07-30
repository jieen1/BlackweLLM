"""Graph-local stream coordination for Laguna MoE shared experts.

The MoE patch itself is deliberately agnostic about CUDA Graph ownership.  A
captured graph opts in by installing one of these sessions around both its
warmup and capture forwards.  That keeps a stream and its fork/join edges
private to the graph and shape that captured them; normal eager forwards stay
on the original sequential path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

_active_session = threading.local()


class MoESharedOverlapSession:
    """Run one shared-expert branch beside the routed-expert branch.

    ``wait_stream`` expresses the CUDA Graph fork/join relationship directly:
    the auxiliary stream waits for the caller stream before consuming hidden
    states, and the caller waits for it before finalization consumes the shared
    output.  PyTorch records these dependencies as graph-internal edges during
    capture, so the session must be owned by exactly one captured graph.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        cuda_api: Any | None = None,
        stream: Any | None = None,
    ) -> None:
        self.device = device
        self._cuda = torch.cuda if cuda_api is None else cuda_api
        self._stream = stream if stream is not None else self._cuda.Stream(device=device)

    def launch(
        self,
        shared_expert: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Fork the shared-expert branch after ``hidden_states`` is ready."""
        producer = self._cuda.current_stream(device=hidden_states.device)
        self._stream.wait_stream(producer)
        with self._cuda.stream(self._stream):
            return shared_expert(hidden_states)

    def join(self, device: torch.device | str) -> None:
        """Join the branch before its output participates in finalization."""
        self._cuda.current_stream(device=device).wait_stream(self._stream)


def active_moe_shared_overlap() -> MoESharedOverlapSession | None:
    """Return the session installed by the current graph capture, if any."""
    return getattr(_active_session, "value", None)


@contextmanager
def moe_shared_overlap_session(session: MoESharedOverlapSession) -> Iterator[None]:
    """Make a graph-owned overlap session visible to patched MoE forwards."""
    previous = active_moe_shared_overlap()
    _active_session.value = session
    try:
        yield
    finally:
        _active_session.value = previous
