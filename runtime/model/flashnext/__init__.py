"""Flash-Next (qwen4_exp) native modules for the self-built runtime.

Keep the package initializer torch-free.  The HTTP format layer imports the
vision preprocessing module while handling ordinary text requests too; an
eager import of ``spec`` here would pull in Torch (and CUDA-only graph code)
before a text-only CPU test or lightweight server module can be collected.
The public graph classes remain available through module-level lazy access.
"""

__all__ = ["FlashNextSpecEngine", "FlashNextVerifyGraph"]


def __getattr__(name: str):
    if name in __all__:
        from runtime.model.flashnext.spec import FlashNextSpecEngine, FlashNextVerifyGraph

        return {
            "FlashNextSpecEngine": FlashNextSpecEngine,
            "FlashNextVerifyGraph": FlashNextVerifyGraph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
