"""Human-readable and JSON rendering of a ``scan.DivergenceReport``.

Pure formatting only -- no capture, no scanning, no I/O. Kept separate from
``scan.py`` so the scan algorithm stays trivially unit-testable without
worrying about presentation, and separate from ``cli.py`` so the text/JSON
shape can be unit-tested without argparse.
"""

from __future__ import annotations

from typing import Any

from bfdiag.divergence.scan import DivergenceReport


def to_json_dict(report: DivergenceReport) -> dict[str, Any]:
    """Machine-readable rendering, used by ``bf divergence --json``."""
    return {
        "first_divergent_layer": report.first_divergent_layer,
        "first_divergent_submodules": list(report.first_divergent_submodules),
        "has_divergence": report.has_divergence,
        "layers": [
            {
                "layer_idx": layer.layer_idx,
                "passed": layer.passed,
                "submodules": [_verdict_dict(verdict) for verdict in layer.submodules],
            }
            for layer in report.layers
        ],
    }


def _verdict_dict(verdict: Any) -> dict[str, Any]:
    return {
        "submodule": verdict.submodule,
        "passed": verdict.passed,
        "cosine_similarity": verdict.cosine_similarity,
        "max_abs_error": verdict.max_abs_error,
        "rel_max_abs_error": verdict.rel_max_abs_error,
        "mean_abs_error": verdict.mean_abs_error,
        "top1_agreement": verdict.top1_agreement,
        "top5_agreement": verdict.top5_agreement,
        "threshold": {
            "min_cosine": verdict.threshold.min_cosine,
            "max_rel_abs_error": verdict.threshold.max_rel_abs_error,
            "min_top1_agreement": verdict.threshold.min_top1_agreement,
            "min_top5_agreement": verdict.threshold.min_top5_agreement,
        },
        "reason": verdict.reason(),
    }


def format_text_report(report: DivergenceReport) -> str:
    """Render the "layer 00-41 cos>=0.99999 / layer 42 ... <- first
    divergent layer / drill-down" style table plus a one-line conclusion."""
    lines: list[str] = []
    first_bad = report.first_divergent_layer
    run_start: int | None = None
    run_min_cos = float("inf")

    def flush_pass_run(end_layer: int) -> None:
        nonlocal run_start, run_min_cos
        if run_start is None:
            return
        if run_start == end_layer:
            lines.append(f"  layer {run_start:02d}      cos >= {run_min_cos:.5f}")
        else:
            lines.append(f"  layer {run_start:02d}-{end_layer:02d}   cos >= {run_min_cos:.5f}")
        run_start = None
        run_min_cos = float("inf")

    for layer in report.layers:
        layer_min_cos = min((item.cosine_similarity for item in layer.submodules), default=1.0)
        if layer.passed:
            if run_start is None:
                run_start = layer.layer_idx
            run_min_cos = min(run_min_cos, layer_min_cos)
            continue
        flush_pass_run(layer.layer_idx - 1)
        marker = "  <- first divergent layer" if layer.layer_idx == first_bad else ""
        lines.append(f"  layer {layer.layer_idx:02d}      cos = {layer_min_cos:.4f}{marker}")
        for verdict in layer.worst_submodules:
            lines.append(f"      |- {verdict.submodule:<28s} FAIL  {verdict.reason()}")
    if report.layers:
        flush_pass_run(report.layers[-1].layer_idx)

    lines.append("-" * 64)
    if report.has_divergence:
        layer_idx = report.first_divergent_layer
        layer = next(item for item in report.layers if item.layer_idx == layer_idx)
        worst = layer.worst_submodules[0]
        conclusion = (
            f"第一个发散点: layer {layer_idx} 的 {worst.submodule}, "
            f"cos={worst.cosine_similarity:.4f}, top1 agreement={worst.top1_agreement:.4f}"
        )
        if len(layer.worst_submodules) > 1:
            others = ", ".join(item.submodule for item in layer.worst_submodules[1:])
            conclusion += f" (还有: {others})"
        lines.append(conclusion)
    else:
        lines.append(f"未发现发散: 全部 {len(report.layers)} 层均在阈值内。")
    return "\n".join(lines)


if __name__ == "__main__":
    from bfdiag.divergence.scan import scan_layers

    demo_oracle = {i: {"self_attn": [1.0, 2.0, 3.0, 4.0]} for i in range(5)}
    demo_candidate = {i: {"self_attn": [1.0, 2.0, 3.0, 4.0]} for i in range(5)}
    demo_candidate[3] = {"self_attn": [-1.0, -2.0, -3.0, -4.0]}
    demo_report = scan_layers(demo_oracle, demo_candidate)
    print(format_text_report(demo_report))
    print()
    print(to_json_dict(demo_report))
