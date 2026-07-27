"""Human- and machine-readable reports for bfprobe routing comparisons."""

from __future__ import annotations

import json
from typing import Any

from bfprobe.routing_compare import RoutingComparisonResult


def to_dict(result: RoutingComparisonResult) -> dict[str, Any]:
    """Serialize a comparison result to a JSON-safe dict."""
    return {
        "num_layers": result.num_layers,
        "num_tokens": result.num_tokens,
        "top_k": result.top_k,
        "top1_match_rate": result.top1_match_rate,
        "set_match_rate": result.set_match_rate,
        "sequence_match_rate": result.sequence_match_rate,
        "mean_jaccard": result.mean_jaccard,
        "first_divergence": (
            {"layer": result.first_divergence.layer, "token": result.first_divergence.token}
            if result.first_divergence is not None
            else None
        ),
        "weight_cosine": result.weight_cosine,
        "weight_max_abs_diff": result.weight_max_abs_diff,
        "verdict": result.verdict,
    }


def render_text(result: RoutingComparisonResult) -> str:
    """Render a short human-readable summary of a comparison result."""
    lines = [
        "bfprobe routing comparison",
        f"  layers={result.num_layers} tokens={result.num_tokens} top_k={result.top_k}",
        f"  top-1 match rate:      {result.top1_match_rate:.4f}",
        f"  set match rate:        {result.set_match_rate:.4f}",
        f"  exact sequence match:  {result.sequence_match_rate:.4f}",
        f"  mean jaccard:          {result.mean_jaccard:.4f}",
    ]
    if result.weight_cosine is not None:
        lines.append(f"  weight cosine:         {result.weight_cosine:.6f}")
        assert result.weight_max_abs_diff is not None
        lines.append(f"  weight max|diff|:      {result.weight_max_abs_diff:.6g}")
    if result.first_divergence is not None:
        lines.append(
            f"  first divergence:      layer={result.first_divergence.layer} "
            f"token={result.first_divergence.token}"
        )
    else:
        lines.append("  first divergence:      none")
    lines.append(f"  verdict: {result.verdict}")
    return "\n".join(lines)


def render_json(result: RoutingComparisonResult) -> str:
    """Render a comparison result as an indented JSON string."""
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False)
