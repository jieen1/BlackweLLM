"""Freeze and verify the vLLM oracle for Laguna's fixed native MoE router.

Artifacts are intentionally local diagnostics, never Git fixtures.  ``capture``
uses vLLM exactly once to write tensors under ``.bfdiag/router-oracle``;
``verify`` then compares the native C ABI bit-for-bit without importing vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

EXPERTS = 256
TOP_K = 10
DEFAULT_ROWS = (0, 1, 2, 3, 4, 8, 16, 64, 8192)
DEFAULT_FAMILIES = (
    "normal",
    "uniform",
    "equal",
    "tie",
    "near_tie",
    "signed_zero",
    "extreme",
    "nonfinite",
    "zero_sum",
)


def parse_rows(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, non-negative router-row matrix."""
    try:
        rows = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from error
    if not rows or any(row < 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must contain at least one non-negative integer")
    if len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("rows must not contain duplicates")
    return rows


def parse_families(value: str) -> tuple[str, ...]:
    """Parse the explicitly named input families for a capture."""
    families = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
    if not families or unknown:
        choices = ", ".join(DEFAULT_FAMILIES)
        raise argparse.ArgumentTypeError(
            f"unknown router input family {unknown}; choices: {choices}"
        )
    if len(set(families)) != len(families):
        raise argparse.ArgumentTypeError("families must not contain duplicates")
    return families


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_root() -> Path:
    return _repo_root() / ".bfdiag" / "router-oracle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True
    ).strip()


def _artifact_fingerprint(
    *, rows: tuple[int, ...], families: tuple[str, ...], biases: dict[str, Any]
) -> str:
    payload = {
        "experts": EXPERTS,
        "top_k": TOP_K,
        "rows": rows,
        "families": families,
        "bias_sha256": {
            name: hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
            for name, tensor in sorted(biases.items())
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _require_torch() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Laguna router oracle requires CUDA")
    return torch


def _make_input(torch: Any, *, family: str, rows: int, device: str) -> Any:
    """Build deterministic FP32 test logits for one semantic input family."""
    logits = torch.empty((rows, EXPERTS), dtype=torch.float32, device=device)
    if rows == 0:
        return logits

    generator = torch.Generator(device=device)
    generator.manual_seed(20_260_729 + rows * 97 + DEFAULT_FAMILIES.index(family))
    if family == "normal":
        logits.normal_(generator=generator)
    elif family == "uniform":
        logits.uniform_(-8.0, 8.0, generator=generator)
    elif family in {"equal", "tie"}:
        logits.zero_()
    elif family == "near_tie":
        logits.zero_()
        base = torch.tensor(0.125, dtype=torch.float32, device=device)
        for expert in range(16):
            target = torch.tensor(float(expert + 1), device=device)
            logits[:, expert] = torch.nextafter(base, target)
    elif family == "signed_zero":
        logits.zero_()
        logits[:, ::2] = -0.0
    elif family == "extreme":
        logits.fill_(-80.0)
        logits[:, 1::4] = 80.0
        logits[:, 2::16] = 20.0
    elif family == "nonfinite":
        logits.zero_()
        logits[:, 0] = float("nan")
        logits[:, 1] = float("inf")
        logits[:, 2] = float("-inf")
    elif family == "zero_sum":
        logits.fill_(float("-inf"))
    else:  # parse_families keeps this defensive branch unreachable.
        raise ValueError(f"unknown router input family: {family}")
    return logits


def _load_biases(torch: Any, checkpoint: Path | None) -> dict[str, Any]:
    from runtime.backends.laguna_sparkinfer_moe import (
        MOE_LAYER_IDS,
        _find_checkpoint,
        load_moe_layer_e_score_correction_bias,
    )

    resolved = checkpoint or _find_checkpoint()
    return {
        f"layer_{layer_idx}": load_moe_layer_e_score_correction_bias(
            resolved, layer_idx, device="cuda"
        ).contiguous()
        for layer_idx in MOE_LAYER_IDS
    }


def _load_live_logits(torch: Any, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("live router logits must be a non-empty tensor mapping")
    logits: dict[str, Any] = {}
    for layer_name, tensor in sorted(payload.items()):
        if (
            not isinstance(layer_name, str)
            or not layer_name.startswith("layer_")
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.ndim != 2
            or tensor.shape[1] != EXPERTS
        ):
            raise ValueError(f"invalid live router logits for {layer_name!r}")
        logits[layer_name] = tensor.contiguous()
    return logits


def _vllm_oracle(torch: Any, logits: Any, bias: Any) -> tuple[Any, Any]:
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    if logits.shape[0] == 0:
        return (
            torch.empty((0, TOP_K), dtype=torch.float32, device=logits.device),
            torch.empty((0, TOP_K), dtype=torch.int32, device=logits.device),
        )
    return fused_topk_bias(
        logits,
        logits,
        "sigmoid",
        bias,
        TOP_K,
        True,
        routed_scaling_factor=1.0,
    )


def _write_artifact(destination: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    import torch

    if destination.exists():
        raise FileExistsError(f"oracle artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tmp-router-oracle-", dir=destination.parent))
    try:
        payload_path = temporary / "oracle.pt"
        torch.save(payload, payload_path)
        metadata["payload_sha256"] = _sha256(payload_path)
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def capture(args: argparse.Namespace) -> Path:
    """Capture a full vLLM oracle matrix into one immutable local artifact."""
    torch = _require_torch()
    biases = _load_biases(torch, args.checkpoint)
    fingerprint = _artifact_fingerprint(rows=args.rows, families=args.families, biases=biases)
    destination = args.output or _default_root() / fingerprint
    references: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}

    for family in args.families:
        for rows in args.rows:
            input_name = f"{family}-m{rows}"
            logits = _make_input(torch, family=family, rows=rows, device="cuda")
            inputs[input_name] = logits.cpu()
            for bias_name, bias in biases.items():
                weights, ids = _vllm_oracle(torch, logits, bias)
                references[f"{input_name}:{bias_name}"] = {
                    "weights": weights.cpu(),
                    "ids": ids.cpu(),
                }

    torch.cuda.synchronize()
    payload = {
        "inputs": inputs,
        "biases": {name: tensor.cpu() for name, tensor in biases.items()},
        "references": references,
    }
    metadata = {
        "artifact_version": 1,
        "capture_git_sha": _git_sha(),
        "experts": EXPERTS,
        "top_k": TOP_K,
        "rows": list(args.rows),
        "families": list(args.families),
        "biases": sorted(biases),
        "fingerprint": fingerprint,
        "vllm_oracle": "fused_topk_bias(sigmoid, renormalize=True, routed_scaling_factor=1.0)",
    }
    _write_artifact(destination, payload, metadata)
    return destination


def capture_live(args: argparse.Namespace) -> Path:
    """Freeze vLLM routing for logits recorded from a production forward."""
    torch = _require_torch()
    live_logits = _load_live_logits(torch, args.logits)
    checkpoint_biases = _load_biases(torch, args.checkpoint)
    unknown_layers = sorted(set(live_logits) - set(checkpoint_biases))
    if unknown_layers:
        raise ValueError(f"live router logits have no checkpoint bias: {unknown_layers}")

    inputs: dict[str, Any] = {}
    biases: dict[str, Any] = {}
    references: dict[str, dict[str, Any]] = {}
    for layer_name, cpu_logits in live_logits.items():
        input_name = f"live-{layer_name}"
        inputs[input_name] = cpu_logits
        biases[layer_name] = checkpoint_biases[layer_name].cpu()
        weights, ids = _vllm_oracle(
            torch, cpu_logits.to(device="cuda"), checkpoint_biases[layer_name]
        )
        references[f"{input_name}:{layer_name}"] = {
            "weights": weights.cpu(),
            "ids": ids.cpu(),
        }

    torch.cuda.synchronize()
    fingerprint_material = b"".join(
        name.encode() + tensor.numpy().tobytes() for name, tensor in live_logits.items()
    )
    fingerprint = hashlib.sha256(fingerprint_material).hexdigest()[:16]
    destination = args.output or _default_root() / f"live-{fingerprint}"
    payload = {"inputs": inputs, "biases": biases, "references": references}
    metadata = {
        "artifact_version": 1,
        "capture_git_sha": _git_sha(),
        "experts": EXPERTS,
        "top_k": TOP_K,
        "fingerprint": fingerprint,
        "kind": "live-production-router-logits",
        "layers": sorted(live_logits),
        "source_logits": str(args.logits),
        "vllm_oracle": "fused_topk_bias(sigmoid, renormalize=True, routed_scaling_factor=1.0)",
    }
    _write_artifact(destination, payload, metadata)
    return destination


def verify(args: argparse.Namespace) -> dict[str, int]:
    """Compare native routing with the frozen oracle or BF16 ABI baseline."""
    import torch

    from runtime.laguna_router import LagunaRouterLibrary

    manifest_path = args.artifact / "manifest.json"
    payload_path = args.artifact / "oracle.pt"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(payload_path) != metadata.get("payload_sha256"):
        raise RuntimeError("router oracle payload SHA256 does not match its manifest")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    router = LagunaRouterLibrary.load()
    checked = 0
    for reference_name, reference in payload["references"].items():
        input_name, bias_name = reference_name.split(":", maxsplit=1)
        cpu_logits = payload["inputs"][input_name]
        logits = cpu_logits.to(device="cuda", dtype=torch.float32)
        weights = torch.empty((logits.shape[0], TOP_K), dtype=torch.float32, device="cuda")
        ids = torch.empty((logits.shape[0], TOP_K), dtype=torch.int32, device="cuda")
        bias = payload["biases"][bias_name].to(device="cuda", dtype=torch.float32)
        if args.bf16_input:
            bf16_logits = logits.to(dtype=torch.bfloat16)
            baseline_weights = torch.empty_like(weights)
            baseline_ids = torch.empty_like(ids)
            expected_weights, expected_ids = router.launch(
                bf16_logits.float(), bias, baseline_weights, baseline_ids
            )
            got_weights, got_ids = router.launch(bf16_logits, bias, weights, ids)
        else:
            got_weights, got_ids = router.launch(logits, bias, weights, ids)
        torch.cuda.synchronize()
        expected_weights = expected_weights.cpu() if args.bf16_input else reference["weights"]
        expected_ids = expected_ids.cpu() if args.bf16_input else reference["ids"]
        if not torch.equal(got_ids.cpu(), expected_ids):
            raise AssertionError(f"router ids differ for {reference_name}")
        if not torch.equal(got_weights.cpu(), expected_weights):
            raise AssertionError(f"router weights differ for {reference_name}")
        sorted_ids = torch.sort(got_ids, dim=1).values
        if logits.shape[0] and torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise AssertionError(f"router duplicated an expert id for {reference_name}")
        if not bool(torch.isfinite(got_weights).all()):
            raise AssertionError(f"router produced non-finite weights for {reference_name}")
        checked += 1
    return {
        "cases": checked,
        "input_dtype": "bf16" if args.bf16_input else "fp32",
        "rows": len(payload["inputs"]),
        "biases": len(payload["biases"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture", help="freeze vLLM router outputs")
    capture_parser.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS)
    capture_parser.add_argument("--families", type=parse_families, default=DEFAULT_FAMILIES)
    capture_parser.add_argument("--checkpoint", type=Path)
    capture_parser.add_argument("--output", type=Path)
    live_parser = subparsers.add_parser(
        "capture-live", help="freeze vLLM routing for production gate logits"
    )
    live_parser.add_argument("--logits", required=True, type=Path)
    live_parser.add_argument("--checkpoint", type=Path)
    live_parser.add_argument("--output", type=Path)
    verify_parser = subparsers.add_parser(
        "verify", help="verify native router against frozen output"
    )
    verify_parser.add_argument("artifact", type=Path)
    verify_parser.add_argument(
        "--bf16-input",
        action="store_true",
        help="compare the BF16 ABI to the FP32 ABI on the same BF16-rounded logits",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "capture":
        print(json.dumps({"artifact": str(capture(args))}, sort_keys=True))
    elif args.command == "capture-live":
        print(json.dumps({"artifact": str(capture_live(args))}, sort_keys=True))
    else:
        print(json.dumps(verify(args), sort_keys=True))


if __name__ == "__main__":
    main()
