"""Toy analysis for the causal-mask / softmax attention-sink mechanism.

This script does not train a language model. It isolates the attention
normalization step by sampling synthetic attention logits, applying a
causal mask, and comparing stock softmax against softmax-1.

Two synthetic regimes are useful:

* ``constant``: every visible logit is identical, i.e. a head has no
  preference among keys. This is the cleanest way to show the forced
  normalization effect of softmax.
* ``gaussian``: logits are i.i.d. Gaussian noise. This is a slightly
  noisier but more realistic "uninformative head" toy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from xai_repro.attention import softmax1


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Mask out future keys for each query position."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def _sample_scores(
    mode: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    gaussian_std: float,
    constant_value: float,
) -> Tensor:
    """Sample synthetic logits for one toy batch."""
    if mode == "constant":
        return torch.full((batch_size, seq_len, seq_len), constant_value, device=device)
    if mode == "gaussian":
        return gaussian_std * torch.randn(batch_size, seq_len, seq_len, device=device)
    raise ValueError(f"unsupported mode: {mode}")


def _apply_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    masked = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    return torch.softmax(masked, dim=-1)


def _apply_softmax1(scores: Tensor, mask: Tensor) -> Tensor:
    masked = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    return softmax1(masked, dim=-1)


def _empty_accumulator(seq_len: int) -> dict[str, Any]:
    zeros = torch.zeros(seq_len, dtype=torch.float64)
    return {
        "n_samples": 0,
        "sum_token0_mass": 0.0,
        "sum_row_sums": 0.0,
        "sum_key_mass": zeros.clone(),
        "sum_token0_by_query": zeros.clone(),
        "sum_row_sum_by_query": zeros.clone(),
    }


def _update(acc: dict[str, Any], weights: Tensor) -> None:
    row_sums = weights.sum(dim=-1)
    batch_size, seq_len, _ = weights.shape
    acc["n_samples"] += batch_size
    acc["sum_token0_mass"] += float(weights[:, :, 0].sum())
    acc["sum_row_sums"] += float(row_sums.sum())
    acc["sum_key_mass"] += weights.sum(dim=(0, 1)).to(torch.float64).cpu()
    acc["sum_token0_by_query"] += weights[:, :, 0].sum(dim=0).to(torch.float64).cpu()
    acc["sum_row_sum_by_query"] += row_sums.sum(dim=0).to(torch.float64).cpu()
    assert seq_len == acc["sum_key_mass"].shape[0]


def _finalize(acc: dict[str, Any], seq_len: int) -> dict[str, Any]:
    n_samples = int(acc["n_samples"])
    if n_samples == 0:
        raise ValueError("no samples accumulated")

    n_rows = n_samples * seq_len
    mean_row_sum_by_query = (acc["sum_row_sum_by_query"] / n_samples).tolist()

    return {
        "n_samples": n_samples,
        "mean_attn_to_token0": acc["sum_token0_mass"] / n_rows,
        "mean_row_sum": acc["sum_row_sums"] / n_rows,
        "mean_null_mass": 1.0 - acc["sum_row_sums"] / n_rows,
        "mean_key_mass": (acc["sum_key_mass"] / n_rows).tolist(),
        "mean_token0_by_query": (acc["sum_token0_by_query"] / n_samples).tolist(),
        "mean_row_sum_by_query": mean_row_sum_by_query,
        "mean_null_mass_by_query": [1.0 - x for x in mean_row_sum_by_query],
    }


@torch.no_grad()
def run_toy_experiment(
    *,
    mode: str,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    gaussian_std: float,
    constant_value: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run one synthetic attention-sink experiment."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    mask = _causal_mask(seq_len, device=device)
    softmax_acc = _empty_accumulator(seq_len)
    softmax1_acc = _empty_accumulator(seq_len)

    remaining = num_samples
    while remaining > 0:
        cur_batch = min(batch_size, remaining)
        scores = _sample_scores(
            mode=mode,
            batch_size=cur_batch,
            seq_len=seq_len,
            device=device,
            gaussian_std=gaussian_std,
            constant_value=constant_value,
        )
        _update(softmax_acc, _apply_softmax(scores, mask))
        _update(softmax1_acc, _apply_softmax1(scores, mask))
        remaining -= cur_batch

    return {
        "mode": mode,
        "seq_len": seq_len,
        "num_samples": num_samples,
        "softmax": _finalize(softmax_acc, seq_len),
        "softmax1": _finalize(softmax1_acc, seq_len),
    }


def _format_summary(result: dict[str, Any]) -> str:
    s0 = result["softmax"]
    s1 = result["softmax1"]
    lines = [
        f"Mode: {result['mode']}",
        (
            "  softmax   : "
            f"mean_attn_to_token0={s0['mean_attn_to_token0']:.6f}, "
            f"mean_row_sum={s0['mean_row_sum']:.6f}, "
            f"mean_null_mass={s0['mean_null_mass']:.6f}"
        ),
        (
            "  softmax-1 : "
            f"mean_attn_to_token0={s1['mean_attn_to_token0']:.6f}, "
            f"mean_row_sum={s1['mean_row_sum']:.6f}, "
            f"mean_null_mass={s1['mean_null_mass']:.6f}"
        ),
        (
            "  delta      : "
            f"token0={s1['mean_attn_to_token0'] - s0['mean_attn_to_token0']:+.6f}, "
            f"row_sum={s1['mean_row_sum'] - s0['mean_row_sum']:+.6f}, "
            f"null_mass={s1['mean_null_mass'] - s0['mean_null_mass']:+.6f}"
        ),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy causal-mask / softmax attention-sink analysis.")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=("constant", "gaussian"),
        default=("constant", "gaussian"),
        help="Synthetic logit regimes to run.",
    )
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gaussian-std", type=float, default=1.0)
    p.add_argument("--constant-value", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'.")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    results = {
        "settings": {
            "seq_len": args.seq_len,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "gaussian_std": args.gaussian_std,
            "constant_value": args.constant_value,
            "seed": args.seed,
            "device": str(device),
        },
        "results": [],
    }

    for mode in args.modes:
        result = run_toy_experiment(
            mode=mode,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            gaussian_std=args.gaussian_std,
            constant_value=args.constant_value,
            device=device,
            seed=args.seed,
        )
        results["results"].append(result)
        print(_format_summary(result))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved JSON to {args.output}")


if __name__ == "__main__":
    main()
