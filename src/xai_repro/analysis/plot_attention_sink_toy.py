"""Plot the toy attention-sink analysis results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot attention-sink toy JSON results.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load(args.input)
    results = payload["results"]

    fig, axes = plt.subplots(2, len(results), figsize=(7 * len(results), 9), constrained_layout=True)
    if len(results) == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, result in enumerate(results):
        mode = result["mode"]
        softmax = result["softmax"]
        softmax1 = result["softmax1"]

        ax_bar = axes[0][col]
        labels = ["token0 mass", "row sum", "null mass"]
        x = range(len(labels))
        width = 0.35
        ax_bar.bar(
            [i - width / 2 for i in x],
            [
                softmax["mean_attn_to_token0"],
                softmax["mean_row_sum"],
                softmax["mean_null_mass"],
            ],
            width=width,
            label="softmax",
            color="#1f77b4",
        )
        ax_bar.bar(
            [i + width / 2 for i in x],
            [
                softmax1["mean_attn_to_token0"],
                softmax1["mean_row_sum"],
                softmax1["mean_null_mass"],
            ],
            width=width,
            label="softmax-1",
            color="#d62728",
        )
        ax_bar.set_xticks(list(x), labels)
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_title(f"{mode}: aggregate metrics")
        ax_bar.grid(True, alpha=0.25)
        ax_bar.legend()

        ax_line = axes[1][col]
        q = list(range(result["seq_len"]))
        ax_line.plot(q, softmax["mean_token0_by_query"], label="softmax token0", color="#1f77b4")
        ax_line.plot(q, softmax1["mean_token0_by_query"], label="softmax-1 token0", color="#d62728")
        ax_line.plot(
            q,
            softmax1["mean_null_mass_by_query"],
            label="softmax-1 null mass",
            color="#2ca02c",
            linestyle="--",
        )
        ax_line.set_title(f"{mode}: by query position")
        ax_line.set_xlabel("Query position")
        ax_line.set_ylabel("Attention mass")
        ax_line.grid(True, alpha=0.25)
        ax_line.legend()

    fig.suptitle("Toy Attention-Sink Mechanism Under Causal Masking", fontsize=18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
