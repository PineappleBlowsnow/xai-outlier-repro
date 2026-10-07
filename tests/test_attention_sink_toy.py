from __future__ import annotations

import math

import pytest
import torch

from xai_repro.analysis.attention_sink_toy import run_toy_experiment


@pytest.mark.parametrize(
    ("method", "expected_token0", "expected_row_sum"),
    (
        ("softmax", (1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0) / 4.0, 1.0),
        (
            "softmax1",
            (1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0) / 4.0,
            (1.0 / 2.0 + 2.0 / 3.0 + 3.0 / 4.0 + 4.0 / 5.0) / 4.0,
        ),
    ),
)
def test_constant_logits_match_closed_form(
    method: str, expected_token0: float, expected_row_sum: float
) -> None:
    result = run_toy_experiment(
        mode="constant",
        seq_len=4,
        num_samples=1,
        batch_size=1,
        gaussian_std=1.0,
        constant_value=0.0,
        device=torch.device("cpu"),
        seed=0,
    )

    summary = result[method]
    assert math.isclose(summary["mean_attn_to_token0"], expected_token0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(summary["mean_row_sum"], expected_row_sum, rel_tol=0, abs_tol=1e-6)
