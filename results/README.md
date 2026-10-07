# Archived local experiments

This directory contains existing files recovered from the local course-project
checkout on 7 October 2026. They supplement the joint report and the previously
published cross-model diagnostics. They retain the project's joint attribution
to Tristan Martin and Ying Jin; adding local files does not establish a new
division of authorship.

## Synthetic attention experiment

The [toy experiment](../src/xai_repro/analysis/attention_sink_toy.py) compares
softmax and softmax-1 on constant or independent Gaussian logits under a causal
mask. It samples logits directly: it does **not** train or evaluate a language
model and cannot establish downstream quality improvements.

| Archived file | Saved configuration |
|---|---|
| [attention_sink_toy.json](toy/attention_sink_toy.json) | Length 64, 1,024 samples, batch 128, seed 0, CPU |
| [attention_sink_toy_local.json](toy/attention_sink_toy_local.json) | Length 256, 2,048 samples, batch 128, seed 0, CPU |
| [attention_sink_toy_local.png](toy/attention_sink_toy_local.png) | Saved two-regime visualization accompanying the local experiment |

For zero constant logits, a query with `q` visible keys assigns `1/q` to each
key under softmax and `1/(q+1)` under softmax-1. The latter also assigns `1/(q+1)`
to an implicit null destination. Averaged across a sequence of length `T`, the
first key therefore receives `H_T/T` versus `(H_(T+1)-1)/T`. This explains the
causal visibility baseline; it does not show that all learned attention sinks
have the same cause or disappear with softmax-1.

Run from the repository root with compatible PyTorch/Transformers and plotting
dependencies installed, and the package on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m pytest tests/test_attention_sink_toy.py tests/test_softmax1.py
PYTHONPATH=src python -m xai_repro.analysis.attention_sink_toy --device cpu --seq-len 64 --num-samples 1024 --batch-size 128 --seed 0 --output runs_smoke/toy.json
PYTHONPATH=src python -m xai_repro.analysis.plot_attention_sink_toy --input runs_smoke/toy.json --output runs_smoke/toy.png
```

On PowerShell, set `$env:PYTHONPATH = 'src'` before the `python` commands instead
of using the shell prefix above. `runs_smoke/` is ignored, so reruns do not
overwrite the historical JSON or figures.

The imported `softmax1` lives in `xai_repro.attention`; the recovered toy script's
obsolete `xai_repro.attention.softmax1` import was corrected for this layout.
The plotting command additionally needs Matplotlib. The validation environment
and actual checks are recorded in [VALIDATION.md](VALIDATION.md).

## Historical training and activation plots — interpret with caution

The [historical directory](historical/) preserves ten PNGs with their original
filenames and pixels. These are archived outputs, **not newly reproduced
training results or verified headline claims**. Complete per-run configuration,
checkpoint and raw training-log provenance has not been reconstructed for
these images.

Several embedded titles are stronger than the plotted evidence:

- `fig2_attention_dominance.png` and `layerwise_dominance.png` say that softmax-1
  eliminates the attention sink, while their curves still show substantial
  first-token attention. The caption is not a supported general conclusion.
- `fig3_kurtosis_depth.png` says OrthoAdam keeps kurtosis at approximately 3,
  while much of its curve remains above that reference.
- `fig4_max_activation_depth.png` and `layerwise_max_Activation.png` say peak
  activations are reduced, while the plotted OrthoAdam curve is higher than the
  baseline in multiple layers.
- `fig1_training_stability.png` and `supp_summary_bars.png` lack a complete
  traceable run bundle here. Do not use them to claim equivalent convergence,
  universal outlier suppression, or an independently reproduced improvement.

| Files | What is archived |
|---|---|
| `atten_per_layer_baseline.png`, `atten_per_layer_orthoadam.png`, `atten_per_layer_softmax-1.png` | Per-layer attention heatmaps |
| `fig1_training_stability.png` | Saved validation-loss curves |
| `fig2_attention_dominance.png`, `layerwise_dominance.png` | First-token dominance plots with different series |
| `fig3_kurtosis_depth.png` | Saved depth-wise kurtosis plot |
| `fig4_max_activation_depth.png`, `layerwise_max_Activation.png` | Saved maximum-activation plots |
| `supp_summary_bars.png` | Saved aggregate metric comparison |

These files are retained as inspectable project history. The toy JSON and its
closed-form checks are a separate kind of evidence from trained-model plots.

## File selection and provenance

[manifest.json](manifest.json) records the original relative path, published
path, SHA-256 digests and any source adjustment for all 16 recovered files.
Text line-ending normalization is recorded where applicable. Historical data
and image bytes are preserved.

The imported set excludes the downloaded reference paper, incomplete report
source fragments (`intro.tex` and `preamble.tex`), an unfinished planning note,
and a standalone cross-architecture image whose provenance was not established.
The joint [report.pdf](../report.pdf) is already present in the repository.
Caches, environments, weights and credentials are not included. The existing
MIT license and upstream history are preserved.
