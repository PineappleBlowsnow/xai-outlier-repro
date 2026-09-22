# Attention Sinks and Activation Outliers in Small Language Models

A collaborative MVA Explainable AI project by **Tristan MARTIN and Ying JIN**. This fork preserves the original project at [Tristan22400/xai-outlier-repro](https://github.com/Tristan22400/xai-outlier-repro). See the [joint course report](../report.pdf) for experiment details and limitations.

## What the project studies

The experiments examine **softmax-1 attention** and **OrthoAdam** in small GPT-2-style language models, separately and jointly. These are reproductions and evaluations of existing methods, not claims to have invented the interventions. The implementation supports baseline, softmax-1, OrthoAdam and joint variants and analyses of attention sinks, activation kurtosis, validation perplexity and INT8 post-training quantization.

## Actual source layout

```text
src/xai_repro/
  attention.py     # softmax-1 and attention replacement
  optim.py         # OrthoAdam and orthogonal transformations
  norm.py          # normalization helpers
  model.py         # configuration/model construction
  data.py          # C4 data pipeline
  train.py         # training entry point
  callbacks.py     # utilization/wall-clock callbacks
  analysis/        # attention, kurtosis, quantization and plotting analyses
configs/gpt2_60m.yaml
scripts/
tests/
```

## Setup and validation commands

From the repository root, in an isolated Python 3.10+ environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests
```

The project metadata pins PyTorch 2.1.2 and constrains Transformers/NumPy. Choose a platform and PyTorch build compatible with those dependencies. A training invocation supported by the local source is:

```bash
python -m xai_repro.train --variant baseline --config configs/gpt2_60m.yaml --output_dir runs/baseline
```

Other variant names are `softmax1`, `orthoadam` and `softmax1_ortho`. Training requires data access, adequate compute and the project's logging configuration. Cluster job files are examples tied to the original environment; inspect paths and resource requests before reuse. The source also provides `--smoke` for a short training run, which still requires the training dependencies and data.

## Evidence and limitations

The report and saved local analyses support a course study of the interventions. Metric percentages should be quoted only with the exact run, definition and checkpoint: the reviewed report and analysis JSON use differing attention-sink representations, so this README does not collapse them into a single improvement claim. Forward/training throughput, validation perplexity, activation statistics and quantization effects are distinct measurements.

The documentation and command paths were checked against the local source on 22 September 2026. Training and the test suite were **not rerun** during documentation preparation. The reviewed artifacts establish joint authorship but do not specify a complete per-file contribution split.

## Attribution and reuse

Keep the upstream history, source references and both project authors. The current upstream and fork include the [MIT license](../LICENSE), which this documentation update preserves unchanged.
