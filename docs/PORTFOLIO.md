# Attention Sinks and Activation Outliers in Small Language Models

A collaborative MVA Explainable AI project by **Tristan MARTIN and Ying JIN**. This fork preserves the original project at [Tristan22400/xai-outlier-repro](https://github.com/Tristan22400/xai-outlier-repro). See the [joint course report](../report.pdf) for experiment details and limitations.

## What the project studies

The experiments examine **softmax-1 attention** and **OrthoAdam** in small GPT-2-style language models, separately and jointly. These are reproductions and evaluations of existing methods, not claims to have invented the interventions. The implementation supports baseline, softmax-1, OrthoAdam and joint variants and analyses of attention sinks, activation kurtosis, validation perplexity and INT8 post-training quantization.

## Ying Jin's documented contribution

[Commit 4327383](https://github.com/PineappleBlowsnow/xai-outlier-repro/commit/4327383cacfd888c8892e8bdcced3624fa63fe56) adds two diagnostic scripts and their saved outputs:

| Contribution | Source | Historical output |
|---|---|---|
| Cross-model attention and activation visualization, including attention-sink scoring and per-token activation kurtosis | [visualize_figure1.py](../src/xai_repro/analysis/visualize_figure1.py) | [Figure 1 replication](../analysis_results_v2/figure1_replication.png) |
| Input-only probes with token replacement, context-length sweeps, and positional-embedding controls | [input_ablation.py](../src/xai_repro/analysis/input_ablation.py) | [Ablation results](../analysis_results_v2/ablation_results.json) |

The visualization script configures Pythia, GPT-2 and Llama models. It requests attention tensors and hidden states through the model's `output_attentions` and `output_hidden_states` options. These scripts do not implement a forward-hook framework or train those pretrained models.

The archived ablation JSON contains 24 experiment records across GPT-2, GPT-2 Medium, Pythia-31M and Pythia-160M. Some records contain skipped conditions, such as a context length exceeding a model's limit or a positional-embedding control that does not apply. Llama support in the source should not be read as an executed Llama ablation result in this JSON.

This Git record establishes a concrete contribution; it is not a complete division of all project work. The training pipeline, optimizer interventions and broader course study retain their joint attribution to Tristan Martin and Ying Jin.

## Reading and rerunning the diagnostics

The input-ablation script can select a small subset of probes:

```bash
python src/xai_repro/analysis/input_ablation.py --models openai-community/gpt2 --experiments baseline swap_at_0 length --out ablation_results/gpt2.json
```

This requires compatible PyTorch/Transformers dependencies and access to the pretrained weights. It downloads models as needed and performs inference; it was not rerun for this documentation update. The Figure 1 script uses its `MODELS` list and runs a much larger sweep; inspect that list and available memory before executing it.

Two attention metrics must be kept separate: **argmax percentage** counts the share of layer/head/query triples whose winning key is the probed position, while **attention-mass percentage** averages the weight assigned to that key. Taking an argmax after averaging attention matrices is a different statistic. The zero-position-embedding experiment is a diagnostic perturbation of a model trained with those embeddings; a disappearing sink under that perturbation does not, by itself, establish a causal mechanism.

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

The contribution history, diagnostic source paths, JSON structure and documentation links were checked on 26 September 2026. Training, pretrained-model inference and the test suite were **not rerun** during this documentation update. Historical outputs are not a new reproduction, and the specific diagnostic contribution above does not imply sole ownership of the joint codebase.

## Attribution and reuse

Keep the upstream history, source references and both project authors. The current upstream and fork include the [MIT license](../LICENSE), which this documentation update preserves unchanged.
