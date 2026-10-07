# Validation of the recovered toy experiment

Date: 7 October 2026. These are new CPU checks performed while preparing the
local files for publication. The files under `toy/` and `historical/` remain
the original saved outputs; no pretrained-model inference or language-model
training was performed for this update.

## Environment

Windows, Python 3.12.14, PyTorch 2.6.0+cpu, Transformers 4.49.0, NumPy 1.26.4,
Matplotlib 3.11.2, pytest 9.1.1. This was an isolated validation environment,
not a recreation of the complete historical training environment. The existing
project metadata pins PyTorch 2.1.2; it was not changed or tested as an exact
dependency lock by this update.

## Checks completed

1. `tests/test_attention_sink_toy.py` and `tests/test_softmax1.py`: **7 passed**.
   The recovered toy test checks constant logits against the closed-form
   first-key mass and row sum for both methods. The existing softmax-1 tests
   cover normalization, a direct reference, large positive logits, masked rows
   and finite gradients.
2. Re-executed both archived toy configurations on CPU, each with constant and
   Gaussian logits, seed 0 and batch size 128:

   | Sequence length | Samples | Compared numeric fields | Maximum absolute difference from archived JSON |
   |---:|---:|---:|---:|
   | 64 | 1,024 | 1,050 | 1.4901161193847656e-08 |
   | 256 | 2,048 | 4,122 | 1.862645149230957e-08 |

   The recursive comparison checked matching JSON structure and non-numeric
   settings, and used an absolute tolerance of `1e-6` for numeric fields.
   Agreement is within tolerance, not a claim of bitwise equality.
3. Executed the plotting command on the length-256 rerun and visually checked
   the generated two-column figure. Outputs were written to ignored
   `runs_smoke/`, separately from the historical archive.
4. Checked file-selection provenance, JSON readability, image readability,
   documentation links and the publication manifest.

The only recovered Python-code change is the import path correction in
`attention_sink_toy.py`, from the obsolete nested module to
`from xai_repro.attention import softmax1`. Git may normalize text line endings;
the manifest records source bytes and published blob bytes separately.

## Scope

This validates the synthetic experiment, its relationship to the stored toy
JSON and the plotting path. It does not validate the historical trained-model
plots, reproduce their training runs, establish a downstream accuracy or
quantization gain, or represent a pass of the entire repository test suite.
See [the archive notes](README.md) for the misleading captions retained in the
historical figures.
