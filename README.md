# Finetuning Guide for Small Models (1B-3B)

This is a hands-on guide for finetuning small models, not a library. The focus is on data curation and using Unsloth for LoRA finetuning. The guide is educational and transparent about what happens under the hood, with the goal of helping you write your own kernels eventually.

## Setup

Use `uv` for dependency management:

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
# For GPU training:
uv pip install -e ".[dev,unsloth]"
# For better dedup:
uv pip install datasketch
```

## How to Run

Open the main guide notebook:

```bash
jupyter notebook notebooks/finetuning_guide.ipynb
```

## Components

- `config.py`: `FinetuneConfig` dataclass - all knobs in one place.
- `data.py`: `DataCurator` - load, filter, dedup, format, report.
- `trainer.py`: `Finetuner` - Unsloth + TRL SFTTrainer wrapper (with transformers+peft fallback).
- `inference.py`: `InferenceRunner` - generate, batch_generate, chat.
- `eval.py`: `Evaluator` - ROUGE-1/L, BLEU-1, exact match (dependency-free).
- `cli.py`: placeholder CLI (points to the notebook).

## Data Curation

Data curation matters for small models. The guide walks through filtering, deduplication, and formatting to ensure high-quality training data.

## Custom Kernels

The guide explains what Unsloth does under the hood and points toward writing your own Triton kernels.

## Sample Dataset

The `data/sample_dataset.jsonl` is intentionally bad. It contains low-quality, duplicate, and empty examples so you can see the curation pipeline in action.
