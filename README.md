# Finetuning Guide for Small Models (1B-3B)

A hands-on guide for finetuning small models with a focus on data curation and evaluation. Educational, not a library. Transparent about what happens under the hood, with the goal of helping you write your own kernels eventually.

## Setup

Use `uv` for dependency management:

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
# For GPU training:
uv pip install -e ".[dev,unsloth]"
# For eval extras (data evaluation, plotting):
uv pip install -e ".[dev,eval]"
# For serious model evaluation:
uv pip install lm-eval
# For better dedup (MinHash LSH):
uv pip install datasketch
```

## How to Run

Open the main guide notebook:

```bash
jupyter notebook notebooks/finetuning_guide.ipynb
```

## The Pipeline

The full pipeline: **Config -> Curate -> Evaluate Data -> Load Model -> LoRA -> Train -> Evaluate Model -> Inference**.

- `config.py`: all knobs in one place.
- `data.py`: load, filter, dedup, format.
- `eval.py` (DataEvaluator): check dataset quality before training.
- `trainer.py`: Unsloth + TRL SFTTrainer wrapper.
- `eval.py` (Evaluator): check model generations after training.
- `inference.py`: generate, batch, chat.

## Data Curation

Data curation matters more for small models than large ones. A 1B model can't brute-force through bad data the way a 70B model can. Every bad example is a larger fraction of the total training signal.

### Quality Filtering

Heuristic scoring (word diversity, repetition, length, ASCII ratio) gives each example a 0-1 quality score. Not a replacement for manual inspection but a fast first pass. The `DataCurator.quality_score` method weights diversity and alphanumeric ratio most heavily, with penalties for repetition and extreme lengths.

### Deduplication

Two stages:

- **Exact dedup**: MD5 hash of normalized (lowercased, whitespace-collapsed) text. Catches identical copies.
- **Near-dup**: shingled Jaccard similarity (or MinHash LSH via `datasketch` for large datasets). Why this matters: paraphrased examples and template-generated data look unique to a hash but teach the same thing. Default threshold is 0.9 Jaccard similarity.

### Formatting

The alpaca template is the default format. Consistent formatting matters because SFTTrainer consumes the `text` field -- if your formatting is inconsistent, the model learns inconsistent patterns. The formatter adds instruction/input/response markers and a system prompt.

### Manual Inspection

The most important step. Read 50-100 examples from your curated dataset. If you can't explain what the model should learn from 20 random examples, the data is too noisy. No automated metric replaces this.

## Data Evaluation

Before training, evaluate your dataset with `DataEvaluator`. This runs on the raw dataset without loading a model.

### Length Distribution

Checks min, max, mean, median, std, and percentiles (p25, p50, p75, p90) of character lengths. Detects bimodality -- if the distribution has two clusters, you're probably mixing task types with different lengths. Warns if max length exceeds `max_seq_length` (padding waste).

### N-gram Repetition

Counts how many examples each 5-gram appears in. If a 5-gram appears in >20% of examples, the model will memorize that phrase regardless of dedup. Reports the top 10 most repeated n-grams and the percentage of examples containing the top one.

### Topic Diversity

Extracts top words per example (excluding stopwords), clusters by shared top words. If one cluster is >50% of the data, that topic will dominate what the model learns. Reports cluster count, sizes, and whether the distribution is imbalanced.

### Quality Distribution

Runs the heuristic quality scorer across every example. Reports mean, median, std, min, max, and count below threshold. Saves a histogram to `quality_distribution.png` (matplotlib Agg backend, no display needed).

### Perplexity

Runs the dataset through the base model and computes perplexity. Requires a loaded model and tokenizer. Interpretation:

- **Very high (>100)**: data is OOD for this model. Likely garbage.
- **Very low (<5)**: model already knows this data. Training won't help much.
- **Moderate (5-100)**: sweet spot. The model is "almost right but not quite" -- good training signal.

### Dedup Audit

Estimates how much dedup would remove before running the full pipeline. Exact dedup via MD5. Near-dup estimate via sampled Jaccard (100 random pairs, extrapolated). Reports exact dups, estimated near dups, and total would-remove count.

### Running It

```python
from ftguide.eval import DataEvaluator
from ftguide.config import FinetuneConfig

config = FinetuneConfig()
evaluator = DataEvaluator(config)
report = evaluator.full_report(dataset)
```

For perplexity (needs a model):

```python
report = evaluator.perplexity_report(dataset, model, tokenizer, num_samples=100)
```

## Model Evaluation

After training, evaluate the model with `Evaluator`.

### Generation Metrics

- **Exact match**: strict string equality. Useful for factual tasks, useless for open-ended generation.
- **ROUGE-1**: unigram overlap F1. Measures word-level similarity.
- **ROUGE-L**: longest common subsequence F1. Measures fluency and ordering.
- **BLEU-1**: unigram precision with brevity penalty. Measures precision of word choice.

All implemented dependency-free in `Evaluator`. Good for a first look. Not sufficient for real evaluation.

### Prediction Diversity

- **Type-token ratio**: unique words / total words across all predictions. Low TTR = repetitive vocabulary.
- **Per-prediction TTR**: average TTR per individual prediction.
- **Self-BLEU-2**: bigram precision treating each prediction as a candidate against all others. High self-BLEU = model is regurgitating the same patterns.

### lm-eval-harness

For serious evaluation, use lm-eval-harness. It provides 60+ academic benchmarks and is the backend for the Open LLM Leaderboard.

```bash
pip install "lm_eval[hf]"
lm_eval --model hf --model_args pretrained=your-model --tasks hellaswag,winogrande,arc_challenge,piqa --device cuda:0 --batch_size 8
```

Key tasks for small models (1B-3B): hellaswag (commonsense reasoning), winogrande (pronoun resolution), arc_challenge (science reasoning), piqa (physical commonsense), mmlu subset (multitask knowledge).

For instruction-tuned models, generation tasks are more meaningful than multiple-choice. Multiple-choice can mask generation quality issues.

### The Eval Split

Always hold out 10-20% of your data. Never train on it. Monitor eval loss alongside train loss. The moment eval loss rises, stop -- you're overfitting.

## Preventing Overfitting

Practical levers, ordered by impact:

- **LoRA rank**: drop r=16 to r=8 if overfitting. This is the biggest lever -- less capacity means less room to memorize.
- **LoRA dropout**: 0.0 is aggressive. Try 0.05-0.1 for regularization.
- **Learning rate**: 2e-4 is default. Too high = sharp minimum that memorizes noise. Try 1e-4 or 5e-5.
- **Epochs**: 1-3 is enough for LoRA. More epochs on small data = guaranteed overfit.
- **Catastrophic forgetting vs overfitting**: different problems, different fixes. Forgetting = lower lr, fewer steps, mix general data. Overfitting = less capacity, more regularization, early stopping.

## Components

- `config.py`: `FinetuneConfig` dataclass -- all knobs in one place.
- `data.py`: `DataCurator` -- load, filter, dedup, format, report.
- `trainer.py`: `Finetuner` -- Unsloth + TRL SFTTrainer wrapper (with transformers+peft fallback).
- `eval.py`: `Evaluator` (ROUGE, BLEU, exact match, diversity) + `DataEvaluator` (length dist, n-gram rep, topic diversity, quality dist, perplexity, dedup audit).
- `inference.py`: `InferenceRunner` -- generate, batch_generate, chat.
- `cli.py`: placeholder CLI (points to the notebook).

## Custom Kernels

The guide explains what Unsloth does under the hood and points toward writing your own Triton kernels. Understanding the mechanics of LoRA forward/backward passes is the first step toward custom optimization.

## Sample Dataset

The `data/sample_dataset.jsonl` is intentionally bad. It contains low-quality, duplicate, and empty examples so you can see the curation pipeline in action.
