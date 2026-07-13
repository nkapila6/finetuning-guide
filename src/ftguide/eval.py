"""
Evaluator: basic generation evaluation for finetuned models.

Provides exact match, ROUGE-1/ROUGE-L, and BLEU-1 metrics using simple
n-gram overlap implementations -- no external packages required.

For serious evaluation, use a proper framework like evaluate or lm-eval-harness.
This module exists to give the reader a transparent, dependency-free look at
what these metrics actually compute.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from ftguide.config import FinetuneConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a finetuned model on a dataset.

    Takes a model and tokenizer (already loaded), runs generation on samples
    from the dataset, and computes basic text-similarity metrics.
    """

    def __init__(self, model, tokenizer, config: FinetuneConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    # ------------------------------------------------------------------
    # Generation over a dataset
    # ------------------------------------------------------------------

    def generate_for_dataset(
        self,
        dataset,
        num_samples: int = 50,
        max_new_tokens: int = 128,
    ) -> tuple[list[str], list[str]]:
        """Generate responses for samples in the eval dataset.

        Expects dataset entries to have an "input" (or "text") field for the
        prompt and an "output" (or "label") field for the reference.

        Returns (predictions, references).
        """
        predictions: list[str] = []
        references: list[str] = []

        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            # Flexible field names -- handle common dataset formats
            prompt = example.get("input") or example.get("text") or example.get("instruction") or ""
            reference = (
                example.get("output") or example.get("label") or example.get("response") or ""
            )

            if not prompt:
                logger.warning("Skipping sample %d: no prompt field found", i)
                continue

            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for eval reproducibility
                pad_token_id=self.tokenizer.eos_token_id,
            )
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0][input_len:]
            generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            predictions.append(generated)
            references.append(reference)

        return predictions, references

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, predictions: list[str], references: list[str]) -> dict[str, Any]:
        """Compute exact match, avg length, ROUGE-1, ROUGE-L, BLEU-1."""
        if not predictions or not references:
            return {"error": "empty predictions or references"}

        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        exact_match_rate = exact_matches / len(predictions)

        avg_pred_length = sum(len(p.split()) for p in predictions) / len(predictions)
        avg_ref_length = sum(len(r.split()) for r in references) / len(references)

        rouge1_scores = [self._rouge_1(p, r) for p, r in zip(predictions, references)]
        rougeL_scores = [self._rouge_l(p, r) for p, r in zip(predictions, references)]
        bleu1_scores = [self._bleu_1(p, r) for p, r in zip(predictions, references)]

        return {
            "num_samples": len(predictions),
            "exact_match_rate": round(exact_match_rate, 4),
            "avg_prediction_length": round(avg_pred_length, 2),
            "avg_reference_length": round(avg_ref_length, 2),
            "rouge1_f1": round(self._avg(rouge1_scores), 4),
            "rougeL_f1": round(self._avg(rougeL_scores), 4),
            "bleu1": round(self._avg(bleu1_scores), 4),
        }

    # ------------------------------------------------------------------
    # ROUGE-1: unigram overlap (F1)
    # ------------------------------------------------------------------

    @staticmethod
    def _rouge_1(pred: str, ref: str) -> float:
        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        if not p_tokens or not r_tokens:
            return 0.0
        overlap = sum((Counter(p_tokens) & Counter(r_tokens)).values())
        precision = overlap / len(p_tokens)
        recall = overlap / len(r_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # ROUGE-L: longest common subsequence based (F1)
    # ------------------------------------------------------------------

    @staticmethod
    def _rouge_l(pred: str, ref: str) -> float:
        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        if not p_tokens or not r_tokens:
            return 0.0
        lcs_len = Evaluator._lcs_length(p_tokens, r_tokens)
        precision = lcs_len / len(p_tokens)
        recall = lcs_len / len(r_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(a: list[str], b: list[str]) -> int:
        """Standard DP for longest common subsequence length."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # ------------------------------------------------------------------
    # BLEU-1: unigram precision with brevity penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _bleu_1(pred: str, ref: str) -> float:
        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        if not p_tokens or not r_tokens:
            return 0.0

        # Count clipped unigrams
        ref_counter = Counter(r_tokens)
        pred_counter = Counter(p_tokens)
        clipped = sum(min(count, ref_counter.get(t, 0)) for t, count in pred_counter.items())

        precision = clipped / len(p_tokens)

        # Brevity penalty: BP = exp(1 - ref_len / pred_len) if pred_len < ref_len else 1
        if len(p_tokens) < len(r_tokens):
            bp = 2.71828 ** (1 - len(r_tokens) / len(p_tokens))
        else:
            bp = 1.0

        return precision * bp

    # ------------------------------------------------------------------
    # Top-level evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset,
        num_samples: int = 50,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        """Run generation + metrics on a dataset.

        Returns a dict with predictions, references, and computed metrics.
        """
        logger.info(
            "Evaluating on %d samples (max_new_tokens=%d)",
            num_samples,
            max_new_tokens,
        )
        predictions, references = self.generate_for_dataset(
            dataset, num_samples=num_samples, max_new_tokens=max_new_tokens
        )
        metrics = self.compute_metrics(predictions, references)
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
        }

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    @staticmethod
    def print_results(results: dict[str, Any]) -> None:
        """Print a readable summary of evaluation results."""
        metrics = results.get("metrics", {})
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        print("=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"  Samples evaluated:  {metrics.get('num_samples', '?')}")
        print(f"  Exact match rate:   {metrics.get('exact_match_rate', '?'):.2%}")
        print(f"  Avg pred length:    {metrics.get('avg_prediction_length', '?')} tokens")
        print(f"  Avg ref length:     {metrics.get('avg_reference_length', '?')} tokens")
        print(f"  ROUGE-1 (F1):       {metrics.get('rouge1_f1', '?')}")
        print(f"  ROUGE-L (F1):       {metrics.get('rougeL_f1', '?')}")
        print(f"  BLEU-1:             {metrics.get('bleu1', '?')}")
        print("-" * 50)

        # Show a few examples
        predictions = results.get("predictions", [])
        references = results.get("references", [])
        for i in range(min(3, len(predictions))):
            print(f"\n  Example {i + 1}:")
            print(f"    Pred: {predictions[i][:120]}")
            print(f"    Ref:  {references[i][:120]}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
