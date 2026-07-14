"""
Evaluator (model evaluation) and DataEvaluator (data quality evaluation).

Evaluator provides exact match, ROUGE-1/ROUGE-L, BLEU-1, and diversity metrics
for evaluating finetuned model generations -- no external packages required.

DataEvaluator evaluates dataset quality before training: length distribution,
n-gram repetition, topic diversity, quality scoring, perplexity, and dedup audit.
No model needed for most checks.

For serious model evaluation, use lm-eval-harness (see Evaluator.lm_eval_guide).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import Counter
from typing import Any

from ftguide.config import FinetuneConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for topic diversity
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "not",
    "this",
    "that",
    "it",
    "with",
    "as",
    "by",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "your",
    "you",
    "i",
    "he",
    "she",
    "they",
    "we",
    "me",
    "him",
    "her",
    "them",
    "us",
    "his",
    "hers",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "theirs",
    "ours",
}


class DataEvaluator:
    """Evaluate dataset quality before training.

    No model needed. Runs statistical and heuristic checks on the raw dataset
    to surface problems before you waste GPU hours.
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Length distribution
    # ------------------------------------------------------------------

    def length_distribution(self, dataset) -> dict[str, Any]:
        """Compute char and approx-token length stats.

        Detects bimodality (two clusters in the middle 50% of sorted lengths).
        Warns if max length exceeds config.max_seq_length.
        """
        lengths = []
        for example in dataset:
            text = self._get_text(example)
            lengths.append(len(text))

        if not lengths:
            logger.warning("length_distribution: empty dataset")
            return {}

        lengths.sort()
        n = len(lengths)
        total = sum(lengths)
        mean = total / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std = math.sqrt(variance)

        def _p(pct: int) -> int:
            return lengths[int(n * pct / 100)]

        # Bimodality detection: look for a large gap in the middle 50%
        mid_start = int(n * 0.25)
        mid_end = int(n * 0.75)
        mid_slice = lengths[mid_start:mid_end]
        bimodal = False
        if len(mid_slice) >= 4:
            gaps = [mid_slice[i + 1] - mid_slice[i] for i in range(len(mid_slice) - 1)]
            max_gap = max(gaps)
            mean_gap = sum(gaps) / len(gaps)
            # If the largest gap is > 3x the mean gap, likely bimodal
            if mean_gap > 0 and max_gap > 3 * mean_gap:
                bimodal = True

        max_len = lengths[-1]
        if max_len > self.config.max_seq_length * 4:  # chars, not tokens
            logger.warning(
                "Max length %d chars (~%d tokens) exceeds max_seq_length=%d. "
                "Padding waste expected.",
                max_len,
                max_len // 4,
                self.config.max_seq_length,
            )
        if bimodal:
            logger.warning(
                "Length distribution appears bimodal. "
                "You may be mixing task types with different lengths."
            )

        return {
            "min": lengths[0],
            "max": max_len,
            "mean": round(mean, 1),
            "median": _p(50),
            "std": round(std, 1),
            "p25": _p(25),
            "p50": _p(50),
            "p75": _p(75),
            "p90": _p(90),
            "bimodal": bimodal,
            "approx_token_min": lengths[0] // 4,
            "approx_token_max": max_len // 4,
            "approx_token_mean": round(mean / 4, 1),
        }

    # ------------------------------------------------------------------
    # 2. N-gram repetition
    # ------------------------------------------------------------------

    def ngram_repetition(self, dataset, n: int = 5) -> dict[str, Any]:
        """Check n-gram repetition across the whole dataset.

        High repetition (>20% of examples sharing an n-gram) means the model
        will memorize that phrase regardless of dedup.
        """
        ngram_example_count: dict[str, int] = {}
        total_examples = 0

        for example in dataset:
            text = self._get_text(example)
            words = text.lower().split()
            if len(words) < n:
                continue
            total_examples += 1
            seen_in_example: set[str] = set()
            for i in range(len(words) - n + 1):
                ng = " ".join(words[i : i + n])
                if ng not in seen_in_example:
                    seen_in_example.add(ng)
                    ngram_example_count[ng] = ngram_example_count.get(ng, 0) + 1

        if not ngram_example_count:
            return {
                "total_unique_ngrams": 0,
                "total_ngram_instances": 0,
                "top_10": [],
                "top_ngram_pct": 0.0,
            }

        total_instances = sum(ngram_example_count.values())
        sorted_ngrams = sorted(ngram_example_count.items(), key=lambda x: -x[1])
        top_10 = [(ng, cnt) for ng, cnt in sorted_ngrams[:10]]
        top_pct = (sorted_ngrams[0][1] / total_examples * 100) if total_examples > 0 else 0.0

        if top_pct > 20:
            logger.warning(
                "Top %d-gram appears in %.1f%% of examples. "
                "Model will likely memorize this phrase.",
                n,
                top_pct,
            )

        return {
            "total_unique_ngrams": len(ngram_example_count),
            "total_ngram_instances": total_instances,
            "top_10": top_10,
            "top_ngram_pct": round(top_pct, 1),
        }

    # ------------------------------------------------------------------
    # 3. Topic diversity
    # ------------------------------------------------------------------

    def topic_diversity(self, dataset) -> dict[str, Any]:
        """Simple topic diversity check via word overlap clustering.

        Extracts top words per example (excluding stopwords), clusters by
        shared top words. Warns if one cluster >50% of data.
        """
        example_topics: list[list[str]] = []
        for example in dataset:
            text = self._get_text(example)
            words = [w for w in text.lower().split() if w not in _STOPWORDS and w.isalpha()]
            if not words:
                example_topics.append([])
                continue
            # Top 5 most frequent words as the topic signature
            top = [w for w, _ in Counter(words).most_common(5)]
            example_topics.append(top)

        if not example_topics:
            return {"num_clusters": 0, "cluster_sizes": [], "imbalanced": False}

        # Simple greedy clustering: each example joins the first cluster it
        # shares at least one top word with.
        clusters: list[list[int]] = []
        for i, topics in enumerate(example_topics):
            if not topics:
                continue
            assigned = False
            for cluster in clusters:
                # Check if this example shares a top word with any member
                for idx in cluster:
                    if set(topics) & set(example_topics[idx]):
                        cluster.append(i)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                clusters.append([i])

        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        largest_pct = (cluster_sizes[0] / len(example_topics) * 100) if example_topics else 0
        imbalanced = largest_pct > 50

        if imbalanced:
            logger.warning(
                "Topic imbalance: largest cluster is %.1f%% of data. "
                "That topic will dominate what the model learns.",
                largest_pct,
            )

        return {
            "num_clusters": len(clusters),
            "cluster_sizes": cluster_sizes,
            "largest_cluster_pct": round(largest_pct, 1),
            "imbalanced": imbalanced,
        }

    # ------------------------------------------------------------------
    # 4. Quality distribution
    # ------------------------------------------------------------------

    def quality_distribution(self, dataset) -> dict[str, Any]:
        """Run DataCurator.quality_score on each example and report distribution.

        Saves a histogram to file (matplotlib Agg backend, no display).
        """
        from ftguide.data import DataCurator

        scores: list[float] = []
        for example in dataset:
            text = self._get_text(example)
            scores.append(DataCurator.quality_score(text))

        if not scores:
            return {}

        scores.sort()
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)
        median = scores[int(n * 0.5)]
        threshold = 0.3
        below_threshold = sum(1 for s in scores if s < threshold)

        # Plot histogram to file
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(scores, bins=30, color="steelblue", edgecolor="white")
            ax.axvline(mean, color="red", linestyle="--", label=f"Mean={mean:.2f}")
            ax.axvline(median, color="green", linestyle="--", label=f"Median={median:.2f}")
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Count")
            ax.set_title("Dataset Quality Score Distribution")
            ax.legend()
            fig.tight_layout()
            fig.savefig("quality_distribution.png", dpi=100)
            plt.close(fig)
            logger.info("Saved quality histogram to quality_distribution.png")
        except Exception:
            logger.warning("Could not plot quality histogram (matplotlib missing?)")

        return {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std": round(std, 4),
            "min": round(scores[0], 4),
            "max": round(scores[-1], 4),
            "count_below_0.3": below_threshold,
        }

    # ------------------------------------------------------------------
    # 5. Perplexity
    # ------------------------------------------------------------------

    def perplexity(self, dataset, model, tokenizer, num_samples: int = 100) -> dict[str, Any]:
        """Compute average perplexity of the dataset under a model.

        Perplexity = exp(average negative log-likelihood).
        Very high = OOD data. Very low = model already knows this.
        Moderate = good training signal.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.error("torch is required for perplexity computation")
            return {"error": "torch not installed"}

        model.eval()
        losses: list[float] = []
        count = 0

        for example in dataset:
            if count >= num_samples:
                break
            text = self._get_text(example)
            if not text:
                continue

            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(model.device)
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                # Cross-entropy loss (already averaged over tokens)
                loss = outputs.loss.item()
                losses.append(loss)
                count += 1
            except Exception as e:
                logger.warning("Perplexity sample %d failed: %s", count, e)
                continue

        if not losses:
            return {"error": "no valid samples for perplexity"}

        losses.sort()
        n = len(losses)
        mean_loss = sum(losses) / n
        variance = sum((l - mean_loss) ** 2 for l in losses) / n
        std_loss = math.sqrt(variance)
        median_loss = losses[int(n * 0.5)]
        mean_ppl = math.exp(mean_loss)
        median_ppl = math.exp(median_loss)
        min_ppl = math.exp(losses[0])
        max_ppl = math.exp(losses[-1])

        # Interpretation
        if mean_ppl > 100:
            logger.warning(
                "Mean perplexity %.1f is very high. Data may be OOD for this model.",
                mean_ppl,
            )
        elif mean_ppl < 5:
            logger.warning(
                "Mean perplexity %.1f is very low. Model already knows this data. "
                "Training may not help.",
                mean_ppl,
            )
        else:
            logger.info(
                "Mean perplexity %.1f is moderate. Good training signal expected.",
                mean_ppl,
            )

        return {
            "mean_perplexity": round(mean_ppl, 2),
            "median_perplexity": round(median_ppl, 2),
            "min_perplexity": round(min_ppl, 2),
            "max_perplexity": round(max_ppl, 2),
            "std_perplexity": round(std_loss, 4),
            "num_samples": n,
        }

    # ------------------------------------------------------------------
    # 6. Dedup audit
    # ------------------------------------------------------------------

    def dedup_audit(self, dataset) -> dict[str, Any]:
        """Audit how much dedup would remove.

        Exact dedup via MD5 hash of normalized text.
        Near-dup estimate via sampled Jaccard (100 pairs, extrapolated).
        """
        from ftguide.data import DataCurator

        # --- Exact dedup ---
        seen_hashes: set[str] = set()
        exact_dups = 0
        total = 0

        for example in dataset:
            text = self._get_text(example)
            normalized = " ".join(text.lower().strip().split())
            h = hashlib.md5(normalized.encode("utf-8")).hexdigest()
            total += 1
            if h in seen_hashes:
                exact_dups += 1
            else:
                seen_hashes.add(h)

        # --- Near-dup estimate (sampled) ---
        texts = [self._get_text(ex) for ex in dataset]
        near_dup_estimate = 0
        if len(texts) >= 20:
            # Sample 100 pairs, compute Jaccard, estimate fraction above threshold
            sampled_pairs = min(100, len(texts) * (len(texts) - 1) // 2)
            above_threshold = 0
            rng = random.Random(42)
            for _ in range(sampled_pairs):
                i = rng.randint(0, len(texts) - 1)
                j = rng.randint(0, len(texts) - 1)
                if i == j:
                    continue
                s1 = DataCurator._shingles(texts[i])
                s2 = DataCurator._shingles(texts[j])
                intersection = s1 & s2
                union = s1 | s2
                if union and len(intersection) / len(union) > self.config.dedup_threshold:
                    above_threshold += 1
            near_dup_fraction = above_threshold / sampled_pairs if sampled_pairs > 0 else 0
            # Extrapolate: fraction of pairs * total pairs = estimated near-dup pairs
            # But we want estimated near-dup examples, not pairs.
            # Rough estimate: near_dup_fraction * total examples
            near_dup_estimate = int(near_dup_fraction * total)

        total_would_remove = exact_dups + near_dup_estimate
        logger.info(
            "Dedup audit: %d exact dups, ~%d near dups, ~%d total would remove (of %d)",
            exact_dups,
            near_dup_estimate,
            total_would_remove,
            total,
        )

        return {
            "exact_dups": exact_dups,
            "estimated_near_dups": near_dup_estimate,
            "total_would_remove": total_would_remove,
            "total_examples": total,
        }

    # ------------------------------------------------------------------
    # 7. Full report (no model needed)
    # ------------------------------------------------------------------

    def full_report(self, dataset) -> dict[str, Any]:
        """Run all data quality checks (except perplexity) and print summary.

        This is the main entry point for data quality assessment before training.
        """
        results: dict[str, Any] = {}

        results["length_distribution"] = self.length_distribution(dataset)
        results["ngram_repetition"] = self.ngram_repetition(dataset)
        results["topic_diversity"] = self.topic_diversity(dataset)
        results["quality_distribution"] = self.quality_distribution(dataset)
        results["dedup_audit"] = self.dedup_audit(dataset)

        self.stats = results
        self._print_full_report(results)
        return results

    def _print_full_report(self, results: dict[str, Any]) -> None:
        """Print a readable summary table of data quality."""
        print("\n" + "=" * 60)
        print("  Data Quality Report")
        print("=" * 60)

        ld = results.get("length_distribution", {})
        if ld:
            print(f"  Length (chars):")
            print(f"    min={ld.get('min')}  max={ld.get('max')}  mean={ld.get('mean')}")
            print(
                f"    p25={ld.get('p25')}  p50={ld.get('p50')}  p75={ld.get('p75')}  p90={ld.get('p90')}"
            )
            print(f"    bimodal={ld.get('bimodal')}")

        nr = results.get("ngram_repetition", {})
        if nr:
            print(f"  N-gram repetition (n=5):")
            print(f"    unique n-grams: {nr.get('total_unique_ngrams')}")
            print(f"    top n-gram in {nr.get('top_ngram_pct')}% of examples")
            if nr.get("top_10"):
                print(f"    top 5: {[ng for ng, _ in nr['top_10'][:5]]}")

        td = results.get("topic_diversity", {})
        if td:
            print(f"  Topic diversity:")
            print(f"    clusters: {td.get('num_clusters')}")
            print(f"    sizes: {td.get('cluster_sizes')[:5]}")
            print(
                f"    largest: {td.get('largest_cluster_pct')}%  imbalanced={td.get('imbalanced')}"
            )

        qd = results.get("quality_distribution", {})
        if qd:
            print(f"  Quality scores:")
            print(f"    mean={qd.get('mean')}  median={qd.get('median')}  std={qd.get('std')}")
            print(f"    min={qd.get('min')}  max={qd.get('max')}")
            print(f"    below 0.3: {qd.get('count_below_0.3')}")

        da = results.get("dedup_audit", {})
        if da:
            print(f"  Dedup audit:")
            print(f"    exact dups: {da.get('exact_dups')}")
            print(f"    near dups (est): {da.get('estimated_near_dups')}")
            print(
                f"    total would remove: {da.get('total_would_remove')} / {da.get('total_examples')}"
            )

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 8. Perplexity report (needs model)
    # ------------------------------------------------------------------

    def perplexity_report(
        self, dataset, model, tokenizer, num_samples: int = 100
    ) -> dict[str, Any]:
        """Run perplexity and print a readable summary.

        Separate from full_report because it needs a loaded model.
        """
        result = self.perplexity(dataset, model, tokenizer, num_samples=num_samples)
        if "error" in result:
            print(f"  Perplexity: {result['error']}")
            return result

        print("\n" + "-" * 50)
        print("  Perplexity Report")
        print("-" * 50)
        print(f"    samples:       {result.get('num_samples')}")
        print(f"    mean perplexity: {result.get('mean_perplexity')}")
        print(f"    median perplexity: {result.get('median_perplexity')}")
        print(f"    min:           {result.get('min_perplexity')}")
        print(f"    max:           {result.get('max_perplexity')}")
        print(f"    std:           {result.get('std_perplexity')}")
        print("-" * 50 + "\n")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_text(self, example: dict) -> str:
        """Extract text from a dataset example, handling common formats."""
        text = example.get("text") or example.get("output") or example.get("response") or ""
        if text:
            return text
        # Alpaca format
        parts = [
            example.get("instruction", ""),
            example.get("input", ""),
            example.get("output", ""),
        ]
        return "\n".join(p for p in parts if p)


class Evaluator:
    """Evaluate a finetuned model on a dataset.

    Takes a model and tokenizer (already loaded), runs generation on samples
    from the dataset, and computes basic text-similarity metrics.

    Also provides prediction diversity metrics and lm-eval-harness guidance.
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
    # Diversity metrics
    # ------------------------------------------------------------------

    def compute_diversity(self, predictions: list[str]) -> dict[str, Any]:
        """Measure prediction diversity.

        - Type-token ratio (unique words / total words) across all predictions
        - Average per-prediction TTR
        - Self-BLEU-2 (how similar predictions are to each other)
        """
        if not predictions:
            return {"error": "empty predictions"}

        all_words: list[str] = []
        per_pred_ttrs: list[float] = []

        for pred in predictions:
            words = pred.lower().split()
            if not words:
                continue
            all_words.extend(words)
            per_pred_ttrs.append(len(set(words)) / len(words))

        total_words = len(all_words)
        unique_words = len(set(all_words))
        global_ttr = unique_words / total_words if total_words > 0 else 0.0
        avg_ttr = sum(per_pred_ttrs) / len(per_pred_ttrs) if per_pred_ttrs else 0.0

        # Self-BLEU-2: treat each prediction as candidate, all others as references
        # Use bigram precision with brevity penalty
        self_bleu_scores: list[float] = []
        for i, pred in enumerate(predictions):
            p_tokens = pred.lower().split()
            if len(p_tokens) < 2:
                continue
            # Collect bigrams from all other predictions as references
            ref_bigrams: Counter[tuple[str, str]] = Counter()
            for j, ref in enumerate(predictions):
                if i == j:
                    continue
                r_tokens = ref.lower().split()
                for k in range(len(r_tokens) - 1):
                    ref_bigrams[(r_tokens[k], r_tokens[k + 1])] += 1

            if not ref_bigrams:
                continue

            pred_bigrams: Counter[tuple[str, str]] = Counter()
            for k in range(len(p_tokens) - 1):
                pred_bigrams[(p_tokens[k], p_tokens[k + 1])] += 1

            clipped = sum(min(count, ref_bigrams.get(bg, 0)) for bg, count in pred_bigrams.items())
            precision = clipped / sum(pred_bigrams.values()) if pred_bigrams else 0.0

            # Brevity penalty
            avg_ref_len = sum(
                len(predictions[j].split()) for j in range(len(predictions)) if j != i
            ) / max(len(predictions) - 1, 1)
            if len(p_tokens) < avg_ref_len and avg_ref_len > 0:
                bp = math.exp(1 - avg_ref_len / len(p_tokens))
            else:
                bp = 1.0

            self_bleu_scores.append(precision * bp)

        avg_self_bleu = sum(self_bleu_scores) / len(self_bleu_scores) if self_bleu_scores else 0.0

        return {
            "global_type_token_ratio": round(global_ttr, 4),
            "avg_per_prediction_ttr": round(avg_ttr, 4),
            "self_bleu_2": round(avg_self_bleu, 4),
        }

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------

    def compute_all(self, predictions: list[str], references: list[str]) -> dict[str, Any]:
        """Run compute_metrics + compute_diversity, return combined dict."""
        metrics = self.compute_metrics(predictions, references)
        diversity = self.compute_diversity(predictions)
        return {**metrics, **diversity}

    # ------------------------------------------------------------------
    # lm-eval-harness guidance
    # ------------------------------------------------------------------

    @staticmethod
    def lm_eval_guide() -> None:
        """Print guidance on using lm-eval-harness for serious evaluation."""
        print("=" * 60)
        print("  lm-eval-harness Guide")
        print("=" * 60)
        print()
        print("  Install:")
        print('    pip install "lm_eval[hf]"')
        print()
        print("  Basic usage:")
        print(
            "    lm_eval --model hf"
            " --model_args pretrained=your-model"
            " --tasks hellaswag,mmlu"
            " --device cuda:0 --batch_size 8"
        )
        print()
        print("  LoRA adapters (via peft):")
        print(
            "    lm_eval --model hf"
            " --model_args pretrained=base-model,peft=adapter-path"
            " --tasks hellaswag --device cuda:0"
        )
        print()
        print("  Key tasks for small models (1B-3B):")
        print("    - hellaswag  (commonsense reasoning)")
        print("    - winogrande (pronoun resolution)")
        print("    - arc_challenge (science reasoning)")
        print("    - piqa      (physical commonsense)")
        print("    - mmlu (subset) (multitask knowledge)")
        print()
        print("  Notes:")
        print("    - lm-eval-harness is the backend for the Open LLM Leaderboard")
        print(
            "    - For instruction-tuned models, generation tasks"
            " (not multiple-choice) are more meaningful"
        )
        print("    - Multiple-choice tasks can mask generation quality issues")
        print("=" * 60)

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
