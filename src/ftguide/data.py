"""
Data curation module for finetuning.

Provides DataCurator: load, filter, deduplicate, format, and save datasets.
Designed to work with Hugging Face datasets and local JSONL files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from ftguide.config import FinetuneConfig

logger = logging.getLogger(__name__)

# Wrap datasets import so module loads without it installed
_HAS_DATASETS = False
try:
    from datasets import Dataset, load_dataset

    _HAS_DATASETS = True
except ImportError:
    pass

# Optional: datasketch for efficient MinHash LSH
_HAS_DATASKETCH = False
try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    pass


class DataCurator:
    """Load, clean, deduplicate, and format a finetuning dataset.

    Usage:
        curator = DataCurator(config)
        dataset = curator.load()
        dataset = curator.curate(dataset)
        curator.save(dataset, "outputs/curated_data")
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.stats: dict[str, Any] = {}
        self.format_type: str | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Dataset:
        """Load dataset from a local JSONL/JSON file or Hugging Face hub.

        Auto-detects format: alpaca (instruction/input/output columns),
        conversational (messages/conversations column), or plain text.
        """
        if not _HAS_DATASETS:
            raise ImportError(
                "The 'datasets' library is required to load datasets. "
                "Install it with: uv pip install datasets"
            )

        path = self.config.dataset_path

        if path.endswith(".jsonl") or path.endswith(".json"):
            logger.info("Loading local file: %s", path)
            dataset = load_dataset("json", data_files={"train": path}, split="train")
        else:
            logger.info("Loading from Hugging Face hub: %s", path)
            dataset = load_dataset(path, split=self.config.dataset_split)

        # Cap dataset size if configured
        if self.config.max_examples is not None:
            n = min(self.config.max_examples, len(dataset))
            dataset = dataset.select(range(n))
            logger.info("Capped dataset to %d examples", n)

        # Auto-detect format
        cols = dataset.column_names
        if "instruction" in cols:
            self.format_type = "alpaca"
        elif "messages" in cols or "conversations" in cols:
            self.format_type = "conversational"
        else:
            self.format_type = "text"

        logger.info(
            "Detected format: %s | %d examples | columns: %s",
            self.format_type,
            len(dataset),
            cols,
        )

        return dataset

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    @staticmethod
    def quality_score(text: str) -> float:
        """Heuristic quality score for a text string, 0.0 (worst) to 1.0 (best).

        Checks: length, word diversity, alphanumeric ratio, ASCII ratio,
        length penalties, and repetition.
        """
        if not text or len(text) < 10:
            return 0.0

        total_chars = len(text)
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return 0.0

        # 1. Word diversity: unique / total. Low diversity = low quality.
        unique_words = len(set(words))
        diversity = unique_words / total_words

        # 2. Alphanumeric ratio: penalize excessive special characters.
        alnum_chars = sum(c.isalnum() for c in text)
        alnum_ratio = alnum_chars / total_chars

        # 3. ASCII ratio: for English datasets, non-ASCII is suspicious.
        ascii_chars = sum(ord(c) < 128 for c in text)
        ascii_ratio = ascii_chars / total_chars

        # 4. Length factor: very short or very long gets penalized.
        length_factor = 1.0
        if total_chars < 50:
            length_factor = total_chars / 50.0
        elif total_chars > 10000:
            length_factor = 0.5  # penalize extremely long texts

        # 5. Repetition: check if any 5-word sequence repeats more than twice.
        repetition_penalty = 1.0
        if total_words >= 10:
            seen: set[str] = set()
            repeat_count = 0
            for i in range(total_words - 4):
                shingle = " ".join(words[i : i + 5])
                if shingle in seen:
                    repeat_count += 1
                else:
                    seen.add(shingle)
            # If more than 10% of 5-grams are repeats, penalize
            total_shingles = total_words - 4
            if total_shingles > 0 and repeat_count / total_shingles > 0.1:
                repetition_penalty = 0.3

        # Weighted average -- diversity and alnum ratio matter most
        score = (
            0.35 * diversity
            + 0.25 * alnum_ratio
            + 0.15 * ascii_ratio
            + 0.15 * length_factor
            + 0.10 * repetition_penalty
        )

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_low_quality(self, dataset: Dataset) -> Dataset:
        """Drop low-quality and over-length examples."""
        original_count = len(dataset)

        def _full_text(example: dict) -> str:
            """Reconstruct the full text from whatever format we detected."""
            if self.format_type == "alpaca":
                parts = [example.get("instruction", "")]
                inp = example.get("input", "")
                if inp and inp.strip():
                    parts.append(inp)
                parts.append(example.get("output", ""))
                return "\n".join(parts)
            elif self.format_type == "conversational":
                msgs = example.get("messages") or example.get("conversations") or []
                return "\n\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in msgs
                )
            else:
                return example.get("text", "")

        quality_dropped = 0
        length_dropped = 0
        keep = []

        for example in dataset:
            text = _full_text(example)
            score = self.quality_score(text)

            if score < self.config.min_quality_score:
                quality_dropped += 1
                continue

            # Rough token estimate: ~4 chars per token
            approx_tokens = len(text) // 4
            if approx_tokens > self.config.max_length_filter:
                length_dropped += 1
                continue

            keep.append(example)

        filtered = dataset.select(range(len(keep))) if keep else dataset.select([])
        # datasets.select() with indices -- need to rebuild
        # Actually, select() takes indices. Let's build the index list.
        # We already have the examples, but we need indices for select().
        # Re-do: collect indices instead.
        # Actually simpler: just use the keep list approach with from_list.
        # But we want to preserve the Dataset type. Let's use from_list.
        from datasets import Dataset as Ds

        filtered = Ds.from_list(keep) if keep else dataset.select([])

        remaining = len(filtered)
        logger.info(
            "Filter: %d total | %d dropped (quality) | %d dropped (length) | %d remaining",
            original_count,
            quality_dropped,
            length_dropped,
            remaining,
        )

        return filtered

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _shingles(text: str, k: int = 5) -> set[str]:
        """Tokenize into k-word shingles for Jaccard similarity."""
        words = text.lower().split()
        if len(words) < k:
            return {text.lower()}  # fallback: whole text as one shingle
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def deduplicate(self, dataset: Dataset) -> Dataset:
        """Two-stage dedup: exact (MD5) then near-dup (Jaccard / MinHash LSH)."""
        original_count = len(dataset)

        # --- Stage 1: Exact dedup ---
        seen_hashes: set[str] = set()
        exact_dup_indices: list[int] = []

        for i, example in enumerate(dataset):
            # Normalize: lowercase, strip, collapse whitespace
            text = example.get("text", "")
            if not text:
                # For non-formatted datasets, reconstruct from fields
                if self.format_type == "alpaca":
                    parts = [example.get("instruction", "")]
                    inp = example.get("input", "")
                    if inp and inp.strip():
                        parts.append(inp)
                    parts.append(example.get("output", ""))
                    text = "\n".join(parts)
                else:
                    text = str(example)

            normalized = " ".join(text.lower().strip().split())
            h = hashlib.md5(normalized.encode("utf-8")).hexdigest()

            if h in seen_hashes:
                exact_dup_indices.append(i)
            else:
                seen_hashes.add(h)

        # Build deduped dataset (keep only non-duplicate indices)
        keep_indices = [i for i in range(original_count) if i not in exact_dup_indices]
        dataset = dataset.select(keep_indices)
        exact_removed = len(exact_dup_indices)
        logger.info("Exact dedup: removed %d duplicates", exact_removed)

        # --- Stage 2: Near-dedup ---
        near_dup_indices: list[int] = []
        threshold = self.config.dedup_threshold

        if _HAS_DATASKETCH:
            logger.info("Using MinHash LSH for near-dedup")
            lsh = MinHashLSH(threshold=threshold, num_perm=128)
            index_map: dict[str, int] = {}  # key -> original index

            for i, example in enumerate(dataset):
                text = self._get_dedup_text(example)
                if not text:
                    continue
                shingles = self._shingles(text)
                m = MinHash(num_perm=128)
                for s in shingles:
                    m.update(s.encode("utf-8"))
                key = f"ex_{i}"
                # Check if similar to any existing entry
                results = lsh.query(m)
                if results:
                    near_dup_indices.append(i)
                else:
                    lsh.insert(key, m)
                    index_map[key] = i
        else:
            logger.warning(
                "datasketch not installed -- using O(n^2) Jaccard comparison. "
                "Install with: uv pip install datasketch"
            )
            kept_shingle_sets: list[tuple[int, set[str]]] = []

            for i, example in enumerate(dataset):
                text = self._get_dedup_text(example)
                if not text:
                    continue
                shingles = self._shingles(text)
                is_dup = False
                for _, kept_shingles in kept_shingle_sets:
                    intersection = shingles & kept_shingles
                    union = shingles | kept_shingles
                    if union and len(intersection) / len(union) > threshold:
                        is_dup = True
                        break
                if is_dup:
                    near_dup_indices.append(i)
                else:
                    kept_shingle_sets.append((i, shingles))

        # Remove near-duplicates
        near_dup_set = set(near_dup_indices)
        final_indices = [i for i in range(len(dataset)) if i not in near_dup_set]
        dataset = dataset.select(final_indices)

        near_removed = len(near_dup_indices)
        logger.info(
            "Near-dedup: removed %d | remaining: %d",
            near_removed,
            len(dataset),
        )

        self.stats["exact_dups_removed"] = exact_removed
        self.stats["near_dups_removed"] = near_removed

        return dataset

    def _get_dedup_text(self, example: dict) -> str:
        """Extract or reconstruct text from an example for dedup comparison."""
        text = example.get("text", "")
        if text:
            return text
        if self.format_type == "alpaca":
            parts = [example.get("instruction", "")]
            inp = example.get("input", "")
            if inp and inp.strip():
                parts.append(inp)
            parts.append(example.get("output", ""))
            return "\n".join(parts)
        if self.format_type == "conversational":
            return "\n".join(m.get("content", "") for m in example.get("messages", []))
        return str(example)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_instructions(self, dataset: Dataset) -> Dataset:
        """Add a 'text' column formatted for SFTTrainer."""

        def _format(example: dict) -> dict:
            if self.format_type == "alpaca":
                instruction = example.get("instruction", "")
                inp = example.get("input", "")
                output = example.get("output", "")

                if inp and inp.strip():
                    text = (
                        "Below is an instruction that describes a task, paired with "
                        "an input that provides further context. Write a response "
                        "that appropriately completes the request.\n\n"
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{output}"
                    )
                else:
                    text = (
                        "Below is an instruction that describes a task, paired with "
                        "an input that provides further context. Write a response "
                        "that appropriately completes the request.\n\n"
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Response:\n{output}"
                    )
                return {"text": text}

            elif self.format_type == "conversational":
                msgs = example.get("messages") or example.get("conversations") or []
                text = "\n\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in msgs
                )
                return {"text": text}

            else:
                # Already has a text field, or use the whole example
                return {"text": example.get("text", str(example))}

        return dataset.map(_format)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, dataset: Dataset, label: str = "") -> dict[str, Any]:
        """Compute and print length/token stats for the dataset.

        Works with both formatted (has 'text' column) and unformatted datasets.
        For unformatted alpaca data, reconstructs the full text.
        """

        def _get_text(ex: dict) -> str:
            if "text" in ex and ex["text"]:
                return ex["text"]
            if self.format_type == "alpaca":
                parts = [ex.get("instruction", "")]
                inp = ex.get("input", "")
                if inp and inp.strip():
                    parts.append(inp)
                parts.append(ex.get("output", ""))
                return "\n".join(parts)
            return str(ex)

        lengths = [len(_get_text(example)) for example in dataset]
        if not lengths:
            logger.info("Report [%s]: empty dataset", label)
            return {}

        lengths.sort()
        n = len(lengths)
        total = sum(lengths)
        mean = total / n
        p25 = lengths[int(n * 0.25)]
        p50 = lengths[int(n * 0.50)]
        p75 = lengths[int(n * 0.75)]

        # Approx tokens (chars // 4)
        token_lengths = [l // 4 for l in lengths]
        t_mean = sum(token_lengths) / n
        t_p25 = token_lengths[int(n * 0.25)]
        t_p50 = token_lengths[int(n * 0.50)]
        t_p75 = token_lengths[int(n * 0.75)]

        header = f" Dataset Report [{label}] " if label else " Dataset Report "
        print(f"\n{'=' * 60}")
        print(f"{header:-^60}")
        print(f"{'=' * 60}")
        print(f"  Count:              {n}")
        print(f"  Char lengths:")
        print(f"    min:   {lengths[0]}")
        print(f"    max:   {lengths[-1]}")
        print(f"    mean:  {mean:.1f}")
        print(f"    25th:  {p25}")
        print(f"    50th:  {p50}")
        print(f"    75th:  {p75}")
        print(f"  Approx token lengths (chars // 4):")
        print(f"    min:   {token_lengths[0]}")
        print(f"    max:   {token_lengths[-1]}")
        print(f"    mean:  {t_mean:.1f}")
        print(f"    25th:  {t_p25}")
        print(f"    50th:  {t_p50}")
        print(f"    75th:  {t_p75}")
        print(f"{'=' * 60}\n")

        return {
            "count": n,
            "char_min": lengths[0],
            "char_max": lengths[-1],
            "char_mean": mean,
            "char_p25": p25,
            "char_p50": p50,
            "char_p75": p75,
            "token_min": token_lengths[0],
            "token_max": token_lengths[-1],
            "token_mean": t_mean,
            "token_p25": t_p25,
            "token_p50": t_p50,
            "token_p75": t_p75,
        }

    # ------------------------------------------------------------------
    # Curation pipeline
    # ------------------------------------------------------------------

    def curate(self, dataset: Dataset) -> Dataset:
        """Run the full curation pipeline: filter -> dedup -> format."""
        logger.info("Starting curation pipeline")

        dataset = self.filter_low_quality(dataset)
        self.stats["after_filter"] = self.report(dataset, label="after filter")

        dataset = self.deduplicate(dataset)
        self.stats["after_dedup"] = self.report(dataset, label="after dedup")

        dataset = self.format_instructions(dataset)
        self.stats["after_format"] = self.report(dataset, label="after format")

        logger.info("Curation pipeline complete")
        return dataset

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, dataset: Dataset, path: str) -> None:
        """Save dataset as JSONL and as a Hugging Face dataset on disk."""
        os.makedirs(path, exist_ok=True)

        # JSONL
        jsonl_path = os.path.join(path, "curated.jsonl")
        with open(jsonl_path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")
        logger.info("Saved JSONL: %s", jsonl_path)

        # HF dataset on disk
        dataset.save_to_disk(path)
        logger.info("Saved HF dataset: %s", path)
