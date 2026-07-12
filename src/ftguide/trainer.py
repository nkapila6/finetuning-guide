"""
Finetuner: wraps Unsloth + TRL SFTTrainer for LoRA finetuning.

Two code paths exist:
  1. Unsloth path (GPU/CUDA) -- uses FastLanguageModel for 4-bit loading,
     LoRA attachment, and training. This is the primary path.
  2. Fallback path (CPU/Mac)  -- uses transformers + peft directly. This lets
     you import, inspect, and even run the code on a machine without a GPU.
     Training will be slow or impossible on CPU, but the code is at least
     exercisable for development and reading.

Why the fallback: this guide is educational. You should be able to read and
tinker with the code on a MacBook without needing a CUDA GPU.
"""

from __future__ import annotations

import logging
from typing import Any

from ftguide.config import FinetuneConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy imports -- wrapped so the module loads on any machine
# ---------------------------------------------------------------------------

_HAS_UNSLOTH = False
_HAS_TRL = False
_HAS_PEFT = False

try:
    import unsloth  # noqa: F401

    _HAS_UNSLOTH = True
except ImportError:
    pass

try:
    import trl  # noqa: F401

    _HAS_TRL = True
except ImportError:
    pass

try:
    import peft  # noqa: F401

    _HAS_PEFT = True
except ImportError:
    pass


class Finetuner:
    """High-level wrapper around Unsloth + TRL SFTTrainer for LoRA finetuning."""

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load base model. Prefers Unsloth (4-bit), falls back to transformers."""
        if _HAS_UNSLOTH:
            from unsloth import FastLanguageModel

            logger.info("Using Unsloth FastLanguageModel for model loading")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "Unsloth not available -- falling back to transformers.AutoModel. "
                "4-bit loading is skipped; model loads in full precision."
            )
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA attachment
    # ------------------------------------------------------------------

    def attach_lora(self, model):
        """Attach LoRA adapters to the model.

        Unsloth path uses FastLanguageModel.get_peft_model (optimized for
        gradient checkpointing). Fallback uses standard peft.LoraConfig.
        """
        if _HAS_UNSLOTH:
            from unsloth import FastLanguageModel

            logger.info("Attaching LoRA via Unsloth FastLanguageModel.get_peft_model")
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            from peft import LoraConfig, get_peft_model

            logger.info("Unsloth not available -- attaching LoRA via peft.LoraConfig")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        return model

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------

    def setup_trainer(self, model, tokenizer, dataset):
        """Build SFTConfig + SFTTrainer from the config and dataset.

        If config.num_train_epochs is set, it overrides max_steps.
        If config.train_on_responses_only is True, applies Unsloth's
        train_on_responses_only helper (silently skipped in fallback).
        """
        if not _HAS_TRL:
            raise ImportError("TRL is required for training. Install it with: uv pip install trl")

        from trl import SFTConfig, SFTTrainer

        # Build training args from config
        training_args = {
            "output_dir": self.config.output_dir,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_steps": self.config.warmup_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "optim": self.config.optim,
            "seed": self.config.seed,
            "max_seq_length": self.config.max_seq_length,
            "dataset_text_field": self.config.dataset_text_field,
            "report_to": "none",  # keep it simple for the guide
            "logging_steps": 1,
        }

        # num_train_epochs overrides max_steps when set
        if self.config.num_train_epochs is not None:
            training_args["num_train_epochs"] = self.config.num_train_epochs
        else:
            training_args["max_steps"] = self.config.max_steps

        sft_config = SFTConfig(**training_args)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )

        # Optional: train only on response tokens (unsloth helper)
        if self.config.train_on_responses_only:
            if _HAS_UNSLOTH:
                from unsloth import train_on_responses_only

                logger.info("Applying train_on_responses_only")
                trainer = train_on_responses_only(trainer, instruction_part="", response_part="")
            else:
                logger.warning(
                    "train_on_responses_only=True but unsloth is not available. "
                    "Skipping -- training on full sequences."
                )

        return trainer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, trainer) -> dict[str, Any]:
        """Run training and return metrics.

        Returns the TrainOutput metrics dict (loss, runtime, etc.).
        """
        logger.info("Starting training...")
        result = trainer.train()
        metrics = {}
        if hasattr(result, "metrics"):
            metrics = result.metrics
        final_loss = metrics.get("train_loss", "unknown")
        logger.info("Training complete. Final loss: %s", final_loss)
        return metrics

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, model, tokenizer, path: str = "outputs/lora_model") -> None:
        """Save LoRA adapters and tokenizer."""
        logger.info("Saving LoRA adapters to %s", path)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    def save_merged(self, model, tokenizer, path: str = "outputs/merged_model") -> None:
        """Save model merged with LoRA weights (16-bit).

        Unsloth provides a convenient save_pretrained_merged. Falls back to
        saving adapters separately if unsloth is not available.
        """
        if _HAS_UNSLOTH:
            logger.info("Saving merged model (16-bit) to %s", path)
            model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")
        else:
            logger.warning("save_merged requires unsloth. Saving adapters separately instead.")
            self.save(model, tokenizer, path=path)

    # ------------------------------------------------------------------
    # Loading for inference
    # ------------------------------------------------------------------

    @classmethod
    def load_for_inference(cls, config: FinetuneConfig, path: str):
        """Load saved LoRA adapters for inference.

        Uses FastLanguageModel if available, otherwise transformers.
        """
        if _HAS_UNSLOTH:
            from unsloth import FastLanguageModel

            logger.info("Loading model for inference from %s", path)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Unsloth not available -- loading with transformers from %s", path)
            model = AutoModelForCausalLM.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model.eval()

        return model, tokenizer
