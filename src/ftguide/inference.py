"""
InferenceRunner: generate text from a finetuned LoRA model.

Two code paths mirror the trainer:
  1. Unsloth path -- FastLanguageModel with 4-bit and for_inference().
  2. Fallback path -- plain transformers (no 4-bit, no unsloth optimizations).

The fallback exists so the code is readable and importable on a Mac without
a GPU. Generation will work on CPU, just slower.
"""

from __future__ import annotations

import logging
from typing import Any

from ftguide.config import FinetuneConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy imports -- module-level try/except so the file loads anywhere
# ---------------------------------------------------------------------------

_HAS_UNSLOTH = False
_HAS_TRL = False

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


class InferenceRunner:
    """Generate text from a finetuned LoRA model.

    Handles both chat-templated (instruct) and raw-prompt models.
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self, path: str = "outputs/lora_model"):
        """Load model + tokenizer from saved LoRA adapters.

        Prefers Unsloth FastLanguageModel (with for_inference() call).
        Falls back to transformers AutoModel.
        """
        if _HAS_UNSLOTH:
            from unsloth import FastLanguageModel

            logger.info("Loading model via Unsloth from %s", path)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Unsloth not available -- loading with transformers from %s", path)
            model = AutoModelForCausalLM.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model.eval()

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    # Shared generation helper
    # ------------------------------------------------------------------

    def _generate_from_text(
        self,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Tokenize text, generate, and return only the new tokens decoded."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)

        with torch_inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        If the tokenizer has a chat_template, the prompt is treated as a
        single user message and wrapped via apply_chat_template. Otherwise
        it is used as raw text.

        Returns the generated text with the prompt portion stripped.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load(path) before generating.")

        # Detect chat-template models
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        return self._generate_from_text(
            text, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs
        )

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def batch_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate responses for a list of prompts."""
        return [
            self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature)
            for p in prompts
        ]

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from a list of chat messages.

        messages: list of {"role": "user"|"assistant"|"system", "content": ...}
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load(path) before generating.")

        if self.tokenizer.chat_template is None:
            logger.warning(
                "Tokenizer has no chat_template. Falling back to raw prompt "
                "using the last user message."
            )
            # Extract the last user message as a raw prompt
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return self._generate_from_text(last_user, max_new_tokens=max_new_tokens)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._generate_from_text(
            text, max_new_tokens=max_new_tokens, temperature=temperature
        )


# ---------------------------------------------------------------------------
# Small helper: torch.inference_mode() without importing torch at module level
# ---------------------------------------------------------------------------


def torch_inference_mode():
    """Context manager for torch.inference_mode().

    Imported lazily so the module loads without torch installed.
    """
    import torch

    return torch.inference_mode()
