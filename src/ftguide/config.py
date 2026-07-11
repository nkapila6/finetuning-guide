from __future__ import annotations
from dataclasses import dataclass, field
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    base_model: str = "unsloth/Llama-3.2-1B"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    load_in_4bit: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    num_train_epochs: float | None = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    seed: int = 3407
    output_dir: str = "outputs"
    chat_template: str = "llama-3.1"
    dataset_text_field: str = "text"
    train_on_responses_only: bool = False
    # data curation
    dataset_path: str = "yahma/alpaca-cleaned"
    dataset_split: str = "train"
    min_quality_score: float = 0.0
    max_length_filter: int = 2048
    dedup_threshold: float = 0.9
    max_examples: int | None = None

    @classmethod
    def from_yaml(cls, path: str) -> FinetuneConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        if len(data) != len(filtered_data):
            unknown_keys = set(data.keys()) - valid_keys
            logger.warning(f"Unknown keys in YAML: {unknown_keys}")
        return cls(**filtered_data)

    def save_yaml(self, path: str) -> None:
        data = {
            "base_model": self.base_model,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "load_in_4bit": self.load_in_4bit,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optim": self.optim,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "chat_template": self.chat_template,
            "dataset_text_field": self.dataset_text_field,
            "train_on_responses_only": self.train_on_responses_only,
            "dataset_path": self.dataset_path,
            "dataset_split": self.dataset_split,
            "min_quality_score": self.min_quality_score,
            "max_length_filter": self.max_length_filter,
            "dedup_threshold": self.dedup_threshold,
            "max_examples": self.max_examples,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)
