"""Pydantic configuration models for training."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = False


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


class TrainingArgsConfig(BaseModel):
    output_dir: str = "training/checkpoints"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    eval_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_seq_length: int = 2048
    packing: bool = False
    report_to: str = "tensorboard"
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 2
    group_by_length: bool = False
    ddp_find_unused_parameters: Optional[bool] = None
    label_names: list[str] = Field(default_factory=lambda: ["labels"])

    @field_validator("output_dir")
    @classmethod
    def resolve_output_dir(cls, v: str) -> str:
        return str(Path(v).resolve())


class DataConfig(BaseModel):
    train_file: str = "data/processed/train.json"
    val_file: str = "data/processed/val.json"
    dataset_format: str = "chatml"


class TrainingConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    training: TrainingArgsConfig = Field(default_factory=TrainingArgsConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
