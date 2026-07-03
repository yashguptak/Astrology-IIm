"""Dataset loading and preprocessing for ChatML conversational format."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


def load_dataset(
    train_file: str | Path,
    val_file: Optional[str | Path] = None,
    dataset_format: str = "chatml",
) -> DatasetDict | Dataset:
    """Load train/val JSON files into HuggingFace Dataset(s).

    Expects each entry to have a "messages" key containing a list of
    {"role": ..., "content": ...} dicts (ChatML format).
    """
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_data = _load_json_entries(train_path)
    train_dataset = Dataset.from_list(train_data)

    if val_file:
        val_path = Path(val_file)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation file not found: {val_path}")
        val_data = _load_json_entries(val_path)
        val_dataset = Dataset.from_list(val_data)
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    return train_dataset


def _load_json_entries(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_chatml_formatting_func(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Callable[[dict[str, Any]], str]:
    """Create a formatting function that applies the tokenizer's chat template.

    For Qwen2.5, this produces:
        <|im_start|>system
        ...<|im_end|>
        <|im_start|>user
        ...<|im_end|>
        <|im_start|>assistant
        ...<|im_end|>
    """

    def format_chat(example: dict[str, Any]) -> str:
        messages = example.get("messages", [])
        if not messages:
            return ""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return format_chat


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Tokenize a dataset with messages using the chat template.

    Only the assistant responses are used for loss computation.
    The SFTTrainer handles this via the `response_template` or by
    using the completion part of the formatted text.
    """

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages", [])
        if not messages:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=False,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].copy(),
        }

    return dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
