"""
Main training entrypoint for QLoRA SFT fine-tuning.

Usage:
    python -m src.training.train
    python -m src.training.train --config configs/training.yaml
    python -m src.training.train --resume_from_checkpoint training/checkpoints/checkpoint-100

Runs on CPU with graceful exit if no GPU is available.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.config import TrainingConfig
from src.training.dataset import load_dataset
from src.training.callbacks import LossLoggerCallback, SavePeftModelCallback


def train(config: TrainingConfig, resume_from_checkpoint: str | None = None):
    """Run the full training loop with the given config."""

    # ------------------------------------------------------------------ #
    # 1. GPU check — graceful exit if no GPU
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        print("=" * 60)
        print("  NO GPU DETECTED")
        print("=" * 60)
        print()
        print("  This script requires a CUDA-capable GPU for QLoRA training.")
        print()
        print("  Options:")
        print("    1. Google Colab (free T4 GPU)")
        print("       Upload this project and run:")
        print("       !python -m src.training.train")
        print()
        print("    2. RunPod / Vast.ai / Lambda Labs")
        print("       Rent a T4 (16GB) instance for ~$0.11/hr")
        print("       Total training cost: ~$0.33")
        print()
        print("    3. Local with GPU")
        print("       If you have a compatible NVIDIA GPU, install CUDA")
        print("       and re-run.")
        print()
        print("=" * 60)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 2. Imports (only after GPU is confirmed)
    # ------------------------------------------------------------------ #
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    # ------------------------------------------------------------------ #
    # 3. Setup
    # ------------------------------------------------------------------ #
    args = config.training
    set_seed(args.seed)

    print("=" * 60)
    print("  QLoRA SFT TRAINING")
    print("=" * 60)
    print(f"  Model:        {config.model.name_or_path}")
    print(f"  LoRA rank:    {config.lora.r}")
    print(f"  Max seq len:  {args.max_seq_length}")
    print(f"  Batch size:   {args.per_device_train_batch_size}")
    print(f"  Grad accum:   {args.gradient_accumulation_steps}")
    print(f"  Effective BS: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Epochs:       {args.num_train_epochs}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Output dir:   {args.output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 4. Tokenizer
    # ------------------------------------------------------------------ #
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify chat template
    if tokenizer.chat_template is None:
        print("  WARNING: No chat template found. Using default ChatML.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    print(f"  Tokenizer: {config.model.name_or_path}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Chat template: {'present' if tokenizer.chat_template else 'missing'}")

    # ------------------------------------------------------------------ #
    # 5. Dataset
    # ------------------------------------------------------------------ #
    print("\n[2/7] Loading dataset...")
    dataset = load_dataset(
        train_file=config.data.train_file,
        val_file=config.data.val_file,
        dataset_format=config.data.dataset_format,
    )
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    # Preview formatted example
    example = dataset["train"][0]
    formatted = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    print(f"\n  Formatted example ({len(formatted)} chars):")
    print(f"  {formatted[:300]}...")
    print()

    # ------------------------------------------------------------------ #
    # 6. Quantization config (4-bit)
    # ------------------------------------------------------------------ #
    print("[3/7] Configuring quantization...")
    quant_config = config.quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
    )

    # ------------------------------------------------------------------ #
    # 7. Load base model
    # ------------------------------------------------------------------ #
    print("[4/7] Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        use_cache=config.model.use_cache,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ------------------------------------------------------------------ #
    # 8. LoRA adapter
    # ------------------------------------------------------------------ #
    print("[5/7] Applying LoRA...")
    lora_config_obj = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_config_obj)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Total params: {total_params:,}")

    # ------------------------------------------------------------------ #
    # 9. Training arguments
    # ------------------------------------------------------------------ #
    print("[6/7] Configuring training arguments...")
    targs = config.training
    training_args = TrainingArguments(
        output_dir=targs.output_dir,
        num_train_epochs=targs.num_train_epochs,
        per_device_train_batch_size=targs.per_device_train_batch_size,
        per_device_eval_batch_size=targs.per_device_eval_batch_size,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
        gradient_checkpointing=targs.gradient_checkpointing,
        learning_rate=targs.learning_rate,
        lr_scheduler_type=targs.lr_scheduler_type,
        warmup_ratio=targs.warmup_ratio,
        weight_decay=targs.weight_decay,
        max_grad_norm=targs.max_grad_norm,
        optim=targs.optim,
        logging_steps=targs.logging_steps,
        save_steps=targs.save_steps,
        eval_steps=targs.eval_steps,
        eval_strategy=targs.eval_strategy,
        save_total_limit=targs.save_total_limit,
        load_best_model_at_end=targs.load_best_model_at_end,
        metric_for_best_model=targs.metric_for_best_model,
        greater_is_better=targs.greater_is_better,
        report_to=targs.report_to,
        seed=targs.seed,
        fp16=targs.fp16,
        bf16=targs.bf16,
        remove_unused_columns=targs.remove_unused_columns,
        dataloader_num_workers=targs.dataloader_num_workers,
        group_by_length=targs.group_by_length,
        ddp_find_unused_parameters=targs.ddp_find_unused_parameters,
        label_names=targs.label_names,
        logging_dir=str(Path(targs.output_dir).parent / "logs"),
        save_safetensors=True,
    )

    # ------------------------------------------------------------------ #
    # 10. Trainer
    # ------------------------------------------------------------------ #
    print("[7/7] Initializing SFTTrainer...")

    # For ChatML format, we let SFTTrainer handle conversation data.
    # The trainer auto-detects "messages" column and applies the chat template,
    # computing loss only on assistant responses.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=targs.max_seq_length,
        packing=targs.packing,
        dataset_kwargs={
            "add_special_tokens": False,
        },
    )

    # Add callbacks
    trainer.add_callback(LossLoggerCallback())
    trainer.add_callback(SavePeftModelCallback())

    print(f"\n  Trainer ready. Starting training...")
    print(f"  Resume from: {resume_from_checkpoint or 'scratch'}")
    print(f"  TensorBoard: tensorboard --logdir {training_args.logging_dir}")
    print()

    # ------------------------------------------------------------------ #
    # 11. Train
    # ------------------------------------------------------------------ #
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ------------------------------------------------------------------ #
    # 12. Save final model
    # ------------------------------------------------------------------ #
    print("\nSaving final model...")
    final_output = Path(targs.output_dir) / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"  Final adapter saved to: {final_output}")

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate final
    print("\nFinal evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(dataset["validation"])
    try:
        eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["eval_perplexity"] = float("inf")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final eval loss: {eval_metrics['eval_loss']:.4f}")
    print(f"  Final perplexity: {eval_metrics['eval_perplexity']:.2f}")
    print(f"{'=' * 60}")

    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from a specific checkpoint path",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = TrainingConfig.from_yaml(config_path)
    train(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
