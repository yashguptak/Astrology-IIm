"""Merge LoRA adapter weights into the base model for deployment."""

import argparse
import sys
from pathlib import Path

import torch

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def merge_lora(
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str = "training/checkpoints/final",
    output_path: str = "training/merged/qwen2.5-3b-astrologer",
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Merge LoRA weights into base model and save in half precision.

    This creates a standalone model that can be loaded without PEFT,
    suitable for vLLM deployment.
    """
    print("=" * 60)
    print("  MERGING LORA ADAPTERS")
    print("=" * 60)
    print(f"  Base model:  {base_model_name}")
    print(f"  Adapter:     {adapter_path}")
    print(f"  Output:      {output_path}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Merge will run on CPU (very slow).")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("\n[1/3] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Base model loaded: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")

    print("\n[2/3] Loading and merging LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()
        print("  Merge successful!")
    except Exception as e:
        print(f"  ERROR during merge: {e}")
        print("  Attempting direct weight merge...")
        base_model.load_adapter(adapter_path)
        merged_model = base_model.merge_and_unload()
        print("  Merge successful (fallback method)!")

    print("\n[3/3] Saving merged model...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(output_dir))

    print(f"  Saved to: {output_dir}")
    print(f"  Files: {list(output_dir.glob('*'))}")

    if push_to_hub and hub_model_id:
        print(f"\n  Pushing to HuggingFace Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("  Done!")

    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="training/checkpoints/final",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="training/merged/qwen2.5-3b-astrologer",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    args = parser.parse_args()

    merge_lora(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
