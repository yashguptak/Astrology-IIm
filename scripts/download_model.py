#!/usr/bin/env python3
"""
Download and cache the model locally.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct --cache-dir ./models

Runs on CPU. Memory: < 4 GB.
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Download model to local cache")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Downloading model: {args.model}")
    print("=" * 60)

    kwargs = {
        "trust_remote_code": True,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    print(f"  Tokenizer loaded. Vocab size: {len(tokenizer)}")

    print("\n[2/2] Downloading model (CPU, float32)...")
    print("  This may take a few minutes depending on your internet speed.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype="auto",
        **kwargs,
    )
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    cache_info = Path(tokenizer._cache_dir) if hasattr(tokenizer, "_cache_dir") else "default cache"
    print(f"\nModel cached at: {cache_info}")
    print("Download complete!")


if __name__ == "__main__":
    main()
