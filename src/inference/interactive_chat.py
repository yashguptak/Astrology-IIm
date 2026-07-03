"""
Interactive terminal chat with the fine-tuned model.

Usage:
    python -m src.inference.interactive_chat
    python -m src.inference.interactive_chat --model Qwen/Qwen2.5-3B-Instruct
    python -m src.inference.interactive_chat --model training/merged/qwen2.5-3b-astrologer

Requires a GPU or CPU (slow). Memory: < 4 GB (CPU) / < 6 GB (GPU).
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.engine import InferenceEngine
from src.inference.chat import Conversation


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with the model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=(
            "You are Vedaz's AI Vedic astrologer. You give compassionate, "
            "balanced, non-fatalistic guidance. You never predict death, "
            "illness, or guaranteed misfortune. Respond in the user's language."
        ),
        help="System prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use transformers backend instead of vLLM",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ASTROLOGER CHAT — Interactive Mode")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print()
    print("  Type 'exit' to quit, 'clear' to reset conversation.")
    print("=" * 60)
    print()

    engine = InferenceEngine(
        model_path=args.model,
        use_vllm=not args.no_vllm,
    )
    chat = Conversation(system_prompt=args.system)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "clear":
            chat.clear()
            print("[Conversation reset]")
            continue

        chat.add_user(user_input)
        print("Assistant: ", end="", flush=True)

        response = engine.generate(
            messages=chat.messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)
        print()
        chat.add_assistant(response)

        # Keep last 5 exchanges to manage context
        chat.trim_to_last_n_turns(5)

    print("\nGoodbye! ✨")


if __name__ == "__main__":
    main()
