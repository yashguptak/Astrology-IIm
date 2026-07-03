"""
Run full evaluation: BLEU, ROUGE, perplexity, latency, hallucination checks.

Usage:
    python -m src.evaluation.run_evaluation --model Qwen/Qwen2.5-3B-Instruct
    python -m src.evaluation.run_evaluation --model training/merged/qwen2.5-3b-astrologer

Requires: GPU recommended. Falls back to CPU.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.engine import InferenceEngine
from src.evaluation.metrics import Evaluator
from src.evaluation.benchmark import Benchmark
from src.evaluation.hallucination import HallucinationDetector


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default="evaluation/reports/eval_report.json")
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--test-samples", type=int, default=5, help="Number of test prompts")
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    print(f"  Model: {args.model}")

    # Load engine
    engine = InferenceEngine(
        model_path=args.model,
        use_vllm=not args.no_vllm,
    )
    tokenizer = engine._tokenizer

    # Test prompts
    test_prompts = [
        [{"role": "user", "content": "Mera naam Rahul hai. DOB 15 March 1995, 8:30 AM, Delhi. Mera career kaisa rahega?"}],
        [{"role": "user", "content": "Hello, can you tell me about my marriage? I was born 4 June 1997, 5:50 AM, Ranchi."}],
        [{"role": "user", "content": "Mera business mein loss ho raha hai. Kya koi upay hai?"}],
        [{"role": "user", "content": "What does my birth chart say about my health? DOB 12 July 1996, 9:10 AM, Delhi."}],
        [{"role": "user", "content": "Meri shadi kab hogi? Bas exact date bata do."}],
    ]

    # Generate responses
    print(f"\n[1/4] Generating {len(test_prompts)} test responses...")
    responses = []
    for i, prompt in enumerate(test_prompts):
        resp = engine.generate(prompt, max_new_tokens=256, temperature=0.7)
        responses.append(resp)
        print(f"  {i+1}. {resp[:80]}...")

    # Evaluate metrics
    print("\n[2/4] Computing BLEU/ROUGE/perplexity...")
    evaluator = Evaluator(model=None, tokenizer=tokenizer)
    eval_results = evaluator.evaluate_dataset([
        {"reference": "", "candidate": r} for r in responses
    ])
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    # Latency benchmark
    print("\n[3/4] Running latency benchmark...")
    benchmark = Benchmark(engine, tokenizer)
    latency = benchmark.measure_latency(
        test_prompts[0], n_runs=3, warmup=1, max_tokens=128
    )
    for k, v in latency.items():
        print(f"  {k}: {v}")

    # Hallucination check
    print("\n[4/4] Running hallucination checks...")
    detector = HallucinationDetector()
    hall_results = detector.evaluate_dataset(responses)
    print(f"  Flagged: {hall_results['flagged_count']}/{hall_results['total_responses']}")
    for flag in hall_results.get("unique_flags", []):
        print(f"  - {flag}")

    # Combined report
    report = {
        "model": args.model,
        "metrics": eval_results,
        "latency": latency,
        "hallucination": hall_results,
        "sample_responses": [
            {"prompt": p[0]["content"][:100], "response": r[:300]}
            for p, r in zip(test_prompts, responses)
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
