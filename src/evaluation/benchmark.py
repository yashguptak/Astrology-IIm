"""Latency, throughput, and memory benchmarking for the model."""

import time
from typing import Any

import torch


class Benchmark:
    """Measures inference latency, throughput, and memory usage."""

    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    def measure_latency(
        self,
        messages: list[dict[str, str]],
        n_runs: int = 5,
        warmup: int = 2,
        max_tokens: int = 128,
    ) -> dict[str, float]:
        times = []
        tokens_generated = []

        for i in range(warmup + n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            start = time.perf_counter()
            text = self.engine.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=0.7,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if i >= warmup:
                times.append(elapsed)
                generated_tokens = len(self.tokenizer.encode(text))
                tokens_generated.append(generated_tokens)

        avg_latency = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0

        return {
            "avg_latency_seconds": round(avg_latency, 3),
            "min_latency_seconds": round(min(times), 3),
            "max_latency_seconds": round(max(times), 3),
            "p50_latency_seconds": round(sorted(times)[len(times) // 2], 3),
            "avg_tokens_generated": round(avg_tokens, 1),
            "tokens_per_second": round(tokens_per_sec, 1),
            "num_runs": n_runs,
        }

    def measure_throughput(
        self,
        messages: list[dict[str, str]],
        n_requests: int = 10,
        max_tokens: int = 128,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        for _ in range(n_requests):
            self.engine.generate(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=0.7,
            )
        total_time = time.perf_counter() - start

        return {
            "total_time_seconds": round(total_time, 3),
            "requests_per_second": round(n_requests / total_time, 2),
            "num_requests": n_requests,
        }

    def measure_memory(self) -> dict[str, Any]:
        memory = {}
        if torch.cuda.is_available():
            memory["cuda_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1e9, 2
            )
            memory["cuda_reserved_gb"] = round(
                torch.cuda.memory_reserved() / 1e9, 2
            )
            memory["cuda_max_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2
            )
        import psutil
        process = psutil.Process()
        memory["cpu_rss_gb"] = round(process.memory_info().rss / 1e9, 2)
        return memory
