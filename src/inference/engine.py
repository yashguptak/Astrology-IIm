"""Inference engine supporting both HuggingFace transformers and vLLM."""

import sys
from pathlib import Path
from typing import Any


class InferenceEngine:
    """Wrapper around model inference that tries vLLM, falls back to transformers.

    Usage:
        engine = InferenceEngine("Qwen/Qwen2.5-3B-Instruct")
        response = engine.generate([{"role": "user", "content": "Hello"}])
        print(response)
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        use_vllm: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.max_model_len = max_model_len
        self._backend = None
        self._tokenizer = None

        if use_vllm:
            self._try_init_vllm(gpu_memory_utilization)
        else:
            self._init_transformers()

    def _try_init_vllm(self, gpu_memory_utilization: float) -> None:
        try:
            from vllm import LLM, SamplingParams

            self._vllm_llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                dtype="bfloat16",
            )
            self._vllm_sampling = SamplingParams
            self._backend = "vllm"

            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            print(f"[InferenceEngine] Using vLLM backend: {self.model_path}")
        except ImportError:
            print("[InferenceEngine] vLLM not installed, falling back to transformers")
            self._init_transformers()
        except Exception as e:
            print(f"[InferenceEngine] vLLM init failed: {e}, falling back to transformers")
            self._init_transformers()

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
        )
        self._model.eval()
        self._backend = "transformers"
        print(f"[InferenceEngine] Using transformers backend: {self.model_path}")

    @property
    def backend(self) -> str | None:
        return self._backend

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> str | Any:
        if self._backend == "vllm":
            return self._generate_vllm(
                messages, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stream
            )
        return self._generate_transformers(
            messages, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def _generate_vllm(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stream: bool,
    ) -> str | Any:
        sampling_params = self._vllm_sampling(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self._vllm_llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _generate_transformers(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> str:
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
