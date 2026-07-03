"""Evaluation metrics: BLEU, ROUGE, Perplexity, response quality."""

import math
from pathlib import Path
from typing import Any

import torch


class Evaluator:
    """Compute evaluation metrics for the fine-tuned model."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def compute_perplexity(self, text: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")

    def compute_bleu(self, reference: str, candidate: str) -> dict[str, float]:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            if not ref_tokens or not cand_tokens:
                return {"bleu": 0.0}
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
            return {"bleu": round(score, 4)}
        except ImportError:
            return {"bleu": -1.0, "error": "nltk not installed"}

    def compute_rouge(self, reference: str, candidate: str) -> dict[str, float]:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return {
                "rouge1_f": round(scores["rouge1"].fmeasure, 4),
                "rouge1_p": round(scores["rouge1"].precision, 4),
                "rouge1_r": round(scores["rouge1"].recall, 4),
                "rouge2_f": round(scores["rouge2"].fmeasure, 4),
                "rougeL_f": round(scores["rougeL"].fmeasure, 4),
            }
        except ImportError:
            return {"rouge1_f": -1.0, "error": "rouge_score not installed"}

    def compute_response_length(self, text: str) -> dict[str, int]:
        words = text.split()
        chars = len(text)
        return {"words": len(words), "chars": chars}

    def evaluate_dataset(
        self,
        reference_pairs: list[dict[str, str]],
    ) -> dict[str, Any]:
        metrics_summary = {
            "bleu_scores": [],
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "response_lengths": [],
            "perplexities": [],
        }

        for pair in reference_pairs:
            ref = pair.get("reference", "")
            cand = pair.get("candidate", "")

            bleu = self.compute_bleu(ref, cand)
            if bleu.get("bleu", -1) >= 0:
                metrics_summary["bleu_scores"].append(bleu["bleu"])

            rouge = self.compute_rouge(ref, cand)
            if "rouge1_f" in rouge and rouge["rouge1_f"] >= 0:
                metrics_summary["rouge1_f"].append(rouge["rouge1_f"])
                metrics_summary["rouge2_f"].append(rouge["rouge2_f"])
                metrics_summary["rougeL_f"].append(rouge["rougeL_f"])

            length = self.compute_response_length(cand)
            metrics_summary["response_lengths"].append(length["words"])

            if pair.get("compute_ppl", False):
                ppl = self.compute_perplexity(cand)
                metrics_summary["perplexities"].append(ppl)

        avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
        return {
            "bleu_avg": avg(metrics_summary["bleu_scores"]),
            "bleu_min": round(min(metrics_summary["bleu_scores"]), 4) if metrics_summary["bleu_scores"] else 0.0,
            "bleu_max": round(max(metrics_summary["bleu_scores"]), 4) if metrics_summary["bleu_scores"] else 0.0,
            "rouge1_avg": avg(metrics_summary["rouge1_f"]),
            "rouge2_avg": avg(metrics_summary["rouge2_f"]),
            "rougeL_avg": avg(metrics_summary["rougeL_f"]),
            "avg_response_words": avg(metrics_summary["response_lengths"]),
            "avg_perplexity": avg(metrics_summary["perplexities"]),
            "num_pairs": len(reference_pairs),
        }
