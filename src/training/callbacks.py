"""Custom training callbacks for logging and early stopping."""

import math
from typing import Any

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class LossLoggerCallback(TrainerCallback):
    """Logs loss, learning rate, and gradient norm at each logging step."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any],
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            loss = logs.get("loss", logs.get("eval_loss", None))
            lr = logs.get("learning_rate", None)
            grad_norm = logs.get("grad_norm", None)
            step = state.global_step

            parts = [f"Step {step}"]
            if loss is not None:
                parts.append(f"Loss: {loss:.4f}")
                if "eval_loss" in logs:
                    parts.append(f"Val Loss: {logs['eval_loss']:.4f}")
                    try:
                        perplexity = math.exp(logs["eval_loss"])
                        parts.append(f"Perplexity: {perplexity:.2f}")
                    except OverflowError:
                        parts.append("Perplexity: inf")
            if lr is not None:
                parts.append(f"LR: {lr:.2e}")
            if grad_norm is not None:
                parts.append(f"Grad Norm: {grad_norm:.4f}")

            print(f"  {' | '.join(parts)}")


class SavePeftModelCallback(TrainerCallback):
    """Ensures PEFT adapter weights are saved properly at each checkpoint."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            print(f"  Checkpoint saved at step {state.global_step}")


class EarlyStoppingCallback(TrainerCallback):
    """Stop training if eval loss doesn't improve for N evaluations."""

    def __init__(self, patience: int = 5, min_improvement: float = 0.0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        **kwargs,
    ) -> None:
        loss = metrics.get("eval_loss")
        if loss is None:
            return

        improvement = self.best_loss - loss
        if improvement > self.min_improvement:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"  Early stopping triggered: no improvement for "
                    f"{self.patience} evaluations (best loss: {self.best_loss:.4f})"
                )
                control.should_training_stop = True
