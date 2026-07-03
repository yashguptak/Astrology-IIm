"""Stratified train/validation splitting of conversation datasets."""

import random
from typing import Any


class SplitReport:
    def __init__(self):
        self.total = 0
        self.train_count = 0
        self.val_count = 0
        self.seed = 42

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "TRAIN/VALIDATION SPLIT REPORT",
            "=" * 60,
            f"Total entries:      {self.total}",
            f"Train split:        {self.train_count} ({self.train_count / self.total * 100:.1f}%)",
            f"Validation split:   {self.val_count} ({self.val_count / self.total * 100:.1f}%)",
            f"Random seed:        {self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


class TrainValSplitter:
    def __init__(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
        stratify_by_language: bool = False,
    ):
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")
        self.val_ratio = val_ratio
        self.seed = seed
        self.stratify_by_language = stratify_by_language

    @staticmethod
    def _detect_language(entry: dict[str, Any]) -> str:
        combined = " ".join(
            m.get("content", "")
            for m in entry.get("messages", [])
            if m.get("role") == "user"
        )
        has_devanagari = any(0x0900 <= ord(c) <= 0x0FFF for c in combined)
        if has_devanagari:
            english_chars = sum(1 for c in combined if c.isascii() and c.isalpha())
            total_chars = sum(1 for c in combined if c.isalpha())
            if total_chars > 0 and english_chars / total_chars > 0.3:
                return "hinglish"
            return "hindi"
        return "english"

    def split(
        self, entries: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitReport]:
        report = SplitReport()
        report.total = len(entries)
        report.seed = self.seed

        if self.stratify_by_language:
            groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for i, e in enumerate(entries):
                lang = self._detect_language(e)
                groups.setdefault(lang, []).append((i, e))

            train: list[dict[str, Any]] = []
            val: list[dict[str, Any]] = []
            rng = random.Random(self.seed)
            for lang, group in groups.items():
                rng.shuffle(group)
                split_point = max(1, int(len(group) * (1 - self.val_ratio)))
                train.extend(e for _, e in group[:split_point])
                val.extend(e for _, e in group[split_point:])
        else:
            indexed = list(enumerate(entries))
            rng = random.Random(self.seed)
            rng.shuffle(indexed)
            split_point = max(1, int(len(indexed) * (1 - self.val_ratio)))
            train = [e for _, e in indexed[:split_point]]
            val = [e for _, e in indexed[split_point:]]

        report.train_count = len(train)
        report.val_count = len(val)
        return train, val, report
