"""Exact and fuzzy deduplication of conversation entries."""

import hashlib
from typing import Any


class DedupReport:
    def __init__(self):
        self.total = 0
        self.exact_duplicates = 0
        self.pairs_removed: list[tuple[int, int]] = []

    @property
    def unique_count(self) -> int:
        return self.total - self.exact_duplicates

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "DEDUPLICATION REPORT",
            "=" * 60,
            f"Total entries:        {self.total}",
            f"Exact duplicates:     {self.exact_duplicates}",
            f"Remaining unique:     {self.unique_count}",
        ]
        if self.pairs_removed:
            lines.append("\nRemoved pairs (kept first, removed second):")
            for first, second in self.pairs_removed[:15]:
                lines.append(f"  - Index {second} is duplicate of index {first}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Deduplicator:
    def __init__(self, content_hash_length: int = 200):
        self.hash_len = content_hash_length

    @staticmethod
    def _content_key(entry: dict[str, Any]) -> str:
        msgs = entry.get("messages", [])
        pairs: list[str] = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")[:200]
            pairs.append(f"{role}:{content}")
        return "|".join(pairs)

    def deduplicate(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], DedupReport]:
        report = DedupReport()
        report.total = len(entries)

        seen: dict[str, bool] = {}
        unique: list[dict[str, Any]] = []

        for i, entry in enumerate(entries):
            key = self._content_key(entry)
            if key in seen:
                report.exact_duplicates += 1
                report.pairs_removed.append((seen[key], i))
            else:
                seen[key] = i
                unique.append(entry)

        return unique, report
