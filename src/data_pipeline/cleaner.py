"""Text cleaning and normalization for conversation messages."""

import re
from typing import Any


class CleaningReport:
    def __init__(self):
        self.total_messages = 0
        self.modified_messages = 0
        self.details: list[str] = []

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "CLEANING REPORT",
            "=" * 60,
            f"Messages scanned:   {self.total_messages}",
            f"Messages modified:  {self.modified_messages}",
            "=" * 60,
        ]
        for d in self.details[:15]:
            lines.append(f"  - {d}")
        if len(self.details) > 15:
            lines.append(f"  ... and {len(self.details) - 15} more")
        return "\n".join(lines)


class TextCleaner:
    def __init__(self, normalize_unicode: bool = True, strip_whitespace: bool = True):
        self.normalize_unicode_flag = normalize_unicode
        self.strip_whitespace_flag = strip_whitespace

    def clean_message(self, text: str) -> str:
        if self.strip_whitespace_flag:
            text = text.strip()
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" \n", "\n", text)
        if self.normalize_unicode_flag:
            text = self._normalize_unicode(text)
        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
            "\u200b": "",
            "\ufeff": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def clean_entry(
        self, entry: dict[str, Any], report: CleaningReport
    ) -> dict[str, Any]:
        entry = entry.copy()
        msgs = entry.get("messages", [])
        for i, msg in enumerate(msgs):
            msg = msg.copy()
            content = msg.get("content", "")
            cleaned = self.clean_message(content)
            if cleaned != content:
                report.modified_messages += 1
                report.details.append(
                    f"Msg {i}: trimmed {len(content) - len(cleaned)} chars"
                )
            msg["content"] = cleaned
            msgs[i] = msg
        entry["messages"] = msgs
        return entry

    def clean_all(
        self, entries: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CleaningReport]:
        report = CleaningReport()
        report.total_messages = sum(len(e.get("messages", [])) for e in entries)
        cleaned = [self.clean_entry(e, report) for e in entries]
        return cleaned, report
