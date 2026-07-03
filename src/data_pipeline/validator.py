"""Schema and structural validation for conversation datasets."""

import json
from pathlib import Path
from typing import Any

VALID_ROLES = {"system", "user", "assistant"}


class ValidationReport:
    def __init__(self):
        self.total_entries = 0
        self.valid_entries = 0
        self.invalid_entries: list[dict[str, Any]] = []
        self.errors: list[str] = []

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total_entries,
            "valid": self.valid_entries,
            "invalid": len(self.invalid_entries),
            "error_count": len(self.errors),
        }

    def __str__(self) -> str:
        s = self.summary
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Total entries:     {s['total']}",
            f"Valid:             {s['valid']}",
            f"Invalid:           {s['invalid']}",
            f"Errors:            {s['error_count']}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")
        if self.invalid_entries:
            lines.append("\nInvalid entry indices:")
            for inv in self.invalid_entries[:10]:
                lines.append(f"  - Entry {inv['index']}: {inv['reason']}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DatasetValidator:
    def __init__(self, min_turns: int = 2):
        self.min_turns = min_turns

    def validate(
        self, entries: list[dict[str, Any]], source: str = "unknown"
    ) -> ValidationReport:
        report = ValidationReport()
        report.total_entries = len(entries)

        for i, entry in enumerate(entries):
            entry_errors: list[str] = []

            if not isinstance(entry, dict):
                entry_errors.append("Entry is not a dict")
                report.invalid_entries.append({"index": i, "reason": "Not a dict"})
                report.errors.append(f"Entry {i}: Not a dict")
                continue

            if "messages" not in entry:
                entry_errors.append("Missing 'messages' field")
                report.invalid_entries.append(
                    {"index": i, "reason": "Missing 'messages'"}
                )
                report.errors.append(f"Entry {i}: Missing 'messages'")
                continue

            msgs = entry["messages"]
            if not isinstance(msgs, list):
                entry_errors.append("'messages' is not a list")
                report.invalid_entries.append(
                    {"index": i, "reason": "'messages' not a list"}
                )
                report.errors.append(f"Entry {i}: 'messages' not a list")
                continue

            if len(msgs) < self.min_turns:
                entry_errors.append(
                    f"Only {len(msgs)} turns, minimum {self.min_turns}"
                )
                report.invalid_entries.append(
                    {"index": i, "reason": f"< {self.min_turns} turns"}
                )
                report.errors.append(
                    f"Entry {i}: {len(msgs)} turns < {self.min_turns}"
                )
                continue

            for j, msg in enumerate(msgs):
                if not isinstance(msg, dict):
                    entry_errors.append(f"Message {j} is not a dict")
                    continue
                if "role" not in msg:
                    entry_errors.append(f"Message {j} missing 'role'")
                elif msg["role"] not in VALID_ROLES:
                    entry_errors.append(
                        f"Message {j} has invalid role '{msg['role']}'"
                    )
                if "content" not in msg:
                    entry_errors.append(f"Message {j} missing 'content'")
                elif not isinstance(msg["content"], str):
                    entry_errors.append(f"Message {j} 'content' is not a string")
                elif len(msg["content"].strip()) == 0:
                    entry_errors.append(f"Message {j} has empty content")

            if msgs[0].get("role") != "system":
                entry_errors.append("First message must have role='system'")

            roles = [m.get("role") for m in msgs if m.get("role") in VALID_ROLES]
            for j in range(1, len(roles)):
                if roles[j] == roles[j - 1] and roles[j] != "system":
                    entry_errors.append(
                        f"Consecutive same roles at position {j}: {roles[j]}"
                    )

            if msgs[-1].get("role") != "assistant":
                entry_errors.append("Last message must have role='assistant'")

            has_user = any(m.get("role") == "user" for m in msgs)
            if not has_user:
                entry_errors.append("No user message found in conversation")

            if entry_errors:
                report.invalid_entries.append(
                    {"index": i, "reason": "; ".join(entry_errors[:3])}
                )
                for err in entry_errors:
                    report.errors.append(f"Entry {i}: {err}")
            else:
                report.valid_entries += 1

        return report


def load_raw_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON entries from file, handling both arrays and line-delimited JSON."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()

    if raw.startswith("["):
        return json.loads(raw)

    entries: list[dict[str, Any]] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch == "{":
            if depth == 0:
                current = []
            depth += 1
        elif ch == "}":
            depth -= 1
        current.append(ch)
        if depth == 0 and current:
            fragment = "".join(current).strip().rstrip(",")
            if fragment:
                try:
                    entries.append(json.loads(fragment))
                except json.JSONDecodeError:
                    pass
            current = []
    return entries
