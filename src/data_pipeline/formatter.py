"""Format conversation entries into ChatML format for Qwen training."""

from typing import Any


class FormatReport:
    def __init__(self):
        self.input_entries = 0
        self.output_entries = 0
        self.total_turns = 0
        self.system_prompts_count = 0

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "FORMATTING REPORT",
            "=" * 60,
            f"Input entries:       {self.input_entries}",
            f"Output entries:      {self.output_entries}",
            f"Total turns:         {self.total_turns}",
            f"System prompts:      {self.system_prompts_count}",
            "=" * 60,
        ]
        return "\n".join(lines)


class ChatMLFormatter:
    """Converts to HuggingFace conversational format (ChatML).

    Output schema:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
    }

    Qwen2.5 uses <|im_start|> and <|im_end|> tokens. The tokenizer
    applies them automatically when data is in this format.
    """

    def __init__(self, include_system: bool = True, max_content_length: int = 4096):
        self.include_system = include_system
        self.max_content_length = max_content_length

    def format_entry(
        self, entry: dict[str, Any], report: FormatReport
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"messages": []}
        msgs = entry.get("messages", [])

        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system" and not self.include_system:
                continue

            if len(content) > self.max_content_length:
                content = content[: self.max_content_length]

            output["messages"].append({"role": role, "content": content})
            report.total_turns += 1
            if role == "system":
                report.system_prompts_count += 1

        return output

    def format_all(
        self, entries: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], FormatReport]:
        report = FormatReport()
        report.input_entries = len(entries)
        formatted = [self.format_entry(e, report) for e in entries]
        report.output_entries = len(formatted)
        return formatted, report
