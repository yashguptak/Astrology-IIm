"""Conversation history manager for multi-turn chat."""

from typing import Any


class Conversation:
    """Maintains a conversation history compatible with ChatML format.

    Usage:
        chat = Conversation(system_prompt="You are a helpful assistant.")
        chat.add_user("Hello!")
        chat.add_assistant("Hi! How can I help?")
        messages = chat.messages  # list of dicts for the model
    """

    def __init__(self, system_prompt: str | None = None, max_turns: int = 10):
        self.max_turns = max_turns
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_system(self, content: str) -> None:
        self._messages.insert(0, {"role": "system", "content": content})

    def get_last_user(self) -> str | None:
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_assistant(self) -> str | None:
        for msg in reversed(self._messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def trim_to_last_n_turns(self, n: int = 10) -> None:
        """Keep only the last N user-assistant exchanges plus system prompt."""
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]
        if len(non_system) > n * 2:
            non_system = non_system[-(n * 2):]
        self._messages = system_msgs + non_system

    def clear(self) -> None:
        self._messages = [m for m in self._messages if m["role"] == "system"]

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self._messages}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        conv = cls()
        conv._messages = data.get("messages", [])
        return conv

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"Conversation({len(self._messages)} messages)"
