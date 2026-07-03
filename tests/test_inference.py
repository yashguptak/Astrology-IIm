"""Tests for the inference and conversation modules."""

from src.inference.chat import Conversation


class TestConversation:
    def test_empty_conversation(self):
        chat = Conversation()
        assert len(chat) == 0
        assert chat.messages == []

    def test_with_system_prompt(self):
        chat = Conversation(system_prompt="You are a helpful assistant.")
        assert len(chat) == 1
        assert chat.messages[0]["role"] == "system"

    def test_add_user_message(self):
        chat = Conversation()
        chat.add_user("Hello")
        assert len(chat) == 1
        assert chat.messages[0] == {"role": "user", "content": "Hello"}

    def test_add_assistant_message(self):
        chat = Conversation()
        chat.add_user("Hi")
        chat.add_assistant("Hello!")
        assert len(chat) == 2
        assert chat.messages[1] == {"role": "assistant", "content": "Hello!"}

    def test_get_last_user(self):
        chat = Conversation()
        chat.add_user("First")
        chat.add_assistant("Response 1")
        chat.add_user("Second")
        assert chat.get_last_user() == "Second"

    def test_get_last_assistant(self):
        chat = Conversation()
        chat.add_user("Hi")
        chat.add_assistant("Hello")
        chat.add_user("How are you?")
        chat.add_assistant("I'm fine!")
        assert chat.get_last_assistant() == "I'm fine!"

    def test_trim_conversation(self):
        chat = Conversation(system_prompt="System")
        for i in range(20):
            chat.add_user(f"User {i}")
            chat.add_assistant(f"Assistant {i}")
        assert len(chat) == 41  # system + 40 messages

        chat.trim_to_last_n_turns(3)
        # System + last 3 turns (6 messages)
        assert len(chat) == 7

    def test_clear(self):
        chat = Conversation(system_prompt="System")
        chat.add_user("Hello")
        chat.add_assistant("Hi")
        chat.clear()
        assert len(chat) == 1
        assert chat.messages[0]["role"] == "system"

    def test_to_dict(self):
        chat = Conversation(system_prompt="System")
        chat.add_user("Hello")
        d = chat.to_dict()
        assert "messages" in d
        assert len(d["messages"]) == 2

    def test_from_dict(self):
        data = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
            ]
        }
        chat = Conversation.from_dict(data)
        assert len(chat) == 2
        assert chat.messages[1]["content"] == "Hello"
