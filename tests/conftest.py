"""Shared test fixtures for the test suite."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_conversation():
    return {
        "messages": [
            {"role": "system", "content": "You are Vedaz AI astrologer."},
            {"role": "user", "content": "Hello, I want to know about my career."},
            {"role": "assistant", "content": "Please share your birth details."},
        ]
    }


@pytest.fixture
def sample_conversations():
    return [
        {
            "messages": [
                {"role": "system", "content": "System prompt A"},
                {"role": "user", "content": "User message 1"},
                {"role": "assistant", "content": "Assistant response 1"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "System prompt B"},
                {"role": "user", "content": "User message 2"},
                {"role": "assistant", "content": "Assistant response 2"},
            ]
        },
    ]


@pytest.fixture
def malformed_conversations():
    return [
        {"messages": "not_a_list"},
        {"no_messages_key": True},
        {
            "messages": [
                {"role": "user", "content": "Missing system prompt"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "assistant", "content": "No user before assistant"},
            ]
        },
    ]


@pytest.fixture
def duplicate_conversations():
    base = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
        ]
    }
    return [base, base, base]  # 3 identical entries


@pytest.fixture
def sample_train_val():
    return {
        "train": [
            {"messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": f"Train user {i}"},
                {"role": "assistant", "content": f"Train assistant {i}"},
            ]}
            for i in range(10)
        ],
        "validation": [
            {"messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": f"Val user {i}"},
                {"role": "assistant", "content": f"Val assistant {i}"},
            ]}
            for i in range(2)
        ],
    }
