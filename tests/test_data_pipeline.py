"""Tests for the data pipeline modules."""

import json
from pathlib import Path

import pytest

from src.data_pipeline.validator import DatasetValidator, load_raw_entries, ValidationReport
from src.data_pipeline.cleaner import TextCleaner, CleaningReport
from src.data_pipeline.deduplicator import Deduplicator, DedupReport
from src.data_pipeline.formatter import ChatMLFormatter, FormatReport
from src.data_pipeline.splitter import TrainValSplitter, SplitReport


class TestValidator:
    def test_valid_conversation(self, sample_conversation):
        validator = DatasetValidator()
        report = validator.validate([sample_conversation])
        assert report.valid_entries == 1
        assert report.invalid_entries == []

    def test_missing_messages(self):
        validator = DatasetValidator()
        report = validator.validate([{"id": "no_messages"}])
        assert report.valid_entries == 0
        assert len(report.invalid_entries) == 1

    def test_malformed_conversations(self, malformed_conversations):
        validator = DatasetValidator()
        report = validator.validate(malformed_conversations)
        assert report.valid_entries == 0
        assert len(report.invalid_entries) == 4

    def test_min_turns(self):
        validator = DatasetValidator(min_turns=5)
        entry = {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
            ]
        }
        report = validator.validate([entry])
        assert report.valid_entries == 0

    def test_consecutive_same_role(self):
        validator = DatasetValidator()
        entry = {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U1"},
                {"role": "user", "content": "U2"},
                {"role": "assistant", "content": "A"},
            ]
        }
        report = validator.validate([entry])
        assert report.valid_entries == 0

    def test_last_message_not_assistant(self):
        validator = DatasetValidator()
        entry = {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ]
        }
        report = validator.validate([entry])
        assert report.valid_entries == 0


class TestCleaner:
    def test_strip_whitespace(self):
        cleaner = TextCleaner()
        result = cleaner.clean_message("  Hello   World  \n\n\nTest  ")
        assert result == "Hello World\n\nTest"

    def test_unicode_normalization(self):
        cleaner = TextCleaner()
        result = cleaner.clean_message("\u201cHello\u201d \u2014 World\u2026")
        assert result == '"Hello" - World...'

    def test_clean_entry(self):
        cleaner = TextCleaner()
        entry = {
            "messages": [
                {"role": "user", "content": "  Hello  "},
            ]
        }
        report = CleaningReport()
        result = cleaner.clean_entry(entry, report)
        assert result["messages"][0]["content"] == "Hello"

    def test_clean_all(self, sample_conversations):
        cleaner = TextCleaner()
        cleaned, report = cleaner.clean_all(sample_conversations)
        assert len(cleaned) == 2
        assert isinstance(report, CleaningReport)


class TestDeduplicator:
    def test_no_duplicates(self, sample_conversations):
        dedup = Deduplicator()
        unique, report = dedup.deduplicate(sample_conversations)
        assert len(unique) == 2
        assert report.exact_duplicates == 0

    def test_with_duplicates(self, duplicate_conversations):
        dedup = Deduplicator()
        unique, report = dedup.deduplicate(duplicate_conversations)
        assert len(unique) == 1
        assert report.exact_duplicates == 2

    def test_report_values(self, duplicate_conversations):
        dedup = Deduplicator()
        _, report = dedup.deduplicate(duplicate_conversations)
        assert report.total == 3
        assert report.unique_count == 1


class TestFormatter:
    def test_format_entry(self, sample_conversation):
        formatter = ChatMLFormatter()
        report = FormatReport()
        result = formatter.format_entry(sample_conversation, report)
        assert "messages" in result
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][-1]["role"] == "assistant"

    def test_format_all(self, sample_conversations):
        formatter = ChatMLFormatter()
        formatted, report = formatter.format_all(sample_conversations)
        assert len(formatted) == 2
        assert report.input_entries == 2

    def test_exclude_system(self, sample_conversation):
        formatter = ChatMLFormatter(include_system=False)
        report = FormatReport()
        result = formatter.format_entry(sample_conversation, report)
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_content_truncation(self):
        formatter = ChatMLFormatter(max_content_length=10)
        entry = {
            "messages": [
                {"role": "user", "content": "This is a very long message that should be truncated"},
            ]
        }
        report = FormatReport()
        result = formatter.format_entry(entry, report)
        assert len(result["messages"][0]["content"]) == 10


class TestSplitter:
    def test_split_ratio(self, sample_conversations):
        splitter = TrainValSplitter(val_ratio=0.5, seed=42)
        train, val, report = splitter.split(sample_conversations)
        assert len(train) + len(val) == len(sample_conversations)
        assert len(val) == 1

    def test_all_data_preserved(self, sample_conversations):
        splitter = TrainValSplitter(val_ratio=0.1, seed=42)
        train, val, report = splitter.split(sample_conversations)
        assert len(train) + len(val) == len(sample_conversations)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            TrainValSplitter(val_ratio=0)

    def test_report(self, sample_conversations):
        splitter = TrainValSplitter(val_ratio=0.5, seed=42)
        _, _, report = splitter.split(sample_conversations)
        assert report.total == 2
        assert report.train_count == 1
        assert report.val_count == 1


class TestLoadRawEntries:
    def test_load_json_array(self, tmp_path):
        data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "assistant", "content": "Hi"}]},
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        entries = load_raw_entries(f)
        assert len(entries) == 2

    def test_load_line_delimited(self, tmp_path):
        data = (
            '{"messages": [{"role": "user", "content": "Hello"}]}\n'
            '{"messages": [{"role": "assistant", "content": "Hi"}]}'
        )
        f = tmp_path / "test.json"
        f.write_text(data)
        entries = load_raw_entries(f)
        assert len(entries) == 2
