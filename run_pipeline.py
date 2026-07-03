#!/usr/bin/env python3
"""
End-to-end data pipeline: raw JSON → validated → cleaned → deduplicated → formatted → split.

Usage:
    python run_pipeline.py                           # uses defaults from configs/dataset.yaml
    python run_pipeline.py --input data/raw/chat_data.json --output-dir data/processed

Memory: < 4 GB RAM for datasets under 100K conversations.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import (
    DatasetValidator,
    TextCleaner,
    Deduplicator,
    ChatMLFormatter,
    TrainValSplitter,
)
from src.data_pipeline.validator import load_raw_entries


def main():
    parser = argparse.ArgumentParser(description="Astrology LLM Data Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/chat_data.json",
        help="Path to raw dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to write processed outputs",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum number of messages (including system)",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Exclude system prompts from output",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = base_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ASTROLOGY LLM — DATA PIPELINE")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading raw dataset...")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)
    raw_entries = load_raw_entries(input_path)
    print(f"  Loaded {len(raw_entries)} entries from {input_path}")

    # Step 2: Validate
    print("\n[2/5] Validating entries...")
    validator = DatasetValidator(min_turns=args.min_turns)
    validation_report = validator.validate(raw_entries)
    print(validation_report)

    if validation_report.valid_entries == 0:
        print("FATAL: No valid entries to process.")
        sys.exit(1)

    valid_entries = [e for i, e in enumerate(raw_entries) if i not in {inv["index"] for inv in validation_report.invalid_entries}]

    # Step 3: Clean
    print("\n[3/5] Cleaning text...")
    cleaner = TextCleaner()
    cleaned_entries, cleaning_report = cleaner.clean_all(valid_entries)
    print(cleaning_report)

    # Step 4: Deduplicate
    print("\n[4/5] Deduplicating...")
    dedup = Deduplicator()
    unique_entries, dedup_report = dedup.deduplicate(cleaned_entries)
    print(dedup_report)

    # Step 5: Format
    print("\n[5/5] Formatting to ChatML...")
    formatter = ChatMLFormatter(include_system=not args.no_system)
    formatted_entries, format_report = formatter.format_all(unique_entries)

    # Save formatted dataset
    formatted_path = output_dir / "dataset_chatml.json"
    with open(formatted_path, "w", encoding="utf-8") as f:
        json.dump(formatted_entries, f, ensure_ascii=False, indent=2)
    print(f"  Saved full dataset: {formatted_path} ({len(formatted_entries)} entries)")

    # Split train/val
    print("\n  Splitting train/validation...")
    splitter = TrainValSplitter(val_ratio=args.val_ratio, seed=args.seed)
    train_entries, val_entries, split_report = splitter.split(formatted_entries)
    print(split_report)

    # Save splits
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)
    print(f"  Saved train:   {train_path} ({len(train_entries)} entries)")
    print(f"  Saved val:     {val_path} ({len(val_entries)} entries)")

    # Save combined report
    report = {
        "validation": validation_report.summary,
        "cleaning": {
            "total_messages": cleaning_report.total_messages,
            "modified_messages": cleaning_report.modified_messages,
        },
        "deduplication": {
            "total": dedup_report.total,
            "duplicates_removed": dedup_report.exact_duplicates,
            "unique": dedup_report.unique_count,
        },
        "formatting": {
            "input": format_report.input_entries,
            "output": format_report.output_entries,
            "total_turns": format_report.total_turns,
        },
        "split": {
            "total": split_report.total,
            "train": split_report.train_count,
            "val": split_report.val_count,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
    }
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved report:  {report_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
