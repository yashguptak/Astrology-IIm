import json
from pathlib import Path

base = Path("data/processed")

with open(base / "dataset_chatml.json", encoding="utf-8") as f:
    data = json.load(f)
print(f"dataset_chatml.json: {len(data)} entries")

with open(base / "train.json", encoding="utf-8") as f:
    train = json.load(f)
print(f"train.json: {len(train)} entries")

with open(base / "val.json", encoding="utf-8") as f:
    val = json.load(f)
print(f"val.json: {len(val)} entries")

with open(base / "pipeline_report.json", encoding="utf-8") as f:
    report = json.load(f)
print(f"\nPipeline report:\n{json.dumps(report, indent=2)}")

# Verify all entries
all_valid = all(
    msgs[0]["role"] == "system" and msgs[-1]["role"] == "assistant"
    for entry in data
    if (msgs := entry.get("messages", []))
)
roles = set()
for entry in data:
    for m in entry.get("messages", []):
        roles.add(m.get("role"))
print(f"\nAll valid ChatML: {all_valid}")
print(f"Roles found: {roles}")

total_chars = sum(
    len(m.get("content", ""))
    for e in data
    for m in e.get("messages", [])
)
print(f"Total chars: {total_chars:,}")
print(f"Est tokens (4:1): {total_chars // 4:,}")

# Language distribution
langs = {"english": 0, "hindi": 0, "hinglish": 0}
for e in data:
    combined = " ".join(
        m.get("content", "")
        for m in e.get("messages", [])
        if m.get("role") == "user"
    )
    has_dev = any(0x0900 <= ord(c) <= 0x0FFF for c in combined)
    if has_dev:
        en = sum(1 for c in combined if c.isascii() and c.isalpha())
        total = sum(1 for c in combined if c.isalpha())
        if total > 0 and en / total > 0.3:
            langs["hinglish"] += 1
        else:
            langs["hindi"] += 1
    else:
        langs["english"] += 1

print("\nLanguage distribution:")
for k, v in langs.items():
    print(f"  {k}: {v} ({v/len(data)*100:.0f}%)")

# Print first entry safely
print("\nFirst entry structure:")
entry = data[0]
print(json.dumps(entry, ensure_ascii=False, indent=2)[:800])
